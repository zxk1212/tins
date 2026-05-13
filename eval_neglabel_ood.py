import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPTokenizer

from utils.common import get_test_labels, setup_seed
from utils.detection_util import get_measures, print_measures
from utils.file_ops import save_as_dataframe, setup_log
from utils.train_eval_util import set_model_clip, set_ood_loader_ImageNet, set_val_loader


OFFICIAL_NEG_PROMPT = "the nice {}"
OFFICIAL_ADJ_PROMPT = "This is a {} photo"


def process_args():
    parser = argparse.ArgumentParser(
        description="Evaluate NegLabel for ImageNet OOD detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--in_dataset",
        default="ImageNet",
        type=str,
        choices=["ImageNet"],
        help="NegLabel reproduction in this script is configured for ImageNet-1K ID.",
    )
    parser.add_argument("--root-dir", default="datasets", type=str, help="Dataset root dir.")
    parser.add_argument(
        "--wordnet-dir",
        default=None,
        type=str,
        help="Directory containing NegLabel WordNet txt files such as noun.*.txt and adj.*.txt.",
    )
    parser.add_argument("--cache-dir", default="cache/neglabel", type=str, help="Directory to cache selected negative embeddings.")
    parser.add_argument("--name", default="eval_neglabel", type=str, help="Unique ID for the run.")
    parser.add_argument("--seed", default=5, type=int, help="Random seed.")
    parser.add_argument("--gpu", default=0, type=int, help="GPU index to use.")
    parser.add_argument("-b", "--batch-size", default=256, type=int, help="Mini-batch size.")
    parser.add_argument(
        "--CLIP_ckpt",
        type=str,
        default="ViT-B/16",
        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14"],
        help="Which pretrained CLIP encoder to use.",
    )
    parser.add_argument(
        "--neg-topk",
        default=0.15,
        type=float,
        help="Fraction of noun and adjective negatives retained after ranking.",
    )
    parser.add_argument(
        "--percentile",
        "--pencentile",
        dest="percentile",
        default=0.95,
        type=float,
        help="Quantile used to rank negative labels against positive labels.",
    )
    parser.add_argument("--text-batch-size", default=1000, type=int, help="Batch size for text encoding and negative ranking.")
    parser.add_argument("--ngroup", default=100, type=int, help="Number of negative groups used by NegLabel.")
    parser.add_argument("--t", default=1.0, type=float, help="Temperature divisor used in grouped softmax.")
    parser.add_argument(
        "--logit-scale",
        default=100.0,
        type=float,
        help="Logit scale applied to cosine similarities, following the official NegLabel code.",
    )
    parser.add_argument("--pos-prompt", default=OFFICIAL_NEG_PROMPT, type=str, help="Prompt template for positive ImageNet labels.")
    parser.add_argument("--neg-prompt", default=OFFICIAL_NEG_PROMPT, type=str, help="Prompt template for noun negative labels.")
    parser.add_argument("--adj-prompt", default=OFFICIAL_ADJ_PROMPT, type=str, help="Prompt template for adjective negative labels.")
    parser.add_argument("--no-cache", action="store_true", help="Disable loading and saving cached negative embeddings.")
    parser.add_argument(
        "--no-image-feature-cache",
        action="store_true",
        help="Disable loading and saving per-dataset CLIP image feature caches.",
    )
    parser.add_argument(
        "--image-feature-cache-dir",
        default=None,
        type=str,
        help="Directory for cached image features. Default: <cache-dir>/image_features",
    )
    args = parser.parse_args()

    if not (0.0 < args.neg_topk <= 1.0):
        raise ValueError("--neg-topk must be in (0, 1].")
    if not (0.0 <= args.percentile <= 1.0):
        raise ValueError("--percentile must be in [0, 1].")
    if args.ngroup <= 0:
        raise ValueError("--ngroup must be positive.")
    if "{}" not in args.pos_prompt or "{}" not in args.neg_prompt or "{}" not in args.adj_prompt:
        raise ValueError("Prompt templates must contain '{}' once.")

    args.score = "NegLabel"
    args.model = "CLIP"
    ckpt_name = args.CLIP_ckpt.replace("/", "-")
    args.log_directory = f"results/{args.in_dataset}/{args.score}/{args.model}_{ckpt_name}_T_{args.t}_ID_{args.name}"
    os.makedirs(args.log_directory, exist_ok=True)
    if args.image_feature_cache_dir is None:
        args.image_feature_cache_dir = str(Path(args.cache_dir) / "image_features")
    return args


def resolve_wordnet_dir(args):
    candidates = []
    if args.wordnet_dir is not None:
        user_path = Path(args.wordnet_dir)
        candidates.append(user_path)
        if not user_path.is_absolute():
            candidates.append(Path(__file__).resolve().parent / user_path)

    repo_root = Path(__file__).resolve().parent
    candidates.extend(
        [
            repo_root / "txtfiles",
            Path.cwd() / "txtfiles",
        ]
    )

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not find the NegLabel WordNet txtfiles directory. "
        "Please pass --wordnet-dir or place the official txtfiles/ folder in the repo root. "
        "Official source: https://github.com/XueJiang16/NegLabel/tree/main/txtfiles"
    )


def compute_dir_signature(directory):
    digest = hashlib.sha1()
    for path in sorted(directory.glob("*.txt")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_cache_path(args, wordnet_dir):
    cache_root = Path(args.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_payload = {
        "clip_ckpt": args.CLIP_ckpt,
        "wordnet_sig": compute_dir_signature(wordnet_dir),
        "neg_topk": args.neg_topk,
        "percentile": args.percentile,
        "neg_prompt": args.neg_prompt,
        "adj_prompt": args.adj_prompt,
    }
    cache_key = hashlib.sha1(json.dumps(cache_payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return cache_root / f"neglabel_{args.in_dataset}_{cache_key}.pt"


def build_image_feature_cache_path(args, split_name, dataset_root, num_samples):
    cache_root = Path(args.image_feature_cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    root_resolved = str(Path(dataset_root).resolve()) if dataset_root else ""
    payload = {
        "clip_ckpt": args.CLIP_ckpt,
        "split": split_name,
        "root": root_resolved,
        "num_samples": num_samples,
    }
    key = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    safe_split = split_name.replace(os.sep, "_").replace("/", "_")
    return cache_root / f"imgfeat_{safe_split}_{key}.pt"


def load_or_cache_image_features(model, loader, args, log, split_name):
    dataset = loader.dataset
    dataset_root = getattr(dataset, "root", None) or ""
    num_samples = len(dataset)
    cache_path = build_image_feature_cache_path(args, split_name, dataset_root, num_samples)

    if cache_path.exists() and not args.no_image_feature_cache:
        blob = torch.load(cache_path, map_location="cpu")
        feats = blob["image_features"]
        if feats.shape[0] == num_samples:
            log.debug(f"Loaded image features for {split_name} from {cache_path}")
            return feats
        log.debug(
            f"Image feature cache size mismatch ({feats.shape[0]} vs {num_samples}), recomputing {split_name}"
        )

    device = next(model.parameters()).device
    chunks = []
    with torch.no_grad():
        for images, _ in tqdm(loader, total=len(loader), desc=f"Encode images [{split_name}]"):
            images = images.to(device)
            image_features = model.get_image_features(pixel_values=images).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            chunks.append(image_features.cpu())
    image_features = torch.cat(chunks, dim=0)

    if not args.no_image_feature_cache:
        torch.save(
            {
                "image_features": image_features,
                "meta": {
                    "clip_ckpt": args.CLIP_ckpt,
                    "split": split_name,
                    "root": str(Path(dataset_root).resolve()) if dataset_root else "",
                    "num_samples": num_samples,
                },
            },
            cache_path,
        )
        log.debug(f"Saved image features for {split_name} to {cache_path}")
    return image_features


def encode_texts(model, tokenizer, texts, batch_size, device, desc):
    all_features = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc=desc):
            batch_texts = texts[start : start + batch_size]
            text_inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            text_features = model.get_text_features(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
            ).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            all_features.append(text_features.cpu())
    return torch.cat(all_features, dim=0)


def collect_negative_texts(wordnet_dir, neg_prompt, adj_prompt):
    dedup = set()
    noun_texts = []
    adj_texts = []

    for path in sorted(wordnet_dir.glob("*.txt")):
        filetype = path.name.split(".")[0]
        if filetype not in {"noun", "adj"}:
            continue
        with path.open("r") as handle:
            for line in handle:
                word = line.strip()
                if not word or word in dedup:
                    continue
                dedup.add(word)
                if filetype == "noun":
                    noun_texts.append(neg_prompt.format(word))
                else:
                    adj_texts.append(adj_prompt.format(word))
    return noun_texts, adj_texts


def build_label_bank(args, model, tokenizer, positive_labels, log):
    device = next(model.parameters()).device
    wordnet_dir = resolve_wordnet_dir(args)
    positive_texts = [args.pos_prompt.format(label) for label in positive_labels]
    positive_features = encode_texts(
        model,
        tokenizer,
        positive_texts,
        batch_size=args.text_batch_size,
        device=device,
        desc="Encoding positive labels",
    )

    cache_path = build_cache_path(args, wordnet_dir)
    if cache_path.exists() and not args.no_cache:
        cache = torch.load(cache_path, map_location="cpu")
        negative_features = cache["negative_features"]
        selected_negative_texts = cache["selected_negative_texts"]
        log.debug(f"Loaded cached negative bank from {cache_path}")
        return positive_features.to(device), negative_features.to(device), positive_texts, selected_negative_texts, cache_path

    noun_texts, adj_texts = collect_negative_texts(wordnet_dir, args.neg_prompt, args.adj_prompt)
    log.debug(f"Collected {len(noun_texts)} noun negatives and {len(adj_texts)} adjective negatives from {wordnet_dir}")

    noun_features = encode_texts(
        model,
        tokenizer,
        noun_texts,
        batch_size=args.text_batch_size,
        device=device,
        desc="Encoding noun negatives",
    )
    adj_features = encode_texts(
        model,
        tokenizer,
        adj_texts,
        batch_size=args.text_batch_size,
        device=device,
        desc="Encoding adjective negatives",
    )

    negative_features_all = torch.cat([noun_features, adj_features], dim=0)
    positive_features_device = positive_features.to(device)
    negative_rank_scores = []
    with torch.no_grad():
        for start in tqdm(range(0, negative_features_all.shape[0], args.text_batch_size), desc="Ranking negative labels"):
            batch_features = negative_features_all[start : start + args.text_batch_size].to(device)
            similarities = batch_features @ positive_features_device.T
            score = torch.quantile(similarities.float(), q=args.percentile, dim=-1)
            negative_rank_scores.append(score.cpu())
    negative_rank_scores = torch.cat(negative_rank_scores, dim=0)

    noun_count = max(1, int(len(noun_texts) * args.neg_topk))
    adj_count = max(1, int(len(adj_texts) * args.neg_topk))
    noun_indices = torch.argsort(negative_rank_scores[: len(noun_texts)])[:noun_count]
    adj_indices = torch.argsort(negative_rank_scores[len(noun_texts) :])[:adj_count]

    selected_negative_texts = [noun_texts[idx] for idx in noun_indices.tolist()]
    selected_negative_texts.extend(adj_texts[idx] for idx in adj_indices.tolist())
    negative_features = torch.cat([noun_features[noun_indices], adj_features[adj_indices]], dim=0)

    if not args.no_cache:
        torch.save(
            {
                "negative_features": negative_features,
                "selected_negative_texts": selected_negative_texts,
            },
            cache_path,
        )
        log.debug(f"Saved negative bank cache to {cache_path}")

    return positive_features_device, negative_features.to(device), positive_texts, selected_negative_texts, cache_path


def grouped_positive_score(pos_logits, neg_logits, ngroup, temperature, permutation):
    drop = neg_logits.shape[1] % ngroup
    if drop > 0:
        neg_logits = neg_logits[:, :-drop]
        permutation = permutation[: neg_logits.shape[1]]

    if neg_logits.shape[1] == 0:
        raise RuntimeError("No negative labels remain after grouping. Increase --neg-topk or decrease --ngroup.")

    grouped_neg_logits = neg_logits[:, permutation].reshape(pos_logits.shape[0], ngroup, -1).contiguous()
    scores = []
    for group_idx in range(ngroup):
        logits = torch.cat([pos_logits, grouped_neg_logits[:, group_idx, :]], dim=-1) / temperature
        probabilities = logits.softmax(dim=-1)
        pos_mass = probabilities[:, : pos_logits.shape[1]].sum(dim=-1)
        scores.append(pos_mass.unsqueeze(-1))
    scores = torch.cat(scores, dim=-1)
    return scores.mean(dim=-1)


def compute_neglabel_scores_from_image_features(image_features, args, positive_features, negative_features):
    """image_features: CPU tensor [N, D], L2-normalized."""
    device = positive_features.device
    effective_neg_count = negative_features.shape[0] - (negative_features.shape[0] % args.ngroup)
    if effective_neg_count <= 0:
        raise RuntimeError("Selected negative labels are fewer than --ngroup.")

    negative_features = negative_features[:effective_neg_count]
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    permutation = torch.randperm(effective_neg_count, device=device)

    all_scores = []
    n = image_features.shape[0]
    with torch.no_grad():
        for start in tqdm(range(0, n, args.batch_size), desc="NegLabel scores", total=(n + args.batch_size - 1) // args.batch_size):
            batch = image_features[start : start + args.batch_size].to(device)
            pos_logits = args.logit_scale * (batch @ positive_features.T)
            neg_logits = args.logit_scale * (batch @ negative_features.T)
            batch_scores = grouped_positive_score(
                pos_logits=pos_logits,
                neg_logits=neg_logits,
                ngroup=args.ngroup,
                temperature=args.t,
                permutation=permutation,
            )
            all_scores.append(batch_scores.cpu())
    return torch.cat(all_scores, dim=0).numpy()


def get_neglabel_scores(args, model, loader, positive_features, negative_features, split_name, log):
    image_features = load_or_cache_image_features(model, loader, args, log, split_name)
    return compute_neglabel_scores_from_image_features(image_features, args, positive_features, negative_features)


def save_selected_labels(log_directory, labels):
    output_path = Path(log_directory) / "selected_neg_labels.txt"
    output_path.write_text("\n".join(labels) + "\n")


def main():
    args = process_args()
    setup_seed(args.seed)
    log = setup_log(args)

    assert torch.cuda.is_available()
    torch.cuda.set_device(args.gpu)

    net, preprocess = set_model_clip(args)
    net.eval()

    val_loader = set_val_loader(args, preprocess)
    positive_labels = [str(label) for label in get_test_labels(args, val_loader)]
    tokenizer = CLIPTokenizer.from_pretrained(args.ckpt)

    positive_features, negative_features, positive_texts, selected_negative_texts, cache_path = build_label_bank(
        args, net, tokenizer, positive_labels, log
    )
    save_selected_labels(args.log_directory, selected_negative_texts)

    log.debug(f"Positive labels: {len(positive_texts)}")
    log.debug(f"Selected negatives: {len(selected_negative_texts)}")
    log.debug(f"Negative bank cache path: {cache_path}")

    in_score = get_neglabel_scores(args, net, val_loader, positive_features, negative_features, "ImageNet_val", log)
    log.debug(
        "ImageNet score stats: min=%.6f max=%.6f mean=%.6f",
        float(np.min(in_score)),
        float(np.max(in_score)),
        float(np.mean(in_score)),
    )

    out_datasets = ["iNaturalist", "SUN", "places365", "dtd"]
    auroc_list = []
    aupr_list = []
    fpr_list = []

    for out_dataset in out_datasets:
        log.debug(f"Evaluating OOD dataset {out_dataset}")
        ood_loader = set_ood_loader_ImageNet(
            args,
            out_dataset,
            preprocess,
            root=os.path.join(args.root_dir, "ImageNet_OOD_dataset"),
        )
        out_score = get_neglabel_scores(args, net, ood_loader, positive_features, negative_features, out_dataset, log)
        log.debug(
            "%s score stats: min=%.6f max=%.6f mean=%.6f",
            out_dataset,
            float(np.min(out_score)),
            float(np.max(out_score)),
            float(np.mean(out_score)),
        )

        auroc, aupr, fpr = get_measures(in_score, out_score)
        auroc_list.append(auroc)
        aupr_list.append(aupr)
        fpr_list.append(fpr)
        print_measures(log, auroc, aupr, fpr, method_name=args.score)

    log.debug("\n\nMean Test Results")
    print_measures(log, np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.score)
    save_as_dataframe(args, out_datasets, fpr_list, auroc_list, aupr_list)


if __name__ == "__main__":
    main()
