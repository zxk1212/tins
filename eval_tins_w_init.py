import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.common import get_test_labels, setup_seed
from utils.detection_util import get_measures, print_measures
from utils.file_ops import setup_log
from utils.train_eval_util import set_ood_loader_ImageNet, set_train_loader, set_val_loader


THIRD_PARTY_DIR = Path(__file__).resolve().parent / "third_party"
if str(THIRD_PARTY_DIR) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIR))
import openai_clip as official_clip


OFFICIAL_NOUN_PROMPT = "The nice {}."
OFFICIAL_ADJ_PROMPT = "This is a {} photo."
INVERSION_PROMPT_TEMPLATE = "a photo of a {}."
INVERSION_PROMPT_TEMPLATE_NO_PERIOD = "a photo of a {}"
NEGATIVE_BANK_CACHE_VERSION = 2
INVERSION_INIT_CACHE_VERSION = 1
DEFAULT_OPENOOD_ROOT = "/disk1/yangyifeng/icml_2024/OpenOOD"
FOUR_OOD_DATASETS = ["iNaturalist", "SUN", "places365", "dtd"]
OPENOOD_IMAGENET1K_GROUPS = {
    "nearood": ["ssb_hard", "ninco"],
    "farood": ["inaturalist", "textures", "openimageo"],
}
OPENOOD_IMAGENET1K_IMGLISTS = {
    "id": "test_imagenet.txt",
    "ssb_hard": "test_ssb_hard.txt",
    "ninco": "test_ninco.txt",
    "inaturalist": "test_inaturalist.txt",
    "textures": "test_textures.txt",
    "openimageo": "test_openimage_o.txt",
}




class ImageListDataset(Dataset):
    def __init__(self, root, imglist_path, transform):
        self.root = str(root)
        self.transform = transform
        self.samples = []
        with open(imglist_path, "r") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rel_path, label = line.split()[:2]
                self.samples.append((os.path.join(self.root, rel_path), int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class MixedStreamDataset(Dataset):
    def __init__(self, id_dataset, ood_dataset, order, root):
        self.id_dataset = id_dataset
        self.ood_dataset = ood_dataset
        self.order = order
        self.root = root

    def __len__(self):
        return len(self.order)

    def __getitem__(self, index):
        is_ood, inner_index = self.order[index]
        if is_ood:
            image, _ = self.ood_dataset[inner_index]
            return image, 1
        image, _ = self.id_dataset[inner_index]
        return image, 0


def process_args():
    parser = argparse.ArgumentParser(
        description="Evaluate tins for ImageNet Four-OOD or OpenOOD ImageNet-1K mixed-stream detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--in_dataset",
        default="ImageNet",
        type=str,
        choices=["ImageNet"],
        help="tins reproduction in this script is configured for ImageNet-1K ID.",
    )
    parser.add_argument("--root-dir", default="datasets", type=str, help="Dataset root dir.")
    parser.add_argument(
        "--eval-protocol",
        default="four_ood",
        type=str,
        choices=["four_ood", "openood_imagenet1k"],
        help="Evaluation protocol. OpenOOD mode still evaluates each OOD dataset via its own mixed ID+OOD stream.",
    )
    parser.add_argument(
        "--openood-root",
        default=DEFAULT_OPENOOD_ROOT,
        type=str,
        help="Root directory of the OpenOOD repository used by the openood_imagenet1k protocol.",
    )
    parser.add_argument(
        "--wordnet-dir",
        default=None,
        type=str,
        help="Directory containing WordNet txt files such as noun.*.txt and adj.*.txt.",
    )
    parser.add_argument(
        "--train-imglist",
        default=None,
        type=str,
        help="Optional official 16-shot ImageNet imglist. If missing, fall back to the first N images per class from ImageFolder.",
    )
    parser.add_argument("--cache-dir", default="cache/tins", type=str, help="Directory for tins caches.")
    parser.add_argument("--name", default="eval_tins_init", type=str, help="Unique ID for the run.")
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    parser.add_argument("--gpu", default=0, type=int, help="GPU index to use.")
    parser.add_argument("-b", "--batch-size", default=256, type=int, help="Mini-batch size.")
    parser.add_argument(
        "--CLIP_ckpt",
        type=str,
        default="ViT-B/16",
        choices=["RN50", "RN101", "ViT-B/32", "ViT-B/16", "ViT-L/14"],
        help="Which pretrained CLIP encoder to use.",
    )
    parser.add_argument(
        "--train-shot-per-class",
        default=16,
        type=int,
        help="Number of ID images per class used to build image prototypes.",
    )
    parser.add_argument(
        "--ood-number",
        default=2000,
        type=int,
        help="Number of inter-modal selected negative texts, corresponding to M in the paper.",
    )
    parser.add_argument(
        "--extra-text-length",
        default=2000,
        type=int,
        help="Maximum number of extra negative text embeddings kept during inference, corresponding to K in the paper.",
    )
    parser.add_argument(
        "--group-num",
        default=5,
        type=int,
        help="Number of negative groups used by tins.",
    )
    parser.add_argument(
        "--ood-threshold",
        default=0.3, # 0.35
        type=float,
        help="Threshold used to trigger modality inversion, corresponding to beta in the paper.",
    )
    parser.add_argument("--tau", default=1.0, type=float, help="Temperature in the paper. Kept for parity with the paper setting.")
    parser.add_argument("--text-batch-size", default=1000, type=int, help="Batch size for text encoding.")
    parser.add_argument(
        "--pos-prompt",
        default=OFFICIAL_NOUN_PROMPT,
        type=str,
        help="Prompt template for ImageNet labels.",
    )
    parser.add_argument(
        "--noun-prompt",
        default=OFFICIAL_NOUN_PROMPT,
        type=str,
        help="Prompt template for noun negatives.",
    )
    parser.add_argument(
        "--adj-prompt",
        default=OFFICIAL_ADJ_PROMPT,
        type=str,
        help="Prompt template for adjective negatives.",
    )
    parser.add_argument(
        "--inversion-token",
        default="$",
        type=str,
        help="Placeholder token used during modality inversion.",
    )
    parser.add_argument(
        "--inversion-steps",
        default=30,
        type=int,
        help="Optimization steps used for modality inversion.",
    )
    parser.add_argument(
        "--inversion-lr",
        default=2e-2,
        type=float,
        help="Learning rate for pseudo-token optimization.",
    )
    parser.add_argument(
        "--inversion-weight-decay",
        default=1e-2,
        type=float,
        help="Weight decay for pseudo-token optimization.",
    )
    parser.add_argument(
        "--inversion-reg-lambda",
        default=0.3,
        type=float,
        help="Lambda coefficient for inversion regularization term.",
    )
    parser.add_argument(
        "--inversion-init-mode",
        default="best_neg_word",
        choices=["best_neg_word", "random"],
        help="Initialization strategy for inversion pseudo tokens.",
    )

    parser.add_argument(
        "--random-permute",
        action="store_true",
        default=True,
        help="Randomly permute negatives before grouping.",
    )
    parser.add_argument(
        "--no-random-permute",
        dest="random_permute",
        action="store_false",
        help="Disable random permutation before grouping negatives.",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable loading and saving caches.")
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
    parser.add_argument(
        "--prototype-batch-size",
        default=256,
        type=int,
        help="Batch size used when encoding prototype images.",
    )
    parser.add_argument(
        "--stream-shuffle",
        default=True,
        type=bool,
        help="Whether to shuffle the mixed stream order (ID+OOD) before feature encoding and scoring.",
    )
    parser.add_argument(
        "--stream-seed",
        default=123,
        type=int,
        help="Seed used to generate the mixed stream order when --stream-shuffle is enabled.",
    )
    parser.add_argument(
        "--save-stream-scores",
        action="store_true",
        help="Save per-sample stream scores and GT (is_ood) to <log_directory>/stream_scores_<ood>.npz.",
    )
    parser.add_argument(
        "--use-buffer",
        dest="use_buffer",
        action="store_true",
        help=(
            "Enable the overflow-buffer management for the dynamic negative bank. "
            "When disabled, the bank is still capped at --extra-text-length and overflow is discarded. "
            "When enabled, the bank is capped at --extra-text-length and overflow is stored in a buffer; "
            "once the buffer reaches --bank-buffer-size, the top-half of the buffer is mixed with the bank, "
            "and --extra-text-length items are randomly re-sampled to form the new bank (buffer is then cleared)."
        ),
    )
    parser.add_argument(
        "--no-use-buffer",
        dest="use_buffer",
        action="store_false",
        help="Disable the overflow-buffer management for the dynamic negative bank.",
    )
    parser.add_argument(
        "--bank-buffer-size",
        default=2000,
        type=int,
        help=(
            "Capacity of the overflow buffer when --use-buffer is set. Features dropped from the active bank "
            "are kept here; once the buffer reaches this size, it is shuffled together with the bank and a "
            "random subset of size --extra-text-length is promoted back into the bank, then the buffer is cleared."
        ),
    )
    parser.set_defaults(use_buffer=True)
    args = parser.parse_args()

    if args.train_shot_per_class <= 0:
        raise ValueError("--train-shot-per-class must be positive.")
    if args.ood_number <= 0:
        raise ValueError("--ood-number must be positive.")
    if args.extra_text_length <= 0:
        raise ValueError("--extra-text-length must be positive.")
    if args.group_num <= 0:
        raise ValueError("--group-num must be positive.")
    if args.inversion_steps < 0:
        raise ValueError("--inversion-steps must be positive.")
    if "{}" not in args.pos_prompt or "{}" not in args.noun_prompt or "{}" not in args.adj_prompt:
        raise ValueError("Prompt templates must contain '{}' once.")

    args.score = "tins"
    args.model = "CLIP"
    args.clip_impl = "openai_official_v1"
    ckpt_name = args.CLIP_ckpt.replace("/", "-")
    args.log_directory = (
        f"results/{args.in_dataset}/{args.score}/{args.eval_protocol}/{args.model}_{ckpt_name}_ID_{args.name}"
    )
    os.makedirs(args.log_directory, exist_ok=True)
    if args.image_feature_cache_dir is None:
        args.image_feature_cache_dir = str(Path(args.cache_dir) / "image_features")
    if args.eval_protocol == "openood_imagenet1k":
        resolve_openood_root(args)
    return args


def resolve_wordnet_dir(args):
    candidates = []
    if args.wordnet_dir is not None:
        user_path = Path(args.wordnet_dir)
        candidates.append(user_path)
        if not user_path.is_absolute():
            candidates.append(Path(__file__).resolve().parent / user_path)

    repo_root = Path(__file__).resolve().parent
    candidates.extend([repo_root / "txtfiles", Path.cwd() / "txtfiles"])

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not find the WordNet txtfiles directory. Please pass --wordnet-dir or place the txtfiles/ folder in the repo root."
    )


def resolve_train_imglist(args):
    candidates = []
    if args.train_imglist is not None:
        user_path = Path(args.train_imglist)
        candidates.append(user_path)
        if not user_path.is_absolute():
            candidates.append(Path(__file__).resolve().parent / user_path)

    repo_root = Path(__file__).resolve().parent
    candidates.extend(
        [
            repo_root / "data" / "benchmark_imglist" / "imagenet" / "train_imagenet_16.txt",
            Path.cwd() / "data" / "benchmark_imglist" / "imagenet" / "train_imagenet_16.txt",
        ]
    )

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    return None


def resolve_openood_root(args):
    user_path = Path(args.openood_root)
    candidates = [user_path]
    if not user_path.is_absolute():
        candidates.append(Path(__file__).resolve().parent / user_path)
        candidates.append(Path.cwd() / user_path)

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        f"Could not find OpenOOD root from --openood-root={args.openood_root!r}. "
        "Please point it to the OpenOOD repo directory."
    )


def compute_dir_signature(directory):
    digest = hashlib.sha1()
    for path in sorted(directory.glob("*.txt")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def compute_file_signature(path):
    digest = hashlib.sha1()
    digest.update(Path(path).name.encode("utf-8"))
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def build_cache_path(cache_root, prefix, payload):
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return cache_root / f"{prefix}_{cache_key}.pt"


def compute_words_signature(words):
    digest = hashlib.sha1()
    for word in words:
        digest.update(word.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def compute_tensor_signature(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha1()
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def build_image_feature_cache_path(args, split_name, dataset_root, num_samples):
    cache_root = Path(args.image_feature_cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    root_resolved = str(Path(dataset_root).resolve()) if dataset_root else ""
    payload = {
        "clip_impl": args.clip_impl,
        "clip_ckpt": args.CLIP_ckpt,
        "split": split_name,
        "root": root_resolved,
        "num_samples": num_samples,
    }
    key = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    safe_split = split_name.replace(os.sep, "_").replace("/", "_")
    return cache_root / f"imgfeat_{safe_split}_{key}.pt"


def get_model_embed_dim(model):
    return int(model.text_projection.shape[1])


def get_model_word_embed_dim(model):
    return int(model.token_embedding.weight.shape[1])


def load_official_clip(args):
    model, preprocess = official_clip.load(args.CLIP_ckpt, device="cuda", jit=False)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, preprocess


def build_imglist_loader(dataset_root, imglist_path, preprocess, batch_size, num_workers=4):
    dataset_root = Path(dataset_root).resolve()
    imglist_path = Path(imglist_path).resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not imglist_path.is_file():
        raise FileNotFoundError(f"Imglist file does not exist: {imglist_path}")
    dataset = ImageListDataset(dataset_root, imglist_path, preprocess)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_openood_imagenet1k_eval_setup(args, preprocess):
    openood_root = resolve_openood_root(args)
    benchmark_dir = openood_root / "data" / "benchmark_imglist" / "imagenet"
    images_largescale_root = openood_root / "data" / "images_largescale"
    images_classic_root = openood_root / "data" / "images_classic"

    id_loader = build_imglist_loader(
        dataset_root=images_largescale_root,
        imglist_path=benchmark_dir / OPENOOD_IMAGENET1K_IMGLISTS["id"],
        preprocess=preprocess,
        batch_size=args.batch_size,
    )

    eval_specs = []
    for group_name, dataset_names in OPENOOD_IMAGENET1K_GROUPS.items():
        for dataset_name in dataset_names:
            dataset_root = images_classic_root if dataset_name == "textures" else images_largescale_root
            eval_specs.append(
                {
                    "dataset_name": dataset_name,
                    "group_name": group_name,
                    "stream_name": f"openood_{group_name}_{dataset_name}",
                    "ood_loader": build_imglist_loader(
                        dataset_root=dataset_root,
                        imglist_path=benchmark_dir / OPENOOD_IMAGENET1K_IMGLISTS[dataset_name],
                        preprocess=preprocess,
                        batch_size=args.batch_size,
                    ),
                }
            )
    return id_loader, eval_specs


def build_eval_setup(args, preprocess):
    if args.eval_protocol == "four_ood":
        id_loader = set_val_loader(args, preprocess)
        eval_specs = []
        for dataset_name in FOUR_OOD_DATASETS:
            eval_specs.append(
                {
                    "dataset_name": dataset_name,
                    "group_name": "all",
                    "stream_name": f"four_ood_{dataset_name}",
                    "ood_loader": set_ood_loader_ImageNet(
                        args,
                        dataset_name,
                        preprocess,
                        root=os.path.join(args.root_dir, "ImageNet_OOD_dataset"),
                    ),
                }
            )
        return id_loader, eval_specs
    if args.eval_protocol == "openood_imagenet1k":
        return build_openood_imagenet1k_eval_setup(args, preprocess)
    raise ValueError(f"Unsupported eval protocol: {args.eval_protocol}")


def metric_triplet_to_percent(fpr, auroc, aupr):
    return [float(f"{100 * metric:.2f}") for metric in (fpr, auroc, aupr)]


def compute_mean_metrics(result_rows):
    if not result_rows:
        raise ValueError("Cannot compute mean metrics from an empty result list.")
    return {
        "fpr": float(np.mean([row["fpr"] for row in result_rows])),
        "auroc": float(np.mean([row["auroc"] for row in result_rows])),
        "aupr": float(np.mean([row["aupr"] for row in result_rows])),
    }


def save_metric_rows(args, dataset_rows, summary_rows=None):
    data = {}
    for row in dataset_rows:
        data[row["name"]] = metric_triplet_to_percent(row["fpr"], row["auroc"], row["aupr"])
    for row in summary_rows or []:
        data[row["name"]] = metric_triplet_to_percent(row["fpr"], row["auroc"], row["aupr"])
    avg_metrics = compute_mean_metrics(dataset_rows)
    data["AVG"] = metric_triplet_to_percent(avg_metrics["fpr"], avg_metrics["auroc"], avg_metrics["aupr"])
    df = pd.DataFrame.from_dict(data, orient="index", columns=["FPR95", "AUROC", "AUPR"])
    df.to_csv(Path(args.log_directory) / f"{args.name}.csv")


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
            image_features = model.encode_image(images).float()
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


def load_or_cache_stream_features_and_gt(model, loader, args, log, split_name):
    dataset = loader.dataset
    dataset_root = getattr(dataset, "root", None) or ""
    num_samples = len(dataset)
    cache_path = build_image_feature_cache_path(args, f"{split_name}_stream", dataset_root, num_samples)

    if cache_path.exists() and not args.no_image_feature_cache:
        blob = torch.load(cache_path, map_location="cpu")
        feats = blob.get("image_features")
        gt = blob.get("is_ood")
        if feats is not None and gt is not None and feats.shape[0] == num_samples and gt.shape[0] == num_samples:
            log.debug(f"Loaded stream features+gt for {split_name} from {cache_path}")
            return feats, gt.numpy().astype(np.int32)
        log.debug(f"Stream cache mismatch, recomputing {split_name}")

    device = next(model.parameters()).device
    chunks = []
    gts = []
    with torch.no_grad():
        for images, is_ood in tqdm(loader, total=len(loader), desc=f"Encode stream [{split_name}]"):
            images = images.to(device)
            image_features = model.encode_image(images).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            chunks.append(image_features.cpu())
            gts.append(is_ood.cpu().to(torch.int32))
    image_features = torch.cat(chunks, dim=0)
    is_ood = torch.cat(gts, dim=0)

    if not args.no_image_feature_cache:
        torch.save(
            {
                "image_features": image_features,
                "is_ood": is_ood,
                "meta": {
                    "clip_ckpt": args.CLIP_ckpt,
                    "split": split_name,
                    "root": str(Path(dataset_root).resolve()) if dataset_root else "",
                    "num_samples": num_samples,
                },
            },
            cache_path,
        )
        log.debug(f"Saved stream features+gt for {split_name} to {cache_path}")

    return image_features, is_ood.numpy().astype(np.int32)


def encode_texts(model, texts, batch_size, device, desc):
    all_features = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc=desc):
            batch_texts = texts[start : start + batch_size]
            tokenized = official_clip.tokenize(batch_texts, truncate=True).to(device)
            text_features = model.encode_text(tokenized).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            all_features.append(text_features.cpu())
    return torch.cat(all_features, dim=0)


def unique_preserve_order(values):
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def collect_negative_words(wordnet_dir, positive_labels):
    noun_words = []
    adj_words = []
    for path in sorted(wordnet_dir.glob("*.txt")):
        filetype = path.name.split(".")[0]
        if filetype not in {"noun", "adj"}:
            continue
        words = []
        with path.open("r") as handle:
            for line in handle:
                word = line.strip()
                if word:
                    words.append(word)
        if filetype == "noun":
            noun_words.extend(words)
        else:
            adj_words.extend(words)

    positive_set = set(positive_labels)
    noun_words = [word for word in unique_preserve_order(noun_words) if word not in positive_set]
    adj_words = [word for word in unique_preserve_order(adj_words) if word not in positive_set]
    noun_set = set(noun_words)
    adj_words = [word for word in adj_words if word not in noun_set]
    return noun_words, adj_words


def encode_words(model, words, prompt_template, batch_size, device, desc):
    texts = [prompt_template.format(word) for word in words]
    return encode_texts(model, texts, batch_size, device, desc)


def score_intermodal_candidates(candidate_features, class_prototypes, base_sim):
    similarities = candidate_features @ class_prototypes.T
    valid_mask = torch.all(similarities < base_sim.unsqueeze(0), dim=1)
    scores = (base_sim.unsqueeze(0) - similarities).mean(dim=1)
    return valid_mask, scores


def select_top_words(words, features, scores, count):
    if count <= 0 or len(words) == 0:
        return [], torch.empty((0, features.shape[1]), dtype=features.dtype)
    count = min(count, len(words))
    top_indices = torch.topk(scores, k=count, largest=True).indices.cpu().tolist()
    selected_words = [words[index] for index in top_indices]
    selected_features = features[top_indices]
    return selected_words, selected_features


def load_or_build_class_prototypes(args, model, preprocess, log):
    train_imglist = resolve_train_imglist(args)
    if train_imglist is not None:
        prototype_source = {
            "mode": "imglist",
            "imglist_sig": compute_file_signature(train_imglist),
            "imglist_path": str(train_imglist),
        }
    else:
        prototype_source = {
            "mode": "first_n_imagefolder",
            "shot_per_class": args.train_shot_per_class,
            "root": str(Path(args.root_dir).resolve()),
        }

    cache_path = build_cache_path(
        Path(args.cache_dir) / "class_prototypes",
        "class_prototypes",
        {
            "clip_impl": args.clip_impl,
            "clip_ckpt": args.CLIP_ckpt,
            "in_dataset": args.in_dataset,
            "source": prototype_source,
        },
    )
    if cache_path.exists() and not args.no_cache:
        blob = torch.load(cache_path, map_location="cpu")
        log.debug(f"Loaded class prototypes from {cache_path}")
        return blob["class_prototypes"], blob["meta"], cache_path

    if train_imglist is not None:
        dataset_root = Path(args.root_dir) / "ImageNet"
        dataset = ImageListDataset(dataset_root, train_imglist, preprocess)
        loader = DataLoader(
            dataset,
            batch_size=args.prototype_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        meta = {
            "mode": "imglist",
            "train_imglist": str(train_imglist),
            "num_samples": len(dataset),
        }
        log.debug(f"Building class prototypes from official imglist {train_imglist}")
    else:
        max_count = getattr(args, "max_count", None)
        args.max_count = args.train_shot_per_class
        loader = set_train_loader(
            args,
            preprocess=preprocess,
            batch_size=args.prototype_batch_size,
            shuffle=False,
            subset=True,
        )
        if max_count is None:
            delattr(args, "max_count")
        else:
            args.max_count = max_count
        meta = {
            "mode": "first_n_imagefolder",
            "shot_per_class": args.train_shot_per_class,
            "num_samples": len(loader.dataset),
        }
        log.debug("Building class prototypes from ImageFolder order fallback")

    num_classes = len(get_test_labels(args, None))
    device = next(model.parameters()).device
    class_sum = torch.zeros((num_classes, get_model_embed_dim(model)), dtype=torch.float32)
    class_count = torch.zeros(num_classes, dtype=torch.long)

    with torch.no_grad():
        for images, labels in tqdm(loader, total=len(loader), desc="Build class prototypes"):
            images = images.to(device)
            labels = labels.to(device)
            image_features = model.encode_image(images).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            for index in range(labels.shape[0]):
                label = int(labels[index].item())
                class_sum[label] += image_features[index].cpu()
                class_count[label] += 1

    missing_classes = torch.nonzero(class_count == 0).flatten().tolist()
    if missing_classes:
        raise RuntimeError(f"Missing prototype samples for classes: {missing_classes[:10]}")

    class_prototypes = class_sum / class_count.unsqueeze(1)
    class_prototypes /= class_prototypes.norm(dim=-1, keepdim=True)

    if not args.no_cache:
        torch.save({"class_prototypes": class_prototypes, "meta": meta}, cache_path)
        log.debug(f"Saved class prototypes to {cache_path}")
    return class_prototypes, meta, cache_path


def load_or_build_negative_bank(args, model, positive_labels, positive_features, class_prototypes, log):
    wordnet_dir = resolve_wordnet_dir(args)
    prototype_meta = {
        "mode": "unknown",
        "train_shot_per_class": args.train_shot_per_class,
    }
    train_imglist = resolve_train_imglist(args)
    if train_imglist is not None:
        prototype_meta["mode"] = "imglist"
        prototype_meta["train_imglist_sig"] = compute_file_signature(train_imglist)
    else:
        prototype_meta["mode"] = "first_n_imagefolder"

    cache_path = build_cache_path(
        Path(args.cache_dir) / "negative_bank",
        "tins_bank",
        {
            "cache_version": NEGATIVE_BANK_CACHE_VERSION,
            "clip_impl": args.clip_impl,
            "clip_ckpt": args.CLIP_ckpt,
            "wordnet_sig": compute_dir_signature(wordnet_dir),
            "ood_number": args.ood_number,
            "pos_prompt": args.pos_prompt,
            "noun_prompt": args.noun_prompt,
            "adj_prompt": args.adj_prompt,
            "prototype_meta": prototype_meta,
        },
    )

    if cache_path.exists() and not args.no_cache:
        cache = torch.load(cache_path, map_location="cpu")
        log.debug(f"Loaded tins negative bank from {cache_path}")
        return cache["negative_features"], cache["selected_negative_texts"], cache["selected_negative_words"], cache_path

    noun_words, adj_words = collect_negative_words(wordnet_dir, positive_labels)
    log.debug(
        f"Collected {len(noun_words)} noun candidates and {len(adj_words)} adjective candidates from {wordnet_dir}"
    )

    device = next(model.parameters()).device
    base_sim = (positive_features * class_prototypes.to(device)).sum(dim=1).cpu()

    adj_features = encode_words(
        model,
        adj_words,
        prompt_template=args.adj_prompt,
        batch_size=args.text_batch_size,
        device=device,
        desc="Encoding adjective candidates",
    )
    adj_mask, adj_scores = score_intermodal_candidates(adj_features, class_prototypes, base_sim)
    selected_adj_words_all = [word for word, keep in zip(adj_words, adj_mask.tolist()) if keep]
    selected_adj_scores_all = adj_scores[adj_mask]
    selected_adj_features_all = adj_features[adj_mask]

    noun_features = encode_words(
        model,
        noun_words,
        prompt_template=args.noun_prompt,
        batch_size=args.text_batch_size,
        device=device,
        desc="Encoding noun candidates",
    )
    noun_mask, noun_scores = score_intermodal_candidates(noun_features, class_prototypes, base_sim)
    selected_noun_words_all = [word for word, keep in zip(noun_words, noun_mask.tolist()) if keep]
    selected_noun_scores_all = noun_scores[noun_mask]
    selected_noun_features_all = noun_features[noun_mask]

    total_selected = len(selected_adj_words_all) + len(selected_noun_words_all)
    if total_selected == 0:
        raise RuntimeError("No negative texts satisfy the tins inter-modal criterion.")

    adj_count = int(args.ood_number * (len(selected_adj_words_all) / total_selected))
    noun_count = args.ood_number - adj_count
    if len(selected_adj_words_all) > 0:
        adj_count = max(1, adj_count)
    if len(selected_noun_words_all) > 0:
        noun_count = max(1, noun_count)

    selected_adj_words, selected_adj_features = select_top_words(
        selected_adj_words_all,
        selected_adj_features_all,
        selected_adj_scores_all,
        adj_count,
    )
    selected_noun_words, selected_noun_features = select_top_words(
        selected_noun_words_all,
        selected_noun_features_all,
        selected_noun_scores_all,
        noun_count,
    )

    selected_negative_texts = [args.adj_prompt.format(word) for word in selected_adj_words]
    selected_negative_texts.extend(args.noun_prompt.format(word) for word in selected_noun_words)
    selected_negative_words = list(selected_adj_words)
    selected_negative_words.extend(selected_noun_words)
    negative_features = torch.cat([selected_adj_features, selected_noun_features], dim=0)

    if negative_features.shape[0] == 0:
        raise RuntimeError("Failed to build the fixed negative text bank.")

    if not args.no_cache:
        torch.save(
            {
                "negative_features": negative_features,
                "selected_negative_texts": selected_negative_texts,
                "selected_negative_words": selected_negative_words,
            },
            cache_path,
        )
        log.debug(f"Saved tins negative bank to {cache_path}")

    return negative_features, selected_negative_texts, selected_negative_words, cache_path


def get_placeholder_token_id(inversion_token):
    placeholder_ids = official_clip.tokenize(inversion_token)[0]
    placeholder_ids = placeholder_ids[placeholder_ids != 0].tolist()
    if len(placeholder_ids) < 3:
        raise RuntimeError(f"Could not find a standalone token id for inversion token '{inversion_token}'.")
    return placeholder_ids[1]


def build_inversion_template(token_or_word):
    return INVERSION_PROMPT_TEMPLATE.format(token_or_word)


def build_placeholder_inversion_template(inversion_token):
    return f"a photo of a {inversion_token}."


def get_trainable_positions_from_placeholder(tokenized_texts, placeholder_token_id):
    placeholder_positions = (tokenized_texts == placeholder_token_id).nonzero(as_tuple=False)
    if placeholder_positions.shape[0] != tokenized_texts.shape[0]:
        raise RuntimeError("Each inversion prompt must contain exactly one placeholder token.")
    return placeholder_positions[:, 1]


def get_last_trainable_token_positions(tokenized_texts, tokenized_without_period):
    token_counts_no_period = (tokenized_without_period != 0).sum(dim=1)
    target_positions = token_counts_no_period - 2
    if torch.any(target_positions < 1):
        raise RuntimeError("Failed to locate the last content token for inversion initialization.")

    token_counts = (tokenized_texts != 0).sum(dim=1)
    if torch.any(target_positions >= (token_counts - 1)):
        raise RuntimeError("Trainable token position falls outside the tokenized inversion prompt.")
    return target_positions.long()


def gather_token_embeddings(model, tokenized_texts, target_positions):
    token_embeddings = model.token_embedding(tokenized_texts).type(model.dtype)
    row_indices = torch.arange(tokenized_texts.shape[0], device=tokenized_texts.device)
    return token_embeddings[row_indices, target_positions]


def encode_with_pseudo_tokens(model, tokenized_texts, pseudo_tokens, target_positions):
    x = model.token_embedding(tokenized_texts).type(model.dtype)
    x = x.clone()
    row_indices = torch.arange(tokenized_texts.shape[0], device=tokenized_texts.device)
    x[row_indices, target_positions] = pseudo_tokens.to(x.dtype)
    x = x + model.positional_embedding.type(model.dtype)
    x = x.permute(1, 0, 2)
    x = model.transformer(x)
    x = x.permute(1, 0, 2)
    x = model.ln_final(x).type(model.dtype)
    x = x[torch.arange(x.shape[0], device=x.device), tokenized_texts.argmax(dim=-1)] @ model.text_projection
    return x


def build_inversion_init_candidates(args, model, selected_negative_words, class_prototypes, device, log):
    if len(selected_negative_words) == 0:
        raise RuntimeError("Cannot build inversion initializations from an empty fixed negative bank.")

    cache_path = build_cache_path(
        Path(args.cache_dir) / "inversion_init",
        "tins_init_candidates",
        {
            "cache_version": INVERSION_INIT_CACHE_VERSION,
            "clip_impl": args.clip_impl,
            "clip_ckpt": args.CLIP_ckpt,
            "text_batch_size": args.text_batch_size,
            "negative_words_sig": compute_words_signature(selected_negative_words),
            "class_prototypes_sig": compute_tensor_signature(class_prototypes),
        },
    )
    if cache_path.exists() and not args.no_cache:
        cache = torch.load(cache_path, map_location="cpu")
        log.debug(f"Loaded inversion init candidates from {cache_path}")
        cache["candidate_features"] = cache["candidate_features"].to(device)
        cache["candidate_reg_loss"] = cache["candidate_reg_loss"].to(device)
        return cache

    candidate_texts = [build_inversion_template(word) for word in selected_negative_words]
    candidate_texts_no_period = [INVERSION_PROMPT_TEMPLATE_NO_PERIOD.format(word) for word in selected_negative_words]
    tokenized_texts = official_clip.tokenize(candidate_texts, truncate=True)
    tokenized_without_period = official_clip.tokenize(candidate_texts_no_period, truncate=True)
    target_positions = get_last_trainable_token_positions(tokenized_texts, tokenized_without_period)

    tokenized_texts_device = tokenized_texts.to(device)
    target_positions_device = target_positions.to(device)
    candidate_feature_batches = []
    with torch.no_grad():
        init_vectors = gather_token_embeddings(model, tokenized_texts_device, target_positions_device).float().cpu()
        for start in tqdm(
            range(0, tokenized_texts_device.shape[0], args.text_batch_size),
            desc="Encoding inversion init candidates",
        ):
            batch_tokens = tokenized_texts_device[start : start + args.text_batch_size]
            batch_features = model.encode_text(batch_tokens).float()
            batch_features /= batch_features.norm(dim=-1, keepdim=True)
            candidate_feature_batches.append(batch_features.cpu())
        candidate_features_cpu = torch.cat(candidate_feature_batches, dim=0)
        candidate_reg_loss_cpu = (1 + (candidate_features_cpu @ class_prototypes.detach().cpu().T)).mean(dim=1)

    result = {
        "tokenized_texts": tokenized_texts,
        "target_positions": target_positions,
        "init_vectors": init_vectors,
        "candidate_features": candidate_features_cpu.to(device),
        "candidate_reg_loss": candidate_reg_loss_cpu.to(device),
    }
    if not args.no_cache:
        torch.save(
            {
                "tokenized_texts": tokenized_texts,
                "target_positions": target_positions,
                "init_vectors": init_vectors,
                "candidate_features": candidate_features_cpu,
                "candidate_reg_loss": candidate_reg_loss_cpu,
            },
            cache_path,
        )
        log.debug(f"Saved inversion init candidates to {cache_path}")
    return result


def initialize_inversion_state(args, model, batch_im_features, placeholder_token_id, init_candidates):
    device = batch_im_features.device
    batch_size = batch_im_features.shape[0]
    embedding_dim = get_model_word_embed_dim(model)

    if args.inversion_init_mode == "random":
        tokenized_texts = official_clip.tokenize(
            [build_placeholder_inversion_template(args.inversion_token)] * batch_size,
            truncate=True,
        ).to(device)
        target_positions = get_trainable_positions_from_placeholder(tokenized_texts, placeholder_token_id)
        pseudo_tokens = torch.empty((batch_size, embedding_dim), device=device, dtype=torch.float32)
        nn.init.normal_(pseudo_tokens, std=0.02)
        return tokenized_texts, target_positions, pseudo_tokens

    if init_candidates is None:
        raise RuntimeError("best_neg_word initialization requires fixed-bank inversion candidates.")

    objective = 1.0 - (batch_im_features.float() @ init_candidates["candidate_features"].T)
    objective = objective + (args.inversion_reg_lambda * init_candidates["candidate_reg_loss"].unsqueeze(0))
    best_indices = torch.argmin(objective, dim=1).cpu()

    tokenized_texts = init_candidates["tokenized_texts"][best_indices].to(device)
    target_positions = init_candidates["target_positions"][best_indices].to(device)
    pseudo_tokens = init_candidates["init_vectors"][best_indices].to(device)
    return tokenized_texts, target_positions, pseudo_tokens


def compute_grouped_positive_score(image_features, positive_features, negative_features, logit_scale, group_num, random_permute):
    pos_logits = logit_scale * (image_features @ positive_features.T)
    neg_logits = logit_scale * (image_features @ negative_features.T)

    drop = neg_logits.shape[1] % group_num
    if drop > 0:
        neg_logits = neg_logits[:, :-drop]

    if neg_logits.shape[1] == 0:
        raise RuntimeError("No negative labels remain after grouping.")

    if random_permute:
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        permutation = torch.randperm(neg_logits.shape[1], device=image_features.device)
        neg_logits = neg_logits[:, permutation]

    grouped_neg_logits = neg_logits.reshape(pos_logits.shape[0], group_num, -1).contiguous()
    scores = []
    class_count = pos_logits.shape[1]
    log_class_count = torch.log(
        torch.tensor(float(class_count), device=pos_logits.device, dtype=pos_logits.dtype)
    )
    for group_index in range(group_num):
        group_neg_logits = grouped_neg_logits[:, group_index, :]
        group_neg_count = group_neg_logits.shape[1]
        log_pos_mass = torch.logsumexp(pos_logits, dim=-1)
        log_neg_mean_mass = torch.logsumexp(group_neg_logits, dim=-1) - np.log(float(group_neg_count))
        log_scaled_neg_mass = log_neg_mean_mass + log_class_count
        log_denom = torch.logaddexp(log_pos_mass, log_scaled_neg_mass)
        pos_mass = torch.exp(log_pos_mass - log_denom)
        scores.append(pos_mass.unsqueeze(-1))
    scores = torch.cat(scores, dim=-1)
    return scores.mean(dim=-1)





def build_inversion_templates(inversion_token):
    return [build_inversion_template(inversion_token)]


def maybe_expand_dynamic_bank(
    args,
    model,
    image_features,
    current_scores,
    class_prototypes,
    base_sim,
    fixed_text_bank,
    bank_features,
    bank_scores,
    placeholder_token_id,
    init_candidates,
    buffer_features=None,
    buffer_scores=None,
):
    # # NOTE: 用户要求关闭 neglabel 动态扩张（dynamic bank expansion）。
    # # 这里直接短路返回，保持函数签名与调用逻辑不变。
    # return None, bank_features, bank_scores, buffer_features, buffer_scores

    use_buffer = bool(getattr(args, "use_buffer", False))

    activate_indicator = current_scores < args.ood_threshold
    if not torch.any(activate_indicator):
        return None, bank_features, bank_scores, buffer_features, buffer_scores

    batch_im_features = image_features[activate_indicator]
    tokenized_texts, target_positions, pseudo_tokens = initialize_inversion_state(
        args=args,
        model=model,
        batch_im_features=batch_im_features,
        placeholder_token_id=placeholder_token_id,
        init_candidates=init_candidates,
    )
    pseudo_tokens = nn.Parameter(pseudo_tokens)
    optimizer = torch.optim.AdamW(
        [pseudo_tokens],
        lr=args.inversion_lr,
        weight_decay=args.inversion_weight_decay,
    )
    class_prototypes = class_prototypes.to(image_features.device)
    base_sim = base_sim.to(image_features.device)

    for _ in range(args.inversion_steps):
        optimizer.zero_grad(set_to_none=True)
        template_features = encode_with_pseudo_tokens(model, tokenized_texts, pseudo_tokens, target_positions).float()
        template_features = template_features / template_features.norm(dim=-1, keepdim=True)

        cos_sim = F.cosine_similarity(template_features, batch_im_features.float(), dim=-1)
        loss_align = (1.0 - cos_sim).mean()

        intermodal_sim = F.cosine_similarity(
            template_features.unsqueeze(1),
            class_prototypes.unsqueeze(0),
            dim=-1,
        )
        loss_reg = (1+intermodal_sim).mean()
  

        loss = loss_align + (args.inversion_reg_lambda * loss_reg)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        template_features = encode_with_pseudo_tokens(model, tokenized_texts, pseudo_tokens, target_positions).float()
        template_features = template_features / template_features.norm(dim=-1, keepdim=True)

        intermodal_sim = template_features @ class_prototypes.T
        mask = torch.all(intermodal_sim < base_sim.unsqueeze(0), dim=1)
        if mask.numel() == 0 or not torch.any(mask):
            return None, bank_features, bank_scores, buffer_features, buffer_scores

        selected_features = template_features[mask].detach().cpu()
        selected_scores = (base_sim.unsqueeze(0) - intermodal_sim[mask]).mean(dim=1).detach().cpu()

        combined_features = torch.cat([bank_features, selected_features.half()], dim=0)
        combined_scores = torch.cat([bank_scores, selected_scores], dim=0)
        sorted_scores, sorted_indices = torch.sort(combined_scores, descending=True)
        sorted_features = combined_features[sorted_indices]

        # 1) 把新候选合并到 bank 中并按分数排序，bank 容量限制为 extra_text_length。
        #    开启 buffer 时，超出的样本作为"溢出"进入缓冲区；关闭时直接丢弃。
        keep = args.extra_text_length
        if sorted_features.shape[0] > keep:
            bank_features = sorted_features[:keep]
            bank_scores = sorted_scores[:keep]
            if use_buffer:
                overflow_features = sorted_features[keep:]
                overflow_scores = sorted_scores[keep:]
                buffer_features = torch.cat([buffer_features, overflow_features], dim=0)
                buffer_scores = torch.cat([buffer_scores, overflow_scores], dim=0)
        else:
            bank_features = sorted_features
            bank_scores = sorted_scores

        # 2) 缓冲区写满后：取缓冲区中 score 最高的前一半 feature 与 bank 合并 shuffle，
        #    随机抽取 extra_text_length 个样本作为新的 bank，然后清空整个缓冲区。
        buffer_capacity = int(getattr(args, "bank_buffer_size", 2000))
        if use_buffer and buffer_capacity > 0 and buffer_features.shape[0] >= buffer_capacity:
            half_size = buffer_features.shape[0] // 2
            if half_size > 0:
                buffer_sorted_scores, buffer_sorted_idx = torch.sort(
                    buffer_scores, descending=True
                )
                top_half_idx = buffer_sorted_idx[:half_size]
                top_half_features = buffer_features[top_half_idx]
                top_half_scores = buffer_sorted_scores[:half_size]
                pool_features = torch.cat([bank_features, top_half_features], dim=0)
                pool_scores = torch.cat([bank_scores, top_half_scores], dim=0)
            else:
                pool_features = bank_features
                pool_scores = bank_scores

            sample_size = min(args.extra_text_length, pool_features.shape[0])
            perm = torch.randperm(pool_features.shape[0])
            sampled_idx = perm[:sample_size]
            bank_features = pool_features[sampled_idx]
            bank_scores = pool_scores[sampled_idx]
            buffer_features = torch.zeros(
                (0, bank_features.shape[1]), dtype=buffer_features.dtype
            )
            buffer_scores = torch.zeros(0, dtype=buffer_scores.dtype)

        dynamic_features = bank_features[: args.extra_text_length].float().to(image_features.device)
        updated_text_bank = torch.cat([fixed_text_bank, dynamic_features], dim=0)
        return updated_text_bank, bank_features, bank_scores, buffer_features, buffer_scores


def compute_tins_scores_from_image_features(
    image_features,
    args,
    model,
    positive_features,
    negative_features,
    inversion_init_candidates,
    class_prototypes,
    base_sim,
):
    device = positive_features.device
    logit_scale = float(model.logit_scale.exp().detach().cpu())
    fixed_text_bank = torch.cat([positive_features, negative_features], dim=0)
    fixed_negative_features = fixed_text_bank[positive_features.shape[0] :]
    placeholder_token_id = None
    if args.inversion_init_mode == "random":
        placeholder_token_id = get_placeholder_token_id(args.inversion_token)

    feature_dim = positive_features.shape[1]
    bank_features = torch.zeros((0, feature_dim), dtype=torch.float16)
    bank_scores = torch.zeros(0, dtype=torch.float32)
    buffer_features = torch.zeros((0, feature_dim), dtype=torch.float16)
    buffer_scores = torch.zeros(0, dtype=torch.float32)
    all_scores = []
    total_steps = (image_features.shape[0] + args.batch_size - 1) // args.batch_size

    for start in tqdm(range(0, image_features.shape[0], args.batch_size), desc="tins scores", total=total_steps):
        batch = image_features[start : start + args.batch_size].to(device)

        if bank_features.shape[0] > 0:
            dynamic_features = bank_features[: args.extra_text_length].float().to(device)
            text_bank = torch.cat([fixed_text_bank, dynamic_features], dim=0)
            current_negative_features = text_bank[positive_features.shape[0] :]
        else:
            text_bank = fixed_text_bank
            current_negative_features = fixed_negative_features

        batch_scores = compute_grouped_positive_score(
            image_features=batch,
            positive_features=positive_features,
            negative_features=current_negative_features,
            # negative_features=fixed_negative_features,
            logit_scale=logit_scale,
            group_num=args.group_num,
            random_permute=args.random_permute,
        )

        (
            updated_text_bank,
            bank_features,
            bank_scores,
            buffer_features,
            buffer_scores,
        ) = maybe_expand_dynamic_bank(
            args=args,
            model=model,
            image_features=batch,
            current_scores=batch_scores,
            class_prototypes=class_prototypes,
            base_sim=base_sim,
            fixed_text_bank=fixed_text_bank,
            bank_features=bank_features,
            bank_scores=bank_scores,
            placeholder_token_id=placeholder_token_id,
            init_candidates=inversion_init_candidates,
            buffer_features=buffer_features,
            buffer_scores=buffer_scores,
        )

        if updated_text_bank is not None:
            current_negative_features = updated_text_bank[positive_features.shape[0] :]
            batch_scores = compute_grouped_positive_score(
                image_features=batch,
                positive_features=positive_features,
                negative_features=current_negative_features,
                logit_scale=logit_scale,
                group_num=args.group_num,
                random_permute=args.random_permute,
            )

        all_scores.append(batch_scores.cpu())

    return torch.cat(all_scores, dim=0).numpy()


def get_tins_scores(
    args,
    model,
    loader,
    positive_features,
    negative_features,
    inversion_init_candidates,
    class_prototypes,
    base_sim,
    split_name,
    log,
):
    image_features = load_or_cache_image_features(model, loader, args, log, split_name)
    return compute_tins_scores_from_image_features(
        image_features=image_features,
        args=args,
        model=model,
        positive_features=positive_features,
        negative_features=negative_features,
        inversion_init_candidates=inversion_init_candidates,
        class_prototypes=class_prototypes,
        base_sim=base_sim,
    )


def build_mixed_stream_loader(args, id_loader, ood_loader, stream_name):
    id_dataset = id_loader.dataset
    ood_dataset = ood_loader.dataset
    order = [(0, i) for i in range(len(id_dataset))] + [(1, i) for i in range(len(ood_dataset))]
    if args.stream_shuffle:
        rng = random.Random(args.stream_seed)
        rng.shuffle(order)
    root = f"mixed_stream::{stream_name}::id={getattr(id_dataset, 'root', '')}::ood={getattr(ood_dataset, 'root', '')}"
    dataset = MixedStreamDataset(id_dataset=id_dataset, ood_dataset=ood_dataset, order=order, root=root)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)


def eval_mixed_stream_one_ood(
    args,
    model,
    id_loader,
    ood_loader,
    positive_features,
    negative_features,
    inversion_init_candidates,
    class_prototypes,
    base_sim,
    stream_name,
    log,
):
    loader = build_mixed_stream_loader(args, id_loader, ood_loader, stream_name)
    image_features, is_ood = load_or_cache_stream_features_and_gt(model, loader, args, log, stream_name)
    scores = compute_tins_scores_from_image_features(
        image_features=image_features,
        args=args,
        model=model,
        positive_features=positive_features,
        negative_features=negative_features,
        inversion_init_candidates=inversion_init_candidates,
        class_prototypes=class_prototypes,
        base_sim=base_sim,
    )

    if args.save_stream_scores:
        out_path = Path(args.log_directory) / f"stream_scores_{stream_name}.npz"
        np.savez_compressed(out_path, scores=scores.astype(np.float32), is_ood=is_ood.astype(np.int32))
        log.debug(f"Saved stream per-sample scores to {out_path}")

    in_score = scores[is_ood == 0]
    out_score = scores[is_ood == 1]
    return in_score, out_score


def save_selected_labels(log_directory, labels):
    output_path = Path(log_directory) / "selected_neg_labels.txt"
    output_path.write_text("\n".join(labels) + "\n")


def main():
    args = process_args()
    setup_seed(args.seed)
    log = setup_log(args)

    assert torch.cuda.is_available()
    torch.cuda.set_device(args.gpu)

    net, preprocess = load_official_clip(args)
    net.eval()
    log.debug(f"Using CLIP backend: {args.clip_impl}")

    val_loader, eval_specs = build_eval_setup(args, preprocess)
    positive_labels = [str(label) for label in get_test_labels(args, val_loader)]
    device = next(net.parameters()).device

    class_prototypes, prototype_meta, prototype_cache_path = load_or_build_class_prototypes(args, net, preprocess, log)
    positive_texts = [args.pos_prompt.format(label) for label in positive_labels]
    positive_features = encode_texts(
        net,
        positive_texts,
        batch_size=args.text_batch_size,
        device=device,
        desc="Encoding positive labels",
    ).to(device)
    class_prototypes = class_prototypes.to(device)
    base_sim = (positive_features * class_prototypes).sum(dim=1)

    negative_features, selected_negative_texts, selected_negative_words, negative_cache_path = load_or_build_negative_bank(
        args=args,
        model=net,
        positive_labels=positive_labels,
        positive_features=positive_features,
        class_prototypes=class_prototypes.cpu(),
        log=log,
    )
    negative_features = negative_features.to(device)
    inversion_init_candidates = None
    if args.inversion_init_mode == "best_neg_word":
        inversion_init_candidates = build_inversion_init_candidates(
            args,
            net,
            selected_negative_words,
            class_prototypes,
            device,
            log,
        )
    save_selected_labels(args.log_directory, selected_negative_texts)

    log.debug(f"Positive labels: {len(positive_texts)}")
    log.debug(f"Selected negatives: {len(selected_negative_texts)}")
    log.debug(f"Prototype cache path: {prototype_cache_path}")
    log.debug(f"Prototype meta: {prototype_meta}")
    log.debug(f"Negative bank cache path: {negative_cache_path}")
    if args.use_buffer:
        log.debug(f"Overflow buffer enabled, bank_buffer_size={args.bank_buffer_size}")
    else:
        log.debug("Overflow buffer disabled (bank grows unbounded, sorted by score).")

    dataset_rows = []
    grouped_results = {}

    for spec in eval_specs:
        out_dataset = spec["dataset_name"]
        group_name = spec["group_name"]
        stream_name = spec["stream_name"]
        ood_loader = spec["ood_loader"]
        log.debug(f"Evaluating OOD dataset {out_dataset} (group={group_name}, stream={stream_name})")
        mixed_in_score, out_score = eval_mixed_stream_one_ood(
            args=args,
            model=net,
            id_loader=val_loader,
            ood_loader=ood_loader,
            positive_features=positive_features,
            negative_features=negative_features,
            inversion_init_candidates=inversion_init_candidates,
            class_prototypes=class_prototypes,
            base_sim=base_sim,
            stream_name=stream_name,
            log=log,
        )
        log.debug(
            "%s score stats: min=%.6f max=%.6f mean=%.6f",
            out_dataset,
            float(np.min(out_score)),
            float(np.max(out_score)),
            float(np.mean(out_score)),
        )

        auroc, aupr, fpr = get_measures(mixed_in_score, out_score)
        print_measures(log, auroc, aupr, fpr, method_name=args.score)
        result_row = {
            "name": out_dataset,
            "group": group_name,
            "fpr": fpr,
            "auroc": auroc,
            "aupr": aupr,
        }
        dataset_rows.append(result_row)
        grouped_results.setdefault(group_name, []).append(result_row)

    summary_rows = []
    if args.eval_protocol == "openood_imagenet1k":
        for group_name in ("nearood", "farood"):
            group_metrics = compute_mean_metrics(grouped_results[group_name])
            summary_rows.append(
                {
                    "name": group_name,
                    "group": "summary",
                    "fpr": group_metrics["fpr"],
                    "auroc": group_metrics["auroc"],
                    "aupr": group_metrics["aupr"],
                }
            )
            log.debug(f"\n\nMean Test Results ({group_name})")
            print_measures(
                log,
                group_metrics["auroc"],
                group_metrics["aupr"],
                group_metrics["fpr"],
                method_name=f"{args.score}_{group_name}",
            )

    overall_metrics = compute_mean_metrics(dataset_rows)
    log.debug("\n\nMean Test Results")
    print_measures(
        log,
        overall_metrics["auroc"],
        overall_metrics["aupr"],
        overall_metrics["fpr"],
        method_name=args.score,
    )
    save_metric_rows(args, dataset_rows, summary_rows=summary_rows)


if __name__ == "__main__":
    main()
