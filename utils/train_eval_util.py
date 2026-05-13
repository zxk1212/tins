
import sys
import os
import numpy as np
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from transformers import CLIPModel
from torchvision import datasets, transforms
import torchvision.transforms as transforms
from dataloaders import StanfordCars, Food101, OxfordIIITPet, Cub2011


def _resolve_clip_checkpoint(clip_ckpt):
    ckpt_mapping = {
        "ViT-B/16": ("openai/clip-vit-base-patch16", "clip-vit-base-patch16"),
        "ViT-B/32": ("openai/clip-vit-base-patch32", "clip-vit-base-patch32"),
        "ViT-L/14": ("openai/clip-vit-large-patch14", "clip-vit-large-patch14"),
    }
    remote_ckpt, local_dir = ckpt_mapping[clip_ckpt]
    local_path = os.path.abspath(local_dir)
    if os.path.isdir(local_path) and os.path.exists(os.path.join(local_path, "config.json")):
        return local_path
    return remote_ckpt


def _resolve_imagenet_ood_root(root):
    parent = os.path.dirname(root.rstrip(os.sep))
    candidates = [
        root,
        os.path.join(root, "ImageNet_OOD_dataset"),
        parent,
        os.path.join(parent, "ImageNet_OOD_dataset"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find ImageNet OOD root under '{root}'. "
        "Expected either 'ImageNet_OOD_dataset' or the datasets root itself."
    )


def _resolve_imagefolder_root(base_root, candidates):
    for relative_path in candidates:
        path = os.path.join(base_root, relative_path)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        f"Could not find any dataset directory under '{base_root}'. Tried: {candidates}"
    )


def set_model_clip(args):
    '''
    load Huggingface CLIP
    '''
    args.ckpt = _resolve_clip_checkpoint(args.CLIP_ckpt)
    model =  CLIPModel.from_pretrained(args.ckpt)
    if args.model == 'CLIP-Linear':
        model.load_state_dict(torch.load(args.finetune_ckpt, map_location=torch.device(args.gpu)))
    model = model.cuda()
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
    val_preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    
    return model, val_preprocess

def set_train_loader(args, preprocess=None, batch_size=None, shuffle=False, subset=False):
    root = args.root_dir
    if preprocess == None:
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
        preprocess = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    kwargs = {'num_workers': 4, 'pin_memory': True}
    if batch_size is None:  # normal case: used for trainign
        batch_size = args.batch_size
        shuffle = True
    if args.in_dataset == "ImageNet":
        path = os.path.join(root, 'ImageNet', 'train')
        dataset = datasets.ImageFolder(path, transform=preprocess)
        if subset:
            from collections import defaultdict
            classwise_count = defaultdict(int)
            indices = []
            for i, label in enumerate(dataset.targets):
                if classwise_count[label] < args.max_count:
                    indices.append(i)
                    classwise_count[label] += 1
            dataset = torch.utils.data.Subset(dataset, indices)
        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif args.in_dataset in ["ImageNet10", "ImageNet20", "ImageNet100"]:
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(
                root, args.in_dataset, 'train'), transform=preprocess),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif args.in_dataset == "car196":
        train_loader = torch.utils.data.DataLoader(StanfordCars(root, split="train", download=True, transform=preprocess),
                                                   batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif args.in_dataset == "food101":
        train_loader = torch.utils.data.DataLoader(Food101(root, split="train", download=True, transform=preprocess),
                                                   batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif args.in_dataset == "pet37":
        train_loader = torch.utils.data.DataLoader(OxfordIIITPet(root, split="trainval", download=True, transform=preprocess),
                                                   batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif args.in_dataset == "bird200":
        train_loader = torch.utils.data.DataLoader(Cub2011(root, train = True, transform=preprocess),
                    batch_size=batch_size, shuffle=shuffle, **kwargs)
    return train_loader


def set_val_loader(args, preprocess=None):
    root = args.root_dir
    if preprocess == None:
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    kwargs = {'num_workers': 4, 'pin_memory': True}
    if args.in_dataset == "ImageNet":
        path = os.path.join(root, 'ImageNet', 'val')
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(path, transform=preprocess),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset in ["ImageNet10", "ImageNet20", "ImageNet100"]:
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(
                root, args.in_dataset, 'val'), transform=preprocess),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == "car196":
        val_loader = torch.utils.data.DataLoader(StanfordCars(root, split="test", download=True, transform=preprocess),
                                                 batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == "food101":
        val_loader = torch.utils.data.DataLoader(Food101(root, split="test", download=True, transform=preprocess),
                                                 batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == "pet37":
        val_loader = torch.utils.data.DataLoader(OxfordIIITPet(root, split="test", download=True, transform=preprocess),
                                                 batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == "bird200":
        val_loader = torch.utils.data.DataLoader(Cub2011(root, train = False, transform=preprocess),
                    batch_size=args.batch_size, shuffle=False, **kwargs)

    return val_loader


def set_ood_loader_ImageNet(args, out_dataset, preprocess, root):
    '''
    set OOD loader for ImageNet scale datasets
    '''
    root = _resolve_imagenet_ood_root(root)
    if out_dataset == 'iNaturalist':
        dataset_root = _resolve_imagefolder_root(root, ['iNaturalist', 'inaturalist', os.path.join('inaturalist', 'images')])
        testsetout = torchvision.datasets.ImageFolder(root=dataset_root, transform=preprocess)
    elif out_dataset == 'SUN':
        dataset_root = _resolve_imagefolder_root(root, ['SUN', 'sun', os.path.join('sun', 'images')])
        testsetout = torchvision.datasets.ImageFolder(root=dataset_root, transform=preprocess)
    elif out_dataset == 'places365': # filtered places
        dataset_root = _resolve_imagefolder_root(root, ['Places', 'places', os.path.join('places', 'images')])
        testsetout = torchvision.datasets.ImageFolder(root=dataset_root, transform=preprocess)
    elif out_dataset == 'placesbg': 
        dataset_root = _resolve_imagefolder_root(root, ['placesbg'])
        testsetout = torchvision.datasets.ImageFolder(root=dataset_root, transform=preprocess)
    elif out_dataset == 'dtd':
        dataset_root = _resolve_imagefolder_root(root, [os.path.join('dtd', 'images'), 'dtd', 'texture'])
        testsetout = torchvision.datasets.ImageFolder(root=dataset_root, transform=preprocess)
        # testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'Textures'),
        #                                 transform=preprocess)
    elif out_dataset == 'ImageNet10':
        testsetout = datasets.ImageFolder(os.path.join(args.root_dir, 'ImageNet10', 'train'), transform=preprocess)
    elif out_dataset == 'ImageNet20':
        testsetout = datasets.ImageFolder(os.path.join(args.root_dir, 'ImageNet20', 'val'), transform=preprocess)
    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size,
                                            shuffle=False, num_workers=4)
    return testloaderOut
