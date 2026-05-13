# TINS: Test-time ID-prototype-separated Negative Semantics Learning for OOD Detection

[![arXiv](https://img.shields.io/badge/arXiv-2605.10756-b31b1b.svg)](https://arxiv.org/abs/2605.10756)
[![Code](https://img.shields.io/badge/Code-TINS-blue.svg)](https://github.com/zxk1212/tins)
[![Python](https://img.shields.io/badge/Python-3.x-green.svg)]()

---

## 🔥 News

- **2026-05-12** — 📄 Paper released on arXiv
- **2026-05-13** — 🚀 Code and evaluation scripts released

---

## 📖 Overview

This repository provides the official implementation of **TINS: Test-time ID-prototype-separated Negative Semantics Learning for OOD Detection**.

**Abstract:** Vision-language models enable OOD detection by comparing image alignment
with ID labels and negative semantics. Existing negative-label-based methods
mainly rely on static negative labels constructed before inference, limiting their
ability to cover diverse and evolving OOD concepts. Although test-time expansion
provides a natural solution, naively learning negative semantics from potential
OOD samples may introduce hard ID contamination. To address this issue, we
propose a Test-time ID-prototype-separated Negative Semantics learning method,
termed TINS. TINS learns sample-specific negative text embeddings via imageto-text modality inversion and introduces ID-prototype-separated regularization to
keep them separated from ID semantics. To further stabilize negative semantics
expansion, TINS employs group-wise aggregation scoring and a buffer update
strategy. Extensive experiments across Four-OOD, OpenOOD, Temporal-shift,
and Various ID settings show consistent improvements over strong baselines.
Notably, on the Four-OOD benchmark with ImageNet-1K as ID, TINS reduces
the average FPR95 from 14.04% to 6.72%.



## 🛠️ Installation

```bash
git clone https://github.com/zxk1212/tins.git
cd tins

pip install -r requirements.txt
```

---

## 📦 Data Preparation

By default, the examples below use `./datasets` as `--root-dir`. You can use another location by passing `--root-dir /path/to/datasets`.

### ImageNet ID Data

Prepare ImageNet-1K in torchvision `ImageFolder` format:

```text
datasets/
  ImageNet/
    train/
      n01440764/
      n01443537/
      ...
    val/
      n01440764/
      n01443537/
      ...
```

`train/` is used to build few-shot ID class prototypes. `val/` is used as ID test data.

### ImageNet Four-OOD Data

For ImageNet Four-OOD evaluation, prepare iNaturalist, SUN, Places, and DTD/Textures under `datasets/ImageNet_OOD_dataset`:

```text
datasets/
  iNaturalist/
  SUN/
  Places/
  dtd/
```

The loader accepts several common names, including `iNaturalist` or `inaturalist`, `SUN` or `sun`, `Places` or `places`, and `dtd` or `texture`.

### OpenOOD Data

For OpenOOD protocols, clone or prepare the OpenOOD repository/data and pass its root with `--openood-root`:

```bash
git clone https://github.com/Jingkang50/OpenOOD.git /path/to/OpenOOD
```

The CIFAR scripts expect OpenOOD imglist files such as:

```text
/path/to/OpenOOD/
  data/
    benchmark_imglist/
      cifar10/
      cifar100/
      ...
```

### Various-ID Data

`eval_tins_w_init_across_id.py` supports `ImageNet`, `Food-101`, `ImageNet-Sketch`, `ImageNet-R`, and `ImageNet-V2`.

The script reads the datasets from these roots by default:

```text
data
  food-101/

${OPENOOD_ROOT}/data/images_largescale
  imagenet-sketch/images/
  imagenet_r/
  imagenet_v2/
```

If your datasets are stored elsewhere, pass `ROOT_DIR`, `FOOD101_ROOT`, or `OPENOOD_IMAGE_ROOT` when launching `run_eval_tins_w_init_across_id.sh`.

---

## 🚀 Evaluation

Run all commands from the repository root:

```bash
cd /path/to/tins
```

### ImageNet Four-OOD

```bash
python ./eval_tins_w_init.py \
  --eval-protocol four_ood \
  --in_dataset ImageNet \
  --root-dir ./datasets \
  --wordnet-dir ./txtfiles \
  --gpu 0 \
  --CLIP_ckpt "ViT-B/16" \
  --name "tins_four_ood_vit_b16" \
  --inversion-steps 30 \
  --inversion-reg-lambda 0.3 \
  --ood-threshold 0.3 \
  --group-num 5 \
  --ood-number 2000 \
  --extra-text-length 2000
```

### OpenOOD CIFAR-10 / CIFAR-100

CIFAR-10 as ID:

```bash
python ./eval_tins_w_init_cifar10_100.py \
  --eval-protocol openood_cifar10 \
  --openood-root /path/to/OpenOOD \
  --root-dir ./datasets \
  --wordnet-dir ./txtfiles \
  --gpu 0 \
  --CLIP_ckpt "ViT-B/16" \
  --train-shot-per-class 16 \
  --name "tins_openood_cifar10_16shot" \
  --inversion-steps 30 \
  --inversion-reg-lambda 0.5 \
  --ood-threshold 0.3 \
  --group-num 5 \
  --ood-number 70000 \
  --bank-buffer-size 2000 \
  --extra-text-length 2000
```

CIFAR-100 as ID:

```bash
python ./eval_tins_w_init_cifar10_100.py \
  --eval-protocol openood_cifar100 \
  --openood-root /path/to/OpenOOD \
  --root-dir ./datasets \
  --wordnet-dir ./txtfiles \
  --gpu 0 \
  --CLIP_ckpt "ViT-B/16" \
  --train-shot-per-class 16 \
  --name "tins_openood_cifar100_16shot" \
  --inversion-steps 30 \
  --inversion-reg-lambda 0.5 \
  --ood-threshold 0.3 \
  --group-num 5 \
  --ood-number 70000 \
  --bank-buffer-size 2000 \
  --extra-text-length 2000
```

### Temporal-Shift Evaluation

Use the provided script to launch multiple OOD orderings:

```bash
ROOT_DIR=./datasets \
BACKBONE_LIST="ViT-B/16" \
GPU_LIST="0 1 2 3" \
bash run_eval_tins_w_init_temporal_shift.sh
```

### Various Backbones

```bash
ROOT_DIR=./datasets \
OPENOOD_ROOT=/path/to/OpenOOD \
BACKBONE_LIST="RN50 RN101 ViT-B/32 ViT-B/16 ViT-L/14" \
GPU_LIST="0 1 2 3 4" \
bash run_eval_tins_w_init_backbones.sh
```

### Various-ID Evaluation

Run the default ID datasets (`Food-101`, `ImageNet-Sketch`, `ImageNet-R`, and `ImageNet-V2`):

```bash
ROOT_DIR=./datasets \
OPENOOD_ROOT=/path/to/OpenOOD \
FOOD101_ROOT=./data \
OPENOOD_IMAGE_ROOT=/path/to/OpenOOD/data/images_largescale \
GPU=0 \
CKPT="ViT-B/16" \
bash run_eval_tins_w_init_across_id.sh
```

To evaluate a specific ID dataset:

```bash
bash run_eval_tins_w_init_across_id.sh ImageNet
bash run_eval_tins_w_init_across_id.sh Food-101
```

---

## 📁 Repository Structure

```text
.
├── eval_tins_w_init.py                 # ImageNet Four-OOD evaluation
├── eval_tins_w_init_cifar10_100.py     # OpenOOD CIFAR-10 / CIFAR-100 evaluation
├── eval_tins_w_init_temporal_shift.py  # Temporal-shift evaluation
├── eval_tins_w_init_across_id.py       # Various-ID evaluation
├── run_eval_tins_w_init_backbones.sh   # Multi-backbone launcher
├── run_eval_tins_w_init_temporal_shift.sh
├── run_eval_tins_w_init_across_id.sh
├── dataloaders/
├── utils/
└── third_party/openai_clip/
```

---

## 📚 Citation

If you find this repository useful, please cite:

```bibtex
@article{yang2026tins,
  title={TINS: Test-time ID-prototype-separated Negative Semantics Learning for OOD Detection},
  author={Yang, Yifeng and Feng, Jubo and Xu, Jing and Wang, Xinbing and Gu, Qinying and Ye, Nanyang},
  journal={arXiv preprint arXiv:2605.10756},
  year={2026}
}
```
