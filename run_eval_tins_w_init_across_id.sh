#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${REPO_ROOT}/eval_tins_w_init_across_id.py"

ROOT_DIR="${ROOT_DIR:-datasets}"
OPENOOD_ROOT="${OPENOOD_ROOT:-/disk1/yangyifeng/icml_2024/OpenOOD}"
FOOD101_ROOT="${FOOD101_ROOT:-data}"
OPENOOD_IMAGE_ROOT="${OPENOOD_IMAGE_ROOT:-${OPENOOD_ROOT}/data/images_largescale}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/cache/tins_across_id}"
GPU="${GPU:-0}"
CKPT="${CKPT:-ViT-B/16}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TEXT_BATCH_SIZE="${TEXT_BATCH_SIZE:-1000}"
PROTOTYPE_BATCH_SIZE="${PROTOTYPE_BATCH_SIZE:-256}"
SEED="${SEED:-0}"

INVERSION_STEPS="${INVERSION_STEPS:-30}"
INVERSION_REG_LAMBDA="${INVERSION_REG_LAMBDA:-0.3}"
OOD_THRESHOLD="${OOD_THRESHOLD:-0.3}"
GROUP_NUM="${GROUP_NUM:-5}"
OOD_NUMBER="${OOD_NUMBER:-2000}"
EXTRA_TEXT_LENGTH="${EXTRA_TEXT_LENGTH:-2000}"

DEFAULT_ID_DATASETS=("Food-101" "ImageNet-Sketch" "ImageNet-R" "ImageNet-V2")
VALID_ID_DATASETS=("ImageNet" "${DEFAULT_ID_DATASETS[@]}")

usage() {
  cat <<'EOF'
Usage:
  bash run_eval_tins_w_init_across_id.sh [ID_DATASET ...]

ID datasets:
  ImageNet Food-101 ImageNet-Sketch ImageNet-R ImageNet-V2

Environment overrides:
  ROOT_DIR              ImageNet root parent; expects ${ROOT_DIR}/ImageNet/val
  OPENOOD_ROOT          OpenOOD repository root
  FOOD101_ROOT          torchvision Food101 root; expects ${FOOD101_ROOT}/food-101
  OPENOOD_IMAGE_ROOT    parent of imagenet-sketch/images, imagenet_r, imagenet_v2
  CACHE_DIR             cache directory
  GPU                   GPU index passed to the Python script
  CKPT                  CLIP backbone: ViT-B/32, ViT-B/16, or ViT-L/14
EOF
}

contains_dataset() {
  local candidate="$1"
  local dataset
  for dataset in "${VALID_ID_DATASETS[@]}"; do
    [[ "${candidate}" == "${dataset}" ]] && return 0
  done
  return 1
}

safe_name() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | tr ' /-' '___'
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ID_DATASETS=("${DEFAULT_ID_DATASETS[@]}")
if [[ "$#" -gt 0 ]]; then
  ID_DATASETS=("$@")
fi

cd "${REPO_ROOT}"

for in_dataset in "${ID_DATASETS[@]}"; do
  if ! contains_dataset "${in_dataset}"; then
    echo "Unsupported ID dataset: ${in_dataset}" >&2
    usage >&2
    exit 1
  fi

  dataset_name="$(safe_name "${in_dataset}")"
  exp_name="across_id_${dataset_name}_inv${INVERSION_STEPS}_thr${OOD_THRESHOLD}_lam${INVERSION_REG_LAMBDA}"

  echo "============================================================"
  echo "Running tins across-ID evaluation"
  echo "  ID dataset: ${in_dataset}"
  echo "  Experiment: ${exp_name}"
  echo "  Root dir: ${ROOT_DIR}"
  echo "  Food-101 root: ${FOOD101_ROOT}"
  echo "  OpenOOD image root: ${OPENOOD_IMAGE_ROOT}"
  echo "  Cache dir: ${CACHE_DIR}"
  echo "============================================================"

  python "${SCRIPT_PATH}" \
    --eval-protocol four_ood \
    --in_dataset "${in_dataset}" \
    --root-dir "${ROOT_DIR}" \
    --openood-root "${OPENOOD_ROOT}" \
    --food101-root "${FOOD101_ROOT}" \
    --openood-image-root "${OPENOOD_IMAGE_ROOT}" \
    --cache-dir "${CACHE_DIR}" \
    --gpu "${GPU}" \
    --CLIP_ckpt "${CKPT}" \
    --batch-size "${BATCH_SIZE}" \
    --text-batch-size "${TEXT_BATCH_SIZE}" \
    --prototype-batch-size "${PROTOTYPE_BATCH_SIZE}" \
    --seed "${SEED}" \
    --name "${exp_name}" \
    --inversion-steps "${INVERSION_STEPS}" \
    --inversion-reg-lambda "${INVERSION_REG_LAMBDA}" \
    --ood-threshold "${OOD_THRESHOLD}" \
    --group-num "${GROUP_NUM}" \
    --ood-number "${OOD_NUMBER}" \
    --extra-text-length "${EXTRA_TEXT_LENGTH}"
done
