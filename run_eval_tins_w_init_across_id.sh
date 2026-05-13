#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${REPO_ROOT}/eval_tins_w_init_across_id.py"

# For ImageNet only, the script still reads <ROOT_DIR>/ImageNet/val.
# Food-101 / ImageNet-Sketch / ImageNet-R / ImageNet-V2 use the fixed paths
# already written inside eval_tins_w_init_across_id.py.
ROOT_DIR="${ROOT_DIR:-datasets}"
OPENOOD_ROOT="${OPENOOD_ROOT:-/disk1/yangyifeng/icml_2024/OpenOOD}"
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

ID_DATASETS=(
  "Food-101"
  "ImageNet-Sketch"
  "ImageNet-R"
  "ImageNet-V2"
)

if [[ $# -gt 0 ]]; then
  ID_DATASETS=("$@")
fi

cd "${REPO_ROOT}"

for in_dataset in "${ID_DATASETS[@]}"; do
  safe_name="$(echo "${in_dataset}" | tr '[:upper:]' '[:lower:]' | tr ' -' '__')"
  exp_name="across_id_${safe_name}_inv${INVERSION_STEPS}_thr${OOD_THRESHOLD}_lam${INVERSION_REG_LAMBDA}"

  echo "============================================================"
  echo "Running tins across-ID evaluation"
  echo "  ID dataset: ${in_dataset}"
  echo "  Experiment: ${exp_name}"
  echo "============================================================"

  python "${SCRIPT_PATH}" \
    --eval-protocol four_ood \
    --in_dataset "${in_dataset}" \
    --root-dir "${ROOT_DIR}" \
    --openood-root "${OPENOOD_ROOT}" \
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
