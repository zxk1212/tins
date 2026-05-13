#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${REPO_ROOT}/eval_tins_w_init_temporal_shift.py"

ROOT_DIR="${ROOT_DIR:-datasets}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/tins_temporal_shift}"

BATCH_SIZE="${BATCH_SIZE:-256}"
TEXT_BATCH_SIZE="${TEXT_BATCH_SIZE:-1000}"
PROTOTYPE_BATCH_SIZE="${PROTOTYPE_BATCH_SIZE:-256}"
SEED="${SEED:-0}"
STREAM_SEED="${STREAM_SEED:-123}"

INVERSION_STEPS="${INVERSION_STEPS:-30}"
INVERSION_REG_LAMBDA="${INVERSION_REG_LAMBDA:-0.3}"
OOD_THRESHOLD="${OOD_THRESHOLD:-0.3}"
GROUP_NUM="${GROUP_NUM:-5}"
OOD_NUMBER="${OOD_NUMBER:-2000}"
EXTRA_TEXT_LENGTH="${EXTRA_TEXT_LENGTH:-2000}"

BACKBONE_LIST="${BACKBONE_LIST:-ViT-B/16}"
GPU_LIST="${GPU_LIST:-0 2 6 7}"
ORDER_LIST="${ORDER_LIST:-I->S->P->T S->P->T->I P->T->I->S T->I->S->P}"

read -r -a BACKBONES <<< "${BACKBONE_LIST}"
read -r -a GPUS <<< "${GPU_LIST}"
read -r -a ORDERS <<< "${ORDER_LIST}"

if [[ "${#BACKBONES[@]}" -ne 1 ]]; then
  echo "This script expects exactly one backbone for four-order temporal-shift execution." >&2
  echo "Backbones: ${#BACKBONES[@]}" >&2
  exit 1
fi

if [[ "${#GPUS[@]}" -ne "${#ORDERS[@]}" ]]; then
  echo "Number of GPUs must match number of temporal orders." >&2
  echo "GPUs: ${#GPUS[@]}, orders: ${#ORDERS[@]}" >&2
  exit 1
fi

mkdir -p "${LOG_ROOT}"
cd "${REPO_ROOT}"

safe_name() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | tr '/' '_' | tr '-' '_' | tr '>' '_'
}

pids=()
job_names=()
backbone="${BACKBONES[0]}"
backbone_safe="$(safe_name "${backbone}")"

for idx in "${!ORDERS[@]}"; do
  order="${ORDERS[$idx]}"
  gpu="${GPUS[$idx]}"
  order_safe="$(safe_name "${order}")"

  exp_name="eval_tins_w_init_temporal_shiftbuffer_False_${order_safe}_${backbone_safe}_inv${INVERSION_STEPS}_thr${OOD_THRESHOLD}_lam${INVERSION_REG_LAMBDA}"
  log_path="${LOG_ROOT}/${exp_name}.log"

  echo "============================================================"
  echo "Launching tins temporal-shift evaluation"
  echo "  Backbone: ${backbone}"
  echo "  GPU: ${gpu}"
  echo "  Temporal order: ${order}"
  echo "  Experiment: ${exp_name}"
  echo "  Log: ${log_path}"
  echo "  ID dataset: ImageNet-1K"
  echo "  OOD datasets: iNaturalist, SUN, Places, Textures"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${gpu}" python "${SCRIPT_PATH}" \
    --root-dir "${ROOT_DIR}" \
    --gpu 0 \
    --eval-protocol temporal_shift \
    --temporal-order "${order}" \
    --CLIP_ckpt "${backbone}" \
    --batch-size "${BATCH_SIZE}" \
    --text-batch-size "${TEXT_BATCH_SIZE}" \
    --prototype-batch-size "${PROTOTYPE_BATCH_SIZE}" \
    --seed "${SEED}" \
    --stream-seed "${STREAM_SEED}" \
    --name "${exp_name}" \
    --inversion-steps "${INVERSION_STEPS}" \
    --inversion-reg-lambda "${INVERSION_REG_LAMBDA}" \
    --no-use-buffer  \
    --ood-threshold "${OOD_THRESHOLD}" \
    --group-num "${GROUP_NUM}" \
    --ood-number "${OOD_NUMBER}" \
    --extra-text-length "${EXTRA_TEXT_LENGTH}" \
    > "${log_path}" 2>&1 &

  pids+=("$!")
  job_names+=("${order}")
done

echo "Started ${#pids[@]} jobs. Waiting for completion..."

failed=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  order="${job_names[$idx]}"
  if wait "${pid}"; then
    echo "[OK] ${order}"
  else
    echo "[FAIL] ${order}" >&2
    failed=1
  fi
done

exit "${failed}"
