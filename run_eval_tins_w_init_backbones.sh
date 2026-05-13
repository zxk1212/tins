#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${REPO_ROOT}/eval_tins_w_init.py"

ROOT_DIR="${ROOT_DIR:-datasets}"
OPENOOD_ROOT="${OPENOOD_ROOT:-/disk1/yangyifeng/icml_2024/OpenOOD}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/cache/tins_w_init_backbones}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/tins_w_init_backbones}"
EVAL_PROTOCOL="${EVAL_PROTOCOL:-four_ood}"

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

BACKBONE_LIST="${BACKBONE_LIST:-RN50 RN101 ViT-B/32 ViT-B/16 ViT-L/14}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4}"

read -r -a BACKBONES <<< "${BACKBONE_LIST}"
read -r -a GPUS <<< "${GPU_LIST}"

if [[ "${#GPUS[@]}" -ne "${#BACKBONES[@]}" ]]; then
  echo "Number of GPUs must match number of backbones." >&2
  echo "GPUs: ${#GPUS[@]}, backbones: ${#BACKBONES[@]}" >&2
  exit 1
fi

mkdir -p "${CACHE_ROOT}" "${LOG_ROOT}"
cd "${REPO_ROOT}"

safe_name() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | tr '/-' '__'
}

pids=()

for idx in "${!BACKBONES[@]}"; do
  backbone="${BACKBONES[$idx]}"
  gpu="${GPUS[$idx]}"
  backbone_safe="$(safe_name "${backbone}")"

  cache_dir="${CACHE_ROOT}/${backbone_safe}"
  exp_name="eval_tins_w_init_${EVAL_PROTOCOL}_${backbone_safe}_inv${INVERSION_STEPS}_thr${OOD_THRESHOLD}_lam${INVERSION_REG_LAMBDA}"
  log_path="${LOG_ROOT}/${exp_name}.log"

  mkdir -p "${cache_dir}"

  echo "============================================================"
  echo "Launching tins OOD w/ init"
  echo "  Backbone: ${backbone}"
  echo "  GPU: ${gpu}"
  echo "  Cache dir: ${cache_dir}"
  echo "  Experiment: ${exp_name}"
  echo "  Log: ${log_path}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${gpu}" python "${SCRIPT_PATH}" \
    --eval-protocol "${EVAL_PROTOCOL}" \
    --openood-root "${OPENOOD_ROOT}" \
    --root-dir "${ROOT_DIR}" \
    --cache-dir "${cache_dir}" \
    --gpu 0 \
    --CLIP_ckpt "${backbone}" \
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
    --extra-text-length "${EXTRA_TEXT_LENGTH}" \
    > "${log_path}" 2>&1 &

  pids+=("$!")
done

echo "Started ${#pids[@]} jobs. Waiting for completion..."

failed=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  backbone="${BACKBONES[$idx]}"
  if wait "${pid}"; then
    echo "[OK] ${backbone}"
  else
    echo "[FAIL] ${backbone}" >&2
    failed=1
  fi
done

exit "${failed}"
