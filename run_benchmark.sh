#!/usr/bin/env bash

set -u

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29600}"
LOG_DIR="${LOG_DIR:-logs}"
RUN_DIR="${RUN_DIR:-runs}"

mkdir -p "$LOG_DIR" "$RUN_DIR"

TASKS=(
  "gsm8k:128"
  "math500:128"
  "aime24:30"
  "aime25:30"
  "humaneval:164"
  "mbpp:128"
  "livecodebench:128"
  "swe-bench:128"
  "mt-bench:80"
  "alpaca:128"
)

MODEL_DRAFT_PAIRS=(
  "Qwen/Qwen3-4B|z-lab/Qwen3-4B-DFlash-b16"
  "Qwen/Qwen3-8B|z-lab/Qwen3-8B-DFlash-b16"
  "Qwen/Qwen3-Coder-30B-A3B-Instruct|z-lab/Qwen3-Coder-30B-A3B-DFlash"
)

TEMPERATURES=(
  "0.0"
  "1.0"
)

COMMON_BENCHMARK_ARGS=(
  --max-new-tokens 2048
)

slugify() {
  local value="$1"
  value="${value//\//_}"
  value="${value//:/_}"
  value="${value// /_}"
  echo "$value"
}

run_benchmark() {
  local dataset_name="$1"
  local max_samples="$2"
  local model_name="$3"
  local draft_name="$4"
  local mode_name="$5"
  local save_path="$6"
  local log_path="$7"
  shift 7

  echo "========================================================"
  echo "Running Benchmark: dataset=${dataset_name} max_samples=${max_samples} model=${model_name} draft=${draft_name} mode=${mode_name}"
  echo "========================================================"

  if [[ -f "${save_path}" ]]; then
    echo "Skipping existing run: ${save_path}"
    return
  fi

  torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    benchmark.py \
    --dataset "${dataset_name}" \
    --max-samples "${max_samples}" \
    --model-name-or-path "${model_name}" \
    --draft-name-or-path "${draft_name}" \
    --save-path "${save_path}" \
    "${COMMON_BENCHMARK_ARGS[@]}" \
    "$@" \
    2>&1 | tee "${log_path}"
}

for task in "${TASKS[@]}"; do
  IFS=':' read -r dataset_name max_samples <<< "${task}"

  for pair in "${MODEL_DRAFT_PAIRS[@]}"; do
    IFS='|' read -r model_name draft_name <<< "${pair}"

    model_slug="$(slugify "${model_name}")"
    draft_slug="$(slugify "${draft_name}")"
    for temperature in "${TEMPERATURES[@]}"; do
      temperature_slug="$(slugify "${temperature}")"
      run_name="${dataset_name}__${model_slug}__${draft_slug}__temp${temperature_slug}"

      run_benchmark \
        "${dataset_name}" \
        "${max_samples}" \
        "${model_name}" \
        "${draft_name}" \
        "sdpa" \
        "${RUN_DIR}/${run_name}__sdpa.pt" \
        "${LOG_DIR}/${run_name}__sdpa.log" \
        --temperature "${temperature}"

      run_benchmark \
        "${dataset_name}" \
        "${max_samples}" \
        "${model_name}" \
        "${draft_name}" \
        "flash_attn" \
        "${RUN_DIR}/${run_name}__flash_attn.pt" \
        "${LOG_DIR}/${run_name}__flash_attn.log" \
        --temperature "${temperature}" \
        --flash-attn
    done
  done
done
