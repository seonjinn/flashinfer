#!/usr/bin/env bash

set -euo pipefail

: "${REPO_ROOT:?set REPO_ROOT}"
: "${RESULT_ROOT:?set RESULT_ROOT}"
: "${SHAPE_SUMMARY:?set SHAPE_SUMMARY}"
: "${SOURCE_SHA:?set SOURCE_SHA}"

REPETITIONS=${REPETITIONS:-1}
BASE_SEED=${BASE_SEED:-20260825}
TOP_K=${TOP_K:-3}
REFINEMENT_ROUNDS=${REFINEMENT_ROUNDS:-3}
EVALUATION_ROUNDS=${EVALUATION_ROUNDS:-5}
THRESHOLD_PCT=${THRESHOLD_PCT:--1}
LIMIT=${LIMIT:-12}
SHARED_CACHE=${SHARED_CACHE:-${RESULT_ROOT}/cache}

mkdir -p "${RESULT_ROOT}"/{analysis,logs,validation}
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export FLASHINFER_WORKSPACE_BASE="${SHARED_CACHE}/flashinfer"
export TORCH_EXTENSIONS_DIR="${SHARED_CACHE}/torch_extensions"

printf '%s\n' "${SOURCE_SHA}" > "${RESULT_ROOT}/source_sha.txt"
sha256sum \
  "${REPO_ROOT}/benchmarks/bench_mxfp8_refinement_policy.py" \
  "${REPO_ROOT}/benchmarks/summarize_mxfp8_refinement_policy.py" \
  "${REPO_ROOT}/benchmarks/run_mxfp8_refinement_policy_lyris.sh" \
  > "${RESULT_ROOT}/harness.sha256"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv \
  > "${RESULT_ROOT}/gpu_metadata.csv"

pids=()
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=${gpu} python3 \
    "${REPO_ROOT}/benchmarks/bench_mxfp8_refinement_policy.py" \
    --shape-summary "${SHAPE_SUMMARY}" \
    --output "${RESULT_ROOT}/validation/policy-${gpu}.jsonl" \
    --repetitions "${REPETITIONS}" \
    --base-seed "${BASE_SEED}" \
    --top-k "${TOP_K}" \
    --refinement-rounds "${REFINEMENT_ROUNDS}" \
    --evaluation-rounds "${EVALUATION_ROUNDS}" \
    --threshold-pct "${THRESHOLD_PCT}" \
    --limit "${LIMIT}" \
    --shard-index "${gpu}" \
    --num-shards 4 \
    > "${RESULT_ROOT}/logs/policy-${gpu}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "${pid}"
done

python3 "${REPO_ROOT}/benchmarks/summarize_mxfp8_refinement_policy.py" \
  --input-dir "${RESULT_ROOT}/validation" \
  --output "${RESULT_ROOT}/analysis/summary.json"
printf '%s\n' \
  "repetitions=${REPETITIONS}" \
  "base_seed=${BASE_SEED}" \
  "top_k=${TOP_K}" \
  "refinement_rounds=${REFINEMENT_ROUNDS}" \
  "evaluation_rounds=${EVALUATION_ROUNDS}" \
  "threshold_pct=${THRESHOLD_PCT}" \
  "limit=${LIMIT}" \
  "shared_cache=${SHARED_CACHE}" \
  > "${RESULT_ROOT}/run_config.txt"
touch "${RESULT_ROOT}/SUCCESS"
