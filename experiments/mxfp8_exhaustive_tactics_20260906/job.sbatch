#!/usr/bin/env bash
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=gb200
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --job-name=coreai_dlalgo_llm-flashinfer.exhaustive-tactics

set -euo pipefail

: "${CONTROL_ROOT:?set CONTROL_ROOT}"
: "${RESULT_ROOT:?set RESULT_ROOT}"
: "${PRUNED_ROOT:?set PRUNED_ROOT}"
: "${EXHAUSTIVE_ROOT:?set EXHAUSTIVE_ROOT}"
: "${PRUNED_SHA:?set PRUNED_SHA}"
: "${EXHAUSTIVE_SHA:?set EXHAUSTIVE_SHA}"

scratch_root="/raid/scratch/${USER}/flashinfer-exhaustive-tactics/${SLURM_JOB_ID}"
mkdir -p "${scratch_root}"
trap 'rm -rf "${scratch_root}"' EXIT

(
  cd "${EXHAUSTIVE_ROOT}"
  export PYTHONPATH="${EXHAUSTIVE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  export FLASHINFER_WORKSPACE_BASE="${scratch_root}/tests/flashinfer"
  export TORCH_EXTENSIONS_DIR="${scratch_root}/tests/torch"
  export PATCH_CSRC="${EXHAUSTIVE_ROOT}/csrc"
  export PATCH_INCLUDE="${EXHAUSTIVE_ROOT}/include"
  export EMPTY_AOT="${scratch_root}/tests/empty-aot"
  mkdir -p "${EMPTY_AOT}"
  CUDA_VISIBLE_DEVICES=0,1 python3 "${CONTROL_ROOT}/run_with_source_jit.py" \
    -m pytest -q -x \
    tests/gemm/test_mm_mxfp8_dynamic_quant.py
) 2>&1 | tee "${RESULT_ROOT}/tests.log"

(
  cd "${EXHAUSTIVE_ROOT}"
  export PYTHONPATH="${EXHAUSTIVE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  export FLASHINFER_WORKSPACE_BASE="${scratch_root}/inventory/flashinfer"
  export TORCH_EXTENSIONS_DIR="${scratch_root}/inventory/torch"
  export PATCH_CSRC="${EXHAUSTIVE_ROOT}/csrc"
  export PATCH_INCLUDE="${EXHAUSTIVE_ROOT}/include"
  export EMPTY_AOT="${scratch_root}/inventory/empty-aot"
  mkdir -p "${EMPTY_AOT}"
  CUDA_VISIBLE_DEVICES=0 python3 "${CONTROL_ROOT}/run_with_source_jit.py" \
    "${CONTROL_ROOT}/candidate_inventory.py" \
    "${RESULT_ROOT}/candidate_inventory.csv"
) 2>&1 | tee "${RESULT_ROOT}/candidate-inventory.log"

REPO_ROOT="${PRUNED_ROOT}" \
RESULT_ROOT="${RESULT_ROOT}/pruned" \
EXPECTED_SHA="${PRUNED_SHA}" \
SOURCE_LABEL=pruned \
SCRATCH_ROOT="${scratch_root}/pruned" \
  "${CONTROL_ROOT}/run_source.sh"

REPO_ROOT="${EXHAUSTIVE_ROOT}" \
RESULT_ROOT="${RESULT_ROOT}/exhaustive" \
EXPECTED_SHA="${EXHAUSTIVE_SHA}" \
SOURCE_LABEL=exhaustive \
SCRATCH_ROOT="${scratch_root}/exhaustive" \
  "${CONTROL_ROOT}/run_source.sh"

python3 "${CONTROL_ROOT}/collect_results.py" "${RESULT_ROOT}"
touch "${RESULT_ROOT}/SUCCESS"
