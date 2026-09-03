#!/usr/bin/env bash
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=gb200
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --job-name=coreai_dlalgo_llm-flashinfer.pr4657-4684

set -euo pipefail

: "${CONTROL_ROOT:?set CONTROL_ROOT}"
: "${RESULT_ROOT:?set RESULT_ROOT}"
: "${PR4657_ROOT:?set PR4657_ROOT}"
: "${COMBINED_ROOT:?set COMBINED_ROOT}"
: "${PR4657_SHA:?set PR4657_SHA}"
: "${COMBINED_SHA:?set COMBINED_SHA}"

scratch_root="/raid/scratch/${USER}/flashinfer-pr4657-4684/${SLURM_JOB_ID}"
mkdir -p "${scratch_root}"
trap 'rm -rf "${scratch_root}"' EXIT

(
  cd "${COMBINED_ROOT}"
  export PYTHONPATH="${COMBINED_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  export FLASHINFER_WORKSPACE_BASE="${scratch_root}/tests/flashinfer"
  export TORCH_EXTENSIONS_DIR="${scratch_root}/tests/torch_extensions"
  CUDA_VISIBLE_DEVICES=0,1 python3 -m pytest -q \
    tests/gemm/test_mm_mxfp8_dynamic_quant.py \
    tests/gemm/test_mm_mxfp8.py \
    tests/benchmarks/test_gemm_args.py
) 2>&1 | tee "${RESULT_ROOT}/combined-tests.log"

REPO_ROOT="${PR4657_ROOT}" \
RESULT_ROOT="${RESULT_ROOT}/pr4657" \
EXPECTED_SHA="${PR4657_SHA}" \
SOURCE_LABEL=pr4657 \
SCRATCH_ROOT="${scratch_root}/pr4657" \
  "${CONTROL_ROOT}/run_source.sh"

REPO_ROOT="${COMBINED_ROOT}" \
RESULT_ROOT="${RESULT_ROOT}/combined" \
EXPECTED_SHA="${COMBINED_SHA}" \
SOURCE_LABEL=combined \
SCRATCH_ROOT="${scratch_root}/combined" \
  "${CONTROL_ROOT}/run_source.sh"

python3 "${CONTROL_ROOT}/collect_results.py" "${RESULT_ROOT}"
touch "${RESULT_ROOT}/SUCCESS"
