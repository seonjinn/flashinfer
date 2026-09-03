#!/usr/bin/env bash
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=gb200
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --job-name=coreai_dlalgo_llm-flashinfer.model-generalization

set -euo pipefail

: "${CONTROL_ROOT:?set CONTROL_ROOT}"
: "${RESULT_ROOT:?set RESULT_ROOT}"
: "${PR4657_ROOT:?set PR4657_ROOT}"
: "${COMBINED_ROOT:?set COMBINED_ROOT}"
: "${PR4657_SHA:?set PR4657_SHA}"
: "${COMBINED_SHA:?set COMBINED_SHA}"

scratch_root="/raid/scratch/${USER}/flashinfer-model-generalization/${SLURM_JOB_ID}"
mkdir -p "${scratch_root}"
trap 'rm -rf "${scratch_root}"' EXIT

REPO_ROOT="${PR4657_ROOT}" \
CONTROL_ROOT="${CONTROL_ROOT}" \
RESULT_ROOT="${RESULT_ROOT}/pr4657" \
EXPECTED_SHA="${PR4657_SHA}" \
SOURCE_LABEL=pr4657 \
INCLUDE_CUTE_DSL=0 \
SCRATCH_ROOT="${scratch_root}/pr4657" \
  "${CONTROL_ROOT}/run_source.sh"

REPO_ROOT="${COMBINED_ROOT}" \
CONTROL_ROOT="${CONTROL_ROOT}" \
RESULT_ROOT="${RESULT_ROOT}/combined" \
EXPECTED_SHA="${COMBINED_SHA}" \
SOURCE_LABEL=combined \
INCLUDE_CUTE_DSL=1 \
SCRATCH_ROOT="${scratch_root}/combined" \
  "${CONTROL_ROOT}/run_source.sh"

python3 "${CONTROL_ROOT}/collect_results.py" "${RESULT_ROOT}"
touch "${RESULT_ROOT}/SUCCESS"
