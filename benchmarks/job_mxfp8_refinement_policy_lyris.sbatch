#!/usr/bin/env bash
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=gb200-backfill
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --job-name=coreai_dlalgo_llm-flashinfer.mxfp8-policy

set -euo pipefail

: "${REPO_ROOT:?set REPO_ROOT}"
: "${RESULT_ROOT:?set RESULT_ROOT}"
: "${SHAPE_SUMMARY:?set SHAPE_SUMMARY}"
: "${SOURCE_SHA:?set SOURCE_SHA}"

srun \
  --container-image=/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/vllm_openai_v0271_aarch64.sqsh \
  --container-mounts=/lustre:/lustre \
  --output="${RESULT_ROOT}/slurm-%j.out" \
  bash -lc "REPO_ROOT='${REPO_ROOT}' RESULT_ROOT='${RESULT_ROOT}' SHAPE_SUMMARY='${SHAPE_SUMMARY}' SOURCE_SHA='${SOURCE_SHA}' REPETITIONS='${REPETITIONS:-1}' LIMIT='${LIMIT:-12}' SHARED_CACHE='${SHARED_CACHE:-${RESULT_ROOT}/cache}' bash '${REPO_ROOT}/benchmarks/run_mxfp8_refinement_policy_lyris.sh'"
