#!/usr/bin/env bash

set -euo pipefail

CONTROL_ROOT=${CONTROL_ROOT:-/home/sna/flashinfer-pr4657-4684-combined/experiments/pr4657_pr4684_combined}
PR4657_ROOT=${PR4657_ROOT:-/home/sna/flashinfer-pr4657-only}
COMBINED_ROOT=${COMBINED_ROOT:-/home/sna/flashinfer-pr4657-4684-combined}
PR4657_SHA=${PR4657_SHA:-9fcad8b9b03e61d6f4af3801f2e7acf4f06dea85}
COMBINED_SHA=${COMBINED_SHA:-$(git -C "${COMBINED_ROOT}" rev-parse HEAD)}
CONTAINER=${CONTAINER:-/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/vllm_openai_v0271_aarch64.sqsh}
STAMP=${STAMP:-$(date +%Y%m%d-%H%M%S)}
RESULT_ROOT=${RESULT_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/flashinfer-pr4657-4684-combined/${STAMP}}

for spec in "${PR4657_ROOT}:${PR4657_SHA}" "${COMBINED_ROOT}:${COMBINED_SHA}"; do
  source_root=${spec%:*}
  expected_sha=${spec#*:}
  actual_sha=$(git -C "${source_root}" rev-parse HEAD)
  if [[ "${actual_sha}" != "${expected_sha}" ]]; then
    echo "Expected ${expected_sha}, found ${actual_sha} in ${source_root}" >&2
    exit 1
  fi
done

mkdir -p "${RESULT_ROOT}"
printf '%s\n' \
  "control_root=${CONTROL_ROOT}" \
  "pr4657_root=${PR4657_ROOT}" \
  "pr4657_sha=${PR4657_SHA}" \
  "combined_root=${COMBINED_ROOT}" \
  "combined_sha=${COMBINED_SHA}" \
  "container=${CONTAINER}" \
  > "${RESULT_ROOT}/submission.txt"

args=(
  --account=coreai_dlalgo_llm
  --partition=gb200
  --nodes=1
  --time=04:00:00
  --job-name=coreai_dlalgo_llm-flashinfer.pr4657-4684
  --output="${RESULT_ROOT}/slurm-%j.out"
  --export="ALL,CONTROL_ROOT=${CONTROL_ROOT},RESULT_ROOT=${RESULT_ROOT},PR4657_ROOT=${PR4657_ROOT},COMBINED_ROOT=${COMBINED_ROOT},PR4657_SHA=${PR4657_SHA},COMBINED_SHA=${COMBINED_SHA}"
)

if [[ "${SBATCH_TEST_ONLY:-0}" == "1" ]]; then
  sbatch --test-only "${args[@]}" "${CONTROL_ROOT}/job.sbatch"
  exit 0
fi

sbatch "${args[@]}" \
  --wrap="srun --container-image=${CONTAINER} --container-mounts=/home:/home,/lustre:/lustre,/raid:/raid bash ${CONTROL_ROOT}/job.sbatch"
