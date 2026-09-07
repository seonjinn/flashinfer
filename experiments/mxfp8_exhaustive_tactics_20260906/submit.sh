#!/usr/bin/env bash

set -euo pipefail

CONTROL_ROOT=${CONTROL_ROOT:-/home/sna/flashinfer-pr4657-4684-combined/experiments/mxfp8_exhaustive_tactics_20260906}
PRUNED_ROOT=${PRUNED_ROOT:-/home/sna/flashinfer-mxfp8-exhaustive-control}
EXHAUSTIVE_ROOT=${EXHAUSTIVE_ROOT:-/home/sna/flashinfer-mxfp8-exhaustive-tactics}
PRUNED_SHA=${PRUNED_SHA:-465a4abafb336a2f01b63e2b1e56d1c51062b7bc}
EXHAUSTIVE_SHA=${EXHAUSTIVE_SHA:-6fab535dc01b18e37a7cdeb70c526a0dcab05aed}
CONTAINER=${CONTAINER:-/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/vllm_openai_v0271_aarch64_20260813_2688476.sqsh}
STAMP=${STAMP:-$(date +%Y%m%d-%H%M%S)}
RESULT_ROOT=${RESULT_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/flashinfer-mxfp8-exhaustive-tactics/${STAMP}}

for spec in "${PRUNED_ROOT}:${PRUNED_SHA}" "${EXHAUSTIVE_ROOT}:${EXHAUSTIVE_SHA}"; do
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
  "pruned_root=${PRUNED_ROOT}" \
  "pruned_sha=${PRUNED_SHA}" \
  "exhaustive_root=${EXHAUSTIVE_ROOT}" \
  "exhaustive_sha=${EXHAUSTIVE_SHA}" \
  "container=${CONTAINER}" \
  > "${RESULT_ROOT}/submission.txt"

args=(
  --account=coreai_dlalgo_llm
  --partition=gb200
  --nodes=1
  --time=04:00:00
  --job-name=coreai_dlalgo_llm-flashinfer.exhaustive-tactics
  --output="${RESULT_ROOT}/slurm-%j.out"
  --export="ALL,CONTROL_ROOT=${CONTROL_ROOT},RESULT_ROOT=${RESULT_ROOT},PRUNED_ROOT=${PRUNED_ROOT},EXHAUSTIVE_ROOT=${EXHAUSTIVE_ROOT},PRUNED_SHA=${PRUNED_SHA},EXHAUSTIVE_SHA=${EXHAUSTIVE_SHA}"
)

if [[ "${SBATCH_TEST_ONLY:-0}" == "1" ]]; then
  sbatch --test-only "${args[@]}" "${CONTROL_ROOT}/job.sbatch"
  exit 0
fi

sbatch "${args[@]}" \
  --wrap="srun --container-image=${CONTAINER} --container-mounts=/home:/home,/lustre:/lustre,/raid:/raid bash ${CONTROL_ROOT}/job.sbatch"
