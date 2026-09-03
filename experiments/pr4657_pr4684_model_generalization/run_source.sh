#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT=${REPO_ROOT:?set REPO_ROOT}
CONTROL_ROOT=${CONTROL_ROOT:?set CONTROL_ROOT}
RESULT_ROOT=${RESULT_ROOT:?set RESULT_ROOT}
EXPECTED_SHA=${EXPECTED_SHA:?set EXPECTED_SHA}
SOURCE_LABEL=${SOURCE_LABEL:?set SOURCE_LABEL}
SCRATCH_ROOT=${SCRATCH_ROOT:?set SCRATCH_ROOT}
REPEATS=${REPEATS:-4}
NUM_ITERS=${NUM_ITERS:-100}
DRY_RUN_ITERS=${DRY_RUN_ITERS:-10}
INCLUDE_CUTE_DSL=${INCLUDE_CUTE_DSL:-0}

mkdir -p "${RESULT_ROOT}"/{logs,raw} "${SCRATCH_ROOT}"/{cache,torch_extensions}
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export FLASHINFER_WORKSPACE_BASE="${SCRATCH_ROOT}/cache/flashinfer"
export TORCH_EXTENSIONS_DIR="${SCRATCH_ROOT}/torch_extensions"

cd "${REPO_ROOT}"
python3 - <<'PY' > "${RESULT_ROOT}/metadata.txt"
import platform

import flashinfer
import torch

print(f"python={platform.python_version()}")
print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
print(f"flashinfer={getattr(flashinfer, '__version__', 'unknown')}")
print(f"device={torch.cuda.get_device_name(0)}")
print(f"capability={torch.cuda.get_device_capability(0)}")
PY
printf 'source_label=%s\nsource_sha=%s\n' "${SOURCE_LABEL}" "${EXPECTED_SHA}" \
  >> "${RESULT_ROOT}/metadata.txt"
cp "${CONTROL_ROOT}/model_shapes.csv" "${RESULT_ROOT}/model_shapes.csv"

generate_testlist() {
  local testlist=$1
  local cache=$2
  TESTLIST="${testlist}" CACHE="${cache}" SHAPES="${CONTROL_ROOT}/model_shapes.csv" \
    INCLUDE_CUTE_DSL="${INCLUDE_CUTE_DSL}" \
    NUM_ITERS="${NUM_ITERS}" DRY_RUN_ITERS="${DRY_RUN_ITERS}" python3 - <<'PY'
import csv
import os
from pathlib import Path

ms = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
modes = [
    ("fixed-8x4", "8x4", "trtllm"),
    ("adaptive", "auto", "trtllm"),
]
if os.environ["INCLUDE_CUTE_DSL"] == "1":
    modes.append(("cute-dsl", "128x4", "cute-dsl"))
common = (
    f"--autotune --autotune_cache {os.environ['CACHE']} "
    f"--num_iters {os.environ['NUM_ITERS']} "
    f"--dry_run_iters {os.environ['DRY_RUN_ITERS']} "
    "--refcheck"
)
with Path(os.environ["SHAPES"]).open(newline="") as stream:
    projections = list(csv.DictReader(stream))

lines = []
for row in projections:
    for m in ms:
        for mode, layout, backend in modes:
            tags = ";".join(
                (
                    f"model={row['model']}",
                    f"topology={row['topology']}",
                    f"projection={row['projection']}",
                    f"mode={mode}",
                    f"shape={m}x{row['n']}x{row['k']}",
                )
            )
            lines.append(
                f"--routine mm_mxfp8 --m {m} --n {row['n']} --k {row['k']} "
                f"--dynamic_quant --dynamic_quant_layout {layout} "
                f"--backends {backend} --case_tag {tags} {common}"
            )
Path(os.environ["TESTLIST"]).write_text("\n".join(lines) + "\n")
PY
}

pids=()
for repetition in $(seq 1 "${REPEATS}"); do
  device=$((repetition - 1))
  testlist="${RESULT_ROOT}/cases-${repetition}.txt"
  cache="${SCRATCH_ROOT}/cache/autotune-${repetition}.json"
  output="${RESULT_ROOT}/raw/repetition-${repetition}.csv"
  log="${RESULT_ROOT}/logs/benchmark-${repetition}.log"
  generate_testlist "${testlist}" "${cache}"
  CUDA_VISIBLE_DEVICES="${device}" python3 benchmarks/flashinfer_benchmark.py \
    --testlist "${testlist}" \
    --output_path "${output}" \
    > "${log}" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

cp "${SCRATCH_ROOT}"/cache/autotune-*.json "${RESULT_ROOT}/"
touch "${RESULT_ROOT}/SUCCESS"
