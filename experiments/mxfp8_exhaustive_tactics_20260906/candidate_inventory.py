#!/usr/bin/env python3

import csv
import sys
from pathlib import Path

from flashinfer.gemm.gemm_base import get_trtllm_gemm_module


MS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
NK_SHAPES = (
    (2304, 8192),
    (2560, 8192),
    (8192, 2560),
    (8192, 4096),
    (8832, 8192),
)


def main() -> None:
    output_path = Path(sys.argv[1])
    module = get_trtllm_gemm_module()
    rows: list[tuple[int, int, int, str, int, int, int]] = []

    for n, k in NK_SHAPES:
        for m in MS:
            for use_8x4 in (True, False):
                pruned = set(module.trtllm_mxfp8_gemm_tactics(m, n, k, use_8x4))
                exhaustive = set(
                    module.trtllm_mxfp8_gemm_tactics(
                        m,
                        n,
                        k,
                        use_8x4,
                        exhaustive=True,
                    )
                )
                if not pruned or not pruned <= exhaustive:
                    raise RuntimeError(
                        f"invalid candidate sets for {(m, n, k, use_8x4)}"
                    )
                rows.append(
                    (
                        m,
                        n,
                        k,
                        "8x4" if use_8x4 else "128x4",
                        len(pruned),
                        len(exhaustive),
                        len(exhaustive - pruned),
                    )
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "m",
                "n",
                "k",
                "scale_layout",
                "pruned_candidates",
                "exhaustive_candidates",
                "added_candidates",
            )
        )
        writer.writerows(rows)


if __name__ == "__main__":
    main()
