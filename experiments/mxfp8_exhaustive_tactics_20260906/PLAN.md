# Plan

1. Confirm the new dynamic runner test fails on the pruned implementation.
2. Expose exhaustive tactic enumeration without changing the existing API.
3. Verify the pruned set is a subset of the exhaustive set on GB200.
4. Compare pruned, exhaustive, fixed 8x4, and fixed 128x4 across observed
   Nemotron 3 Ultra dense MXFP8 GEMM shapes.
5. Promote exhaustive tuning only if the selected tactics improve latency enough
   to justify the added one-time tuning cost.
