# MXFP8 exhaustive tactic validation

This experiment compares the existing pruned TensorRT-LLM tactic list with all
manifest tactics that pass `isValidConfig()` for dynamic MXFP8 quantization.

The two source trees share the same PR 4657 and FlashInfer main merge base:

- `465a4aba`: pruned tactic control
- `6fab535d`: exhaustive tactic candidate

The GB200 run records candidate counts, selected tactics, one-time autotuning
wall time, kernel latency, and reference-check results. Build and JIT caches use
node-local `/raid/scratch`; durable CSV, JSON, metadata, and logs use `/lustre`.
The harness forces the TRTLLM wrapper through source JIT so an installed AOT
module cannot hide C++ changes in the checkout.

Run on Lyris from the experiment checkout:

```bash
SBATCH_TEST_ONLY=1 ./experiments/mxfp8_exhaustive_tactics_20260906/submit.sh
./experiments/mxfp8_exhaustive_tactics_20260906/submit.sh
```
