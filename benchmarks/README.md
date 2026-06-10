# Prover benchmark campaign

`run_benchmarks.py` measures end-to-end proof times (prove + verify, including trace
generation and setup) for every runnable example, comparing pre-built test binaries:

- **baseline** — `dev` (latest starkware-libs/stwo + the pull-and-pin commit), with the
  benchmark tests and an env-gated proof-hash dump patched in (no prover changes),
- **optimized-cpu** — this branch without the `metal` feature (CPU/SIMD work only),
- **optimized+metal** — this branch with the `metal` feature (Apple-GPU kernels).

Configurations are interleaved per (program, size) so ambient load affects all sides
equally; every run is status-checked, and the proof hashes of all configurations are
compared — the campaign fails on any mismatch, so a passing run certifies that the
optimized prover produces **bit-identical proofs** to the baseline on every
experiment.

## Running

Build the three test binaries (`cargo test -p stwo-examples --release --features
parallel,slow-tests[,metal] --no-run` in each checkout), then:

```
uv run benchmarks/run_benchmarks.py \
    --baseline <baseline test binary> \
    --metal <optimized metal binary> \
    --nometal <optimized cpu-only binary> \
    --pairs 3
```

## Artifacts

Each run writes `benchmarks/results/<timestamp>/`:

- `table.md` — median times per configuration, speedup, and the bit-identity verdict
- `results.csv` — every individual run (program, size, config, pair, status, seconds, hash)
- `runs.jsonl` — the same records as JSON lines
- `env.txt` — machine, chip, and binary provenance

Programs that a configuration cannot run (e.g. `simd_poseidon` is unsupported by the
lifted protocol upstream) are recorded with their failure status and excluded from the
table rather than silently dropped.
