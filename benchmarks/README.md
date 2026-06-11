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

## AIR-shape throughput sweep

`run_air_shapes.py` measures prover throughput in **Mcells/s** over a grid of trace
shapes (2^14–2^22 rows x 16–256 columns) using the parametric `air_shape` example
(SimdBackend), in two modes:

- **constrained** — one degree-2 constraint per derived column (real-AIR workload),
- **unconstrained** — masks only, measuring the commitment/FRI envelope (in the
  spirit of kkrt-labs/rookie-numbers' frequency benchmark).

The timed section is the protocol itself (twiddle precompute through proof),
excluding trace generation. Same interleaving and bit-identity guarantees as the
main campaign. Note the SimdBackend path does not route through the Metal kernels
(the GPU chained commit currently specializes the CpuBackend route only), so this
sweep isolates the CPU/SIMD-side gains.

```
uv run benchmarks/run_air_shapes.py --baseline <bin> --metal <bin> --nometal <bin>
```
