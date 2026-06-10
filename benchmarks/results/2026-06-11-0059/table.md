| program | size | baseline | optimized+metal | optimized-cpu | speedup (metal) | bit-identical |
|---|---|---|---|---|---|---|
| cpu_wide_fib | 14 | 0.15s | 0.14s | 0.02s | **1.1x** | yes |
| cpu_wide_fib | 16 | 0.58s | 0.18s | 0.06s | **3.2x** | yes |
| cpu_wide_fib | 18 | 2.42s | 0.28s | 0.22s | **8.6x** | yes |
| cpu_wide_fib | 20 | 9.50s | 0.58s | 0.78s | **16.4x** | yes |
| cpu_wide_fib | 22 | 39.35s | 1.77s | 2.70s | **22.2x** | yes |
| simd_blake | 10 | 0.76s | 0.56s | 0.57s | **1.4x** | yes |
| simd_blake | 12 | 1.04s | 0.69s | 0.69s | **1.5x** | yes |
| simd_blake | 14 | 1.96s | 1.19s | — | **1.6x** | yes |
| simd_blake | 16 | 6.01s | 3.24s | — | **1.9x** | yes |
| simd_plonk | 12 | 0.03s | 0.02s | 0.02s | **1.5x** | yes |
| simd_plonk | 14 | 0.07s | 0.06s | 0.06s | **1.2x** | yes |
| simd_plonk | 16 | 0.22s | 0.18s | 0.17s | **1.2x** | yes |
| simd_wide_fib | fixed | 0.01s | 0.01s | 0.01s | **1.0x** | yes |
| state_machine | fixed | 0.00s | 0.00s | 0.00s | — | yes |
