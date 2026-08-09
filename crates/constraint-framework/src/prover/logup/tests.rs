use std::array;
use std::hint::black_box;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Instant;

use num_traits::Zero;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use stwo::core::fields::m31::P;
use stwo::core::fields::qm31::SecureField;
use stwo::core::fields::FieldExpOps;
use stwo::prover::backend::simd::m31::{LOG_N_LANES, N_LANES};
use stwo::prover::backend::simd::qm31::PackedSecureField;
use stwo::prover::backend::Column;

use crate::prover::logup::LogupTraceGenerator;
use crate::{m31, qm31};

#[test]
fn test_frac_writer() {
    let expected_sum = (qm31!(1, 2, 3, 4) * qm31!(5, 6, 7, 8).inverse()) * m31!(1 << 6);

    let mut log_gen = LogupTraceGenerator::new(6);
    let mut col_gen = log_gen.new_col();
    for writer in col_gen.iter_mut() {
        let num = PackedSecureField::broadcast(qm31!(1, 2, 3, 4));
        let den = PackedSecureField::broadcast(qm31!(5, 6, 7, 8));
        writer.write_frac(num, den);
    }

    col_gen.finalize_col();
    let (_, sum) = log_gen.finalize_last();
    assert_eq!(sum, expected_sum);
}

#[test]
fn test_col_from_iter() {
    let log_size = 8;
    let expected_sum = (qm31!(1, 2, 3, 4) * qm31!(5, 6, 7, 8).inverse()) * m31!(1 << log_size);

    let mut log_gen = LogupTraceGenerator::new(log_size);
    let col_iter = (0..1 << (log_size - LOG_N_LANES)).map(|_| {
        let num = PackedSecureField::broadcast(qm31!(1, 2, 3, 4));
        let den = PackedSecureField::broadcast(qm31!(5, 6, 7, 8));
        (num, den)
    });
    log_gen.col_from_iter(col_iter);

    let (_, sum) = log_gen.finalize_last();
    assert_eq!(sum, expected_sum);
}

fn denominator_with_one_zero_lane(zero: SecureField) -> PackedSecureField {
    let mut lanes = array::from_fn(|lane| {
        qm31!(
            lane as u32 + 1,
            lane as u32 + 2,
            lane as u32 + 3,
            lane as u32 + 4
        )
    });
    lanes[N_LANES / 2] = zero;
    PackedSecureField::from_array(lanes)
}

fn single_fraction_rejects(denominator: PackedSecureField) -> bool {
    catch_unwind(AssertUnwindSafe(|| {
        let mut generator = LogupTraceGenerator::new(LOG_N_LANES);
        generator.col_from_iter(std::iter::once((
            PackedSecureField::broadcast(qm31!(1, 2, 3, 4)),
            denominator,
        )));
    }))
    .is_err()
}

/// This check is intentionally unconditional: this same test is run under
/// `cargo test --release` so a pole cannot turn into a silent zero inverse
/// when debug assertions are compiled out.
#[test]
fn test_one_zero_denominator_lane_amid_nonzero_lanes_is_rejected() {
    assert!(single_fraction_rejects(denominator_with_one_zero_lane(
        SecureField::zero()
    )));
}

#[test]
fn test_every_raw_zero_coordinate_variant_is_rejected() {
    // M31 permits both 0 and P as raw representatives of zero. Every one
    // of the 2^4 coordinate combinations below is therefore a QM31 pole.
    for p_mask in 0_u32..1 << 4 {
        let raw = |coordinate: u32| {
            if p_mask & (1 << coordinate) == 0 {
                0
            } else {
                P
            }
        };
        let zero = SecureField::from_u32_unchecked(raw(0), raw(1), raw(2), raw(3));
        assert!(
            single_fraction_rejects(denominator_with_one_zero_lane(zero)),
            "raw P mask {p_mask:#06b} was accepted"
        );
    }
}

#[test]
fn test_cols_from_fn_rejects_a_zero_denominator_lane() {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut generator = LogupTraceGenerator::new(LOG_N_LANES);
        generator.cols_from_fn(1, |_, _| {
            (
                PackedSecureField::broadcast(qm31!(1, 2, 3, 4)),
                denominator_with_one_zero_lane(qm31!(P, 0, P, 0)),
            )
        });
    }));
    assert!(result.is_err());
}

#[test]
fn test_cols_from_fn_matches_legacy_batching_for_odd_and_even_entry_counts() {
    fn entry_fraction(entry: usize, vec_row: usize) -> (PackedSecureField, PackedSecureField) {
        let numerator = array::from_fn(|lane| {
            let raw_zero = if (entry + vec_row + lane) % 31 == 0 {
                stwo::core::fields::m31::P
            } else {
                0
            };
            qm31!(
                raw_zero,
                (entry * 17 + vec_row * 11 + lane + 2) as u32,
                (entry * 13 + vec_row * 3 + lane + 3) as u32,
                (entry * 5 + vec_row * 19 + lane + 4) as u32
            )
        });
        // A non-zero real coordinate makes every lane's denominator non-zero,
        // while all four extension coordinates vary across entries and rows.
        let denominator = array::from_fn(|lane| {
            qm31!(
                (entry * 97 + vec_row * 23 + lane + 5) as u32,
                (entry * 29 + vec_row * 31 + lane + 6) as u32,
                (entry * 37 + vec_row * 41 + lane + 7) as u32,
                (entry * 43 + vec_row * 47 + lane + 8) as u32
            )
        });
        let [numerator, denominator] = [numerator, denominator].map(PackedSecureField::from_array);
        (numerator, denominator)
    }

    fn batched_fraction(
        entry_count: usize,
        batch: usize,
        output_col: usize,
        vec_row: usize,
    ) -> (PackedSecureField, PackedSecureField) {
        let first = output_col * batch;
        let (n0, d0) = entry_fraction(first, vec_row);
        if batch == 1 || first + 1 == entry_count {
            (n0, d0)
        } else {
            let (n1, d1) = entry_fraction(first + 1, vec_row);
            (n0 * d1 + n1 * d0, d0 * d1)
        }
    }

    // 2^14 / 16 = 1024 packed rows, crossing the implementation's
    // 512-row band boundary exactly once.
    let log_size = 14;
    let packed_len = 1 << (log_size - LOG_N_LANES);
    for (batch, entry_count) in [(1_usize, 3_usize), (1, 4), (2, 3), (2, 4)] {
        let output_cols = entry_count.div_ceil(batch);

        let mut legacy = LogupTraceGenerator::new(log_size);
        for output_col in 0..output_cols {
            let mut col = legacy.new_col();
            for vec_row in 0..packed_len {
                let (numerator, denominator) =
                    batched_fraction(entry_count, batch, output_col, vec_row);
                col.write_frac(vec_row, numerator, denominator);
            }
            col.finalize_col();
        }
        let (legacy_trace, legacy_sum) = legacy.finalize_last();

        let mut fused = LogupTraceGenerator::new(log_size);
        fused.cols_from_fn(output_cols, |output_col, vec_row| {
            batched_fraction(entry_count, batch, output_col, vec_row)
        });
        let (fused_trace, fused_sum) = fused.finalize_last();

        assert_eq!(
            legacy_sum, fused_sum,
            "batch={batch}, entry_count={entry_count}"
        );
        assert_eq!(legacy_trace.len(), fused_trace.len());
        for (legacy_col, fused_col) in legacy_trace.iter().zip(&fused_trace) {
            assert_eq!(legacy_col.domain, fused_col.domain);
            assert_eq!(
                legacy_col.values.to_cpu(),
                fused_col.values.to_cpu(),
                "batch={batch}, entry_count={entry_count}"
            );
        }
    }
}

fn pseudo_random_word(mut state: u64) -> u32 {
    state ^= state >> 30;
    state = state.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    state ^= state >> 27;
    state = state.wrapping_mul(0x94d0_49bb_1331_11eb);
    ((state ^ (state >> 31)) % P as u64) as u32
}

fn randomized_fraction(
    seed: u64,
    output_col: usize,
    vec_row: usize,
) -> (PackedSecureField, PackedSecureField) {
    let lane_values = |is_denominator: bool| {
        array::from_fn(|lane| {
            let mut coordinates: [u32; 4] = array::from_fn(|coordinate| {
                let key = seed
                    ^ ((output_col as u64) << 48)
                    ^ ((vec_row as u64) << 16)
                    ^ ((lane as u64) << 8)
                    ^ coordinate as u64;
                let word = pseudo_random_word(key);
                // Exercise the non-canonical raw zero representative in
                // both numerators and otherwise-valid denominators.
                if (key.wrapping_mul(0x9e37_79b9) & 7) == 0 {
                    P
                } else {
                    word
                }
            });
            if is_denominator {
                // Keep every denominator out of the pole set while still
                // allowing 0/P in each of its other raw coordinates.
                coordinates[3] =
                    pseudo_random_word(seed ^ (output_col as u64) ^ (vec_row as u64) ^ lane as u64)
                        % (P - 1)
                        + 1;
            }
            SecureField::from_u32_unchecked(
                coordinates[0],
                coordinates[1],
                coordinates[2],
                coordinates[3],
            )
        })
    };
    let numerator = PackedSecureField::from_array(lane_values(false));
    let denominator = PackedSecureField::from_array(lane_values(true));
    (numerator, denominator)
}

#[test]
fn test_cols_from_fn_randomized_raw_parity_across_full_and_partial_bands() {
    let mut rng = SmallRng::seed_from_u64(0x5e71_5eed_cafe_f00d);
    // Packed row counts are powers of two. Logs below BAND exercise a
    // partial band; log 13 is exact; log 14 spans two complete bands.
    for log_size in [8_u32, 12, 13, 14] {
        for _ in 0..3 {
            let n_cols = rng.gen_range(1_usize..=10);
            let seed = rng.gen::<u64>();
            let packed_len = 1 << (log_size - LOG_N_LANES);

            let mut legacy = LogupTraceGenerator::new(log_size);
            for output_col in 0..n_cols {
                legacy.col_from_iter(
                    (0..packed_len).map(|vec_row| randomized_fraction(seed, output_col, vec_row)),
                );
            }
            let (legacy_trace, legacy_sum) = legacy.finalize_last();

            let mut fused = LogupTraceGenerator::new(log_size);
            fused.cols_from_fn(n_cols, |output_col, vec_row| {
                randomized_fraction(seed, output_col, vec_row)
            });
            let (fused_trace, fused_sum) = fused.finalize_last();

            assert_eq!(
                legacy_sum, fused_sum,
                "raw claimed-sum mismatch: log_size={log_size}, n_cols={n_cols}, seed={seed:#x}"
            );
            assert_eq!(legacy_trace.len(), fused_trace.len());
            for (coordinate, (legacy_col, fused_col)) in
                legacy_trace.iter().zip(&fused_trace).enumerate()
            {
                assert_eq!(legacy_col.domain, fused_col.domain);
                assert_eq!(
                    legacy_col.values.to_cpu(),
                    fused_col.values.to_cpu(),
                    "raw trace mismatch: log_size={log_size}, n_cols={n_cols}, seed={seed:#x}, coordinate={coordinate}"
                );
            }
        }
    }
}

/// Bounded microbenchmark for the exact multi-column generator. At log 20
/// and eight QM31 outputs its peak live storage remains far below 1 GiB.
#[test]
#[ignore = "manual bounded LogUp performance measurement"]
fn benchmark_cols_from_fn_log20_eight_columns() {
    const LOG_SIZE: u32 = 20;
    const N_COLS: usize = 8;
    const SAMPLES: usize = 5;
    let mut elapsed = Vec::with_capacity(SAMPLES);
    for sample in 0..SAMPLES {
        let started = Instant::now();
        let mut generator = LogupTraceGenerator::new(LOG_SIZE);
        generator.cols_from_fn(N_COLS, |output_col, vec_row| {
            let value = ((output_col * 131 + vec_row * 17) as u32 % (P - 1)) + 1;
            (
                PackedSecureField::broadcast(SecureField::from_u32_unchecked(
                    value,
                    P,
                    value.wrapping_mul(3) % P,
                    0,
                )),
                PackedSecureField::broadcast(SecureField::from_u32_unchecked(
                    value,
                    value.wrapping_mul(5) % P,
                    P,
                    value.wrapping_mul(7) % P,
                )),
            )
        });
        let (trace, claimed_sum) = generator.finalize_last();
        black_box(trace.len());
        black_box(claimed_sum);
        let duration = started.elapsed();
        eprintln!("logup synthetic sample {sample}: {duration:?}");
        elapsed.push(duration);
    }
    elapsed.sort_unstable();
    eprintln!("logup synthetic median: {:?}", elapsed[SAMPLES / 2]);
}

#[cfg(feature = "parallel")]
#[test]
fn test_col_from_par_iter() {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    let log_size = 8;
    let expected_sum = (qm31!(1, 2, 3, 4) * qm31!(5, 6, 7, 8).inverse()) * m31!(1 << log_size);

    let mut log_gen = LogupTraceGenerator::new(log_size);
    let col_iter = (0..1 << (log_size - LOG_N_LANES)).into_par_iter().map(|_| {
        let num = PackedSecureField::broadcast(qm31!(1, 2, 3, 4));
        let den = PackedSecureField::broadcast(qm31!(5, 6, 7, 8));
        (num, den)
    });
    log_gen.col_from_par_iter(col_iter);

    let (_, sum) = log_gen.finalize_last();
    assert_eq!(sum, expected_sum);
}

#[cfg(feature = "parallel")]
#[test]
fn test_parallel_frac_writer() {
    use std::array;

    use rayon::prelude::*;
    // Sequential version.
    let mut log_gen_seq = LogupTraceGenerator::new(6);
    let mut col_gen_seq = log_gen_seq.new_col();
    col_gen_seq.iter_mut().enumerate().for_each(|(i, writer)| {
        let num = array::from_fn(|j| qm31!(i as u32, j as u32, 0, 1));
        let den = array::from_fn(|j| qm31!(i as u32, j as u32, 2, 3));
        let [num, den] = [num, den].map(PackedSecureField::from_array);
        writer.write_frac(num, den);
    });
    col_gen_seq.finalize_col();
    let (_, sum_seq) = log_gen_seq.finalize_last();

    // Parallel version.
    let mut log_gen_par = LogupTraceGenerator::new(6);
    let mut col_gen_par = log_gen_par.new_col();
    col_gen_par
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, writer)| {
            let num = array::from_fn(|j| qm31!(i as u32, j as u32, 0, 1));
            let den = array::from_fn(|j| qm31!(i as u32, j as u32, 2, 3));
            let [num, den] = [num, den].map(PackedSecureField::from_array);
            writer.write_frac(num, den);
        });
    col_gen_par.finalize_col();
    let (_, sum_par) = log_gen_par.finalize_last();

    assert_eq!(sum_seq, sum_par);
}
