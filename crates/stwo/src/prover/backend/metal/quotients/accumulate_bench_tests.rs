use std::time::{Duration, Instant};

use rayon::prelude::*;

use super::accumulate::{
    accumulate_numerators_metal, accumulate_numerators_metal_serial, NumeratorBatch,
};
use crate::core::fields::m31::{BaseField, P};
use crate::core::fields::qm31::SecureField;
use crate::prover::backend::simd::m31::{PackedM31, N_LANES};

type Shape = (u32, usize, &'static [usize]);

const SHA_2KIB: &[Shape] = &[
    (16, 80, &[80, 8, 8]),
    (17, 73, &[73, 4, 4]),
    (18, 9, &[9, 4, 4]),
    (19, 7, &[7, 4, 4]),
    (20, 22, &[30, 8]),
];
const KECCAK_2KIB: &[Shape] = &[
    (16, 80, &[80, 8, 8]),
    (17, 80, &[80, 8, 8]),
    (18, 151, &[151, 8, 8]),
    (19, 9, &[9, 4, 4]),
    (20, 22, &[30, 8]),
];
const SHA_2KIB_WIDE: &[Shape] = &[
    (15, 560, &[560, 16, 16]),
    (16, 80, &[80, 8, 8]),
    (17, 73, &[73, 4, 4]),
    (18, 9, &[9, 4, 4]),
    (19, 7, &[7, 4, 4]),
    (20, 22, &[30, 8]),
];
const KECCAK_2KIB_WIDE: &[Shape] = &[
    (14, 475, &[475, 8, 8]),
    (15, 68, &[68, 8, 8]),
    (16, 80, &[80, 8, 8]),
    (17, 80, &[80, 8, 8]),
    (18, 151, &[151, 8, 8]),
    (19, 9, &[9, 4, 4]),
    (20, 22, &[30, 8]),
];

fn build_storage(shapes: &[Shape]) -> Vec<Vec<Vec<BaseField>>> {
    shapes
        .iter()
        .enumerate()
        .map(|(group, &(log_rows, columns, _))| {
            (0..columns)
                .map(|column| {
                    (0..1usize << log_rows)
                        .map(|row| {
                            BaseField::from_u32_unchecked(
                                (row as u32)
                                    .wrapping_mul(0x045d_9f3b ^ group as u32)
                                    .wrapping_add(17 + column as u32 * 101)
                                    % P,
                            )
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

fn build_batches<'a>(
    shapes: &[Shape],
    storage: &'a [Vec<Vec<BaseField>>],
) -> Vec<NumeratorBatch<'a>> {
    let mut batch_index = 0usize;
    let mut batches = Vec::new();
    for (&(log_rows, group_columns, batch_counts), columns) in shapes.iter().zip(storage) {
        for &column_count in batch_counts {
            let coeffs = (0..column_count)
                .map(|column| {
                    let seed = 1 + batch_index as u32 * 97 + column as u32 * 13;
                    SecureField::from_u32_unchecked(seed, seed + 2, seed + 4, seed + 6)
                })
                .collect();
            batches.push(NumeratorBatch {
                columns: (0..column_count)
                    .map(|column| columns[column % group_columns].as_slice())
                    .collect(),
                coeffs,
                neg_b_sum: SecureField::from_u32_unchecked(
                    11 + batch_index as u32,
                    23 + batch_index as u32,
                    37 + batch_index as u32,
                    53 + batch_index as u32,
                ),
                n_rows: 1 << log_rows,
            });
            batch_index += 1;
        }
    }
    batches
}

fn run_packed_cpu(batches: &[NumeratorBatch<'_>]) -> Duration {
    const CHUNK_ROWS: usize = 1 << 12;
    let started = Instant::now();
    let outputs = batches
        .iter()
        .map(|batch| {
            let mut output: [Vec<BaseField>; 4] =
                std::array::from_fn(|_| vec![BaseField::default(); batch.n_rows]);
            let mut chunks = {
                let [c0, c1, c2, c3] = &mut output;
                c0.chunks_mut(CHUNK_ROWS)
                    .zip(c1.chunks_mut(CHUNK_ROWS))
                    .zip(c2.chunks_mut(CHUNK_ROWS))
                    .zip(c3.chunks_mut(CHUNK_ROWS))
                    .enumerate()
                    .map(|(index, (((d0, d1), d2), d3))| (index * CHUNK_ROWS, [d0, d1, d2, d3]))
                    .collect::<Vec<_>>()
            };
            chunks.par_iter_mut().for_each(|(start, destination)| {
                let rows = destination[0].len();
                assert!(rows.is_multiple_of(N_LANES));
                let groups = rows / N_LANES;
                let init = batch.neg_b_sum.to_m31_array();
                let mut acc: [Vec<PackedM31>; 4] =
                    init.map(|value| vec![PackedM31::broadcast(value); groups]);
                for (column, coefficient) in batch.columns.iter().zip(&batch.coeffs) {
                    let packed_coefficients = coefficient.to_m31_array().map(PackedM31::broadcast);
                    for (group, values) in column[*start..*start + rows]
                        .chunks_exact(N_LANES)
                        .enumerate()
                    {
                        let value = PackedM31::from_array(values.try_into().unwrap());
                        for coordinate in 0..4 {
                            acc[coordinate][group] += value * packed_coefficients[coordinate];
                        }
                    }
                }
                for coordinate in 0..4 {
                    for (group, value) in acc[coordinate].iter().enumerate() {
                        destination[coordinate][group * N_LANES..(group + 1) * N_LANES]
                            .copy_from_slice(&value.to_array());
                    }
                }
            });
            output
        })
        .collect::<Vec<_>>();
    std::hint::black_box(outputs);
    started.elapsed()
}

fn run_serial_metal(batches: &[NumeratorBatch<'_>]) -> Duration {
    let started = Instant::now();
    let outputs = batches
        .iter()
        .map(|batch| {
            accumulate_numerators_metal_serial(batch)
                .expect("serial Metal control command must succeed")
                .expect("serial Metal control shape must dispatch")
        })
        .collect::<Vec<_>>();
    std::hint::black_box(outputs);
    started.elapsed()
}

fn run_bulk_metal(batches: &[NumeratorBatch<'_>]) -> Duration {
    let started = Instant::now();
    let outputs = accumulate_numerators_metal(batches)
        .expect("bulk Metal command must succeed")
        .expect("bulk Metal shape must dispatch");
    std::hint::black_box(outputs);
    started.elapsed()
}

fn median(mut times: [Duration; 5]) -> Duration {
    times.sort_unstable();
    times[2]
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn benchmark_workload(name: &str, shapes: &[Shape]) {
    let storage = build_storage(shapes);
    let batches = build_batches(shapes, &storage);
    let batch_count = batches.len();
    let _ = run_packed_cpu(&batches);
    let _ = run_serial_metal(&batches);
    let _ = run_bulk_metal(&batches);

    let mut cpu = [Duration::ZERO; 5];
    let mut serial = [Duration::ZERO; 5];
    let mut bulk = [Duration::ZERO; 5];
    for trial in 0..5 {
        match trial % 3 {
            0 => {
                cpu[trial] = run_packed_cpu(&batches);
                serial[trial] = run_serial_metal(&batches);
                bulk[trial] = run_bulk_metal(&batches);
            }
            1 => {
                serial[trial] = run_serial_metal(&batches);
                bulk[trial] = run_bulk_metal(&batches);
                cpu[trial] = run_packed_cpu(&batches);
            }
            _ => {
                bulk[trial] = run_bulk_metal(&batches);
                cpu[trial] = run_packed_cpu(&batches);
                serial[trial] = run_serial_metal(&batches);
            }
        }
    }
    let (cpu, serial, bulk) = (median(cpu), median(serial), median(bulk));
    std::println!(
        "{name} exact numerator batches={batch_count}: packed CPU={:.3}ms; old {batch_count}-command \
         Metal={:.3}ms; bulk one-command Metal={:.3}ms; CPU-to-bulk={:.2}x; old-to-bulk={:.2}x",
        milliseconds(cpu),
        milliseconds(serial),
        milliseconds(bulk),
        cpu.as_secs_f64() / bulk.as_secs_f64(),
        serial.as_secs_f64() / bulk.as_secs_f64(),
    );

    let mut start = 0usize;
    for &(log_rows, _, batch_counts) in shapes {
        let end = start + batch_counts.len();
        let group = &batches[start..end];
        let mut cpu = [Duration::ZERO; 5];
        let mut bulk = [Duration::ZERO; 5];
        for trial in 0..5 {
            if trial % 2 == 0 {
                cpu[trial] = run_packed_cpu(group);
                bulk[trial] = run_bulk_metal(group);
            } else {
                bulk[trial] = run_bulk_metal(group);
                cpu[trial] = run_packed_cpu(group);
            }
        }
        let (cpu, bulk) = (median(cpu), median(bulk));
        let terms = batch_counts.iter().sum::<usize>();
        std::println!(
            "  subdomain-log{log_rows} terms={terms}: packed CPU={:.3}ms; bulk Metal={:.3}ms; \
             CPU-to-Metal={:.2}x",
            milliseconds(cpu),
            milliseconds(bulk),
            cpu.as_secs_f64() / bulk.as_secs_f64(),
        );
        start = end;
    }
}

#[test]
#[ignore = "manual bounded Apple-GPU stage benchmark"]
fn exact_sha_keccak_2kib_bulk_numerator_bench() {
    benchmark_workload("SHA-256/2KiB", SHA_2KIB);
    benchmark_workload("Keccak/2KiB", KECCAK_2KIB);
    benchmark_workload("SHA-256/2KiB log15+", SHA_2KIB_WIDE);
    benchmark_workload("Keccak/2KiB log14+", KECCAK_2KIB_WIDE);
}
