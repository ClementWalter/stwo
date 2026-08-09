use std::time::{Duration, Instant};

use metal::{Buffer, MTLResourceOptions, MTLResourceUsage, MTLSize};
use rayon::prelude::*;

use super::*;
use crate::core::fields::batch_inverse_in_place;

struct PreparedGpuCombine {
    accumulation_buffers: Vec<Buffer>,
    address_buffer: Buffer,
    xs_buffer: Buffer,
    ys_buffer: Buffer,
    direct_outputs: Vec<Buffer>,
    batched_outputs: Vec<Buffer>,
    pole_report: Buffer,
    samples: Vec<super::super::SampleParams>,
    params: super::super::CombineParams,
}

fn run_gpu(
    context: &super::super::QuotientContext,
    prepared: &PreparedGpuCombine,
    direct: bool,
) -> Duration {
    let started = Instant::now();
    let command_buffer = context.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    let pipeline = if direct {
        context.direct_combine_pipeline.as_ref().unwrap()
    } else {
        context.combine_pipeline.as_ref().unwrap()
    };
    let outputs = if direct {
        &prepared.direct_outputs
    } else {
        &prepared.batched_outputs
    };
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&prepared.address_buffer), 0);
    encoder.set_buffer(1, Some(&prepared.xs_buffer), 0);
    encoder.set_buffer(2, Some(&prepared.ys_buffer), 0);
    for (coordinate, buffer) in outputs.iter().enumerate() {
        encoder.set_buffer(3 + coordinate as u64, Some(buffer), 0);
    }
    encoder.set_bytes(
        7,
        std::mem::size_of_val(prepared.samples.as_slice()) as u64,
        prepared.samples.as_ptr() as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        8,
        std::mem::size_of::<super::super::CombineParams>() as u64,
        &prepared.params as *const _ as *const std::ffi::c_void,
    );
    if !direct {
        encoder.set_buffer(9, Some(&prepared.pole_report), 0);
    }
    for buffer in &prepared.accumulation_buffers {
        encoder.use_resource(buffer, MTLResourceUsage::Read);
    }
    if direct {
        encoder.dispatch_threads(
            MTLSize::new(prepared.params.n_rows as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    } else {
        encoder.dispatch_thread_groups(
            MTLSize::new((prepared.params.n_rows as usize).div_ceil(256) as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    }
    encoder.end_encoding();
    command_buffer.commit();
    super::super::super::context::wait_for_completion(command_buffer)
        .expect("timed quotient command must complete");
    started.elapsed()
}

fn run_cpu(
    accumulations: &[AccumulatedNumerators<CpuBackend>],
    sample_points: &[CirclePoint<SecureField>],
    points: &[CirclePoint<BaseField>],
    denominators_buffer: &mut [CM31],
    inverses: &mut [CM31],
    output: &mut [SecureField],
    log_size: u32,
) -> Duration {
    const CHUNK_ROWS: usize = 4096;
    let n_samples = accumulations.len();
    let started = Instant::now();
    denominators_buffer
        .par_chunks_mut(n_samples)
        .enumerate()
        .for_each(|(row, destination)| {
            for (slot, denominator) in denominators(sample_points, points[row]).enumerate() {
                destination[slot] = denominator;
            }
        });
    denominators_buffer
        .par_chunks(CHUNK_ROWS * n_samples)
        .zip(inverses.par_chunks_mut(CHUNK_ROWS * n_samples))
        .for_each(|(source, destination)| batch_inverse_in_place(source, destination));
    output.par_iter_mut().enumerate().for_each(|(row, value)| {
        let point = points[row];
        let mut quotient = SecureField::default();
        for (sample, accumulation) in accumulations.iter().enumerate() {
            let log_ratio = log_size - accumulation.partial_numerators_acc.len().ilog2();
            let lifted = ((row >> (log_ratio + 1)) << 1) | (row & 1);
            let numerator = accumulation.partial_numerators_acc.at(lifted)
                - accumulation.first_linear_term_acc * point.y;
            quotient += numerator.mul_cm31(inverses[row * n_samples + sample]);
        }
        *value = quotient;
    });
    let elapsed = started.elapsed();
    std::hint::black_box(output);
    elapsed
}

fn summary(mut samples: [Duration; 7]) -> (Duration, Duration) {
    samples.sort_unstable();
    (samples[3], samples[6])
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

/// Manual bounded stage benchmark. All Rust vectors and Metal data buffers are
/// allocated before warmup/timing; timed GPU wall includes only command creation,
/// encoding, submission, checked wait, and kernel execution.
#[test]
#[ignore]
fn quotient_batch_bench() {
    const LOG_SIZE: u32 = 18;
    const TRIALS: usize = 7;
    let subdomain = CanonicCoset::new(LOG_SIZE).circle_domain();
    let n_rows = subdomain.size();
    let accumulations = (0..6)
        .map(|sample| seeded_accumulation(LOG_SIZE, [0, 1, 3][sample % 3], sample))
        .collect::<Vec<_>>();
    let (xs, ys) = domain_xy(subdomain);
    let points = xs
        .iter()
        .copied()
        .zip(ys.iter().copied())
        .map(|(x, y)| CirclePoint { x, y })
        .collect::<Vec<_>>();
    let sample_points = accumulations
        .iter()
        .map(|accumulation| accumulation.sample_point)
        .collect::<Vec<_>>();

    let context = super::super::context().expect("benchmark requires unified-memory Metal");
    let mut context = context.lock().unwrap();
    super::super::ensure_combine_pipeline(&mut context).expect("batched pipeline must compile");
    super::super::ensure_direct_combine_pipeline(&mut context)
        .expect("direct control pipeline must compile");
    let samples = accumulations
        .iter()
        .map(|accumulation| {
            let pair = |value: CM31| [value.0 .0, value.1 .0];
            let flt = accumulation.first_linear_term_acc.to_m31_array();
            super::super::SampleParams {
                prx: pair(accumulation.sample_point.x.0),
                pix: pair(accumulation.sample_point.x.1),
                pry: pair(accumulation.sample_point.y.0),
                piy: pair(accumulation.sample_point.y.1),
                flt0: [flt[0].0, flt[1].0],
                flt1: [flt[2].0, flt[3].0],
                log_ratio: LOG_SIZE - accumulation.partial_numerators_acc.len().ilog2(),
                _pad: 0,
            }
        })
        .collect::<Vec<_>>();
    let mut accumulation_buffers = Vec::new();
    let mut addresses = Vec::new();
    for accumulation in &accumulations {
        for column in &accumulation.partial_numerators_acc.columns {
            let buffer = super::super::bind_input(&context.device, column);
            addresses.push(buffer.gpu_address());
            accumulation_buffers.push(buffer);
        }
    }
    let address_buffer = context.device.new_buffer_with_data(
        addresses.as_ptr() as *const std::ffi::c_void,
        std::mem::size_of_val(addresses.as_slice()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let xs_buffer = super::super::bind_input(&context.device, &xs);
    let ys_buffer = super::super::bind_input(&context.device, &ys);
    let output_bytes = (n_rows * std::mem::size_of::<BaseField>()) as u64;
    let direct_outputs = (0..4)
        .map(|_| {
            context
                .device
                .new_buffer(output_bytes, MTLResourceOptions::StorageModeShared)
        })
        .collect();
    let batched_outputs = (0..4)
        .map(|_| {
            context
                .device
                .new_buffer(output_bytes, MTLResourceOptions::StorageModeShared)
        })
        .collect();
    let initial_pole_report = [
        0,
        u32::MAX,
        u32::MAX,
        u32::MAX,
        u32::MAX,
        u32::MAX,
        u32::MAX,
    ];
    let pole_report = context.device.new_buffer_with_data(
        initial_pole_report.as_ptr() as *const std::ffi::c_void,
        std::mem::size_of_val(&initial_pole_report) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let prepared = PreparedGpuCombine {
        accumulation_buffers,
        address_buffer,
        xs_buffer,
        ys_buffer,
        direct_outputs,
        batched_outputs,
        pole_report,
        samples,
        params: super::super::CombineParams {
            n_rows: n_rows as u32,
            n_samples: accumulations.len() as u32,
        },
    };
    let mut denominator_buffer = vec![CM31::default(); n_rows * accumulations.len()];
    let mut inverses = denominator_buffer.clone();
    let mut cpu_output = vec![SecureField::default(); n_rows];

    let _ = run_cpu(
        &accumulations,
        &sample_points,
        &points,
        &mut denominator_buffer,
        &mut inverses,
        &mut cpu_output,
        LOG_SIZE,
    );
    let _ = run_gpu(&context, &prepared, true);
    let _ = run_gpu(&context, &prepared, false);

    let mut cpu_times = [Duration::ZERO; TRIALS];
    let mut direct_times = [Duration::ZERO; TRIALS];
    let mut batched_times = [Duration::ZERO; TRIALS];
    for trial in 0..TRIALS {
        match trial % 3 {
            0 => {
                cpu_times[trial] = run_cpu(
                    &accumulations,
                    &sample_points,
                    &points,
                    &mut denominator_buffer,
                    &mut inverses,
                    &mut cpu_output,
                    LOG_SIZE,
                );
                direct_times[trial] = run_gpu(&context, &prepared, true);
                batched_times[trial] = run_gpu(&context, &prepared, false);
            }
            1 => {
                direct_times[trial] = run_gpu(&context, &prepared, true);
                batched_times[trial] = run_gpu(&context, &prepared, false);
                cpu_times[trial] = run_cpu(
                    &accumulations,
                    &sample_points,
                    &points,
                    &mut denominator_buffer,
                    &mut inverses,
                    &mut cpu_output,
                    LOG_SIZE,
                );
            }
            _ => {
                batched_times[trial] = run_gpu(&context, &prepared, false);
                cpu_times[trial] = run_cpu(
                    &accumulations,
                    &sample_points,
                    &points,
                    &mut denominator_buffer,
                    &mut inverses,
                    &mut cpu_output,
                    LOG_SIZE,
                );
                direct_times[trial] = run_gpu(&context, &prepared, true);
            }
        }
    }
    let (cpu_median, cpu_p95) = summary(cpu_times);
    let (direct_median, direct_p95) = summary(direct_times);
    let (batched_median, batched_p95) = summary(batched_times);
    std::println!(
        "quotient-combine log={LOG_SIZE} samples=6 trials={TRIALS}: CPU reference median={:.3}ms p95={:.3}ms; direct Metal median={:.3}ms p95={:.3}ms; batched Metal median={:.3}ms p95={:.3}ms; direct-to-batched={:.2}x",
        milliseconds(cpu_median), milliseconds(cpu_p95),
        milliseconds(direct_median), milliseconds(direct_p95),
        milliseconds(batched_median), milliseconds(batched_p95),
        direct_median.as_secs_f64() / batched_median.as_secs_f64(),
    );

    for outputs in [&prepared.direct_outputs, &prepared.batched_outputs] {
        for (coordinate, buffer) in outputs.iter().enumerate() {
            // SAFETY: each live StorageModeShared buffer has exactly `n_rows`
            // u32-aligned words, the last checked command completed, and no GPU or host
            // mutation occurs while this immutable slice is compared.
            let actual = unsafe {
                std::slice::from_raw_parts(buffer.contents() as *const BaseField, n_rows)
            };
            for (row, value) in actual.iter().enumerate() {
                assert_eq!(*value, cpu_output[row].to_m31_array()[coordinate]);
            }
        }
    }
}
