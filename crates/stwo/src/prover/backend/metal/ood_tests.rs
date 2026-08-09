use std::hint::black_box;
use std::time::Instant;

use itertools::Itertools;

use super::{eval_m31_columns_metal, should_batch_barycentric_on_metal, MIN_METAL_OOD_LOG_SIZE};
use crate::core::circle::{CirclePoint, CirclePointIndex, SECURE_FIELD_CIRCLE_GEN};
use crate::core::constraints::point_vanishing;
use crate::core::fields::m31::{BaseField, P};
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldExpOps;
use crate::core::poly::circle::CanonicCoset;
use crate::prover::backend::metal::{MetalRequirement, MetalSession};
use crate::prover::backend::CpuBackend;
use crate::prover::poly::circle::{BarycentricEvalGroup, CircleEvaluation, PolyOps};
use crate::prover::poly::BitReversedOrder;

fn test_values(log_size: u32, seed: u32) -> Vec<BaseField> {
    let n = 1 << log_size;
    (0..n)
        .map(|i| {
            let raw = (i as u32).wrapping_mul(0x045d_9f3b).wrapping_add(seed) & P;
            match i {
                0 => BaseField::from_u32_unchecked(0),
                1 | 257 => BaseField::from_u32_unchecked(P),
                _ => BaseField::from_u32_unchecked(raw),
            }
        })
        .collect()
}

fn evaluations(log_size: u32) -> Vec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> {
    let domain = CanonicCoset::new(log_size).circle_domain();
    [19, 3, 11]
        .into_iter()
        .map(|seed| CircleEvaluation::new(domain, test_values(log_size, seed)))
        .collect()
}

#[test]
fn batched_barycentric_metal_matches_legacy_across_logs_points_and_raw_zero() {
    let session = MetalSession::admit().expect("Metal OOD test requires a unified-memory GPU");
    let points = [
        SECURE_FIELD_CIRCLE_GEN,
        SECURE_FIELD_CIRCLE_GEN.mul(1_234_567),
    ];

    for log_size in [MIN_METAL_OOD_LOG_SIZE, MIN_METAL_OOD_LOG_SIZE + 1] {
        let evals = evaluations(log_size);
        let refs = evals.iter().collect_vec();
        let columns = refs
            .iter()
            .map(|evaluation| evaluation.values.as_slice())
            .collect_vec();
        for point in points {
            let weights =
                <CpuBackend as PolyOps>::barycentric_weights(CanonicCoset::new(log_size), point);
            let expected = refs
                .iter()
                .map(|eval| <CpuBackend as PolyOps>::barycentric_eval_at_point(eval, &weights))
                .collect_vec();
            let actual = eval_m31_columns_metal(&columns, &weights)
                .expect("eligible barycentric batch must dispatch to Metal");
            assert_eq!(actual, expected, "log_size={log_size}, point={point:?}");
        }
    }

    let report = session.finish();
    assert_eq!(report.failed_submissions, 0);
    assert!(report.successful_submissions >= 4);
    MetalRequirement::ParticipationRequired
        .validate(&report)
        .expect("checked Metal dispatches must succeed");
}

#[test]
fn batched_dot_metal_reduces_raw_p_representatives_exactly() {
    let log_size = MIN_METAL_OOD_LOG_SIZE;
    let evals = evaluations(log_size);
    let refs = evals.iter().collect_vec();
    let columns = refs
        .iter()
        .map(|evaluation| evaluation.values.as_slice())
        .collect_vec();
    let weights = (0..1 << log_size)
        .map(|i| match i {
            0 => SecureField::from_u32_unchecked(0, P, 0, P),
            1 | 257 => SecureField::from_u32_unchecked(P, 0, P, 0),
            _ => SecureField::from_u32_unchecked(
                i as u32 & P,
                (i as u32).wrapping_mul(3) & P,
                (i as u32).wrapping_mul(5) & P,
                (i as u32).wrapping_mul(7) & P,
            ),
        })
        .collect_vec();
    let expected = refs
        .iter()
        .map(|eval| <CpuBackend as PolyOps>::barycentric_eval_at_point(eval, &weights))
        .collect_vec();
    let actual = eval_m31_columns_metal(&columns, &weights)
        .expect("eligible raw-representation batch must dispatch to Metal");
    assert_eq!(actual, expected);
}

#[test]
fn barycentric_placement_gate_matches_measured_crossover() {
    assert!(!should_batch_barycentric_on_metal(1 << 20, 7));
    assert!(!should_batch_barycentric_on_metal(1 << 20, 69));
    assert!(!should_batch_barycentric_on_metal(1 << 23, 32));
    assert!(should_batch_barycentric_on_metal(1 << 21, 64));
    assert!(should_batch_barycentric_on_metal(1 << 21, 148));
    assert!(should_batch_barycentric_on_metal(1 << 23, 69));
}

#[test]
fn barycentric_block_tables_match_bit_reversed_domain_points() {
    for log_size in [8, 9, 16, 21, 23] {
        let domain = CanonicCoset::new(log_size).circle_domain();
        let tables = super::barycentric_point_tables(log_size);
        let n = 1usize << log_size;
        let mut rows = vec![0, 1, 2, 127, 128, 255, 256.min(n - 1), n / 2, n - 2, n - 1];
        rows.extend((0..64).map(|i| (i * 0x9e37_79b9usize + log_size as usize * 17) & (n - 1)));
        for row in rows {
            let block = row >> 8;
            let lane = row & 255;
            let base = tables.bases[block];
            let offset = tables.offsets[lane >> 1];
            let mut actual = CirclePoint {
                x: BaseField::from_u32_unchecked(base[0]),
                y: BaseField::from_u32_unchecked(base[1]),
            } + CirclePoint {
                x: BaseField::from_u32_unchecked(offset[0]),
                y: BaseField::from_u32_unchecked(offset[1]),
            };
            if !lane.is_multiple_of(2) {
                actual = actual.conjugate();
            }
            let expected = domain.at(super::bit_reverse_index(row, log_size));
            assert_eq!(actual, expected, "log_size={log_size}, row={row}");
        }
    }
}

#[test]
fn barycentric_inverse_uses_circle_group_subtraction() {
    let p = SECURE_FIELD_CIRCLE_GEN.mul(1_234_567);
    let q = CirclePoint::<SecureField>::zero();
    let h = p - q;
    let required = (SecureField::from(1u32) + h.x) / h.y;
    let point_vanishing_inverse = point_vanishing(q, p).inverse();
    let coordinate_difference_shortcut = p.x / p.y;
    assert_eq!(required, point_vanishing_inverse);
    assert_ne!(required, coordinate_difference_shortcut);
}

#[test]
fn resident_barycentric_weights_match_simd_reference() {
    let session = MetalSession::admit().expect("Metal OOD test requires a unified-memory GPU");
    for log_size in [MIN_METAL_OOD_LOG_SIZE, MIN_METAL_OOD_LOG_SIZE + 1] {
        let coset = CanonicCoset::new(log_size);
        for point in [
            SECURE_FIELD_CIRCLE_GEN,
            SECURE_FIELD_CIRCLE_GEN.mul(1_234_567),
        ] {
            let expected = <CpuBackend as PolyOps>::barycentric_weights(coset, point);
            let actual = super::barycentric_weights_metal(coset, point)
                .expect("eligible weight column must dispatch to Metal");
            assert_eq!(actual, expected, "log_size={log_size}, point={point:?}");
        }
    }
    let report = session.finish();
    assert_eq!(report.failed_submissions, 0);
    assert!(report.successful_submissions >= 4);
    MetalRequirement::ParticipationRequired
        .validate(&report)
        .expect("checked Metal dispatches must succeed");
}

#[test]
fn resident_barycentric_groups_match_legacy_in_one_submission() {
    let evals16 = evaluations(MIN_METAL_OOD_LOG_SIZE);
    let evals17 = evaluations(MIN_METAL_OOD_LOG_SIZE + 1);
    let p0 = SECURE_FIELD_CIRCLE_GEN;
    let p1 = SECURE_FIELD_CIRCLE_GEN.mul(1_234_567);
    let groups = [
        BarycentricEvalGroup {
            coset: CanonicCoset::new(MIN_METAL_OOD_LOG_SIZE + 1),
            point: p1,
            evals: evals17.iter().rev().collect_vec(),
        },
        BarycentricEvalGroup {
            coset: CanonicCoset::new(MIN_METAL_OOD_LOG_SIZE),
            point: p0,
            evals: evals16.iter().collect_vec(),
        },
        BarycentricEvalGroup {
            coset: CanonicCoset::new(MIN_METAL_OOD_LOG_SIZE + 1),
            point: p0,
            evals: evals17.iter().collect_vec(),
        },
        BarycentricEvalGroup {
            coset: CanonicCoset::new(MIN_METAL_OOD_LOG_SIZE),
            point: p1,
            evals: evals16.iter().rev().collect_vec(),
        },
    ];
    let expected = groups
        .iter()
        .map(|group| {
            let weights = <CpuBackend as PolyOps>::barycentric_weights(group.coset, group.point);
            group
                .evals
                .iter()
                .map(|eval| <CpuBackend as PolyOps>::barycentric_eval_at_point(eval, &weights))
                .collect_vec()
        })
        .collect_vec();

    let session = MetalSession::admit().expect("Metal OOD test requires a unified-memory GPU");
    let actual = super::barycentric_eval_groups_metal(&groups);
    assert_eq!(actual, expected.into_iter().map(Some).collect_vec());
    let report = session.finish();
    assert_eq!(report.failed_submissions, 0);
    assert_eq!(report.successful_submissions, 1);
    MetalRequirement::ParticipationRequired
        .validate(&report)
        .expect("fused resident Metal dispatch must succeed");
}

#[test]
fn resident_barycentric_rejects_domain_points_before_submission() {
    let log_size = MIN_METAL_OOD_LOG_SIZE;
    let coset = CanonicCoset::new(log_size);
    let evals = evaluations(log_size);
    let q = coset.circle_domain().at(31);
    assert_ne!(q.conjugate(), q.antipode());

    let session = MetalSession::admit().expect("Metal OOD test requires a unified-memory GPU");
    for point in [
        q.into_ef::<SecureField>(),
        q.antipode().into_ef::<SecureField>(),
    ] {
        let group = BarycentricEvalGroup {
            coset,
            point,
            evals: evals.iter().collect_vec(),
        };
        assert!(super::barycentric_weights_metal(coset, point).is_none());
        assert_eq!(super::barycentric_eval_groups_metal(&[group]), vec![None]);
    }
    let report = session.finish();
    assert_eq!(report.successful_submissions, 0);
    assert_eq!(report.failed_submissions, 0);
}

#[test]
fn resident_barycentric_rejects_same_size_domain_mismatch() {
    let log_size = MIN_METAL_OOD_LOG_SIZE;
    let coset = CanonicCoset::new(log_size);
    let shifted_domain = coset.circle_domain().shift(CirclePointIndex::generator());
    let evaluation = CircleEvaluation::new(shifted_domain, test_values(log_size, 7));
    let group = BarycentricEvalGroup {
        coset,
        point: SECURE_FIELD_CIRCLE_GEN,
        evals: vec![&evaluation],
    };

    let session = MetalSession::admit().expect("Metal OOD test requires a unified-memory GPU");
    assert_eq!(super::barycentric_eval_groups_metal(&[group]), vec![None]);
    let report = session.finish();
    assert_eq!(report.successful_submissions, 0);
    assert_eq!(report.failed_submissions, 0);
}

/// `STWO_OOD_BENCH_LOG=23 cargo test -p stwo
/// --features prover,metal,parallel --release --lib
/// barycentric_weights_metal_bench -- --ignored --nocapture`
#[test]
#[ignore = "manual Metal barycentric-weight benchmark"]
fn barycentric_weights_metal_bench() {
    let log_size: u32 = std::env::var("STWO_OOD_BENCH_LOG")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(20);
    let runs: u32 = std::env::var("STWO_OOD_BENCH_RUNS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(5);
    let coset = CanonicCoset::new(log_size);
    let point = CirclePoint::get_point(98_347_592_211);
    let expected = <CpuBackend as PolyOps>::barycentric_weights(coset, point);
    let warm = super::barycentric_weights_metal(coset, point)
        .expect("eligible weight column must dispatch to Metal");
    assert_eq!(warm, expected);

    let cpu_start = Instant::now();
    for _ in 0..runs {
        black_box(<CpuBackend as PolyOps>::barycentric_weights(coset, point));
    }
    let cpu = cpu_start.elapsed() / runs;
    let metal_start = Instant::now();
    for _ in 0..runs {
        black_box(
            super::barycentric_weights_metal(coset, point)
                .expect("eligible weight column must dispatch to Metal"),
        );
    }
    let metal = metal_start.elapsed() / runs;
    println!(
        "barycentric weights: log={log_size}, cpu={cpu:?}, metal={metal:?}, speedup={:.2}x",
        cpu.as_secs_f64() / metal.as_secs_f64()
    );
}

/// `STWO_OOD_BENCH_LOG=20 STWO_OOD_BENCH_COLS=7 cargo test -p stwo
/// --features prover,metal,parallel --release --lib
/// resident_barycentric_group_bench -- --ignored --nocapture --test-threads=1`
#[test]
#[ignore = "manual fused resident barycentric-group benchmark"]
fn resident_barycentric_group_bench() {
    let log_size: u32 = std::env::var("STWO_OOD_BENCH_LOG")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(20);
    let n_cols: usize = std::env::var("STWO_OOD_BENCH_COLS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(7);
    let runs: u32 = std::env::var("STWO_OOD_BENCH_RUNS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(5);
    let domain = CanonicCoset::new(log_size).circle_domain();
    let evals = (0..n_cols)
        .map(|seed| CircleEvaluation::new(domain, test_values(log_size, seed as u32 + 1)))
        .collect_vec();
    let point = CirclePoint::get_point(98_347_592_211);
    let group = BarycentricEvalGroup {
        coset: CanonicCoset::new(log_size),
        point,
        evals: evals.iter().collect_vec(),
    };
    let groups = [group];
    let weights = <CpuBackend as PolyOps>::barycentric_weights(groups[0].coset, point);
    let expected = groups[0]
        .evals
        .iter()
        .map(|eval| <CpuBackend as PolyOps>::barycentric_eval_at_point(eval, &weights))
        .collect_vec();
    let warm = super::barycentric_eval_groups_metal(&groups);
    assert_eq!(warm, vec![Some(expected)]);

    let legacy_start = Instant::now();
    for _ in 0..runs {
        let weights =
            <CpuBackend as PolyOps>::barycentric_weights(groups[0].coset, groups[0].point);
        #[cfg(feature = "parallel")]
        let values = {
            use rayon::prelude::*;
            groups[0]
                .evals
                .par_iter()
                .map(|eval| <CpuBackend as PolyOps>::barycentric_eval_at_point(eval, &weights))
                .collect::<Vec<_>>()
        };
        #[cfg(not(feature = "parallel"))]
        let values = groups[0]
            .evals
            .iter()
            .map(|eval| <CpuBackend as PolyOps>::barycentric_eval_at_point(eval, &weights))
            .collect_vec();
        black_box(values);
    }
    let legacy = legacy_start.elapsed() / runs;

    let metal_start = Instant::now();
    for _ in 0..runs {
        black_box(super::barycentric_eval_groups_metal(&groups));
    }
    let metal = metal_start.elapsed() / runs;
    println!(
        "resident barycentric group: log={log_size}, cols={n_cols}, legacy={legacy:?}, metal={metal:?}, speedup={:.2}x",
        legacy.as_secs_f64() / metal.as_secs_f64()
    );
}

/// `cargo test -p stwo --features prover,metal,parallel --release --lib
/// barycentric_eval_many_metal_bench -- --ignored --nocapture`
#[test]
#[ignore = "manual Metal micropath benchmark"]
fn barycentric_eval_many_metal_bench() {
    let log_size: u32 = std::env::var("STWO_OOD_BENCH_LOG")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(18);
    let n_cols: usize = std::env::var("STWO_OOD_BENCH_COLS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(32);
    let runs: u32 = std::env::var("STWO_OOD_BENCH_RUNS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(7);

    let domain = CanonicCoset::new(log_size).circle_domain();
    let evals: Vec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> = (0..n_cols)
        .map(|seed| CircleEvaluation::new(domain, test_values(log_size, seed as u32 + 1)))
        .collect_vec();
    let refs = evals.iter().collect_vec();
    let columns = refs
        .iter()
        .map(|evaluation| evaluation.values.as_slice())
        .collect_vec();
    let weights = <CpuBackend as PolyOps>::barycentric_weights(
        CanonicCoset::new(log_size),
        CirclePoint::get_point(98_347_592_211),
    );

    let expected = refs
        .iter()
        .map(|eval| <CpuBackend as PolyOps>::barycentric_eval_at_point(eval, &weights))
        .collect_vec();
    let warm = eval_m31_columns_metal(&columns, &weights).expect("Metal dispatch required");
    assert_eq!(warm, expected);

    let legacy_start = Instant::now();
    for _ in 0..runs {
        #[cfg(feature = "parallel")]
        let values = {
            use rayon::prelude::*;
            refs.par_iter()
                .map(|eval| <CpuBackend as PolyOps>::barycentric_eval_at_point(eval, &weights))
                .collect::<Vec<_>>()
        };
        #[cfg(not(feature = "parallel"))]
        let values = refs
            .iter()
            .map(|eval| <CpuBackend as PolyOps>::barycentric_eval_at_point(eval, &weights))
            .collect_vec();
        black_box(values);
    }
    let legacy = legacy_start.elapsed() / runs;

    let metal_start = Instant::now();
    for _ in 0..runs {
        black_box(eval_m31_columns_metal(&columns, &weights).expect("Metal dispatch required"));
    }
    let metal = metal_start.elapsed() / runs;
    println!(
        "evaluation OOD micropath: log={log_size}, cols={n_cols}, legacy={legacy:?}, metal={metal:?}, speedup={:.2}x",
        legacy.as_secs_f64() / metal.as_secs_f64()
    );
}
