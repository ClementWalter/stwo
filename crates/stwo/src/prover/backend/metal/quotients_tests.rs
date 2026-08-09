use crate::core::circle::{CirclePoint, SECURE_FIELD_CIRCLE_GEN};
use crate::core::fields::cm31::CM31;
use crate::core::fields::m31::{BaseField, P};
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldExpOps;
use crate::core::pcs::quotients::denominators;
use crate::core::poly::circle::{CanonicCoset, CircleDomain};
use crate::core::utils::bit_reverse_index;
use crate::prover::backend::CpuBackend;
use crate::prover::pcs::quotient_ops::AccumulatedNumerators;
use crate::prover::secure_column::SecureColumnByCoords;

fn seeded_accumulation(
    subdomain_log_size: u32,
    log_ratio: u32,
    sample: usize,
) -> AccumulatedNumerators<CpuBackend> {
    let column_len = 1usize << (subdomain_log_size - log_ratio);
    let columns = std::array::from_fn(|coordinate| {
        (0..column_len)
            .map(|row| {
                let raw = (row as u32)
                    .wrapping_mul(0x045d_9f3b ^ (sample as u32 * 0x0101_0101))
                    .wrapping_add(17 + coordinate as u32 * 101 + sample as u32 * 1009)
                    % P;
                BaseField::from_u32_unchecked(raw)
            })
            .collect()
    });
    AccumulatedNumerators {
        sample_point: SECURE_FIELD_CIRCLE_GEN.mul(1_234_567 + sample as u128 * 97),
        partial_numerators_acc: SecureColumnByCoords { columns },
        first_linear_term_acc: SecureField::from_u32_unchecked(
            13 + sample as u32,
            29 + sample as u32,
            47 + sample as u32,
            71 + sample as u32,
        ),
    }
}

fn direct_reference(
    accumulations: &[AccumulatedNumerators<CpuBackend>],
    subdomain: CircleDomain,
) -> SecureColumnByCoords<CpuBackend> {
    let log_size = subdomain.log_size();
    let mut result = SecureColumnByCoords::zeros(subdomain.size());
    let sample_points = accumulations
        .iter()
        .map(|accumulation| accumulation.sample_point)
        .collect::<Vec<_>>();
    for row in 0..subdomain.size() {
        let point = subdomain.at(bit_reverse_index(row, log_size));
        let mut quotient = SecureField::default();
        for (accumulation, denominator) in accumulations
            .iter()
            .zip(denominators(&sample_points, point))
        {
            let log_ratio = log_size - accumulation.partial_numerators_acc.len().ilog2();
            let lifted = ((row >> (log_ratio + 1)) << 1) | (row & 1);
            let numerator = accumulation.partial_numerators_acc.at(lifted)
                - accumulation.first_linear_term_acc * point.y;
            quotient += numerator.mul_cm31(denominator.inverse());
        }
        result.set(row, quotient);
    }
    result
}

fn batched_reference(
    accumulations: &[AccumulatedNumerators<CpuBackend>],
    subdomain: CircleDomain,
) -> SecureColumnByCoords<CpuBackend> {
    let log_size = subdomain.log_size();
    let mut xs = Vec::with_capacity(subdomain.size());
    let mut ys = Vec::with_capacity(subdomain.size());
    for row in 0..subdomain.size() {
        let point = subdomain.at(bit_reverse_index(row, log_size));
        xs.push(point.x);
        ys.push(point.y);
    }
    batched_reference_with_xy(accumulations, log_size, &xs, &ys)
}

fn batched_reference_with_xy(
    accumulations: &[AccumulatedNumerators<CpuBackend>],
    log_size: u32,
    xs: &[BaseField],
    ys: &[BaseField],
) -> SecureColumnByCoords<CpuBackend> {
    const CHUNK_ROWS: usize = 4096;
    let sample_points = accumulations
        .iter()
        .map(|accumulation| accumulation.sample_point)
        .collect::<Vec<_>>();
    let mut result = SecureColumnByCoords::zeros(xs.len());
    for start in (0..xs.len()).step_by(CHUNK_ROWS) {
        let end = (start + CHUNK_ROWS).min(xs.len());
        let points = (start..end)
            .map(|row| CirclePoint {
                x: xs[row],
                y: ys[row],
            })
            .collect::<Vec<_>>();
        let row_denominators = points
            .iter()
            .flat_map(|&point| denominators(&sample_points, point))
            .collect::<Vec<CM31>>();
        let inverses = CM31::batch_inverse(&row_denominators);
        for (offset, &point) in points.iter().enumerate() {
            let row = start + offset;
            let mut quotient = SecureField::default();
            for (sample, accumulation) in accumulations.iter().enumerate() {
                let log_ratio = log_size - accumulation.partial_numerators_acc.len().ilog2();
                let lifted = ((row >> (log_ratio + 1)) << 1) | (row & 1);
                let numerator = accumulation.partial_numerators_acc.at(lifted)
                    - accumulation.first_linear_term_acc * point.y;
                quotient += numerator.mul_cm31(inverses[offset * accumulations.len() + sample]);
            }
            result.set(row, quotient);
        }
    }
    result
}

fn domain_xy(subdomain: CircleDomain) -> (Vec<BaseField>, Vec<BaseField>) {
    let log_size = subdomain.log_size();
    (0..subdomain.size())
        .map(|row| subdomain.at(bit_reverse_index(row, log_size)))
        .map(|point| (point.x, point.y))
        .unzip()
}

#[test]
fn batched_one_sample_log16_matches_direct_cpu() {
    let log_size = super::MIN_METAL_QUOTIENT_LOG_SIZE;
    let subdomain = CanonicCoset::new(log_size).circle_domain();
    let accumulations = vec![seeded_accumulation(log_size, 0, 0)];
    let expected = direct_reference(&accumulations, subdomain);
    let actual = super::combine_quotients_metal(&accumulations, subdomain, subdomain.size())
        .expect("submitted Metal quotient command must succeed")
        .expect("eligible quotient shape must dispatch");
    let direct = super::combine_quotients_direct_metal(&accumulations, subdomain)
        .expect("submitted direct-control command must succeed")
        .expect("direct-control shape must dispatch");
    assert_eq!(actual.columns, expected.columns);
    assert_eq!(direct.columns, expected.columns);
}

#[test]
fn batched_metal_matches_cpu_across_required_shape_matrix() {
    for log_size in [16, 17, 18] {
        let subdomain = CanonicCoset::new(log_size).circle_domain();
        for n_samples in [1, 2, 6] {
            for log_ratio in [0, 1, 3] {
                let accumulations = (0..n_samples)
                    .map(|sample| seeded_accumulation(log_size, log_ratio, sample))
                    .collect::<Vec<_>>();
                let expected = batched_reference(&accumulations, subdomain);
                let actual =
                    super::combine_quotients_metal(&accumulations, subdomain, subdomain.size())
                        .expect("submitted Metal quotient command must succeed")
                        .expect("eligible quotient shape must dispatch");
                assert_eq!(
                    actual.columns, expected.columns,
                    "log_size={log_size}, n_samples={n_samples}, log_ratio={log_ratio}"
                );
            }
        }
    }
}

#[test]
fn batched_metal_canonicalizes_raw_zero_across_all_load_paths() {
    let log_size = 16;
    let subdomain = CanonicCoset::new(log_size).circle_domain();
    let (mut canonical_xs, mut canonical_ys) = domain_xy(subdomain);
    let (mut raw_xs, mut raw_ys) = (canonical_xs.clone(), canonical_ys.clone());
    let zero_representations = [(0, 0), (P, 0), (0, P), (P, P)];
    for (&row, &(raw_x, raw_y)) in [0, 127, 32768, subdomain.size() - 1]
        .iter()
        .zip(&zero_representations)
    {
        canonical_xs[row] = BaseField::from_u32_unchecked(0);
        canonical_ys[row] = BaseField::from_u32_unchecked(0);
        raw_xs[row] = BaseField::from_u32_unchecked(raw_x);
        raw_ys[row] = BaseField::from_u32_unchecked(raw_y);
    }

    let mut canonical = (0..6)
        .map(|sample| seeded_accumulation(log_size, [0, 1, 3][sample % 3], sample))
        .collect::<Vec<_>>();
    let mut raw = canonical.clone();
    for sample in 0..canonical.len() {
        let (raw_a, raw_b) = zero_representations[sample % zero_representations.len()];
        let pix = [11 + sample as u32, 17 + sample as u32];
        let pry = [23 + sample as u32, 29 + sample as u32];
        let piy = [31 + sample as u32, 37 + sample as u32];
        canonical[sample].sample_point = CirclePoint {
            x: SecureField::from_u32_unchecked(0, 0, pix[0], pix[1]),
            y: SecureField::from_u32_unchecked(pry[0], pry[1], piy[0], piy[1]),
        };
        raw[sample].sample_point = CirclePoint {
            x: SecureField::from_u32_unchecked(raw_a, raw_b, pix[0], pix[1]),
            y: canonical[sample].sample_point.y,
        };
        let flt = canonical[sample].first_linear_term_acc.to_m31_array();
        canonical[sample].first_linear_term_acc =
            SecureField::from_u32_unchecked(0, 0, flt[2].0, flt[3].0);
        raw[sample].first_linear_term_acc =
            SecureField::from_u32_unchecked(raw_a, raw_b, flt[2].0, flt[3].0);

        let column_len = canonical[sample].partial_numerators_acc.len();
        for (offset, &(coordinate_a, coordinate_b)) in zero_representations.iter().enumerate() {
            let row = [0, 1, 257, column_len - 1][offset];
            canonical[sample].partial_numerators_acc.columns[0][row] =
                BaseField::from_u32_unchecked(0);
            canonical[sample].partial_numerators_acc.columns[1][row] =
                BaseField::from_u32_unchecked(0);
            raw[sample].partial_numerators_acc.columns[0][row] =
                BaseField::from_u32_unchecked(coordinate_a);
            raw[sample].partial_numerators_acc.columns[1][row] =
                BaseField::from_u32_unchecked(coordinate_b);
        }
    }

    let expected = batched_reference_with_xy(&canonical, log_size, &canonical_xs, &canonical_ys);
    let actual =
        super::combine_quotients_metal_with_xy(&raw, log_size, raw_xs.len(), &raw_xs, &raw_ys)
            .expect("submitted raw-representation command must succeed")
            .expect("raw-representation shape must dispatch");
    assert_eq!(actual.columns, expected.columns);
}

#[test]
fn raw_p_axis_trials_cover_every_loaded_coordinate() {
    let log_size = 16;
    let subdomain = CanonicCoset::new(log_size).circle_domain();
    let (xs, ys) = domain_xy(subdomain);
    let check = |canonical: &[AccumulatedNumerators<CpuBackend>],
                 raw: &[AccumulatedNumerators<CpuBackend>],
                 canonical_xs: &[BaseField],
                 canonical_ys: &[BaseField],
                 raw_xs: &[BaseField],
                 raw_ys: &[BaseField],
                 case: &str| {
        let expected = batched_reference_with_xy(canonical, log_size, canonical_xs, canonical_ys);
        let actual =
            super::combine_quotients_metal_with_xy(raw, log_size, raw_xs.len(), raw_xs, raw_ys)
                .unwrap_or_else(|error| panic!("{case}: unexpected terminal error: {error}"))
                .unwrap_or_else(|| panic!("{case}: eligible shape declined"));
        assert_eq!(actual.columns, expected.columns, "{case}");
    };

    // All four partial-numerator coordinate buffers.
    let mut canonical = vec![seeded_accumulation(log_size, 3, 0)];
    let mut raw = canonical.clone();
    for coordinate in 0..4 {
        let row = 19 + coordinate * 257;
        canonical[0].partial_numerators_acc.columns[coordinate][row] =
            BaseField::from_u32_unchecked(0);
        raw[0].partial_numerators_acc.columns[coordinate][row] = BaseField::from_u32_unchecked(P);
        assert_eq!(raw[0].partial_numerators_acc.columns[coordinate][row].0, P);
    }
    check(&canonical, &raw, &xs, &ys, &xs, &ys, "partial coordinates");

    // All four first-linear-term parameter coordinates.
    let mut canonical = vec![seeded_accumulation(log_size, 1, 1)];
    let mut raw = canonical.clone();
    canonical[0].first_linear_term_acc = SecureField::from_u32_unchecked(0, 0, 0, 0);
    raw[0].first_linear_term_acc = SecureField::from_u32_unchecked(P, P, P, P);
    assert!(raw[0]
        .first_linear_term_acc
        .to_m31_array()
        .iter()
        .all(|value| value.0 == P));
    check(&canonical, &raw, &xs, &ys, &xs, &ys, "FLT coordinates");

    // Each of prx/pix/pry/piy has two M31 words: all eight sample-point words.
    for axis in 0..8 {
        let mut canonical = vec![seeded_accumulation(log_size, 0, 2)];
        let mut raw = canonical.clone();
        let x = canonical[0].sample_point.x.to_m31_array();
        let y = canonical[0].sample_point.y.to_m31_array();
        let mut canonical_words = [
            x[0].0, x[1].0, x[2].0, x[3].0, y[0].0, y[1].0, y[2].0, y[3].0,
        ];
        let mut raw_words = canonical_words;
        canonical_words[axis] = 0;
        raw_words[axis] = P;
        let point = |words: [u32; 8]| CirclePoint {
            x: SecureField::from_u32_unchecked(words[0], words[1], words[2], words[3]),
            y: SecureField::from_u32_unchecked(words[4], words[5], words[6], words[7]),
        };
        canonical[0].sample_point = point(canonical_words);
        raw[0].sample_point = point(raw_words);
        let raw_x = raw[0].sample_point.x.to_m31_array();
        let raw_y = raw[0].sample_point.y.to_m31_array();
        assert_eq!(
            if axis < 4 {
                raw_x[axis].0
            } else {
                raw_y[axis - 4].0
            },
            P
        );
        check(
            &canonical,
            &raw,
            &xs,
            &ys,
            &xs,
            &ys,
            &format!("sample-point axis {axis}"),
        );
    }

    // Both domain coordinate loads, including a shared row where both are raw P.
    let canonical = vec![seeded_accumulation(log_size, 3, 3)];
    let raw = canonical.clone();
    let (mut canonical_xs, mut canonical_ys) = (xs.clone(), ys.clone());
    let (mut raw_xs, mut raw_ys) = (xs.clone(), ys.clone());
    for row in [0, 257] {
        canonical_xs[row] = BaseField::from_u32_unchecked(0);
        raw_xs[row] = BaseField::from_u32_unchecked(P);
    }
    for row in [1, 257] {
        canonical_ys[row] = BaseField::from_u32_unchecked(0);
        raw_ys[row] = BaseField::from_u32_unchecked(P);
    }
    assert_eq!(raw_xs[0].0, P);
    assert_eq!(raw_ys[1].0, P);
    assert_eq!((raw_xs[257].0, raw_ys[257].0), (P, P));
    check(
        &canonical,
        &raw,
        &canonical_xs,
        &canonical_ys,
        &raw_xs,
        &raw_ys,
        "domain x/y coordinates",
    );
}

#[test]
fn inactive_tail_lanes_are_identity_and_cross_all_barriers() {
    let log_size = 16;
    let subdomain = CanonicCoset::new(log_size).circle_domain();
    let n_rows = subdomain.size() - 13;
    let (xs, ys) = domain_xy(subdomain);
    let accumulations = (0..6)
        .map(|sample| seeded_accumulation(log_size, [0, 1, 3][sample % 3], sample))
        .collect::<Vec<_>>();
    let expected =
        batched_reference_with_xy(&accumulations, log_size, &xs[..n_rows], &ys[..n_rows]);
    let actual = super::combine_quotients_metal_with_xy(&accumulations, log_size, n_rows, &xs, &ys)
        .expect("tail-lane command must succeed")
        .expect("tail-lane shape must dispatch");
    assert_eq!(actual.columns, expected.columns);
}

#[test]
fn malformed_shapes_decline_before_submission() {
    let log_size = 16;
    let subdomain = CanonicCoset::new(log_size).circle_domain();
    let (xs, ys) = domain_xy(subdomain);
    let accumulation = seeded_accumulation(log_size, 0, 0);
    for (n_rows, xs_len, ys_len) in [
        (0, xs.len(), ys.len()),
        (xs.len() + 1, xs.len(), ys.len()),
        (xs.len(), xs.len() - 1, ys.len()),
        (ys.len(), xs.len(), ys.len() - 1),
    ] {
        let result = super::combine_quotients_metal_with_xy(
            std::slice::from_ref(&accumulation),
            log_size,
            n_rows,
            &xs[..xs_len],
            &ys[..ys_len],
        )
        .expect("malformed shape must be a pre-submit decline");
        assert!(result.is_none());
    }

    let mut uneven = accumulation.clone();
    uneven.partial_numerators_acc.columns[3].pop();
    assert!(
        super::combine_quotients_metal_with_xy(&[uneven], log_size, xs.len(), &xs, &ys,)
            .expect("uneven coordinate lengths must decline pre-submit")
            .is_none()
    );
}

#[path = "quotients/pole_tests.rs"]
mod pole_tests;

#[cfg(feature = "parallel")]
#[path = "quotients/bench_tests.rs"]
mod bench_tests;
