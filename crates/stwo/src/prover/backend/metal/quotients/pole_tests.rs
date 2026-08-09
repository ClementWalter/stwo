use super::*;

fn tangent_sample_point(
    subdomain: CircleDomain,
    row: usize,
    raw_zero: (u32, u32),
) -> CirclePoint<SecureField> {
    let point = subdomain.at(bit_reverse_index(row, subdomain.log_size()));
    // The tangent x0*x + y0*y - 1 has exactly one intersection with the circle.
    // Realizing it as the quotient denominator gives a controlled single-row pole:
    // prx*piy - pry*pix = -1, -piy = x0, and pix = y0.
    let y_inverse = point.y.inverse();
    CirclePoint {
        x: SecureField::from_u32_unchecked(raw_zero.0, raw_zero.1, point.y.0, 0),
        y: SecureField::from_u32_unchecked(y_inverse.0, 0, (-point.x).0, 0),
    }
}

#[test]
fn pole_reports_each_sample_at_first_middle_and_last_rows() {
    let log_size = 16;
    let subdomain = CanonicCoset::new(log_size).circle_domain();
    let rows = [0, subdomain.size() / 2, subdomain.size() - 1];
    let zero_representations = [(0, 0), (P, 0), (0, P), (P, P)];
    for sample in 0..6 {
        for &row in &rows {
            let mut accumulations = (0..6)
                .map(|index| seeded_accumulation(log_size, index as u32 % 4, index))
                .collect::<Vec<_>>();
            accumulations[sample].sample_point = tangent_sample_point(
                subdomain,
                row,
                zero_representations[sample % zero_representations.len()],
            );
            let error =
                super::super::combine_quotients_metal(&accumulations, subdomain, subdomain.size())
                    .expect_err("a denominator pole must be terminal after submission");
            match error {
                super::super::QuotientMetalError::Pole {
                    sample_mask,
                    first_row,
                    sample_first_rows,
                } => {
                    assert_eq!(sample_mask, 1 << sample);
                    assert_eq!(first_row, row);
                    assert_eq!(sample_first_rows[sample], Some(row));
                    for (other, first) in sample_first_rows.iter().enumerate() {
                        if other != sample {
                            assert_eq!(*first, None);
                        }
                    }
                }
                other => panic!("unexpected terminal error: {other}"),
            }
        }
    }
}

#[test]
fn pole_report_preserves_distinct_samples_at_distinct_rows() {
    let log_size = 16;
    let subdomain = CanonicCoset::new(log_size).circle_domain();
    let rows = [
        0,
        19,
        257,
        subdomain.size() / 2,
        subdomain.size() - 2,
        subdomain.size() - 1,
    ];
    let zero_representations = [(0, 0), (P, 0), (0, P), (P, P)];
    let mut accumulations = (0..6)
        .map(|sample| seeded_accumulation(log_size, sample as u32 % 4, sample))
        .collect::<Vec<_>>();
    for sample in 0..6 {
        accumulations[sample].sample_point = tangent_sample_point(
            subdomain,
            rows[sample],
            zero_representations[sample % zero_representations.len()],
        );
    }
    let error = super::super::combine_quotients_metal(&accumulations, subdomain, subdomain.size())
        .expect_err("all six distinct poles must be terminal after submission");
    match error {
        super::super::QuotientMetalError::Pole {
            sample_mask,
            first_row,
            sample_first_rows,
        } => {
            assert_eq!(sample_mask, 0b11_1111);
            assert_eq!(first_row, rows[0]);
            assert_eq!(sample_first_rows, rows.map(Some));
        }
        other => panic!("unexpected terminal error: {other}"),
    }
}

#[test]
fn degenerate_all_row_pole_is_terminal_at_row_zero() {
    let log_size = 16;
    let subdomain = CanonicCoset::new(log_size).circle_domain();
    let mut accumulation = seeded_accumulation(log_size, 0, 0);
    // pix=piy=0 makes the denominator identically zero. Raw P in every loaded
    // coordinate confirms canonical-zero diagnostics do not depend on representation.
    accumulation.sample_point = CirclePoint {
        x: SecureField::from_u32_unchecked(P, P, P, P),
        y: SecureField::from_u32_unchecked(P, P, P, P),
    };
    let error = super::super::combine_quotients_metal(&[accumulation], subdomain, subdomain.size())
        .expect_err("an all-row denominator pole must be terminal");
    match error {
        super::super::QuotientMetalError::Pole {
            sample_mask,
            first_row,
            sample_first_rows,
        } => {
            assert_eq!(sample_mask, 1);
            assert_eq!(first_row, 0);
            assert_eq!(sample_first_rows, [Some(0), None, None, None, None, None]);
        }
        other => panic!("unexpected terminal error: {other}"),
    }
}
