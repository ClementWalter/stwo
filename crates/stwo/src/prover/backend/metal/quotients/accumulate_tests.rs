use std::collections::BTreeSet;

use super::accumulate::{
    accumulate_numerators_metal, input_key_for_test, preflight_for_test, NumeratorBatch,
};
use crate::core::fields::m31::{BaseField, P};
use crate::core::fields::qm31::SecureField;
use crate::prover::backend::CpuBackend;
use crate::prover::secure_column::SecureColumnByCoords;

fn canonical_m31(value: BaseField) -> BaseField {
    BaseField::from_u32_unchecked(value.0 % P)
}

fn canonical_secure(value: SecureField) -> SecureField {
    SecureField::from_m31_array(value.to_m31_array().map(canonical_m31))
}

fn cpu_reference(batch: &NumeratorBatch<'_>) -> SecureColumnByCoords<CpuBackend> {
    let mut result = SecureColumnByCoords::zeros(batch.n_rows);
    for row in 0..batch.n_rows {
        let mut accumulator = canonical_secure(batch.neg_b_sum);
        for (column, coefficient) in batch.columns.iter().zip(&batch.coeffs) {
            accumulator += canonical_m31(column[row]) * canonical_secure(*coefficient);
        }
        result.set(row, accumulator);
    }
    result
}

fn seeded_column(log_rows: u32, salt: u32) -> Vec<BaseField> {
    (0..1usize << log_rows)
        .map(|row| {
            BaseField::from_u32_unchecked(
                (row as u32)
                    .wrapping_mul(0x045d_9f3b ^ salt)
                    .wrapping_add(17 + salt)
                    % P,
            )
        })
        .collect()
}

fn coefficient(batch: usize, column: usize) -> SecureField {
    let seed = 1 + batch as u32 * 97 + column as u32 * 13;
    SecureField::from_u32_unchecked(seed, seed + 2, seed + 4, seed + 6)
}

fn batches_for_shapes<'a>(
    storage: &'a [Vec<BaseField>],
    shapes: &[(u32, &[usize])],
) -> Vec<NumeratorBatch<'a>> {
    let mut batch_index = 0usize;
    let mut batches = Vec::new();
    for ((log_rows, column_counts), column) in shapes.iter().zip(storage) {
        assert_eq!(column.len(), 1usize << log_rows);
        for &column_count in *column_counts {
            batches.push(NumeratorBatch {
                columns: std::iter::repeat_n(column.as_slice(), column_count).collect(),
                coeffs: (0..column_count)
                    .map(|column| coefficient(batch_index, column))
                    .collect(),
                neg_b_sum: SecureField::from_u32_unchecked(
                    11 + batch_index as u32,
                    23 + batch_index as u32,
                    37 + batch_index as u32,
                    53 + batch_index as u32,
                ),
                n_rows: column.len(),
            });
            batch_index += 1;
        }
    }
    batches
}

fn assert_batches_match_cpu(batches: &[NumeratorBatch<'_>]) {
    let expected = batches.iter().map(cpu_reference).collect::<Vec<_>>();
    let actual = accumulate_numerators_metal(batches)
        .expect("submitted bulk numerator command must succeed")
        .expect("valid Metal numerator shape must dispatch");
    assert_eq!(actual.len(), expected.len());
    for (batch, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(actual.columns, expected.columns, "stable batch {batch}");
    }
}

#[test]
#[ignore = "exact 2KiB batch inventory; run bounded and serial during Metal review"]
fn exact_sha_and_keccak_2kib_selected_batches_match_cpu() {
    const SHA: &[(u32, &[usize])] = &[
        (15, &[560, 16, 16]),
        (16, &[80, 8, 8]),
        (17, &[73, 4, 4]),
        (18, &[9, 4, 4]),
        (19, &[7, 4, 4]),
        (20, &[30, 8]),
    ];
    const KECCAK: &[(u32, &[usize])] = &[
        (14, &[475, 8, 8]),
        (15, &[68, 8, 8]),
        (16, &[80, 8, 8]),
        (17, &[80, 8, 8]),
        (18, &[151, 8, 8]),
        (19, &[9, 4, 4]),
        (20, &[30, 8]),
    ];
    for (shapes, expected_batches, expected_dispatches) in [(SHA, 17, 60), (KECCAK, 20, 71)] {
        let storage = shapes
            .iter()
            .enumerate()
            .map(|(index, (log_rows, _))| seeded_column(*log_rows, index as u32 + 1))
            .collect::<Vec<_>>();
        let batches = batches_for_shapes(&storage, shapes);
        assert_eq!(batches.len(), expected_batches);
        assert_eq!(
            preflight_for_test(&batches),
            Some((1 << 20, expected_dispatches))
        );
        assert_batches_match_cpu(&batches);
    }
}

#[test]
fn column_chunk_boundaries_and_mixed_logs_match_cpu() {
    const COUNTS: &[usize] = &[1, 15, 16, 17, 31, 32, 33];
    const SHAPES: &[(u32, &[usize])] = &[(8, COUNTS), (12, COUNTS), (16, COUNTS)];
    let storage = SHAPES
        .iter()
        .enumerate()
        .map(|(index, (log_rows, _))| seeded_column(*log_rows, 71 + index as u32))
        .collect::<Vec<_>>();
    let batches = batches_for_shapes(&storage, SHAPES);
    assert_eq!(preflight_for_test(&batches), Some((1 << 16, 36)));
    assert_batches_match_cpu(&batches);
}

#[test]
fn distinct_allocations_across_multiple_batches_match_cpu() {
    const SHAPES: &[(u32, usize)] = &[(8, 3), (9, 17), (8, 5)];
    let storage = SHAPES
        .iter()
        .enumerate()
        .map(|(batch, &(log_rows, column_count))| {
            (0..column_count)
                .map(|column| seeded_column(log_rows, 101 + (batch * 31 + column) as u32))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let batches = storage
        .iter()
        .zip(SHAPES)
        .enumerate()
        .map(
            |(batch, (columns, &(log_rows, column_count)))| NumeratorBatch {
                columns: columns.iter().map(Vec::as_slice).collect(),
                coeffs: (0..column_count)
                    .map(|column| coefficient(batch, column))
                    .collect(),
                neg_b_sum: SecureField::from_u32_unchecked(
                    7 + batch as u32,
                    11 + batch as u32,
                    13 + batch as u32,
                    17 + batch as u32,
                ),
                n_rows: 1 << log_rows,
            },
        )
        .collect::<Vec<_>>();

    let input_keys = batches
        .iter()
        .flat_map(|batch| {
            batch
                .columns
                .iter()
                .map(|column| input_key_for_test(column))
        })
        .collect::<Vec<_>>();
    assert_eq!(
        input_keys.iter().copied().collect::<BTreeSet<_>>().len(),
        input_keys.len(),
        "every input must have a distinct live allocation"
    );
    assert_eq!(preflight_for_test(&batches), Some((1 << 9, 4)));
    assert_batches_match_cpu(&batches);
}

#[test]
fn raw_p_is_canonicalized_on_columns_coefficients_init_and_output() {
    const N_ROWS: usize = 256;
    const N_COLS: usize = 17;
    let mut canonical_columns = (0..N_COLS)
        .map(|column| {
            (0..N_ROWS)
                .map(|row| BaseField::from((row + column + 1) as u32))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut raw_columns = canonical_columns.clone();
    canonical_columns[0][19] = BaseField::from_u32_unchecked(0);
    raw_columns[0][19] = BaseField::from_u32_unchecked(P);

    let mut canonical_coeffs = (0..N_COLS)
        .map(|column| coefficient(3, column))
        .collect::<Vec<_>>();
    let mut raw_coeffs = canonical_coeffs.clone();
    for coordinate in 0..4 {
        let column = coordinate + 1;
        let mut canonical = canonical_coeffs[column].to_m31_array();
        let mut raw = raw_coeffs[column].to_m31_array();
        canonical[coordinate] = BaseField::from_u32_unchecked(0);
        raw[coordinate] = BaseField::from_u32_unchecked(P);
        canonical_coeffs[column] = SecureField::from_m31_array(canonical);
        raw_coeffs[column] = SecureField::from_m31_array(raw);
    }
    let canonical = NumeratorBatch {
        columns: canonical_columns.iter().map(Vec::as_slice).collect(),
        coeffs: canonical_coeffs,
        neg_b_sum: SecureField::from_u32_unchecked(0, 0, 0, 0),
        n_rows: N_ROWS,
    };
    let raw = NumeratorBatch {
        columns: raw_columns.iter().map(Vec::as_slice).collect(),
        coeffs: raw_coeffs,
        neg_b_sum: SecureField::from_u32_unchecked(P, P, P, P),
        n_rows: N_ROWS,
    };
    assert_eq!(raw.columns[0][19].0, P);
    let canonical_output = accumulate_numerators_metal(&[canonical])
        .unwrap()
        .unwrap()
        .pop()
        .unwrap();
    let raw_output = accumulate_numerators_metal(&[raw])
        .unwrap()
        .unwrap()
        .pop()
        .unwrap();
    assert_eq!(raw_output.columns, canonical_output.columns);
    assert!(raw_output.columns.iter().flatten().all(|value| value.0 < P));
}

#[test]
fn returned_results_keep_caller_batch_order() {
    let storage = [
        seeded_column(8, 1),
        seeded_column(9, 2),
        seeded_column(8, 3),
    ];
    let sentinels = [17, 29, 43];
    let batches = storage
        .iter()
        .zip(sentinels)
        .map(|(column, sentinel)| NumeratorBatch {
            columns: vec![column.as_slice()],
            coeffs: vec![SecureField::default()],
            neg_b_sum: SecureField::from_u32_unchecked(sentinel, 0, 0, 0),
            n_rows: column.len(),
        })
        .collect::<Vec<_>>();
    let outputs = accumulate_numerators_metal(&batches).unwrap().unwrap();
    assert_eq!(outputs.len(), sentinels.len());
    for (output, sentinel) in outputs.iter().zip(sentinels) {
        assert!(output.columns[0].iter().all(|value| value.0 == sentinel));
    }
}

#[test]
fn malformed_shapes_decline_during_preflight_before_submission() {
    let column = seeded_column(8, 9);
    let empty = NumeratorBatch {
        columns: vec![],
        coeffs: vec![],
        neg_b_sum: SecureField::default(),
        n_rows: 256,
    };
    assert!(preflight_for_test(&[empty]).is_none());

    let mismatch = NumeratorBatch {
        columns: vec![column.as_slice()],
        coeffs: vec![],
        neg_b_sum: SecureField::default(),
        n_rows: 256,
    };
    assert!(preflight_for_test(&[mismatch]).is_none());

    for n_rows in [0, 127, 512] {
        let malformed = NumeratorBatch {
            columns: vec![column.as_slice()],
            coeffs: vec![SecureField::default()],
            neg_b_sum: SecureField::default(),
            n_rows,
        };
        assert!(preflight_for_test(&[malformed]).is_none());
    }
}

#[test]
fn input_deduplication_key_is_exact_pointer_and_bound_length() {
    let first = seeded_column(9, 17);
    let second = first.clone();
    assert_eq!(
        input_key_for_test(&first[..256]),
        input_key_for_test(&first[..256])
    );
    assert_ne!(
        input_key_for_test(&first[..256]),
        input_key_for_test(&first[..512])
    );
    assert_ne!(
        input_key_for_test(&first[..256]),
        input_key_for_test(&second[..256])
    );
}

#[test]
fn bulk_numerator_admission_boundary_is_pinned() {
    assert_eq!(super::MIN_METAL_NUMERATOR_LOG_SIZE, 14);
    assert!(!super::should_use_metal_numerators(13));
    assert!(super::should_use_metal_numerators(14));
    assert!(super::should_use_metal_numerators(20));
}

#[test]
#[ignore = "session-counter assertion must run alone"]
fn one_bulk_command_records_one_checked_session_submission() {
    let column = seeded_column(12, 101);
    let malformed = NumeratorBatch {
        columns: vec![column.as_slice()],
        coeffs: vec![],
        neg_b_sum: SecureField::default(),
        n_rows: column.len(),
    };
    let before_decline = super::super::context::submission_counters();
    assert!(accumulate_numerators_metal(&[malformed])
        .expect("malformed work must be a pre-submit decline")
        .is_none());
    assert_eq!(
        super::super::context::submission_counters().delta_from(before_decline),
        Default::default()
    );

    let batch = NumeratorBatch {
        columns: vec![column.as_slice(); 33],
        coeffs: (0..33).map(|column| coefficient(7, column)).collect(),
        neg_b_sum: SecureField::default(),
        n_rows: column.len(),
    };
    let before = super::super::context::submission_counters();
    assert!(accumulate_numerators_metal(&[batch]).unwrap().is_some());
    let delta = super::super::context::submission_counters().delta_from(before);
    assert_eq!(delta.successful, 1);
    assert_eq!(delta.failed, 0);
}
