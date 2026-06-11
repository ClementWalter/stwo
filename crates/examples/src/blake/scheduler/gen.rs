//! Trace generation for the Blake scheduler component.
//!
//! Row generation is embarrassingly parallel: every vec-row writes a disjoint
//! slot of each column (and a disjoint range of the round inputs vector), so
//! the columns are split into contiguous row chunks generated independently.

use std::simd::u32x16;

use itertools::{chain, Itertools};
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::ColumnVec;
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::backend::simd::m31::{PackedBaseField, LOG_N_LANES};
use stwo::prover::backend::simd::qm31::PackedSecureField;
use stwo::prover::backend::simd::{blake2s, SimdBackend};
use stwo::prover::backend::Column;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_constraint_framework::{LogupTraceGenerator, Relation, ORIGINAL_TRACE_IDX};
use tracing::{span, Level};

use super::{blake_scheduler_info, BlakeElements};
use crate::blake::round::{BlakeRoundInput, RoundElements};
use crate::blake::{to_felts, N_ROUNDS, N_ROUND_INPUT_FELTS, STATE_SIZE};

#[derive(Copy, Clone, Default)]
pub struct BlakeInput {
    pub v: [u32x16; STATE_SIZE],
    pub m: [u32x16; STATE_SIZE],
}

pub struct BlakeSchedulerLookupData {
    pub round_lookups: [[BaseColumn; N_ROUND_INPUT_FELTS]; N_ROUNDS],
    pub blake_lookups: [BaseColumn; N_ROUND_INPUT_FELTS],
}
impl BlakeSchedulerLookupData {
    fn new(log_size: u32) -> Self {
        Self {
            round_lookups: std::array::from_fn(|_| {
                std::array::from_fn(|_| unsafe { BaseColumn::uninitialized(1 << log_size) })
            }),
            blake_lookups: std::array::from_fn(|_| unsafe {
                BaseColumn::uninitialized(1 << log_size)
            }),
        }
    }
}

/// Mutable view over a contiguous range of vec-rows of all generated columns.
struct SchedulerChunkView<'a> {
    trace: Vec<&'a mut [PackedBaseField]>,
    round_lookups: Vec<Vec<&'a mut [PackedBaseField]>>,
    blake_lookups: Vec<&'a mut [PackedBaseField]>,
    /// Round inputs for this chunk's rows: `N_ROUNDS` consecutive entries per row.
    round_inputs: &'a mut [BlakeRoundInput],
}

/// Generates one chunk of scheduler rows starting at `row_offset`.
fn generate_chunk(mut view: SchedulerChunkView<'_>, row_offset: usize, inputs: &[BlakeInput]) {
    let n_rows = view.trace[0].len();
    for local_row in 0..n_rows {
        let vec_row = row_offset + local_row;
        let mut col_index = 0;

        let trace = &mut view.trace;
        let mut write_u32_array = |x: [u32x16; STATE_SIZE], col_index: &mut usize| {
            x.iter().for_each(|x| {
                to_felts(x).iter().for_each(|x| {
                    trace[*col_index][local_row] = *x;
                    *col_index += 1;
                });
            });
        };

        let BlakeInput { mut v, m } = inputs.get(vec_row).copied().unwrap_or_default();
        let initial_v = v;
        write_u32_array(m, &mut col_index);
        write_u32_array(v, &mut col_index);

        for r in 0..N_ROUNDS {
            let prev_v = v;
            blake2s::round(&mut v, m, r);
            write_u32_array(v, &mut col_index);

            let round_m = blake2s::SIGMA[r].map(|i| m[i as usize]);
            view.round_inputs[local_row * N_ROUNDS + r] = BlakeRoundInput {
                v: prev_v,
                m: round_m,
            };

            chain![
                prev_v.iter().flat_map(to_felts),
                v.iter().flat_map(to_felts),
                round_m.iter().flat_map(to_felts)
            ]
            .enumerate()
            .for_each(|(i, val)| view.round_lookups[r][i][local_row] = val);
        }

        chain![
            initial_v.iter().flat_map(to_felts),
            v.iter().flat_map(to_felts),
            m.iter().flat_map(to_felts)
        ]
        .enumerate()
        .for_each(|(i, val)| view.blake_lookups[i][local_row] = val);
    }
}

pub fn gen_trace(
    log_size: u32,
    inputs: &[BlakeInput],
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    BlakeSchedulerLookupData,
    Vec<BlakeRoundInput>,
) {
    let _span = span!(Level::INFO, "Scheduler Generation").entered();
    let mut lookup_data = BlakeSchedulerLookupData::new(log_size);
    let n_vec_rows: usize = 1 << (log_size - LOG_N_LANES);
    let mut round_inputs = vec![BlakeRoundInput::default(); n_vec_rows * N_ROUNDS];

    let mut trace = (0..blake_scheduler_info().mask_offsets[ORIGINAL_TRACE_IDX].len())
        .map(|_| unsafe { BaseColumn::uninitialized(1 << log_size) })
        .collect_vec();

    #[cfg(feature = "parallel")]
    let n_chunks = rayon::current_num_threads().clamp(1, n_vec_rows);
    #[cfg(not(feature = "parallel"))]
    let n_chunks = 1;
    let chunk_size = n_vec_rows.div_ceil(n_chunks);

    // Split every column into disjoint per-chunk row ranges.
    let mut trace_chunks = trace
        .iter_mut()
        .map(|c| c.data.chunks_mut(chunk_size))
        .collect_vec();
    let mut round_lookup_chunks = lookup_data
        .round_lookups
        .iter_mut()
        .map(|cols| {
            cols.iter_mut()
                .map(|c| c.data.chunks_mut(chunk_size))
                .collect_vec()
        })
        .collect_vec();
    let mut blake_lookup_chunks = lookup_data
        .blake_lookups
        .iter_mut()
        .map(|c| c.data.chunks_mut(chunk_size))
        .collect_vec();
    let mut round_input_chunks = round_inputs.chunks_mut(chunk_size * N_ROUNDS);

    let views = (0..n_vec_rows.div_ceil(chunk_size))
        .map(|_| SchedulerChunkView {
            trace: trace_chunks
                .iter_mut()
                .map(|it| it.next().unwrap())
                .collect(),
            round_lookups: round_lookup_chunks
                .iter_mut()
                .map(|cols| cols.iter_mut().map(|it| it.next().unwrap()).collect())
                .collect(),
            blake_lookups: blake_lookup_chunks
                .iter_mut()
                .map(|it| it.next().unwrap())
                .collect(),
            round_inputs: round_input_chunks.next().unwrap(),
        })
        .collect_vec();

    #[cfg(feature = "parallel")]
    views
        .into_par_iter()
        .enumerate()
        .for_each(|(chunk_idx, view)| generate_chunk(view, chunk_idx * chunk_size, inputs));
    #[cfg(not(feature = "parallel"))]
    views
        .into_iter()
        .enumerate()
        .for_each(|(chunk_idx, view)| generate_chunk(view, chunk_idx * chunk_size, inputs));

    let domain = CanonicCoset::new(log_size).circle_domain();
    let trace = trace
        .into_iter()
        .map(|eval| CircleEvaluation::new(domain, eval))
        .collect();

    (trace, lookup_data, round_inputs)
}

pub fn gen_interaction_trace(
    log_size: u32,
    lookup_data: BlakeSchedulerLookupData,
    round_lookup_elements: &RoundElements,
    blake_lookup_elements: &BlakeElements,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let _span = span!(Level::INFO, "Generate scheduler interaction trace").entered();

    let mut logup_gen = LogupTraceGenerator::new(log_size);

    // One logup column per pair of round lookups, plus a final column for the blake
    // lookup (combined with the last round lookup when the number of rounds is odd,
    // as in blake3).
    let n_pairs = N_ROUNDS / 2;
    logup_gen.cols_from_fn(n_pairs + 1, |col, vec_row| {
        if col < n_pairs {
            let l0 = &lookup_data.round_lookups[2 * col];
            let l1 = &lookup_data.round_lookups[2 * col + 1];
            let p0: PackedSecureField =
                round_lookup_elements.combine(&l0.each_ref().map(|l| l.data[vec_row]));
            let p1: PackedSecureField =
                round_lookup_elements.combine(&l1.each_ref().map(|l| l.data[vec_row]));
            (p0 + p1, p0 * p1)
        } else {
            let p_blake: PackedSecureField = blake_lookup_elements.combine(
                &lookup_data
                    .blake_lookups
                    .each_ref()
                    .map(|l| l.data[vec_row]),
            );
            if N_ROUNDS % 2 == 1 {
                let p_round: PackedSecureField = round_lookup_elements.combine(
                    &lookup_data.round_lookups[N_ROUNDS - 1]
                        .each_ref()
                        .map(|l| l.data[vec_row]),
                );
                // TODO(alont): Remove.
                (p_blake, p_round * p_blake)
            } else {
                // TODO(alont): Remove.
                (PackedSecureField::zero(), p_blake)
            }
        }
    });

    logup_gen.finalize_last()
}
