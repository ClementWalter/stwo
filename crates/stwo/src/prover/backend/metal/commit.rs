//! Single-submission column commitment: interpolation, low-degree extension, Merkle
//! leaves and every above-threshold tree layer encode into one GPU submission with one
//! wait. Building the whole chain in one place keeps every CPU read after the wait,
//! avoiding the read-race class that ad-hoc deferred waits would reintroduce.

use crate::core::vcs::blake2_hash::Blake2sHash;
use crate::prover::backend::simd::blake2s_lifted::build_next_layer_simd;
use crate::prover::backend::CpuBackend;
use crate::prover::poly::circle::EvalsOrCoeffs;
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::Poly;

/// Commits the columns in one chained submission. On success returns the polynomials
/// and, when the tree could be chained, the full tree layers (leaves first). Returns
/// the columns untouched when the GPU path can't take them.
#[allow(clippy::type_complexity)]
pub(crate) fn commit_polynomials_metal(
    columns: Vec<EvalsOrCoeffs<CpuBackend>>,
    log_blowup_factor: u32,
    twiddles: &TwiddleTree<CpuBackend>,
    store_polynomials_coefficients: bool,
    lifting_log_size: Option<u32>,
    is_m31_output: bool,
) -> Result<(Vec<Poly<CpuBackend>>, Option<Vec<Vec<Blake2sHash>>>), Vec<EvalsOrCoeffs<CpuBackend>>>
{
    if columns.is_empty() {
        return Err(columns);
    }
    let base_logs: Vec<u32> = columns
        .iter()
        .map(|column| match column {
            EvalsOrCoeffs::Evals(evals) => evals.domain.log_size(),
            EvalsOrCoeffs::Coeffs(coeffs) => coeffs.log_size(),
        })
        .collect();
    // Avoid paying GPU setup for an all-small commitment. Mixed commitments are
    // admitted when at least one transform is large enough to amortize the epoch.
    if base_logs.iter().copied().max().unwrap_or_default() < super::fft::MIN_METAL_FFT_LOG_SIZE {
        return Err(columns);
    }
    let Some(ext_logs): Option<Vec<u32>> = base_logs
        .iter()
        .map(|&log| log.checked_add(log_blowup_factor))
        .collect()
    else {
        return Err(columns);
    };
    let max_ext_log = ext_logs.iter().copied().max().unwrap_or_default();
    let lifting = lifting_log_size.unwrap_or(max_ext_log);
    if lifting < max_ext_log {
        return Err(columns);
    }
    let uniform_at_lifting = ext_logs.iter().all(|&log| log == lifting);
    let mut canonical_order: Vec<usize> = (0..columns.len()).collect();
    canonical_order.sort_by_key(|&index| (ext_logs[index], index));

    let (polys, pending) = super::fft::fused_transform_metal_chained(
        columns,
        log_blowup_factor,
        twiddles,
        None,
        store_polynomials_coefficients,
        |command_buffer, out_buffers| {
            if uniform_at_lifting {
                super::blake2s::encode_tree(
                    command_buffer,
                    out_buffers,
                    1usize.checked_shl(lifting)?,
                    is_m31_output,
                )
            } else {
                let ordered_buffers = canonical_order
                    .iter()
                    .map(|&index| out_buffers[index].clone())
                    .collect::<Vec<_>>();
                let ordered_logs = canonical_order
                    .iter()
                    .map(|&index| ext_logs[index])
                    .collect::<Vec<_>>();
                super::blake2s::encode_tree_compact(
                    command_buffer,
                    &ordered_buffers,
                    &ordered_logs,
                    lifting,
                    is_m31_output,
                )
            }
        },
    )?;

    let layers = pending.map(|pending| {
        let mut layers = pending.finish();
        // Finish the sub-threshold tail on the SIMD path.
        while (layers.len() as u32) < lifting + 1 {
            let next = if is_m31_output {
                build_next_layer_simd::<true>(layers.last().unwrap())
            } else {
                build_next_layer_simd::<false>(layers.last().unwrap())
            };
            layers.push(next);
        }
        layers
    });
    Ok((polys, layers))
}
