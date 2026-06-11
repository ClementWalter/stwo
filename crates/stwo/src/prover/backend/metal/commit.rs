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
    // The chained tree requires uniform column sizes whose extension equals the
    // lifting size (the common commit shape); otherwise fall back entirely.
    let mut logs = columns.iter().map(|column| match column {
        EvalsOrCoeffs::Evals(evals) => evals.domain.log_size(),
        EvalsOrCoeffs::Coeffs(coeffs) => coeffs.log_size(),
    });
    let Some(first_log) = logs.next() else {
        return Err(columns);
    };
    let uniform = logs.all(|log| log == first_log);
    let ext_log = first_log + log_blowup_factor;
    let lifting = lifting_log_size.unwrap_or(ext_log);
    if !uniform || lifting != ext_log {
        return Err(columns);
    }

    let (polys, pending) = super::fft::fused_transform_metal_chained(
        columns,
        log_blowup_factor,
        twiddles,
        None,
        store_polynomials_coefficients,
        |command_buffer, out_buffers| {
            super::blake2s::encode_tree(
                command_buffer,
                out_buffers,
                1usize << ext_log,
                is_m31_output,
            )
        },
    )?;

    let layers = pending.map(|pending| {
        let mut layers = pending.finish();
        // Finish the sub-threshold tail on the SIMD path.
        while (layers.len() as u32) < ext_log + 1 {
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
