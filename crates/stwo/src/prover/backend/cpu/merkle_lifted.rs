use std::any::TypeId;

use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SECURE_EXTENSION_DEGREE;
use crate::core::vcs::blake2_hash::Blake2sHash;
use crate::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasherGeneric;
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::core::vcs_lifted::verifier::PACKED_LEAF_SIZE;
use crate::parallel_iter;
use crate::prover::backend::simd::blake2s_lifted::{
    build_leaves_from_flat_columns, build_next_layer_simd,
};
use crate::prover::backend::{Col, CpuBackend};
use crate::prover::vcs_lifted::ops::{MerkleOpsLifted, PackLeavesOps};

impl<H: MerkleHasherLifted + Send + Sync + 'static> MerkleOpsLifted<H> for CpuBackend {
    /// Computes the leaves of the Merkle tree. This is the core logic of the lifted Merkle
    /// commitment. The input columns are assumed to be in increasing order of length.
    ///
    /// The columns are interpreted as evaluations of polynomials in bit reversed order.
    /// For example, consider a polynomial that on the canonical circle domain of size 8 has
    /// evaluations (in natural order and bit reversed respectively):
    ///     a   a
    ///     b   e
    ///     c   c
    ///     d   g
    ///     e   b
    ///     f   f
    ///     g   d
    ///     h   h
    /// Then the evaluations of its lifted polynomial on the canonical circle domain of size 16 are
    /// (in natural and bit reversed order respectively):
    ///     a   a
    ///     b   e
    ///     c   a
    ///     d   e
    ///     a   c
    ///     b   g
    ///     c   c
    ///     d   g
    ///     e   b
    ///     f   f
    ///     g   b
    ///     h   f
    ///     e   d
    ///     f   h
    ///     g   d
    ///     h   h
    fn build_leaves(columns: &[&Vec<BaseField>], lifting_log_size: u32) -> Vec<H::Hash> {
        let hasher = H::default();
        if columns.is_empty() {
            return vec![hasher.finalize()];
        }

        // Blake2s fast path: the merkle ops are backend-specific implementations of the
        // same commitment, and a `Vec<BaseField>` column is the same flat power-of-two
        // run of u32 representatives the 16-way SIMD leaf builder reads, so blake2s
        // trees dispatch to it. The resulting leaves are identical to the generic
        // row-by-row absorption below.
        if columns[0].len() >= 16 {
            if let Some(leaves) = blake2s_fast_path::<H>(columns, lifting_log_size) {
                return leaves;
            }
        }

        assert!(columns[0].len() >= 2, "A column must be of length >= 2.");
        let mut prev_layer: Vec<H> = vec![hasher; 2];
        let mut prev_layer_log_size: u32 = 1;
        for (log_size, group) in columns.iter().chunk_by(|c| c.len().ilog2()).into_iter() {
            let log_ratio = log_size - prev_layer_log_size;
            prev_layer = parallel_iter!(0..1 << log_size)
                // We only clone when starting a column chunk of different size.
                .map(|idx| prev_layer[(idx >> (log_ratio + 1) << 1) + (idx & 1)].clone())
                .collect();

            // We chunk by 16 — the amount of M31 elements that triggers a hash permutation
            // in Blake2s (block = 64 bytes = 16 × 4) and matches Poseidon252's absorption rate.
            // For Keccak256 the rate is larger (136 bytes ≈ 34 M31s), so this chunking is
            // suboptimal but still correct. Rows absorb independently; each row gathers its
            // chunk values into a stack buffer.
            for chunk in &group.into_iter().chunks(16) {
                let vec = chunk.into_iter().collect_vec();
                let update_row = |(i, hasher): (usize, &mut H)| {
                    let mut row_values = [BaseField::default(); 16];
                    for (slot, col) in row_values.iter_mut().zip(vec.iter()) {
                        *slot = col[i];
                    }
                    hasher.update_leaf(&row_values[..vec.len()]);
                };
                #[cfg(feature = "parallel")]
                prev_layer.par_iter_mut().enumerate().for_each(update_row);
                #[cfg(not(feature = "parallel"))]
                prev_layer.iter_mut().enumerate().for_each(update_row);
            }
            prev_layer_log_size = log_size;
        }

        let log_ratio = lifting_log_size - prev_layer_log_size;
        if log_ratio > 0 {
            prev_layer = parallel_iter!(0..1 << lifting_log_size)
                .map(|idx| prev_layer[(idx >> (log_ratio + 1) << 1) + (idx & 1)].clone())
                .collect();
        }
        #[cfg(feature = "parallel")]
        return prev_layer.into_par_iter().map(|x| x.finalize()).collect();
        #[cfg(not(feature = "parallel"))]
        prev_layer.into_iter().map(|x| x.finalize()).collect()
    }

    fn build_packed_tree(
        columns: &[&Vec<BaseField>; SECURE_EXTENSION_DEGREE],
        _n_layers: u32,
    ) -> Option<Vec<Vec<H::Hash>>> {
        // Blake2s fast path: leaf kernel reads the coordinate columns in packed order
        // directly (no packing pass), then the layer chain — one GPU submission.
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            let is_m31 = TypeId::of::<H>() == TypeId::of::<Blake2sMerkleHasherGeneric<true>>();
            let is_bytes = TypeId::of::<H>() == TypeId::of::<Blake2sMerkleHasherGeneric<false>>();
            if is_m31 || is_bytes {
                let coords: [&[BaseField]; SECURE_EXTENSION_DEGREE] =
                    std::array::from_fn(|i| columns[i].as_slice());
                if let Some(layers) =
                    crate::prover::backend::metal::blake2s::build_packed_tree_metal(coords, is_m31)
                {
                    // Safety: TypeId equality makes this an identity conversion.
                    return Some(unsafe {
                        std::mem::transmute::<Vec<Vec<Blake2sHash>>, Vec<Vec<H::Hash>>>(layers)
                    });
                }
            }
        }
        let _ = columns;
        None
    }

    fn build_layers(leaves: Vec<H::Hash>, n_layers: u32) -> Vec<Vec<H::Hash>> {
        // Blake2s fast path: chain every large layer in one GPU submission.
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            let is_m31 = TypeId::of::<H>() == TypeId::of::<Blake2sMerkleHasherGeneric<true>>();
            let is_bytes = TypeId::of::<H>() == TypeId::of::<Blake2sMerkleHasherGeneric<false>>();
            if is_m31 || is_bytes {
                let leaves_b: Vec<Blake2sHash> = unsafe { std::mem::transmute(leaves) };
                match crate::prover::backend::metal::blake2s::build_layers_metal(
                    leaves_b, n_layers, is_m31,
                ) {
                    Ok(layers) => {
                        // Finish the sub-threshold tail on the CPU/SIMD path.
                        let mut layers: Vec<Vec<H::Hash>> = unsafe { std::mem::transmute(layers) };
                        while (layers.len() as u32) < n_layers + 1 {
                            let next = <Self as MerkleOpsLifted<H>>::build_next_layer(
                                layers.last().unwrap(),
                            );
                            layers.push(next);
                        }
                        return layers;
                    }
                    Err(leaves_b) => {
                        let leaves: Vec<H::Hash> = unsafe { std::mem::transmute(leaves_b) };
                        let mut layers = vec![leaves];
                        (0..n_layers).for_each(|_| {
                            let next = <Self as MerkleOpsLifted<H>>::build_next_layer(
                                layers.last().unwrap(),
                            );
                            layers.push(next);
                        });
                        return layers;
                    }
                }
            }
        }
        let mut layers = vec![leaves];
        (0..n_layers).for_each(|_| {
            let next = <Self as MerkleOpsLifted<H>>::build_next_layer(layers.last().unwrap());
            layers.push(next);
        });
        layers
    }

    fn build_next_layer(prev_layer: &Vec<H::Hash>) -> Vec<H::Hash> {
        // Blake2s fast path; see `build_leaves`.
        if TypeId::of::<H>() == TypeId::of::<Blake2sMerkleHasherGeneric<true>>() {
            let prev: &Vec<Blake2sHash> = unsafe { std::mem::transmute(prev_layer) };
            #[cfg(all(feature = "metal", target_os = "macos"))]
            if let Some(next) =
                crate::prover::backend::metal::blake2s::build_next_layer_metal(prev, true)
            {
                return unsafe { std::mem::transmute::<Vec<Blake2sHash>, Vec<H::Hash>>(next) };
            }
            let next = build_next_layer_simd::<true>(prev);
            return unsafe { std::mem::transmute::<Vec<Blake2sHash>, Vec<H::Hash>>(next) };
        }
        if TypeId::of::<H>() == TypeId::of::<Blake2sMerkleHasherGeneric<false>>() {
            let prev: &Vec<Blake2sHash> = unsafe { std::mem::transmute(prev_layer) };
            #[cfg(all(feature = "metal", target_os = "macos"))]
            if let Some(next) =
                crate::prover::backend::metal::blake2s::build_next_layer_metal(prev, false)
            {
                return unsafe { std::mem::transmute::<Vec<Blake2sHash>, Vec<H::Hash>>(next) };
            }
            let next = build_next_layer_simd::<false>(prev);
            return unsafe { std::mem::transmute::<Vec<Blake2sHash>, Vec<H::Hash>>(next) };
        }

        let log_size: u32 = prev_layer.len().ilog2() - 1;
        parallel_iter!(0..(1 << log_size))
            .map(|i| H::hash_children((prev_layer[2 * i], prev_layer[2 * i + 1])))
            .collect()
    }
}

/// Computes blake2s lifted-tree leaves with the shared 16-way SIMD implementation when
/// `H` is a blake2s merkle hasher, reading the scalar columns in place. Returns `None`
/// for other hashers.
fn blake2s_fast_path<H: MerkleHasherLifted + 'static>(
    columns: &[&Vec<BaseField>],
    lifting_log_size: u32,
) -> Option<Vec<H::Hash>> {
    let is_m31 = TypeId::of::<H>() == TypeId::of::<Blake2sMerkleHasherGeneric<true>>();
    let is_bytes = TypeId::of::<H>() == TypeId::of::<Blake2sMerkleHasherGeneric<false>>();
    if !is_m31 && !is_bytes {
        return None;
    }
    // `BaseField` is a transparent u32 wrapper, so a column is a flat run of u32s.
    let flat_columns: Vec<&[u32]> = columns
        .iter()
        .map(|column| unsafe {
            std::slice::from_raw_parts(column.as_ptr() as *const u32, column.len())
        })
        .collect();
    // Apple-GPU path for large uniform-size commitments; bit-identical output, with
    // the SIMD builder as fallback (no device / unsupported shape).
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        use crate::prover::backend::metal::blake2s::{build_leaves_metal, MIN_METAL_LOG_SIZE};
        let n_rows = flat_columns[0].len();
        if n_rows >= 1 << MIN_METAL_LOG_SIZE
            && flat_columns.iter().all(|c| c.len() == n_rows)
            && n_rows == 1 << lifting_log_size
        {
            if let Some(leaves) = build_leaves_metal(&flat_columns, is_m31) {
                return Some(unsafe {
                    std::mem::transmute::<Vec<Blake2sHash>, Vec<H::Hash>>(leaves)
                });
            }
        }
    }
    let leaves = if is_m31 {
        build_leaves_from_flat_columns::<true>(&flat_columns, lifting_log_size)
    } else {
        build_leaves_from_flat_columns::<false>(&flat_columns, lifting_log_size)
    };
    Some(unsafe { std::mem::transmute::<Vec<Blake2sHash>, Vec<H::Hash>>(leaves) })
}

impl PackLeavesOps for CpuBackend {
    fn pack_leaves_input(
        values: &[&Col<Self, BaseField>; SECURE_EXTENSION_DEGREE],
    ) -> [Col<Self, BaseField>; SECURE_EXTENSION_DEGREE * PACKED_LEAF_SIZE] {
        let len_m31 = values[0].len();
        assert!(values.iter().all(|c| c.len() == len_m31));
        assert!(len_m31.is_multiple_of(PACKED_LEAF_SIZE));
        let packed_len = len_m31 / PACKED_LEAF_SIZE;
        // Each output slot is a pure function of (packed_row, offset, coord); fill the
        // packed columns in parallel, reading the borrowed inputs directly.
        let mut packed_cpu: [Vec<BaseField>; SECURE_EXTENSION_DEGREE * PACKED_LEAF_SIZE] =
            core::array::from_fn(|_| vec![BaseField::default(); packed_len]);

        let fill = |column_idx: usize, column: &mut Vec<BaseField>| {
            let coord = column_idx % SECURE_EXTENSION_DEGREE;
            let offset = column_idx / SECURE_EXTENSION_DEGREE;
            let src: &[BaseField] = values[coord];
            for (packed_row, slot) in column.iter_mut().enumerate() {
                *slot = src[packed_row * PACKED_LEAF_SIZE + offset];
            }
        };
        #[cfg(feature = "parallel")]
        packed_cpu
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, col)| fill(i, col));
        #[cfg(not(feature = "parallel"))]
        packed_cpu
            .iter_mut()
            .enumerate()
            .for_each(|(i, col)| fill(i, col));

        packed_cpu
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use super::*;
    use crate::core::fields::m31::M31;
    use crate::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasherGeneric;
    use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
    use crate::prover::vcs_lifted::ops::MerkleOpsLifted;

    /// Delegates to the blake2s hasher but, having a distinct type, takes the generic
    /// row-by-row absorption path instead of the 16-way fast path.
    #[derive(Debug, Default, Clone)]
    struct ReferenceBlake(Blake2sMerkleHasherGeneric<false>);
    impl MerkleHasherLifted for ReferenceBlake {
        type Hash = <Blake2sMerkleHasherGeneric<false> as MerkleHasherLifted>::Hash;

        fn hash_children(children_hashes: (Self::Hash, Self::Hash)) -> Self::Hash {
            Blake2sMerkleHasherGeneric::<false>::hash_children(children_hashes)
        }

        fn update_leaf(&mut self, column_values: &[BaseField]) {
            self.0.update_leaf(column_values)
        }

        fn finalize(self) -> Self::Hash {
            self.0.finalize()
        }
    }

    /// The blake2s fast path must produce exactly the generic absorption's leaves and
    /// layers, including with mixed column sizes and extra lifting.
    #[test]
    fn blake2s_fast_path_matches_generic_absorption() {
        const MAX_LOG_N_ROWS: u32 = 9;
        let mut cols: Vec<Vec<BaseField>> = (0..40u32)
            .map(|i| {
                (0..1 << MAX_LOG_N_ROWS)
                    .map(|j| M31::from(100 * i + j))
                    .collect_vec()
            })
            .collect();
        cols[0] = (0..1 << (MAX_LOG_N_ROWS - 3))
            .map(M31::from_u32_unchecked)
            .collect_vec();
        cols[1] = (0..1 << (MAX_LOG_N_ROWS - 2))
            .map(M31::from_u32_unchecked)
            .collect_vec();
        let col_refs = cols.iter().collect_vec();

        let fast = <CpuBackend as MerkleOpsLifted<Blake2sMerkleHasherGeneric<false>>>::build_leaves(
            &col_refs,
            MAX_LOG_N_ROWS + 1,
        );
        let reference = <CpuBackend as MerkleOpsLifted<ReferenceBlake>>::build_leaves(
            &col_refs,
            MAX_LOG_N_ROWS + 1,
        );
        assert_eq!(fast, reference);

        let fast_next =
            <CpuBackend as MerkleOpsLifted<Blake2sMerkleHasherGeneric<false>>>::build_next_layer(
                &fast,
            );
        let reference_next =
            <CpuBackend as MerkleOpsLifted<ReferenceBlake>>::build_next_layer(&reference);
        assert_eq!(fast_next, reference_next);
    }
}
