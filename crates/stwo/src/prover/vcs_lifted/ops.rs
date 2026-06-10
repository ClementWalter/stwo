use serde::{Deserialize, Serialize};

use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SECURE_EXTENSION_DEGREE;
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::core::vcs_lifted::verifier::PACKED_LEAF_SIZE;
use crate::prover::backend::{Col, ColumnOps};

/// Trait for performing Merkle operations on a commitment scheme.
pub trait MerkleOpsLifted<H: MerkleHasherLifted>:
    ColumnOps<BaseField> + ColumnOps<H::Hash> + PackLeavesOps + for<'de> Deserialize<'de> + Serialize
{
    /// Computes the leaves of the lifted Merkle commitment.
    fn build_leaves(columns: &[&Col<Self, BaseField>], lifting_log_size: u32)
        -> Col<Self, H::Hash>;

    /// Given a layer of hashes as input, computes a new layer by hashing pairs
    /// of adjacent elements of the input, as in a standard Merkle tree.
    fn build_next_layer(prev_layer: &Col<Self, H::Hash>) -> Col<Self, H::Hash>;

    /// Builds the whole packed-leaf tree (four secure-coordinate columns, four rows
    /// per leaf) in one batch, layers leaves-first. `None` (the default) keeps the
    /// separate pack + leaves + layers path.
    fn build_packed_tree(
        _columns: &[&Col<Self, BaseField>; SECURE_EXTENSION_DEGREE],
        _n_layers: u32,
    ) -> Option<Vec<Col<Self, H::Hash>>> {
        None
    }

    /// Builds all `n_layers` tree layers above `leaves` (leaves first). Backends can
    /// override to batch the whole chain (e.g. one GPU submission).
    fn build_layers(leaves: Col<Self, H::Hash>, n_layers: u32) -> Vec<Col<Self, H::Hash>> {
        let mut layers = vec![leaves];
        (0..n_layers).for_each(|_| {
            layers.push(Self::build_next_layer(layers.last().unwrap()));
        });
        layers
    }
}

pub trait PackLeavesOps: ColumnOps<BaseField> {
    /// Given a column of QM31s (represented as 4 columns of M31s), reshapes it into 4 columns of
    /// QM31s (represented as 16 columns of M31s). Denoting the input column as [v₀, v₁, v₂, v₃,
    /// ...] where vᵢ ∈ QM31, the output is [[v₀, v₄, v₈, ...], [v₁, v₅, v₉, ...], [v₂, v₆, v₁₀,
    /// ...], [v₃, v₇, v₁₁, ...]].
    fn pack_leaves_input(
        values: &[&Col<Self, BaseField>; SECURE_EXTENSION_DEGREE],
    ) -> [Col<Self, BaseField>; SECURE_EXTENSION_DEGREE * PACKED_LEAF_SIZE];
}
