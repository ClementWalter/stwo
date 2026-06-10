//! Zero-knowledge salting primitives for the lifted Merkle commitment.
//!
//! When the `zk` feature is enabled and a [`crate::core::pcs::PcsConfig`] opts
//! in, every leaf of a witness-bearing commitment tree is committed as
//! `hash_children(leaf, salt)` with a secret per-leaf salt. This hides the
//! committed columns — the root and unopened leaves reveal nothing — while
//! preserving binding, which reduces to collision-resistance of `hash_children`,
//! the same assumption the rest of the tree relies on. See `docs/zk.md`.
//!
//! The salt derivation and fold use only the [`MerkleHasherLifted`] and
//! [`Column`] surfaces, so they are backend- and hasher-agnostic (Blake2s on CPU
//! and SIMD, Poseidon, and downstream custom hashers all work unchanged). Only
//! [`ZkRng`], which draws secret seeds from a CSPRNG, requires the `zk` feature.

use crate::core::fields::m31::BaseField;
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::prover::backend::{Col, Column};
use crate::prover::vcs_lifted::ops::MerkleOpsLifted;

/// Number of base-field elements in a per-tree salt seed. `8 * 31 = 248` bits of
/// entropy, comfortably above the 128-bit hiding target.
pub const ZK_SALT_SEED_LEN: usize = 8;

/// A secret per-commitment-tree salt seed. Per-leaf salts are derived from it.
///
/// The seed is secret prover state: it never appears in a proof. The verifier
/// receives the derived per-leaf salts for opened positions only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SaltSeed(pub [BaseField; ZK_SALT_SEED_LEN]);

/// Derives the salt for a single leaf as `H(seed ‖ M31(index))`.
///
/// The index is a leaf position in `[0, 2^lifting_log_size)`; since the lifting
/// log size is at most the circle log order minus one, it fits a single M31 limb.
pub fn derive_salt<H: MerkleHasherLifted>(seed: &SaltSeed, index: usize) -> H::Hash {
    debug_assert!(
        (index as u64) < crate::core::fields::m31::P as u64,
        "leaf index must fit a single base-field element"
    );
    let mut input = [BaseField::default(); ZK_SALT_SEED_LEN + 1];
    input[..ZK_SALT_SEED_LEN].copy_from_slice(&seed.0);
    input[ZK_SALT_SEED_LEN] = BaseField::from_u32_unchecked(index as u32);

    let mut hasher = H::default();
    hasher.update_leaf(&input);
    hasher.finalize()
}

/// Replaces each leaf hash `leaf_i` with the salted leaf
/// `hash_children(leaf_i, salt_i)`. The tree built over the result has the same
/// height and inner-layer structure as the unsalted tree; only the leaf contents
/// change.
pub fn fold_salts<B, H>(leaves: &Col<B, H::Hash>, seed: &SaltSeed) -> Col<B, H::Hash>
where
    H: MerkleHasherLifted,
    B: MerkleOpsLifted<H>,
{
    (0..leaves.len())
        .map(|i| H::hash_children((leaves.at(i), derive_salt::<H>(seed, i))))
        .collect()
}

/// Derives the salts for a set of (deduplicated, increasing) opened leaf
/// positions, in iteration order. The order must match the order in which the
/// verifier rebuilds leaves in [`crate::core::vcs_lifted::verifier`].
pub fn salts_for_positions<H: MerkleHasherLifted>(
    seed: &SaltSeed,
    deduped_positions: impl Iterator<Item = usize>,
) -> std_shims::Vec<H::Hash> {
    deduped_positions
        .map(|index| derive_salt::<H>(seed, index))
        .collect()
}

/// Cryptographic RNG for zero-knowledge salts (ChaCha-based [`StdRng`]).
///
/// Seeded from OS entropy in production. Salt randomness MUST be independent of
/// the Fiat-Shamir transcript: transcript-derived values are public and provide
/// no hiding.
#[cfg(feature = "zk")]
#[derive(Debug)]
pub struct ZkRng(rand::rngs::StdRng);

#[cfg(feature = "zk")]
impl ZkRng {
    /// Seeds from operating-system entropy. Use in production.
    pub fn from_os() -> Self {
        use rand::SeedableRng;
        Self(rand::rngs::StdRng::from_entropy())
    }

    /// Seeds deterministically. For tests only — a fixed seed provides no hiding.
    pub fn from_test_seed(seed: u64) -> Self {
        use rand::SeedableRng;
        Self(rand::rngs::StdRng::seed_from_u64(seed))
    }

    /// Draws a fresh secret salt seed for one commitment tree, uniform over the
    /// base field in each limb.
    pub fn draw_salt_seed(&mut self) -> SaltSeed {
        use rand::Rng;
        SaltSeed(core::array::from_fn(|_| {
            BaseField::from_u32_unchecked(self.0.gen_range(0..crate::core::fields::m31::P))
        }))
    }
}
