//! Implements a FRI polynomial commitment scheme.
//!
//! This is a protocol where the prover can commit on a set of polynomials and then prove their
//! opening on a set of points.
//! Note: This implementation is not really a polynomial commitment scheme, because we are not in
//! the unique decoding regime. This is enough for a STARK proof though, where we only want to imply
//! the existence of such polynomials, and are ok with having a small decoding list.
//! Note: Opened points cannot come from the commitment domain.

pub mod quotients;
pub mod utils;
mod verifier;

use serde::{Deserialize, Serialize};

pub use self::utils::TreeVec;
pub use self::verifier::CommitmentSchemeVerifier;
use super::channel::Channel;
use super::fields::qm31::SecureField;
use super::fri::FriConfig;

#[derive(Copy, Debug, Clone, PartialEq, Eq)]
pub struct TreeSubspan {
    pub tree_index: usize,
    pub col_start: usize,
    pub col_end: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
/// Configuration parameters for the committment scheme prover.
pub struct PcsConfig {
    /// The number of proof of work bits before the FRI queries.
    pub pow_bits: u32,
    pub fri_config: FriConfig,
    /// An optional integer which controls the size of the lifting domain (This size includes the
    /// `log_blowup_factor`). When specified, the prover lifts all polynomials to the domain of
    /// given log size.
    /// If `None`, the prover lifts each tree’s polynomials to the largest domain within that tree
    /// (an implicit assumption here is that the largest domains are all of equal size across
    /// trees, except possibly for the preprocessed tree).
    pub lifting_log_size: Option<u32>,
    /// Enables zero-knowledge: hiding (salted) commitments for the witness-bearing trees. Off by
    /// default; only the outermost proof of a recursion chain needs it. When false (or when the
    /// `zk` feature is disabled), proofs and the Fiat-Shamir transcript are byte-identical to
    /// non-zk stwo. See `docs/zk.md`.
    #[cfg(feature = "zk")]
    pub zk: bool,
}
impl PcsConfig {
    pub const fn security_bits(&self) -> u32 {
        self.pow_bits + self.fri_config.security_bits()
    }

    pub fn mix_into(&self, channel: &mut impl Channel) {
        let PcsConfig {
            pow_bits,
            fri_config,
            lifting_log_size,
            #[cfg(feature = "zk")]
            zk,
        } = self;
        let FriConfig {
            log_blowup_factor,
            n_queries,
            log_last_layer_degree_bound,
            fold_step,
        } = fri_config;

        // The zk flag occupies a previously-zero slot of the config commitment, so the transcript
        // is bit-identical to non-zk stwo whenever zk is off (or the feature is disabled).
        #[cfg(feature = "zk")]
        let zk_marker = *zk as u32;
        #[cfg(not(feature = "zk"))]
        let zk_marker = 0;

        channel.mix_felts(&[
            SecureField::from_u32_unchecked(
                *pow_bits,
                *log_blowup_factor,
                *n_queries as u32,
                *log_last_layer_degree_bound,
            ),
            SecureField::from_u32_unchecked(
                *fold_step,
                lifting_log_size.unwrap_or(0),
                zk_marker,
                0,
            ),
        ]);
    }
}

impl Default for PcsConfig {
    fn default() -> Self {
        Self {
            pow_bits: 10,
            fri_config: FriConfig::new(0, 1, 3, 1),
            lifting_log_size: None,
            #[cfg(feature = "zk")]
            zk: false,
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_security_bits() {
        let config = super::PcsConfig {
            pow_bits: 42,
            fri_config: super::FriConfig::new(10, 10, 70, 1),
            lifting_log_size: None,
            ..Default::default()
        };
        assert!(config.security_bits() == 10 * 70 + 42);
    }
}
