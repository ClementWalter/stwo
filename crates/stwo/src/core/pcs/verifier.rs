use core::iter::zip;

use itertools::Itertools;
use std_shims::Vec;

use super::super::circle::CirclePoint;
use super::super::fields::qm31::SecureField;
use super::super::fri::{CirclePolyDegreeBound, FriVerifier};
use super::quotients::{fri_answers, PointSample};
use super::utils::TreeVec;
use super::PcsConfig;
use crate::core::channel::{Channel, MerkleChannel};
use crate::core::pcs::quotients::CommitmentSchemeProof;
use crate::core::pcs::utils::prepare_preprocessed_query_positions;
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::core::vcs_lifted::verifier::MerkleVerifierLifted;
use crate::core::verifier::VerificationError;
use crate::core::ColumnVec;

/// The verifier side of a FRI polynomial commitment scheme. See [super].
#[derive(Default)]
pub struct CommitmentSchemeVerifier<MC: MerkleChannel> {
    pub trees: TreeVec<MerkleVerifierLifted<MC::H>>,
    pub config: PcsConfig,
}

impl<MC: MerkleChannel> CommitmentSchemeVerifier<MC> {
    pub fn new(config: PcsConfig) -> Self {
        Self {
            trees: TreeVec::default(),
            config,
        }
    }

    /// A [TreeVec<ColumnVec>] of the log sizes of each column in each commitment tree.
    fn column_log_sizes(&self) -> TreeVec<ColumnVec<u32>> {
        self.trees
            .as_ref()
            .map(|tree| tree.column_log_sizes.clone())
    }

    /// Reads a commitment from the prover.
    pub fn commit(
        &mut self,
        commitment: <MC::H as MerkleHasherLifted>::Hash,
        log_sizes: &[u32],
        channel: &mut MC::C,
    ) {
        MC::mix_root(channel, commitment);
        let extended_log_sizes = log_sizes
            .iter()
            .map(|&log_size| log_size + self.config.fri_config.log_blowup_factor)
            .collect();
        let verifier =
            MerkleVerifierLifted::new(commitment, extended_log_sizes, self.config.lifting_log_size);
        self.trees.push(verifier);
    }

    pub fn verify_values(
        &self,
        sampled_points: TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
        proof: CommitmentSchemeProof<MC::H>,
        channel: &mut MC::C,
    ) -> Result<(), VerificationError> {
        channel.mix_felts(&proof.sampled_values.clone().flatten_cols());

        let lifting_log_size = self.trees.last().unwrap().height;

        // Zero-knowledge: clone the FRI blinding mask opening, mix its commitment into the channel
        // before the batching challenge (matching the prover), and prepare its Merkle verifier.
        // Fail closed if a zk verifier is given a proof without a mask. See `docs/zk.md`, Phase 3.
        #[cfg(feature = "zk")]
        let zk_mask = proof.fri_mask.clone();
        #[cfg(feature = "zk")]
        let zk_mask_verifier = if self.config.zk {
            let mask = zk_mask.as_ref().ok_or_else(|| {
                VerificationError::InvalidStructure(std_shims::ToString::to_string(
                    &"zero-knowledge enabled but the proof carries no FRI mask",
                ))
            })?;
            MC::mix_root(channel, mask.commitment);
            let degree = crate::core::fields::qm31::SECURE_EXTENSION_DEGREE;
            Some(MerkleVerifierLifted::<MC::H>::new(
                mask.commitment,
                (0..degree).map(|_| lifting_log_size).collect(),
                None,
            ))
        } else {
            None
        };

        let random_coeff = channel.draw_secure_felt();
        let bound =
            CirclePolyDegreeBound::new(lifting_log_size - self.config.fri_config.log_blowup_factor);

        // FRI commitment phase on OODS quotients.
        let mut fri_verifier =
            FriVerifier::<MC>::commit(channel, self.config.fri_config, proof.fri_proof, bound)?;

        // Verify proof of work.
        if !channel.verify_pow_nonce(self.config.pow_bits, proof.proof_of_work) {
            return Err(VerificationError::ProofOfWork);
        }
        channel.mix_u64(proof.proof_of_work);
        // Get FRI query positions.
        let query_positions = fri_verifier.sample_query_positions(channel);
        let preprocessed_query_positions = prepare_preprocessed_query_positions(
            &query_positions,
            lifting_log_size,
            self.trees[0].height,
        );

        // Build the query positions tree: the preprocessed tree needs a different treatment than
        // the other trees.
        let query_positions_tree = TreeVec::new(
            self.trees
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    if i == 0 {
                        preprocessed_query_positions.as_slice()
                    } else {
                        query_positions.as_slice()
                    }
                })
                .collect::<Vec<_>>(),
        );
        // Fail-closed: when zk is enabled, every witness-bearing (non-preprocessed, non-empty)
        // tree must carry per-leaf salts. A non-salted decommitment cannot pass against a salted
        // root anyway, but this also rejects a non-hiding proof presented to a zk verifier. See
        // `docs/zk.md`.
        #[cfg(feature = "zk")]
        if self.config.zk {
            for (i, (tree, decommitment)) in self
                .trees
                .iter()
                .zip(proof.decommitments.iter())
                .enumerate()
            {
                if i != crate::core::verifier::PREPROCESSED_TRACE_IDX
                    && tree.height > 0
                    && decommitment.salts.is_empty()
                {
                    return Err(VerificationError::InvalidStructure(
                        std_shims::ToString::to_string(
                            &"zero-knowledge enabled but a witness tree carries no salts",
                        ),
                    ));
                }
            }
        }

        // Verify decommitments.
        self.trees
            .as_ref()
            .zip_eq(proof.decommitments)
            .zip_eq(proof.queried_values.clone())
            .zip_eq(query_positions_tree)
            .map(
                |(((tree, decommitment), queried_values), query_positions)| {
                    tree.verify(query_positions, queried_values, decommitment)
                },
            )
            .0
            .into_iter()
            .collect::<Result<(), _>>()?;
        // Answer FRI queries.
        let samples = sampled_points.zip_cols(proof.sampled_values).map_cols(
            |(sampled_points, sampled_values)| {
                zip(sampled_points, sampled_values)
                    .map(|(point, value)| PointSample { point, value })
                    .collect_vec()
            },
        );

        let fri_answers = fri_answers(
            self.column_log_sizes(),
            samples,
            random_coeff,
            &query_positions,
            proof.queried_values,
            lifting_log_size,
        )?;

        // Zero-knowledge: verify the FRI mask opening and add `R` back into the answers so they
        // equal the committed FRI input `B = Q + R`. See `docs/zk.md`, Phase 3.
        #[cfg(feature = "zk")]
        let fri_answers = {
            let mut fri_answers = fri_answers;
            if let (Some(mask), Some(mask_verifier)) = (&zk_mask, &zk_mask_verifier) {
                mask_verifier.verify(
                    &query_positions,
                    mask.queried_values.clone(),
                    mask.decommitment.clone(),
                )?;
                for (i, answer) in fri_answers.iter_mut().enumerate() {
                    let mask_value = SecureField::from_m31_array(core::array::from_fn(|c| {
                        mask.queried_values[c][i]
                    }));
                    *answer += mask_value;
                }
            }
            fri_answers
        };

        fri_verifier.decommit(fri_answers)?;

        Ok(())
    }
}
