use hashbrown::HashMap;
use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use tracing::{info, span, Level};

use crate::core::channel::{Channel, MerkleChannel};
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
#[cfg(feature = "zk")]
use crate::core::pcs::quotients::FriMaskProof;
use crate::core::pcs::quotients::{
    CommitmentSchemeProof, CommitmentSchemeProofAux, ExtendedCommitmentSchemeProof, PointSample,
};
use crate::core::pcs::utils::prepare_preprocessed_query_positions;
use crate::core::pcs::{PcsConfig, TreeSubspan, TreeVec};
use crate::core::poly::circle::CanonicCoset;
use crate::core::utils::MaybeOwned;
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::core::vcs_lifted::verifier::ExtendedMerkleDecommitmentLifted;
#[cfg(feature = "zk")]
use crate::core::verifier::PREPROCESSED_TRACE_IDX;
use crate::core::ColumnVec;
use crate::prover::air::component_prover::{Poly, Trace, WeightsHashMap};
use crate::prover::backend::{BackendForChannel, Col};
use crate::prover::fri::{FriDecommitResult, FriProver};
use crate::prover::mempool::BaseColumnPool;
use crate::prover::pcs::quotient_ops::compute_fri_quotients;
#[cfg(feature = "zk")]
use crate::prover::poly::circle::SecureCirclePoly;
use crate::prover::poly::circle::{CircleCoefficients, CircleEvaluation};
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;
use crate::prover::vcs_lifted::prover::MerkleProverLifted;
use crate::prover::vcs_lifted::zk::SaltSeed;
#[cfg(feature = "zk")]
use crate::prover::vcs_lifted::zk::ZkRng;

pub mod quotient_ops;

/// The prover side of a FRI polynomial commitment scheme. See [super].
pub struct CommitmentSchemeProver<'a, B: BackendForChannel<MC>, MC: MerkleChannel> {
    pub trees: TreeVec<MaybeOwned<'a, CommitmentTreeProver<B, MC>>>,
    pub config: PcsConfig,
    pub twiddles: &'a TwiddleTree<B>,
    pub store_polynomials_coefficients: bool,
    /// Pre-allocated base field column pool for polynomial evaluation during commit.
    pub base_column_pool: MaybeOwned<'a, BaseColumnPool<B>>,
    /// Secret CSPRNG for zero-knowledge salts. `Some` iff `config.zk`. Used to draw a fresh salt
    /// seed for each witness-bearing (non-preprocessed) commitment tree. See `docs/zk.md`.
    #[cfg(feature = "zk")]
    zk_rng: Option<ZkRng>,
}
impl<'a, B: BackendForChannel<MC>, MC: MerkleChannel> CommitmentSchemeProver<'a, B, MC> {
    /// Creates a new empty commitment scheme prover with the given configuration and twiddles. The
    /// commitment scheme does not store the polynomials coefficients by default.
    pub fn new(config: PcsConfig, twiddles: &'a TwiddleTree<B>) -> Self {
        CommitmentSchemeProver {
            trees: TreeVec::default(),
            config,
            twiddles,
            store_polynomials_coefficients: false,
            base_column_pool: MaybeOwned::Owned(BaseColumnPool::new()),
            #[cfg(feature = "zk")]
            zk_rng: config.zk.then(ZkRng::from_os),
        }
    }

    pub fn with_memory_pool(
        config: PcsConfig,
        twiddles: &'a TwiddleTree<B>,
        base_column_pool: &'a BaseColumnPool<B>,
    ) -> Self {
        CommitmentSchemeProver {
            trees: TreeVec::default(),
            config,
            twiddles,
            store_polynomials_coefficients: false,
            base_column_pool: MaybeOwned::Borrowed(base_column_pool),
            #[cfg(feature = "zk")]
            zk_rng: config.zk.then(ZkRng::from_os),
        }
    }

    /// Draws a fresh secret salt seed for the next commitment tree, or `None` when zk is disabled
    /// or the tree is the preprocessed tree (index 0, which is public and never salted).
    #[cfg_attr(not(feature = "zk"), allow(unused_variables))]
    fn draw_tree_salt_seed(&mut self, tree_index: usize) -> Option<SaltSeed> {
        #[cfg(feature = "zk")]
        {
            if tree_index != PREPROCESSED_TRACE_IDX {
                if let Some(rng) = self.zk_rng.as_mut() {
                    return Some(rng.draw_salt_seed());
                }
            }
        }
        None
    }

    /// Sets the commitment scheme to store the polynomials coefficients starting from the next
    /// commit.
    pub const fn set_store_polynomials_coefficients(&mut self) {
        self.store_polynomials_coefficients = true;
    }

    /// Evaluates the given polynomials, commits them into a Merkle tree, mixes the root into
    /// the channel, and appends the resulting tree to the scheme.
    fn commit(&mut self, polynomials: ColumnVec<CircleCoefficients<B>>, channel: &mut MC::C) {
        let _span = span!(Level::INFO, "Commitment").entered();
        let salt_seed = self.draw_tree_salt_seed(self.trees.len());
        let tree = CommitmentTreeProver::new_impl(
            polynomials,
            self.config.fri_config.log_blowup_factor,
            self.twiddles,
            self.store_polynomials_coefficients,
            self.config.lifting_log_size,
            &self.base_column_pool,
            salt_seed,
        );
        MC::mix_root(channel, tree.commitment.root());
        self.trees.push(MaybeOwned::Owned(tree));
    }

    /// Appends an externally constructed [`CommitmentTreeProver`] to the scheme and mixes its
    /// Merkle root into the channel. Accepts both owned and borrowed trees.
    pub fn commit_tree(
        &mut self,
        tree: MaybeOwned<'a, CommitmentTreeProver<B, MC>>,
        channel: &mut MC::C,
    ) {
        MC::mix_root(channel, tree.commitment.root());
        self.trees.push(tree);
    }

    pub fn tree_builder(&mut self) -> TreeBuilder<'_, 'a, B, MC> {
        TreeBuilder {
            tree_index: self.trees.len(),
            commitment_scheme: self,
            polys: Vec::default(),
        }
    }

    pub fn roots(&self) -> TreeVec<<MC::H as MerkleHasherLifted>::Hash> {
        self.trees.as_ref().map(|tree| tree.commitment.root())
    }

    pub fn polynomials(&self) -> TreeVec<ColumnVec<&Poly<B>>> {
        self.trees
            .as_ref()
            .map(|tree| tree.polynomials.iter().collect())
    }

    pub fn evaluations(
        &self,
    ) -> TreeVec<ColumnVec<&CircleEvaluation<B, BaseField, BitReversedOrder>>> {
        self.trees
            .as_ref()
            .map(|tree| tree.polynomials.iter().map(|poly| &poly.evals).collect())
    }

    pub fn trace(&self) -> Trace<'_, B> {
        let polys = self.polynomials();
        Trace { polys }
    }

    pub fn build_weights_hash_map(
        &self,
        sampled_points: &TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
        max_log_size: u32,
    ) -> WeightsHashMap<B>
    where
        Col<B, SecureField>: Send + Sync,
    {
        let weights_dashmap = WeightsHashMap::<B>::new();

        self.polynomials()
            .zip_cols(sampled_points)
            .map_cols(|(poly, points)| {
                let compute_weights = |(log_size, point): (u32, CirclePoint<SecureField>)| {
                    weights_dashmap.entry((log_size, point)).or_insert_with(|| {
                        CircleEvaluation::<B, BaseField, BitReversedOrder>::barycentric_weights(
                            CanonicCoset::new(log_size),
                            point,
                        )
                    });
                };

                let log_size = poly.evals.domain.log_size();
                // For each sample point, compute the weights needed to evaluate the polynomial at
                // the folded sample point.
                // TODO(Leo): the computation `point.repeated_double(max_log_size - log_size)` is
                // likely repeated a bunch of times in a typical flat air. Consider moving it
                // outside the loop.
                #[cfg(not(feature = "parallel"))]
                points.iter().for_each(|&point| {
                    compute_weights((log_size, point.repeated_double(max_log_size - log_size)))
                });

                #[cfg(feature = "parallel")]
                points.par_iter().for_each(|&point| {
                    compute_weights((log_size, point.repeated_double(max_log_size - log_size)))
                });
            });

        weights_dashmap
    }

    pub fn prove_values(
        mut self,
        sampled_points: TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
        channel: &mut MC::C,
    ) -> ExtendedCommitmentSchemeProof<MC::H> {
        // Evaluate polynomials on open points.
        let span = span!(
            Level::INFO,
            "Evaluate columns out of domain",
            class = "EvaluateOutOfDomain"
        )
        .entered();

        let lifting_log_size = self.trees.last().unwrap().commitment.layers.len() as u32 - 1;
        let weights_hash_map = if self.store_polynomials_coefficients {
            None
        } else {
            Some(self.build_weights_hash_map(&sampled_points, lifting_log_size))
        };

        // Lambda that evaluates a polynomial on a collection of circle points and returns a vector
        // of point samples.
        let eval_at_points = |(poly, points): (&Poly<B>, &Vec<CirclePoint<SecureField>>)| {
            points
                .iter()
                .map(|&point| PointSample {
                    point,
                    value: poly.eval_at_point(
                        point.repeated_double(lifting_log_size - poly.evals.domain.log_size()),
                        weights_hash_map.as_ref(),
                    ),
                })
                .collect_vec()
        };

        #[cfg(not(feature = "parallel"))]
        let samples: TreeVec<Vec<Vec<PointSample>>> = self
            .polynomials()
            .zip_cols(&sampled_points)
            .map_cols(eval_at_points);
        #[cfg(feature = "parallel")]
        let samples: TreeVec<Vec<Vec<PointSample>>> = self
            .polynomials()
            .zip_cols(&sampled_points)
            .par_map_cols(eval_at_points);

        span.exit();
        let sampled_values = samples
            .as_cols_ref()
            .map_cols(|x| x.iter().map(|o| o.value).collect());
        channel.mix_felts(&sampled_values.clone().flatten_cols());

        // Zero-knowledge: commit a uniformly random low-degree codeword `R` BEFORE drawing the
        // quotient-batching challenge, and add it to the batched DEEP quotient (`B = Q + R`) below.
        // Because `v_H` is even, the trace randomizer's entropy halves at every FRI fold; the
        // independent mask `R` is what keeps the FRI transcript (inner layers and last-layer
        // polynomial) free of witness information. Mixing `R`'s root here binds it ahead of the
        // challenge, so the batched combination is an affine shift (sound by correlated agreement).
        // See `docs/zk.md`, Phase 3.
        #[cfg(feature = "zk")]
        let zk_fri_mask = if self.config.zk {
            let mask_log_degree_bound = lifting_log_size - self.config.fri_config.log_blowup_factor;
            let n_coeffs = 1usize << mask_log_degree_bound;
            let degree = crate::core::fields::qm31::SECURE_EXTENSION_DEGREE;
            let rng = self.zk_rng.as_mut().expect("zk config requires a zk RNG");
            let seed = rng.draw_salt_seed();
            let coeffs: Vec<BaseField> = (0..degree * n_coeffs)
                .map(|_| rng.draw_base_felt())
                .collect();
            let mask_poly = SecureCirclePoly::<B>(core::array::from_fn(|c| {
                CircleCoefficients::new(
                    coeffs[c * n_coeffs..(c + 1) * n_coeffs]
                        .iter()
                        .copied()
                        .collect(),
                )
            }));
            let mask_eval = mask_poly.evaluate_with_twiddles(
                CanonicCoset::new(lifting_log_size).circle_domain(),
                self.twiddles,
            );
            let mask_columns: Vec<&Col<B, BaseField>> = mask_eval.values.columns.iter().collect();
            let mask_tree = MerkleProverLifted::<B, MC::H>::commit_salted(
                mask_columns,
                lifting_log_size,
                0,
                &seed,
            );
            MC::mix_root(channel, mask_tree.root());
            Some((mask_eval, mask_tree, seed))
        } else {
            None
        };

        let columns = self.evaluations();
        print_column_size_histogram::<B, MC>(&columns);
        // Compute oods quotients for boundary constraints on the sampled points.
        let quotients = compute_fri_quotients(
            &columns,
            &samples,
            channel.draw_secure_felt(),
            lifting_log_size,
            self.twiddles,
            self.config.fri_config.log_blowup_factor,
        );

        // Blind the batched quotient with the zk mask: `B = Q + R` (no-op when zk is off).
        #[cfg(feature = "zk")]
        let quotients = {
            let mut quotients = quotients;
            if let Some((mask_eval, ..)) = &zk_fri_mask {
                for i in 0..quotients.len() {
                    let blinded = quotients.at(i) + mask_eval.at(i);
                    quotients.set(i, blinded);
                }
            }
            quotients
        };

        // Run FRI commitment phase on the (possibly blinded) oods quotients.
        let fri_prover =
            FriProver::<B, MC>::commit(channel, self.config.fri_config, &quotients, self.twiddles);

        // Proof of work.
        let span1 = span!(Level::INFO, "Grind", class = "Queries POW").entered();
        let proof_of_work = B::grind(channel, self.config.pow_bits);
        span1.exit();
        channel.mix_u64(proof_of_work);

        // FRI decommitment phase.
        let FriDecommitResult {
            fri_proof,
            query_positions,
            unsorted_query_locations,
        } = fri_prover.decommit(channel);

        // Open the zk mask at the FRI query positions so the verifier can re-add `R` to the
        // batched-quotient answers and reconstruct the committed FRI input.
        #[cfg(feature = "zk")]
        let fri_mask = zk_fri_mask.map(|(mask_eval, mask_tree, seed)| {
            let mask_columns: Vec<&Col<B, BaseField>> = mask_eval.values.columns.iter().collect();
            let (queried_values, ext_decommitment) =
                mask_tree.decommit_salted(&query_positions, mask_columns, &seed);
            FriMaskProof {
                commitment: mask_tree.root(),
                queried_values,
                decommitment: ext_decommitment.decommitment,
            }
        });
        // Build the query position tree.
        let preprocessed_query_positions = prepare_preprocessed_query_positions(
            &query_positions,
            lifting_log_size,
            self.trees[0].commitment.layers.len() as u32 - 1,
        );
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
        let commitments = self.roots();
        let (queried_values, decommitments, aux): (Vec<_>, Vec<_>, Vec<_>) = self
            .trees
            .as_ref()
            .zip_eq(query_positions_tree)
            .map(|(tree, query_positions)| tree.decommit(query_positions))
            .0
            .into_iter()
            .map(|(v, x)| (v, x.decommitment, x.aux))
            .multiunzip();

        // Return evaluation buffers to the memory pool for reuse (owned trees only).
        for tree in &mut self.trees.0 {
            if let MaybeOwned::Owned(tree) = tree {
                for poly in tree.polynomials.drain(..) {
                    let log_size = poly.evals.domain.log_size();
                    self.base_column_pool.give_back(log_size, poly.evals.values);
                }
            }
        }

        ExtendedCommitmentSchemeProof {
            proof: CommitmentSchemeProof {
                commitments,
                sampled_values,
                decommitments: TreeVec(decommitments),
                queried_values: TreeVec(queried_values),
                proof_of_work,
                fri_proof: fri_proof.proof,
                config: self.config,
                #[cfg(feature = "zk")]
                fri_mask,
            },
            aux: CommitmentSchemeProofAux {
                unsorted_query_locations,
                trace_decommitment: TreeVec(aux),
                fri: fri_proof.aux,
            },
        }
    }
}

/// Helper struct for aggregating polynomials and evaluations for a commitment tree.
pub struct TreeBuilder<'a, 'b, B: BackendForChannel<MC>, MC: MerkleChannel> {
    tree_index: usize,
    commitment_scheme: &'a mut CommitmentSchemeProver<'b, B, MC>,
    polys: ColumnVec<CircleCoefficients<B>>,
}
impl<B: BackendForChannel<MC>, MC: MerkleChannel> TreeBuilder<'_, '_, B, MC> {
    pub fn extend_evals(
        &mut self,
        columns: Vec<CircleEvaluation<B, BaseField, BitReversedOrder>>,
    ) -> TreeSubspan {
        let span = span!(Level::INFO, "Interpolation for commitment").entered();
        let polys = B::interpolate_columns(columns, self.commitment_scheme.twiddles);
        span.exit();

        self.extend_polys(polys)
    }

    pub fn extend_polys(
        &mut self,
        columns: impl IntoIterator<Item = CircleCoefficients<B>>,
    ) -> TreeSubspan {
        let col_start = self.polys.len();
        self.polys.extend(columns);
        let col_end = self.polys.len();
        TreeSubspan {
            tree_index: self.tree_index,
            col_start,
            col_end,
        }
    }

    pub fn commit(self, channel: &mut MC::C) {
        let _span = span!(Level::INFO, "Commitment").entered();
        self.commitment_scheme.commit(self.polys, channel);
    }
}

/// Prover data for a single commitment tree in a commitment scheme. The commitment scheme allows to
/// commit on a set of polynomials at a time. This corresponds to such a set.
pub struct CommitmentTreeProver<B: BackendForChannel<MC>, MC: MerkleChannel> {
    pub polynomials: ColumnVec<Poly<B>>,
    pub commitment: MerkleProverLifted<B, MC::H>,
    /// Secret salt seed used to commit this tree's leaves; `Some` iff the tree is hiding. Retained
    /// so [`Self::decommit`] can reveal the salts for opened positions. See `docs/zk.md`.
    pub salt_seed: Option<SaltSeed>,
}

impl<B: BackendForChannel<MC>, MC: MerkleChannel> CommitmentTreeProver<B, MC> {
    pub fn new(
        polynomials: ColumnVec<CircleCoefficients<B>>,
        log_blowup_factor: u32,
        twiddles: &TwiddleTree<B>,
        store_polynomials_coefficients: bool,
        lifting_log_size: Option<u32>,
        base_column_pool: &BaseColumnPool<B>,
    ) -> Self {
        Self::new_impl(
            polynomials,
            log_blowup_factor,
            twiddles,
            store_polynomials_coefficients,
            lifting_log_size,
            base_column_pool,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_impl(
        polynomials: ColumnVec<CircleCoefficients<B>>,
        log_blowup_factor: u32,
        twiddles: &TwiddleTree<B>,
        store_polynomials_coefficients: bool,
        lifting_log_size: Option<u32>,
        base_column_pool: &BaseColumnPool<B>,
        salt_seed: Option<SaltSeed>,
    ) -> Self {
        let span = span!(Level::INFO, "Extension").entered();
        let polynomials = B::evaluate_polynomials(
            polynomials,
            log_blowup_factor,
            twiddles,
            store_polynomials_coefficients,
            base_column_pool,
        );
        span.exit();

        let _span = span!(Level::INFO, "Merkle").entered();
        let max_log_domain_size = polynomials
            .iter()
            .map(|poly| poly.evals.domain.log_size())
            .max()
            .unwrap_or_default();
        let lifting_log_size = lifting_log_size.unwrap_or(max_log_domain_size);
        let columns: Vec<&Col<B, BaseField>> = polynomials
            .iter()
            .map(|poly: &Poly<B>| &poly.evals.values)
            .collect();
        let tree = match &salt_seed {
            #[cfg(feature = "zk")]
            Some(seed) => MerkleProverLifted::commit_salted(columns, lifting_log_size, 0, seed),
            _ => MerkleProverLifted::commit(columns, lifting_log_size, 0),
        };

        CommitmentTreeProver {
            polynomials,
            commitment: tree,
            salt_seed,
        }
    }

    /// Decommits the merkle tree on the given query positions.
    /// Returns the values at the queried positions and the decommitment.
    /// The queries are given as a mapping from the log size of the layer size to the queried
    /// positions on each column of that size.
    fn decommit(
        &self,
        queries: &[usize],
    ) -> (
        ColumnVec<Vec<BaseField>>,
        ExtendedMerkleDecommitmentLifted<MC::H>,
    ) {
        let eval_vec = self
            .polynomials
            .iter()
            .map(|poly| &poly.evals.values)
            .collect_vec();
        match &self.salt_seed {
            #[cfg(feature = "zk")]
            Some(seed) => self.commitment.decommit_salted(queries, eval_vec, seed),
            _ => self.commitment.decommit(queries, eval_vec),
        }
    }
}

fn print_column_size_histogram<B: BackendForChannel<MC>, MC: MerkleChannel>(
    columns_per_tree: &TreeVec<ColumnVec<&CircleEvaluation<B, BaseField, BitReversedOrder>>>,
) {
    let mut log_size_histogram = HashMap::new();
    for columns in columns_per_tree.iter() {
        for column in columns {
            *log_size_histogram
                .entry(column.domain.log_size())
                .or_insert(0) += 1;
        }
    }
    for (log_size, count) in log_size_histogram {
        info!("Log size {log_size}: {count}");
    }
}
