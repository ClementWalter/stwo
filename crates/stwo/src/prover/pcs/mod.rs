use hashbrown::HashMap;
use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use tracing::{info, span, Level};

use crate::core::channel::{Channel, MerkleChannel};
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::quotients::{
    CommitmentSchemeProof, CommitmentSchemeProofAux, ExtendedCommitmentSchemeProof, PointSample,
};
use crate::core::pcs::utils::prepare_preprocessed_query_positions;
use crate::core::pcs::{PcsConfig, TreeSubspan, TreeVec};
use crate::core::poly::circle::CanonicCoset;
use crate::core::utils::MaybeOwned;
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::core::vcs_lifted::verifier::ExtendedMerkleDecommitmentLifted;
use crate::core::ColumnVec;
use crate::prover::air::component_prover::{Poly, Trace, WeightsHashMap};
use crate::prover::backend::{Backend, BackendForChannel, Col};
use crate::prover::fri::{FriDecommitResult, FriProver};
use crate::prover::mempool::BaseColumnPool;
use crate::prover::pcs::quotient_ops::compute_fri_quotients;
use crate::prover::poly::circle::{
    BarycentricEvalGroup, CircleCoefficients, CircleEvaluation, EvalsOrCoeffs,
};
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;
use crate::prover::vcs_lifted::prover::MerkleProverLifted;

pub mod quotient_ops;

/// Safely moves an owned value across a runtime type check. Unlike a
/// `TypeId`-guarded `transmute_copy`, a failed check returns the original value and
/// never creates duplicate ownership.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn downcast_owned<T: 'static, U: 'static>(value: T) -> Result<U, T> {
    let value: Box<dyn std::any::Any> = Box::new(value);
    match value.downcast::<U>() {
        Ok(value) => Ok(*value),
        Err(value) => Err(*value.downcast::<T>().unwrap_or_else(|_| {
            panic!("owned type-erasure recovery must preserve its input type")
        })),
    }
}

#[cfg(all(test, feature = "metal", target_os = "macos"))]
mod metal_type_erasure_tests {
    #[test]
    fn owned_downcast_moves_or_recovers_once() {
        assert_eq!(super::downcast_owned::<_, u32>(7u32), Ok(7));
        assert_eq!(super::downcast_owned::<_, u64>(11u32), Err(11));
    }
}

fn batch_evaluation_samples<B: Backend>(
    polys: &TreeVec<ColumnVec<&Poly<B>>>,
    sampled_points: &TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
    weights_map: Option<&WeightsHashMap<B>>,
    lifting_log_size: u32,
) -> TreeVec<Vec<Vec<PointSample>>> {
    use num_traits::Zero;

    // (tree index, column index, point index) entries grouped per shared weights.
    type SampleGroups = HashMap<(u32, CirclePoint<SecureField>), Vec<(usize, usize, usize)>>;

    let mut groups: SampleGroups = HashMap::new();
    for (t, (cols, pts_cols)) in polys.iter().zip(sampled_points.iter()).enumerate() {
        for (c, (poly, points)) in cols.iter().zip(pts_cols.iter()).enumerate() {
            let log_size = poly.evals.domain.log_size();
            for (pi, &point) in points.iter().enumerate() {
                let folded = point.repeated_double(lifting_log_size - log_size);
                groups
                    .entry((log_size, folded))
                    .or_default()
                    .push((t, c, pi));
            }
        }
    }

    // Scatter grouped results back into the original tree/column/point positions;
    // HashMap iteration order therefore cannot affect transcript sample order.
    let mut samples: TreeVec<Vec<Vec<PointSample>>> =
        sampled_points.as_cols_ref().map_cols(|points| {
            points
                .iter()
                .map(|&point| PointSample {
                    point,
                    value: SecureField::zero(),
                })
                .collect_vec()
        });
    let mut groups = groups.into_iter().collect_vec();
    groups.sort_by_key(|((log_size, folded), _)| (*log_size, folded.x, folded.y));
    let backend_groups = groups
        .iter()
        .map(|&((log_size, folded), ref entries)| BarycentricEvalGroup {
            coset: CanonicCoset::new(log_size),
            point: folded,
            evals: entries
                .iter()
                .map(|&(t, c, _)| &polys[t][c].evals)
                .collect_vec(),
        })
        .collect_vec();
    let resident_start = std::time::Instant::now();
    let resident_values = B::barycentric_eval_many_groups(&backend_groups);
    assert_eq!(resident_values.len(), groups.len());
    let resident_groups = resident_values
        .iter()
        .filter(|values| values.is_some())
        .count();
    if resident_groups > 0 {
        tracing::info!(
            "OOD resident batch: groups={resident_groups}/{} in {:?}",
            groups.len(),
            resident_start.elapsed()
        );
    }

    for (group_index, ((log_size, folded), entries)) in groups.into_iter().enumerate() {
        let group_start = std::time::Instant::now();
        let values = if let Some(values) = &resident_values[group_index] {
            values.clone()
        } else {
            let cached_weights = weights_map.and_then(|map| map.get(&(log_size, folded)));
            if let Some(weights) = cached_weights {
                B::barycentric_eval_many_at_point(&backend_groups[group_index].evals, &weights)
            } else {
                // Either an eligible resident submission failed, or the caller
                // deliberately supplied a partial cache. Recompute before the
                // transcript observes sampled values.
                let weights = B::barycentric_weights(CanonicCoset::new(log_size), folded);
                B::barycentric_eval_many_at_point(&backend_groups[group_index].evals, &weights)
            }
        };
        assert_eq!(values.len(), entries.len());
        for (&(t, c, pi), value) in entries.iter().zip(values) {
            samples[t][c][pi].value = value;
        }
        tracing::info!(
            "OOD evaluation group: log_size={log_size} cols={} in {:?}",
            entries.len(),
            group_start.elapsed()
        );
    }
    samples
}

/// The prover side of a FRI polynomial commitment scheme. See [super].
pub struct CommitmentSchemeProver<'a, B: BackendForChannel<MC>, MC: MerkleChannel> {
    pub trees: TreeVec<MaybeOwned<'a, CommitmentTreeProver<B, MC>>>,
    pub config: PcsConfig,
    pub twiddles: &'a TwiddleTree<B>,
    pub store_polynomials_coefficients: bool,
    /// Pre-allocated base field column pool for polynomial evaluation during commit.
    pub base_column_pool: MaybeOwned<'a, BaseColumnPool<B>>,
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
        }
    }

    /// Sets the commitment scheme to store the polynomials coefficients starting from the next
    /// commit.
    pub const fn set_store_polynomials_coefficients(&mut self) {
        self.store_polynomials_coefficients = true;
    }

    /// Interpolates and evaluates the given columns, commits them into a Merkle tree,
    /// mixes the root into the channel, and appends the resulting tree to the scheme.
    fn commit(&mut self, columns: ColumnVec<EvalsOrCoeffs<B>>, channel: &mut MC::C) {
        let _span = span!(Level::INFO, "Commitment").entered();
        let tree = CommitmentTreeProver::new(
            columns,
            self.config.fri_config.log_blowup_factor,
            self.twiddles,
            self.store_polynomials_coefficients,
            self.config.lifting_log_size,
            &self.base_column_pool,
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
        self.build_weights_hash_map_below(sampled_points, max_log_size, None)
    }

    /// Prebuilds only weights below `exclusive_log_limit`. Resident backends use
    /// this to retain cheap small-domain fallback weights without duplicating the
    /// large columns generated directly on the accelerator.
    fn build_weights_hash_map_below(
        &self,
        sampled_points: &TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
        max_log_size: u32,
        exclusive_log_limit: Option<u32>,
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
                if exclusive_log_limit.is_some_and(|limit| log_size >= limit) {
                    return;
                }
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

    /// Builds, for every distinct (coefficients log size, folded sample point) pair, the
    /// FFT-basis column used to evaluate stored coefficients out of domain. The map has
    /// the same shape as the barycentric weights map; which of the two a value is depends
    /// on `store_polynomials_coefficients`.
    pub fn build_eval_basis_map(
        &self,
        sampled_points: &TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
        max_log_size: u32,
    ) -> WeightsHashMap<B>
    where
        Col<B, SecureField>: Send + Sync,
    {
        let basis_map = WeightsHashMap::<B>::new();

        self.polynomials()
            .zip_cols(sampled_points)
            .map_cols(|(poly, points)| {
                let Some(coeffs) = &poly.coeffs else {
                    return;
                };
                let log_size = coeffs.log_size();
                let eval_log_size = poly.evals.domain.log_size();
                #[cfg(not(feature = "parallel"))]
                points.iter().for_each(|&point| {
                    let folded = point.repeated_double(max_log_size - eval_log_size);
                    basis_map
                        .entry((log_size, folded))
                        .or_insert_with(|| B::eval_basis_at_point(log_size, folded));
                });
                #[cfg(feature = "parallel")]
                points.par_iter().for_each(|&point| {
                    let folded = point.repeated_double(max_log_size - eval_log_size);
                    basis_map
                        .entry((log_size, folded))
                        .or_insert_with(|| B::eval_basis_at_point(log_size, folded));
                });
            });

        basis_map
    }

    /// Evaluates every committed polynomial at its sampled points from stored
    /// coefficients, grouping all same-size columns sampled at one folded point so the
    /// backend streams their shared FFT-basis column once per group.
    fn batched_coefficient_samples(
        &self,
        sampled_points: &TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
        basis_map: &WeightsHashMap<B>,
        lifting_log_size: u32,
    ) -> TreeVec<Vec<Vec<PointSample>>> {
        use num_traits::Zero;

        // (tree index, column index, point index) entries grouped per shared basis.
        type SampleGroups = HashMap<(u32, CirclePoint<SecureField>), Vec<(usize, usize, usize)>>;

        let polys = self.polynomials();
        let mut groups: SampleGroups = HashMap::new();
        for (t, (cols, pts_cols)) in polys.iter().zip(sampled_points.iter()).enumerate() {
            for (c, (poly, points)) in cols.iter().zip(pts_cols.iter()).enumerate() {
                let coeffs = poly.coeffs.as_ref().expect("coefficients stored");
                for (pi, &point) in points.iter().enumerate() {
                    let folded =
                        point.repeated_double(lifting_log_size - poly.evals.domain.log_size());
                    groups
                        .entry((coeffs.log_size(), folded))
                        .or_default()
                        .push((t, c, pi));
                }
            }
        }

        let mut samples: TreeVec<Vec<Vec<PointSample>>> =
            sampled_points.as_cols_ref().map_cols(|points| {
                points
                    .iter()
                    .map(|&point| PointSample {
                        point,
                        value: SecureField::zero(),
                    })
                    .collect_vec()
            });
        // Groups are few (one per (size, folded point)); each backend call parallelizes
        // internally over rows.
        for ((log_size, folded), entries) in groups {
            let group_start = std::time::Instant::now();
            let group_polys = entries
                .iter()
                .map(|&(t, c, _)| polys[t][c].coeffs.as_ref().unwrap())
                .collect_vec();
            let basis = basis_map
                .get(&(log_size, folded))
                .expect("basis built for all sampled points");
            let values = B::eval_many_at_point_with_basis(&group_polys, &basis);
            for (&(t, c, pi), value) in entries.iter().zip(values) {
                samples[t][c][pi].value = value;
            }
            tracing::info!(
                "OOD group: log_size={log_size} cols={} in {:?}",
                entries.len(),
                group_start.elapsed()
            );
        }
        samples
    }

    /// Evaluates committed evaluation-form polynomials at their sampled points,
    /// grouping all same-domain columns sampled at one folded point so the backend
    /// streams the shared barycentric-weight column once per group.
    fn batched_evaluation_samples(
        &self,
        sampled_points: &TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
        weights_map: Option<&WeightsHashMap<B>>,
        lifting_log_size: u32,
    ) -> TreeVec<Vec<Vec<PointSample>>> {
        let polys = self.polynomials();
        batch_evaluation_samples(&polys, sampled_points, weights_map, lifting_log_size)
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
        let ood_start = std::time::Instant::now();

        let lifting_log_size = self.trees.last().unwrap().commitment.layers.len() as u32 - 1;
        let basis_span = span!(Level::INFO, "OOD basis", class = "OodBasis").entered();
        let weights_hash_map = if self.store_polynomials_coefficients {
            // With stored coefficients, share one FFT-basis column per
            // (coefficient size, folded point) across all polynomials sampled there.
            Some(self.build_eval_basis_map(&sampled_points, lifting_log_size))
        } else if let Some(min_log_size) = B::resident_barycentric_min_log_size() {
            Some(self.build_weights_hash_map_below(
                &sampled_points,
                lifting_log_size,
                Some(min_log_size),
            ))
        } else {
            Some(self.build_weights_hash_map(&sampled_points, lifting_log_size))
        };
        basis_span.exit();

        let samples: TreeVec<Vec<Vec<PointSample>>> = if self.store_polynomials_coefficients {
            // All same-size columns sampled at one (folded) point share an FFT-basis
            // column; grouping them lets the backend stream that basis once for the
            // whole group instead of once per column.
            self.batched_coefficient_samples(
                &sampled_points,
                weights_hash_map.as_ref().unwrap(),
                lifting_log_size,
            )
        } else {
            self.batched_evaluation_samples(
                &sampled_points,
                weights_hash_map.as_ref(),
                lifting_log_size,
            )
        };
        // Basis/weight columns are only needed to produce `samples`. Release them
        // before quotient and FRI allocations reach their high-water mark.
        drop(weights_hash_map);
        tracing::info!("OOD sampling total in {:?}", ood_start.elapsed());

        span.exit();
        let sampled_values = samples
            .as_cols_ref()
            .map_cols(|x| x.iter().map(|o| o.value).collect());
        channel.mix_felts(&sampled_values.clone().flatten_cols());

        let columns = self.evaluations();
        print_column_size_histogram::<B, MC>(&columns);
        // Compute oods quotients for boundary constraints on the sampled points.
        let quotient_start = std::time::Instant::now();
        let quotients = compute_fri_quotients(
            &columns,
            &samples,
            channel.draw_secure_felt(),
            lifting_log_size,
            self.twiddles,
            self.config.fri_config.log_blowup_factor,
        );
        tracing::info!("FRI quotient evaluation in {:?}", quotient_start.elapsed());

        // Run FRI commitment phase on the oods quotients.
        let fri_commit_start = std::time::Instant::now();
        let fri_prover =
            FriProver::<B, MC>::commit(channel, self.config.fri_config, &quotients, self.twiddles);
        tracing::info!("FRI commitment in {:?}", fri_commit_start.elapsed());

        // Proof of work.
        let grind_start = std::time::Instant::now();
        let span1 = span!(Level::INFO, "Grind", class = "Queries POW").entered();
        let proof_of_work = B::grind(channel, self.config.pow_bits);
        span1.exit();
        tracing::info!("Query grind in {:?}", grind_start.elapsed());
        channel.mix_u64(proof_of_work);

        // FRI decommitment phase.
        let fri_decommit_start = std::time::Instant::now();
        let FriDecommitResult {
            fri_proof,
            query_positions,
            unsorted_query_locations,
        } = fri_prover.decommit(channel);
        tracing::info!("FRI decommitment in {:?}", fri_decommit_start.elapsed());
        // Build the query position tree.
        let openings_start = std::time::Instant::now();
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
        tracing::info!("Trace openings in {:?}", openings_start.elapsed());

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
    polys: ColumnVec<EvalsOrCoeffs<B>>,
}
impl<B: BackendForChannel<MC>, MC: MerkleChannel> TreeBuilder<'_, '_, B, MC> {
    /// Registers evaluations for commitment. Interpolation happens fused with the
    /// low-degree extension when the tree is committed.
    pub fn extend_evals(
        &mut self,
        columns: Vec<CircleEvaluation<B, BaseField, BitReversedOrder>>,
    ) -> TreeSubspan {
        let col_start = self.polys.len();
        self.polys
            .extend(columns.into_iter().map(EvalsOrCoeffs::Evals));
        let col_end = self.polys.len();
        TreeSubspan {
            tree_index: self.tree_index,
            col_start,
            col_end,
        }
    }

    pub fn extend_polys(
        &mut self,
        columns: impl IntoIterator<Item = CircleCoefficients<B>>,
    ) -> TreeSubspan {
        let col_start = self.polys.len();
        self.polys
            .extend(columns.into_iter().map(EvalsOrCoeffs::Coeffs));
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
}

impl<B: BackendForChannel<MC>, MC: MerkleChannel> CommitmentTreeProver<B, MC> {
    pub fn new(
        columns: ColumnVec<EvalsOrCoeffs<B>>,
        log_blowup_factor: u32,
        twiddles: &TwiddleTree<B>,
        store_polynomials_coefficients: bool,
        lifting_log_size: Option<u32>,
        base_column_pool: &BaseColumnPool<B>,
    ) -> Self {
        // Apple-GPU chained commitment: interpolation, extension, Merkle leaves and
        // tree layers in one submission with one wait (CpuBackend + blake2s only).
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let columns = {
            use std::any::{Any, TypeId};

            use crate::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasherGeneric;
            let is_m31 = TypeId::of::<MC::H>() == TypeId::of::<Blake2sMerkleHasherGeneric<true>>();
            let is_bytes =
                TypeId::of::<MC::H>() == TypeId::of::<Blake2sMerkleHasherGeneric<false>>();
            if is_m31 || is_bytes {
                type CpuCols = Vec<EvalsOrCoeffs<crate::prover::backend::CpuBackend>>;
                match downcast_owned::<_, CpuCols>(columns) {
                    Ok(cpu_columns) => {
                        // A successful owned downcast proves `B = CpuBackend`; the
                        // twiddle tree has the same backend parameter by construction.
                        let cpu_twiddles = (twiddles as &dyn Any)
                            .downcast_ref::<TwiddleTree<crate::prover::backend::CpuBackend>>()
                            .unwrap_or_else(|| {
                                panic!("CPU columns must be paired with CPU twiddles")
                            });
                        let span = span!(Level::INFO, "Extension").entered();
                        let result =
                            crate::prover::backend::metal::commit::commit_polynomials_metal(
                                cpu_columns,
                                log_blowup_factor,
                                cpu_twiddles,
                                store_polynomials_coefficients,
                                lifting_log_size,
                                is_m31,
                            );
                        span.exit();
                        match result {
                            Ok((cpu_polys, Some(mut cpu_layers))) => {
                                let _span = span!(Level::INFO, "Merkle").entered();
                                cpu_layers.reverse();
                                let polynomials: ColumnVec<Poly<B>> = downcast_owned(cpu_polys)
                                    .unwrap_or_else(|_| {
                                        panic!("CPU polynomial downcast must match backend")
                                    });
                                let layers: Vec<Col<B, <MC::H as MerkleHasherLifted>::Hash>> =
                                    downcast_owned(cpu_layers).unwrap_or_else(|_| {
                                        panic!("Blake2s layer downcast must match channel")
                                    });
                                return CommitmentTreeProver {
                                    polynomials,
                                    commitment:
                                        crate::prover::vcs_lifted::prover::MerkleProverLifted {
                                            layers,
                                        },
                                };
                            }
                            Ok((cpu_polys, None)) => {
                                // Transforms ran; only the tree fell back. Build it normally.
                                let polynomials: ColumnVec<Poly<B>> = downcast_owned(cpu_polys)
                                    .unwrap_or_else(|_| {
                                        panic!("CPU polynomial downcast must match backend")
                                    });
                                return Self::commit_polynomials(polynomials, lifting_log_size);
                            }
                            Err(cpu_columns) => downcast_owned(cpu_columns).unwrap_or_else(|_| {
                                panic!("CPU column recovery must match backend")
                            }),
                        }
                    }
                    Err(columns) => columns,
                }
            } else {
                columns
            }
        };

        let span = span!(Level::INFO, "Extension").entered();
        let polynomials = B::interpolate_and_evaluate_polynomials(
            columns,
            log_blowup_factor,
            twiddles,
            store_polynomials_coefficients,
            base_column_pool,
        );
        span.exit();

        Self::commit_polynomials(polynomials, lifting_log_size)
    }

    /// Builds the Merkle commitment over already-extended polynomials.
    fn commit_polynomials(polynomials: ColumnVec<Poly<B>>, lifting_log_size: Option<u32>) -> Self {
        let _span = span!(Level::INFO, "Merkle").entered();
        let max_log_domain_size = polynomials
            .iter()
            .map(|poly| poly.evals.domain.log_size())
            .max()
            .unwrap_or_default();
        let lifting_log_size = lifting_log_size.unwrap_or(max_log_domain_size);
        let tree = MerkleProverLifted::commit(
            polynomials
                .iter()
                .map(|poly: &Poly<B>| &poly.evals.values)
                .collect(),
            lifting_log_size,
            0,
        );

        CommitmentTreeProver {
            polynomials,
            commitment: tree,
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
        self.commitment.decommit(queries, eval_vec)
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

#[cfg(test)]
mod tests {
    use crate::core::circle::{CirclePoint, SECURE_FIELD_CIRCLE_GEN};
    use crate::core::fields::m31::{BaseField, P};
    use crate::core::fields::qm31::SecureField;
    use crate::core::pcs::TreeVec;
    use crate::core::poly::circle::CanonicCoset;
    use crate::prover::air::component_prover::{Poly, WeightsHashMap};
    use crate::prover::backend::CpuBackend;
    use crate::prover::poly::circle::{CircleEvaluation, PolyOps};
    use crate::prover::poly::BitReversedOrder;

    fn test_poly(log_size: u32, seed: u32) -> Poly<CpuBackend> {
        let domain = CanonicCoset::new(log_size).circle_domain();
        let values = (0..1 << log_size)
            .map(|i| match i {
                0 => BaseField::from_u32_unchecked(0),
                1 => BaseField::from_u32_unchecked(P),
                _ => BaseField::from((i as u32).wrapping_mul(17).wrapping_add(seed)),
            })
            .collect();
        Poly::new(None, CircleEvaluation::new(domain, values))
    }

    #[test]
    fn grouped_evaluation_samples_match_legacy_in_canonical_order() {
        const LIFTING_LOG_SIZE: u32 = 6;
        let owned = TreeVec(vec![
            vec![test_poly(5, 1), test_poly(6, 2), test_poly(5, 3)],
            vec![test_poly(6, 4), test_poly(5, 5)],
        ]);
        let polys = owned.as_cols_ref();
        let p0 = SECURE_FIELD_CIRCLE_GEN;
        let p1 = SECURE_FIELD_CIRCLE_GEN.mul(1_234_567);
        let sampled_points = TreeVec(vec![
            vec![vec![p0, p1], vec![p1], vec![p0, p0, p1]],
            vec![vec![p1, p0], vec![p1]],
        ]);

        // Model the resident scheduler's partial cache: log-5 groups are
        // prebuilt, while log-6 groups must use the checked on-demand fallback.
        let weights: WeightsHashMap<CpuBackend> = dashmap::DashMap::new();
        for (cols, point_cols) in polys.iter().zip(sampled_points.iter()) {
            for (poly, points) in cols.iter().zip(point_cols) {
                let log_size = poly.evals.domain.log_size();
                if log_size >= LIFTING_LOG_SIZE {
                    continue;
                }
                for point in points {
                    let folded = point.repeated_double(LIFTING_LOG_SIZE - log_size);
                    weights.entry((log_size, folded)).or_insert_with(|| {
                        CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::
                            barycentric_weights(CanonicCoset::new(log_size), folded)
                    });
                }
            }
        }

        let actual = super::batch_evaluation_samples(
            &polys,
            &sampled_points,
            Some(&weights),
            LIFTING_LOG_SIZE,
        );
        for (tree_index, (cols, point_cols)) in polys.iter().zip(sampled_points.iter()).enumerate()
        {
            for (column_index, (poly, points)) in cols.iter().zip(point_cols).enumerate() {
                let log_size = poly.evals.domain.log_size();
                for (point_index, point) in points.iter().enumerate() {
                    let folded = point.repeated_double(LIFTING_LOG_SIZE - log_size);
                    let reference_weights = <CpuBackend as PolyOps>::barycentric_weights(
                        CanonicCoset::new(log_size),
                        folded,
                    );
                    let expected = <CpuBackend as PolyOps>::barycentric_eval_at_point(
                        &poly.evals,
                        &reference_weights,
                    );
                    let sample = &actual[tree_index][column_index][point_index];
                    assert_eq!(sample.point, *point);
                    assert_eq!(sample.value, expected);
                }
            }
        }

        let original_order = sampled_points
            .0
            .iter()
            .flatten()
            .flatten()
            .copied()
            .collect::<Vec<CirclePoint<SecureField>>>();
        let actual_order = actual
            .0
            .iter()
            .flatten()
            .flatten()
            .map(|sample| sample.point)
            .collect::<Vec<_>>();
        assert_eq!(actual_order, original_order);
    }
}
