# Zero-Knowledge for stwo

Status: **design + Phase 1 (hiding commitments) implemented behind the `zk`
feature flag.** The soundness-critical mechanisms (FRI mask, witness/composition
randomization) are specified here and gated on human approval before
implementation. Turning on the `zk` feature and enabling it in a `PcsConfig`
today buys **hiding vector commitments only** — it is a necessary but **not
sufficient** condition for zero-knowledge. Do not represent a proof produced by
the current code as zero-knowledge until Phases 3–5 land.

## Why stwo is not zero-knowledge today

stwo is a succinct, transparent STARK. It is not witness-hiding: the proof and
its commitments are deterministic functions of the witness, so they leak.
Concretely, every field of `CommitmentSchemeProof`
(`core/pcs/quotients.rs`) except `config` and `proof_of_work` is
witness-dependent:

1. **Unsalted Merkle roots** (`commitments`). The leaf hash is the raw little
   endian bytes of the column values (`core/vcs_lifted/blake2_merkle.rs`,
   `update_leaf`). The root is a deterministic commitment to the whole trace, so
   a guessed witness can be confirmed against a published root (brute-force /
   confirmation attack), fatal for low-entropy witnesses even if nothing is ever
   opened.
2. **Queried values** (`queried_values`). `n_queries` LDE evaluations per column
   per proof — known linear functionals of the trace. Re-proving the same
   witness leaks fresh equations each time.
3. **OODS samples** (`sampled_values`). Each QM31 sample is 4 base-field linear
   functionals of a column.
4. **FRI layers** (`fri_proof`): first/inner-layer openings, `fri_witness` coset
   siblings, and `last_layer_poly` sent in the clear — all deterministic linear
   images of the witness.
5. **LogUp `claimed_sum`** — statement-level leakage, not fixable in the PCS.
6. **Hygiene**: `mempool.rs` reuses buffers without zeroization; the prover has
   no secret randomness source at all (every "random" value is Fiat-Shamir
   derived, hence public).

Two structural properties already hold and must be preserved as invariants:

- **Domain disjointness** `D ∩ H = ∅`. The trace domain (canonic coset of order
  `2^{n+1}`) and the commitment domain (canonic coset of order `2^{n+b+1}`)
  consist of points of different orders, hence disjoint. The Circle STARKs paper
  Appendix-C "superset domain" optimization is *provably incompatible with ZK*;
  stwo does not use it and must not in the zk path.
- **The first FRI layer is committed and opened, never elided.** Eliding it (and
  recomputing it from openings) would re-expose masked values. stwo never had
  this optimization; the zk path must not add it.

## Normative construction

The construction is Haböck & Al-Kindi, *"A note on adding zero-knowledge to
STARKs"*, [eprint 2024/1037](https://eprint.iacr.org/2024/1037). The Circle
STARKs paper ([eprint 2024/278](https://eprint.iacr.org/2024/278), §5.4) states
the randomization "carries over to the circle without changes" and defers all
detail to that note. The reference implementation precedent is Plonky3's FRI ZK
stack (`HidingFriPcs` + `MerkleTreeHidingMmcs`, PRs #517/#536/#643), *not*
PR #1767 (HVZK-WHIR, a different multilinear-PCS construction).

Target: **statistical honest-verifier zero-knowledge** at the IOP level plus
**hiding vector commitments**; the BCS16 transform then yields a zk-SNARG in the
ROM and Fiat-Shamir disposes of malicious-verifier concerns. Statistical (not
perfect) is deliberate — perfect ZK for LogUp requires muting transition
constraints at challenge-collision points (note, Appendix A), which is not worth
the cost over QM31.

Four mechanisms are required. **Partial ZK is broken ZK.**

Notation: `H` trace domain, `|H| = 2^n`; `D` commitment domain; `e = [QM31:M31]
= 4`; `n_F` OODS sample points per column; `n_D = FriConfig.n_queries`
(deduplicated). For stark-v's secure config `n_D = 193`.

### Mechanism 1 — witness randomization (Phase 4, deferred)

Commit `ŵᵢ = wᵢ + v_H·rᵢ`, `v_H` = vanishing polynomial of `H`, `rᵢ` a random
**base-field** polynomial with `h ≥ 2·(e·n_F + n_D)` degrees of freedom
(note eq. 10). The factor `e` is Lemma 1 (each extension-field sample costs 4
base-field query-equivalents via the Galois closure); the factor 2 covers the
composition polynomial's `g`-translate. Applies to main + interaction (LogUp)
columns; preprocessed columns are public and untouched.

**Circle-specific crux — power-of-two rigidity.** `v_H·r` pushes the column out
of the dimension-`2^n` circle FFT space. Two integration routes:

- **Route A (conservative, Plonky3 #643):** commit randomized columns at
  `log_size n+1`, same blowup. Transparent to all degree accounting; the lifted
  VCS already handles mixed column sizes. Cost ~2× on trace commit.
- **Route B (note Protocol 2, the `1 + O(1/log|H|)` claim):** keep columns on
  their current domain (blowup ≥ 2× already gives room), let the degree overflow
  surface only in the batched DEEP quotient, which the prover splits into two
  standard-bound polynomials before FRI (the same `split_at_mid` pattern stwo
  already uses for the composition). Per-column bounds enforced at `2^n + h` via
  affine correlated agreement. Cost ~one extra FFT-equivalent per column.

Plan: ship Route A first (matches a merged, reviewed precedent), graduate to
Route B for performance. **Route B's mapping onto stwo's lifted PCS is a
derivation, not paper-stated for the circle — validate against the paper before
coding (repo rule: no proceeding on intuition).**

### Mechanism 2 — composition randomization (Phase 5, deferred)

stwo's `split_at_mid` into 8 coordinate columns (`prover/mod.rs`) is the note's
canonical decomposition with `d = 2`. Randomize PLONK-style (note §4.1):
`q̂₀ = q₀ + B·t`, `q̂₁ = q₁ − t` with random `t` (telescoping cancels; `B` is the
split basis). Avoids the splitting-field simulator of the eq. (3) route.

### Mechanism 3 — FRI mask polynomial R (Phase 3, deferred)

Commit a uniformly random codeword `R` of the full batch degree **before** the
quotient-batching challenge `α` is drawn (currently `prover/pcs/mod.rs`, the
`channel.draw_secure_felt()` at the quotient step), and run FRI on
`B(X) = R(X) + Σ αᵏ·Qₖ(X)`. Non-negotiable: `v_H` is even, so the randomizer
space halves in dimension at every FRI fold while witness information does not —
without `R`, inner layers and `last_layer_poly` leak regardless of `h` (note
Lemma 2, the Decoupling Lemma). Soundness of the affine shift rests on
correlated agreement for **affine** spaces (BCIKS, eprint 2020/654; circle
analogue in Circle STARKs Appendix A). `R` is QM31-valued, committed as 4
base-field coordinate columns in a new randomizer tree. Realizing `R` as `D`
independent base-field polynomials gives **statistical, not perfect** ZK (the
same documented caveat as Plonky3 #643). The FRI layer trees are also salted in
this phase (they live in `fri.rs` alongside `R`).

### Mechanism 4 — hiding commitments + salted Fiat-Shamir (Phase 1, IMPLEMENTED)

See "Phase 1 implementation" below.

### What does not change

Field arithmetic, constraint definitions, AIR semantics, PoW, security
parameters (`security_bits()` is unaffected; zk *adds* validation of the
randomizer budget, it does not reduce soundness), and the verifier's `no_std`
property.

## Phase 1 implementation — hiding (salted) commitments

### Scope

Phase 1 salts the **PCS commitment trees** (main trace, interaction/LogUp trace,
composition polynomial) — every committed tree except the preprocessed tree
(index 0, public). FRI layer trees are salted in Phase 3 (they live in `fri.rs`
together with the FRI mask `R`). Salting alone hides the roots and unopened
leaves; it does **not** hide opened values — that needs Phases 3–4.

### Design

The leaf hash today is `leaf_i = H(col_0[i] ‖ … ‖ col_{m-1}[i])`. We commit to
the **salted leaf** `salted_i = hash_children(leaf_i, salt_i)` where `salt_i` is
a secret per-leaf hash, then build the tree over `salted_i`. The Merkle tree
height and all inner-layer logic are unchanged (the salt is folded at the leaf
*before* the first pairing, 1:1, not as an extra layer).

This design lives **entirely** in `MerkleProverLifted::commit` (prover) and
`MerkleVerifierLifted::verify` (verifier), using only the existing
`MerkleHasherLifted` surface (`hash_children`, `update_leaf`, `finalize`) and the
`Column` surface (`at`, `len`, `FromIterator`). It is therefore **backend- and
hasher-agnostic**: Blake2s (CPU + SIMD), Poseidon252, and stark-v's custom
`Poseidon2M31` hasher all get salting with no per-backend `build_leaves`
changes.

- **Salt derivation (prover):** per tree, draw a secret seed
  `[BaseField; ZK_SALT_SEED_LEN]` from a CSPRNG. `salt_i = H{ update_leaf(seed ‖
  M31(i)); finalize() }`. The index `i < 2^30` fits one M31 limb. The seed is
  secret, so `salt_i` is unpredictable; the verifier never sees the seed or the
  derivation.
- **Decommitment:** `MerkleDecommitmentLifted` carries `salts: Vec<H::Hash>`,
  one per deduplicated opened position, in the order the verifier rebuilds
  leaves. The verifier recomputes `leaf_i` from `queried_values`, folds in the
  provided `salt_i` via `hash_children`, then walks the existing `hash_witness`
  path.

**Binding** is preserved: opening position `i` to different values `v' ≠ v`
requires `hash_children(leaf(v'), salt') = hash_children(leaf(v), salt_i)` for
the committed `salted_i` — a collision, ruled out under the same
collision-resistance assumption the rest of the tree relies on. **Hiding**:
`salted_i` reveals nothing about `leaf_i` given a uniform secret `salt_i`
(standard salted commitment).

**Fail-closed:** when `PcsConfig.zk` is set, the PCS verifier requires every
non-preprocessed tree's decommitment to carry the expected number of salts; a
non-salted (or wrong-count) decommitment is rejected. Conversely the committed
root is over salted leaves, so a prover cannot produce an accepting zk proof
without the correct salts.

### Feature flag and byte-identical guarantee

- A `zk` Cargo feature on the `stwo` crate compiles in the scaffolding. **With
  the feature off** (ethproofs), `PcsConfig` is structurally identical to today,
  no salt code is compiled, and proofs are byte-identical to current stwo.
- **With the feature on but the runtime flag `PcsConfig.zk == false`** (inner
  recursion proofs), no salt layer is added, the root and the entire Fiat-Shamir
  transcript are byte-identical to the non-zk path, and the recursion circuit's
  transcript replay is unaffected. Decommitment salts are empty. Only the outer
  proof sets `zk == true`.
- The zk flag is mixed into the channel **only when true**, as a domain
  separator, so the `zk == false` transcript is unchanged.
- `zk` implies `prover` + std (salt generation needs a CSPRNG). Verifying zk
  proofs in a pure `no_std` build is not supported yet.

### Randomness

The prover uses a CSPRNG (`ZkRng` over ChaCha via `rand::rngs::StdRng`), seeded
from OS entropy at proof time, with a deterministic seed override for tests.
Salt randomness MUST be independent of the Fiat-Shamir transcript (transcript-
derived "randomness" is public and provides no hiding). This is prover-only; the
verifier consumes salts and needs no RNG.

### Files touched (Phase 1)

- `crates/stwo/Cargo.toml` — `zk` feature.
- `crates/stwo/src/core/pcs/mod.rs` — `PcsConfig.zk` (cfg-gated), `mix_into`.
- `crates/stwo/src/prover/vcs_lifted/zk.rs` — `ZkRng`, `SaltSeed`, salt derivation
  and fold.
- `crates/stwo/src/prover/vcs_lifted/prover.rs` — salt-fold in `commit`, salts in
  `decommit`.
- `crates/stwo/src/core/vcs_lifted/verifier.rs` — `salts` field, salt-fold in
  `verify`.
- `crates/stwo/src/prover/pcs/mod.rs` — per-tree salt seeds (non-preprocessed),
  store seed for decommit.
- `crates/stwo/src/core/pcs/verifier.rs` — fail-closed salt-presence check.

## Open soundness items for review (Phases 3–5)

1. Route B circle degree-accounting is a derivation; validate against the paper.
2. The affine correlated-agreement statement for circle codes behind the FRI
   mask `R` needs an explicit citation/statement (Circle STARKs Appendix A
   proves the linear version).
3. LogUp `claimed_sum` leakage is an application contract, not PCS-fixable.
4. ZK failures are *silent* (no test fails when hiding breaks). Assurance leans
   on fail-closed types, distinguisher smoke tests, and external audit.
