//! # nusy-forward-chain — forward-chaining inference with derivation provenance
//!
//! The **VOY-V18-1** derivation engine (EX-4588): iterate a set of Y2 rules over a
//! seed of Y1 ground triples until no new facts appear (a **fixpoint**), recording
//! for every derived fact *why* it holds — the rule that fired and the premises it
//! consumed. That provenance is the substrate the proof-tree layer (EX-4592) and the
//! provable gate (VOY-V18-5) build on: a derived claim carries its derivation, so the
//! gate can return a proof rather than assert.
//!
//! ## Built on `nusy-unify`, not duplicating it
//!
//! Rule-LHS matching (the relational join over shared variables) is
//! [`nusy_unify::match_conjunction`]; this crate adds only the **fixpoint loop** and
//! **provenance recording** on top. `nusy-unify` stays a pure matching primitive.
//!
//! ## Termination
//!
//! Forward chaining terminates because the rules are **range-restricted** Horn rules
//! over flat terms: they introduce no new constants, so the set of derivable ground
//! triples (the Herbrand base over the seed + rule constants) is finite. Each round
//! adds at least one new fact or stops; the fact set is monotonic and bounded.
//!
//! ## Example
//!
//! ```
//! use nusy_forward_chain::{forward_chain, IdRule};
//! use nusy_unify::{Rule, Triple, TriplePattern};
//!
//! // grandparent(?x,?z) :- parent(?x,?y), parent(?y,?z)
//! let gp = IdRule::new(
//!     "grandparent",
//!     Rule::new(
//!         vec![
//!             TriplePattern::parse("?x", "parent", "?y"),
//!             TriplePattern::parse("?y", "parent", "?z"),
//!         ],
//!         vec![TriplePattern::parse("?x", "grandparent", "?z")],
//!     ),
//! );
//! let seed = vec![Triple::new("a", "parent", "b"), Triple::new("b", "parent", "c")];
//! let sat = forward_chain(&[gp], seed);
//!
//! let gpac = Triple::new("a", "grandparent", "c");
//! assert!(sat.contains(&gpac));
//! let proof = sat.derivation_of(&gpac).unwrap();
//! assert_eq!(proof.rule_id, "grandparent");
//! assert_eq!(proof.premises.len(), 2); // parent(a,b), parent(b,c)
//! ```

use std::collections::{HashMap, HashSet};

use nusy_unify::{Rule, Triple, match_conjunction};

pub mod arrow_match;
pub mod batch;
pub mod proof;
pub use proof::ProofTree;

use arrow_match::{match_conjunction_arrow, match_conjunction_arrow_delta};
use batch::{DerivationBatch, TripleBatch};

/// A rule tagged with a stable identifier, so each derived fact can name the rule
/// that produced it. The identifier flows into [`Derivation::rule_id`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IdRule {
    /// Stable rule identifier (e.g. a Y2 rule IRI or guideline-rule name).
    pub id: String,
    /// The Horn rule (LHS body ⊢ RHS head).
    pub rule: Rule,
}

impl IdRule {
    /// Construct an identified rule.
    pub fn new(id: impl Into<String>, rule: Rule) -> Self {
        Self {
            id: id.into(),
            rule,
        }
    }
}

/// Why a derived fact holds: the rule that fired and the ground premises it consumed.
///
/// This is one derivation step. EX-4592 chains these into a full proof tree by
/// recursively resolving each premise that is itself derived (rather than a seed fact).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Derivation {
    /// The newly derived fact.
    pub conclusion: Triple,
    /// The [`IdRule::id`] of the rule that produced it.
    pub rule_id: String,
    /// The ground LHS facts that satisfied the rule body for this conclusion.
    pub premises: Vec<Triple>,
}

/// The result of saturating a rule set over a seed: the closed fact set plus the
/// provenance for every *derived* fact (seed facts have no derivation).
#[derive(Debug, Clone, Default)]
pub struct Saturation {
    /// All facts after reaching the fixpoint, in discovery order (seed first).
    pub facts: Vec<Triple>,
    /// One [`Derivation`] per derived fact — the *first* derivation found for it.
    pub derivations: Vec<Derivation>,
}

impl Saturation {
    /// Is `t` in the saturated fact set?
    pub fn contains(&self, t: &Triple) -> bool {
        self.facts.contains(t)
    }

    /// The derivation of `t`, if `t` was *derived* (seed facts return `None`).
    pub fn derivation_of(&self, t: &Triple) -> Option<&Derivation> {
        self.derivations.iter().find(|d| &d.conclusion == t)
    }

    /// Number of derived (non-seed) facts.
    pub fn derived_count(&self) -> usize {
        self.derivations.len()
    }
}

/// Forward-chain `rules` over `seed` to a fixpoint, recording derivation provenance.
///
/// Each round, every rule is matched against the current fact set; for each solution
/// of the rule body, each RHS head is instantiated. A head that grounds to a *new*
/// fact is added, and its [`Derivation`] (rule id + the ground body premises) is
/// recorded. The loop stops when a full round adds nothing.
///
/// Non-range-restricted RHS atoms (a head variable unbound by the body) cannot ground
/// and are silently skipped — the engine never invents constants. The *first*
/// derivation found for a fact is kept; alternative derivations of the same fact are
/// not recorded (EX-4592 can enumerate them if proof multiplicity is ever needed).
///
/// **Substrate (EX-4670):** since VY-4667 phase 3 the loop runs on the **Arrow**
/// substrate — [`forward_chain_arrow`] over a [`TripleBatch`] with Arrow-native
/// matching ([`arrow_match`]) — and materializes the final [`Saturation`] once at the
/// fixpoint. The public contract is unchanged: same signature, same fact membership,
/// same per-round discovery order (seed first, then each round's delta). Within a
/// round, enumeration order follows the Arrow hash-join rather than the old nested
/// scan; no API consumer observes order within a round as part of the contract.
pub fn forward_chain(rules: &[IdRule], seed: Vec<Triple>) -> Saturation {
    forward_chain_arrow(rules, seed).to_saturation()
}

/// The pre-EX-4670 Vec-based engine, retained as the **reference oracle** for
/// differential tests and Arrow-vs-Vec benchmarks. Semantics match [`forward_chain`]
/// (same fact set and per-round deltas; within-round order may differ). Not used by
/// the engine itself; scheduled for removal once the incremental engine (EX-4593)
/// lands with its own oracle suite.
pub fn forward_chain_vec(rules: &[IdRule], seed: Vec<Triple>) -> Saturation {
    let mut seen: HashSet<Triple> = seed.iter().cloned().collect();
    let mut sat = Saturation {
        facts: seed,
        derivations: Vec::new(),
    };

    loop {
        let mut round: Vec<Derivation> = Vec::new();
        let mut round_seen: HashSet<Triple> = HashSet::new();

        for r in rules {
            for sol in match_conjunction(&r.rule.lhs, &sat.facts) {
                // The ground body facts that satisfied this solution.
                let premises: Vec<Triple> =
                    r.rule.lhs.iter().filter_map(|p| sol.ground(p)).collect();

                for head in &r.rule.rhs {
                    let Some(conclusion) = sol.ground(head) else {
                        continue; // unbound head var (non-range-restricted) → skip
                    };
                    if seen.contains(&conclusion) || round_seen.contains(&conclusion) {
                        continue; // already known, or already derived this round
                    }
                    round_seen.insert(conclusion.clone());
                    round.push(Derivation {
                        conclusion,
                        rule_id: r.id.clone(),
                        premises: premises.clone(),
                    });
                }
            }
        }

        if round.is_empty() {
            break; // fixpoint reached
        }
        for d in round {
            seen.insert(d.conclusion.clone());
            sat.facts.push(d.conclusion.clone());
            sat.derivations.push(d);
        }
    }

    sat
}

/// An Arrow-native saturation: the closed fact set as a [`TripleBatch`] (seed round +
/// one delta round per fixpoint iteration — the seam the semi-naive evaluator EX-4593
/// restricts matching to) and the derivations as a [`DerivationBatch`] (conclusion
/// terms + rule id + premise row refs). Downstream layers (provenance, gate,
/// cog-computable — EX-4671) consume these batches directly; [`to_saturation`]
/// (ArrowSaturation::to_saturation) materializes the Vec form for the stable API.
#[derive(Debug, Clone)]
pub struct ArrowSaturation {
    facts: TripleBatch,
    derivations: DerivationBatch,
    /// First-occurrence global fact row per triple (membership + premise resolution).
    fact_row: HashMap<Triple, u64>,
    /// Derivation-batch row per derived conclusion (first derivation kept).
    derivation_row: HashMap<Triple, usize>,
}

impl ArrowSaturation {
    /// The fact store: seed round + one [`RecordBatch`](arrow::record_batch::RecordBatch)
    /// delta per fixpoint round.
    pub fn facts(&self) -> &TripleBatch {
        &self.facts
    }

    /// The derivations as an Arrow batch (phase-1 schema).
    pub fn derivation_batch(&self) -> &DerivationBatch {
        &self.derivations
    }

    /// Is `t` in the saturated fact set?
    pub fn contains(&self, t: &Triple) -> bool {
        self.fact_row.contains_key(t)
    }

    /// Number of derived (non-seed) facts.
    pub fn derived_count(&self) -> usize {
        self.derivations.len()
    }

    /// The derivation of `t`, decoded from the derivation batch (`None` for seed facts
    /// and unknown triples).
    pub fn derivation_of(&self, t: &Triple) -> Option<Derivation> {
        let row = *self.derivation_row.get(t)?;
        Some(
            self.derivations
                .decode_row(row, &self.facts)
                .expect("derivation batch was encoded against these facts"),
        )
    }

    /// Build the full proof tree for `target`, recursively expanding derived premises
    /// down to seed axioms (the Arrow-side equivalent of [`Saturation::proof_of`]
    /// (proof::ProofTree)). `None` if `target` is not in the saturated fact set.
    pub fn proof_of(&self, target: &Triple) -> Option<ProofTree> {
        if let Some(d) = self.derivation_of(target) {
            let premises = d
                .premises
                .iter()
                .map(|p| {
                    self.proof_of(p)
                        .expect("premises of a recorded derivation are facts")
                })
                .collect();
            Some(ProofTree::Derived {
                conclusion: target.clone(),
                rule_id: d.rule_id,
                premises,
            })
        } else if self.contains(target) {
            Some(ProofTree::Axiom(target.clone()))
        } else {
            None
        }
        // Termination: forward_chain_arrow derives a fact only from premises present in
        // an earlier round, so the derivation graph is acyclic and the recursion finite.
    }

    /// Materialize the stable Vec form: facts in global row order (seed first, then
    /// each round's delta in discovery order), derivations in derivation-row order.
    pub fn to_saturation(&self) -> Saturation {
        #[cfg(feature = "materialization-counter")]
        crate::matcount::bump();
        Saturation {
            facts: self.facts.to_triples(),
            derivations: self
                .derivations
                .to_derivations(&self.facts)
                .expect("derivation batch was encoded against these facts"),
        }
    }
}

/// Test-only instrumentation (EX-4671, behind the `materialization-counter` feature):
/// counts how many times the Arrow→Vec boundary [`ArrowSaturation::to_saturation`] is
/// crossed. Zero-copy-adoption tests reset it, run the Arrow hot path, and assert the
/// count stayed `0` — a precise proof that no `Vec<Triple>` saturation was materialized.
/// The feature is never enabled in production builds, so this is a no-op there.
#[cfg(feature = "materialization-counter")]
pub mod matcount {
    use std::sync::atomic::{AtomicUsize, Ordering};

    static MATERIALIZATIONS: AtomicUsize = AtomicUsize::new(0);

    pub(crate) fn bump() {
        MATERIALIZATIONS.fetch_add(1, Ordering::Relaxed);
    }

    /// Materializations recorded since the last [`reset`].
    pub fn count() -> usize {
        MATERIALIZATIONS.load(Ordering::Relaxed)
    }

    /// Reset the counter at the start of a measured section.
    pub fn reset() {
        MATERIALIZATIONS.store(0, Ordering::Relaxed);
    }
}

/// Forward-chain `rules` over `seed` to a fixpoint **on the Arrow substrate**,
/// returning the saturation as Arrow batches (EX-4670, VY-4667 phase 3).
///
/// Each round matches every rule body via [`match_conjunction_arrow`] against the
/// accumulated [`TripleBatch`], grounds new conclusions, and appends them as that
/// round's **delta batch** — explicitly addressable via
/// [`TripleBatch::rounds`](batch::TripleBatch::rounds), which is the seam EX-4593's
/// semi-naive strategy plugs into (restrict matching to the last delta) without
/// restructuring this loop. Dedup and premise row resolution use a triple→row hash
/// index maintained alongside the batches (an index over the Arrow store, not a
/// second store). Derivations are encoded into the [`DerivationBatch`] at the fixpoint.
pub fn forward_chain_arrow(rules: &[IdRule], seed: Vec<Triple>) -> ArrowSaturation {
    let mut fact_row: HashMap<Triple, u64> = HashMap::new();
    for (row, t) in seed.iter().enumerate() {
        fact_row.entry(t.clone()).or_insert(row as u64);
    }
    let mut facts = TripleBatch::from_triples(&seed);
    let mut all_derivations: Vec<Derivation> = Vec::new();

    loop {
        let mut delta: Vec<Triple> = Vec::new();
        let mut round_derivations: Vec<Derivation> = Vec::new();
        let mut round_seen: HashSet<Triple> = HashSet::new();

        for r in rules {
            for sol in match_conjunction_arrow(&r.rule.lhs, &facts) {
                let premises: Vec<Triple> =
                    r.rule.lhs.iter().filter_map(|p| sol.ground(p)).collect();

                for head in &r.rule.rhs {
                    let Some(conclusion) = sol.ground(head) else {
                        continue; // unbound head var (non-range-restricted) → skip
                    };
                    if fact_row.contains_key(&conclusion) || round_seen.contains(&conclusion) {
                        continue; // already known, or already derived this round
                    }
                    round_seen.insert(conclusion.clone());
                    delta.push(conclusion.clone());
                    round_derivations.push(Derivation {
                        conclusion,
                        rule_id: r.id.clone(),
                        premises: premises.clone(),
                    });
                }
            }
        }

        if delta.is_empty() {
            break; // fixpoint reached
        }
        let base = facts.len() as u64;
        for (i, t) in delta.iter().enumerate() {
            fact_row.insert(t.clone(), base + i as u64);
        }
        facts.append_triples(&delta); // one RecordBatch per round — the EX-4593 delta seam
        all_derivations.extend(round_derivations);
    }

    let derivations = DerivationBatch::from_derivations(&all_derivations, &facts)
        .expect("every premise of a recorded derivation is a fact row");
    let derivation_row = (0..derivations.len())
        .map(|i| (derivations.conclusion_at(i), i))
        .collect();

    ArrowSaturation {
        facts,
        derivations,
        fact_row,
        derivation_row,
    }
}

/// Incrementally extend a `prior` saturation with `new_seed` facts — **semi-naive**
/// re-derivation (EX-4593, VY-4667 phase 5).
///
/// Instead of re-running the full fixpoint over `prior ++ new_seed`, this seeds the new
/// facts as the **first delta** and, each round, fires rules only over solutions that use
/// at least one delta fact ([`match_conjunction_arrow_delta`]). Facts whose every premise
/// pre-dated the delta are never re-derived — so the cost scales with what actually
/// *changed*, not with the size of the closed graph. The common case: add a fact (or a
/// freshly-perceived batch) to an already-saturated being and maintain its conclusions.
///
/// **Equivalence (the correctness contract):** the result — fact set and provability of
/// every claim — is identical to `forward_chain_arrow(rules, prior_seed ++ new_seed)`.
/// Prior derivations are carried forward; new ones are appended. Verified by differential
/// tests against the full-refire engine (`tests/incremental_differential.rs`).
pub fn forward_chain_arrow_incremental(
    prior: &ArrowSaturation,
    rules: &[IdRule],
    new_seed: Vec<Triple>,
) -> ArrowSaturation {
    // Working fact set: every prior fact (already closed) keeps its identity; then the
    // genuinely-new seed facts become the first delta.
    let mut fact_row: HashMap<Triple, u64> = HashMap::new();
    let mut all: Vec<Triple> = Vec::new();
    for t in prior.facts.to_triples() {
        fact_row.entry(t.clone()).or_insert(all.len() as u64);
        all.push(t);
    }
    let mut delta: Vec<Triple> = Vec::new();
    for t in new_seed {
        if !fact_row.contains_key(&t) {
            fact_row.insert(t.clone(), all.len() as u64);
            all.push(t.clone());
            delta.push(t);
        }
    }

    // Carry the prior proofs forward; the fixpoint appends only the delta's consequences.
    let mut all_derivations: Vec<Derivation> = prior
        .derivations
        .to_derivations(&prior.facts)
        .expect("prior derivation batch was encoded against prior facts");
    let mut facts = TripleBatch::from_triples(&all);

    while !delta.is_empty() {
        let delta_batch = TripleBatch::from_triples(&delta);
        let mut next_delta: Vec<Triple> = Vec::new();
        let mut round_seen: HashSet<Triple> = HashSet::new();

        for r in rules {
            for sol in match_conjunction_arrow_delta(&r.rule.lhs, &facts, &delta_batch) {
                let premises: Vec<Triple> =
                    r.rule.lhs.iter().filter_map(|p| sol.ground(p)).collect();

                for head in &r.rule.rhs {
                    let Some(conclusion) = sol.ground(head) else {
                        continue; // unbound head var (non-range-restricted) → skip
                    };
                    if fact_row.contains_key(&conclusion) || round_seen.contains(&conclusion) {
                        continue; // already known (prior or this round)
                    }
                    round_seen.insert(conclusion.clone());
                    next_delta.push(conclusion.clone());
                    all_derivations.push(Derivation {
                        conclusion,
                        rule_id: r.id.clone(),
                        premises: premises.clone(),
                    });
                }
            }
        }

        if next_delta.is_empty() {
            break; // incremental fixpoint reached
        }
        let base = facts.len() as u64;
        for (i, t) in next_delta.iter().enumerate() {
            fact_row.insert(t.clone(), base + i as u64);
        }
        facts.append_triples(&next_delta);
        delta = next_delta;
    }

    let derivations = DerivationBatch::from_derivations(&all_derivations, &facts)
        .expect("every premise of a recorded derivation is a fact row");
    let derivation_row = (0..derivations.len())
        .map(|i| (derivations.conclusion_at(i), i))
        .collect();

    ArrowSaturation {
        facts,
        derivations,
        fact_row,
        derivation_row,
    }
}
