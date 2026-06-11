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

use std::collections::HashSet;

use nusy_unify::{Rule, Triple, match_conjunction};

pub mod batch;
pub mod proof;
pub use proof::ProofTree;

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
pub fn forward_chain(rules: &[IdRule], seed: Vec<Triple>) -> Saturation {
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
