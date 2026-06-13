//! # nusy-clinical-fixtures — executable clinical-guideline gold cases (EX-4615)
//!
//! The **VOY-V18-6** fixture harness. `Usage-Scenarios-CDS.md` is a workflow *taxonomy*;
//! this crate turns clinical guideline logic into **runnable** fixtures, each carrying the
//! six elements §12.4 (Finding E) requires:
//!
//! 1. **patient facts** — FHIR-style ground triples (`Patient`/`Condition`/`Observation`…);
//! 2. **computable rule(s)** — the guideline as forward-chaining Horn rules;
//! 3. **expected recommendation(s)** — facts that *must* be derived (with a proof);
//! 4. **contraindications** — recommendations the guideline must **not** fire;
//! 5. **proof path** — the derivation tree behind each recommendation (via [`proof_path`]);
//! 6. **negative controls** — claims that must be **unprovable**, so the gate abstains.
//!
//! It runs on the landed VOY-1 stack ([`nusy_forward_chain`] over [`nusy_unify`]) — a
//! recommendation is *only* asserted if it is derivable, and it carries the rule chain that
//! proves it. This is the gold-case set the VOY-6 eval battery (EX-4617) scores against, and
//! the end-to-end being demo (EX-4616) must satisfy it.
//!
//! ```
//! use nusy_clinical_fixtures::{run_all, gold_cases};
//! // Every gold-case fixture passes: expected recs derived w/ proof, contraindications
//! // suppressed, negative controls unprovable.
//! assert!(run_all().iter().all(|r| r.passed()), "a clinical gold case failed");
//! assert!(!gold_cases().is_empty());
//! ```

use nusy_forward_chain::{Saturation, forward_chain};
use nusy_unify::Triple;

pub use nusy_forward_chain::IdRule;
pub use nusy_unify::{Rule, TriplePattern};

mod cases;
pub use cases::gold_cases;

/// One executable clinical gold case (§12.4 Finding E — the six elements).
#[derive(Debug, Clone)]
pub struct ClinicalFixture {
    /// Human-readable scenario name.
    pub name: String,
    /// (1) Patient facts as FHIR-style ground triples.
    pub patient_facts: Vec<Triple>,
    /// (2) The guideline in computable form (forward-chaining rules).
    pub rules: Vec<IdRule>,
    /// (3) Recommendations that MUST be derived (each with a proof).
    pub expected_recommendations: Vec<Triple>,
    /// (4) Recommendations the guideline must NOT fire (clinical contraindications).
    pub contraindicated: Vec<Triple>,
    /// (6) Claims that must be unprovable — the gate must flag/abstain, never assert.
    pub negative_controls: Vec<Triple>,
}

/// A node in a recommendation's proof tree: a derived conclusion with the rule that
/// produced it and the sub-proofs of its premises, or a seed fact (a leaf).
///
/// (5) The "expected proof path". Built by recursively resolving each premise's own
/// derivation — a seed fact bottoms out as [`ProofNode::Fact`]. The canonical proof-tree
/// type is EX-4592; this is the harness's lightweight walk for asserting structure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProofNode {
    /// A seed fact — an asserted premise, not derived.
    Fact(Triple),
    /// A derived conclusion: the rule that fired and the proofs of its premises.
    Derived {
        /// The derived fact.
        conclusion: Triple,
        /// The rule that produced it.
        rule_id: String,
        /// Sub-proofs of the premises that satisfied the rule body.
        premises: Vec<ProofNode>,
    },
}

impl ProofNode {
    /// Every rule id used anywhere in this proof tree (deepest-first, with repeats).
    pub fn rule_ids(&self) -> Vec<&str> {
        let mut out = Vec::new();
        self.collect_rule_ids(&mut out);
        out
    }

    fn collect_rule_ids<'a>(&'a self, out: &mut Vec<&'a str>) {
        if let ProofNode::Derived {
            rule_id, premises, ..
        } = self
        {
            for p in premises {
                p.collect_rule_ids(out);
            }
            out.push(rule_id);
        }
    }

    /// Proof depth: a seed fact is 0; a derivation is 1 + its deepest premise.
    pub fn depth(&self) -> usize {
        match self {
            ProofNode::Fact(_) => 0,
            ProofNode::Derived { premises, .. } => {
                1 + premises.iter().map(ProofNode::depth).max().unwrap_or(0)
            }
        }
    }
}

/// Reconstruct the proof tree for `fact` from a saturation's derivation provenance.
/// A fact with no derivation (a seed fact) is a [`ProofNode::Fact`] leaf.
pub fn proof_path(sat: &Saturation, fact: &Triple) -> ProofNode {
    match sat.derivation_of(fact) {
        Some(d) => ProofNode::Derived {
            conclusion: fact.clone(),
            rule_id: d.rule_id.clone(),
            premises: d.premises.iter().map(|p| proof_path(sat, p)).collect(),
        },
        None => ProofNode::Fact(fact.clone()),
    }
}

/// The result of running one fixture: any element that did not behave as specified.
#[derive(Debug, Clone)]
pub struct FixtureReport {
    /// The fixture's name.
    pub name: String,
    /// One message per violated expectation; empty means the fixture passed.
    pub failures: Vec<String>,
}

impl FixtureReport {
    /// Did every expected/contraindicated/negative-control check hold?
    pub fn passed(&self) -> bool {
        self.failures.is_empty()
    }
}

/// Run one fixture: forward-chain the guideline over the patient facts, then check that
/// every expected recommendation is **derived with a proof**, every contraindication is
/// **suppressed**, and every negative control is **unprovable**.
pub fn run_fixture(fx: &ClinicalFixture) -> FixtureReport {
    let sat = forward_chain(&fx.rules, fx.patient_facts.clone());
    let mut failures = Vec::new();

    for rec in &fx.expected_recommendations {
        if !sat.contains(rec) {
            failures.push(format!("expected recommendation not derived: {rec:?}"));
        } else if sat.derivation_of(rec).is_none() {
            failures.push(format!(
                "recommendation derived but carries no proof: {rec:?}"
            ));
        }
    }
    for c in &fx.contraindicated {
        if sat.contains(c) {
            failures.push(format!(
                "contraindicated recommendation WAS derived — guideline fired when it must not: {c:?}"
            ));
        }
    }
    for n in &fx.negative_controls {
        if sat.contains(n) {
            failures.push(format!(
                "negative control was derived — must be unprovable / abstained: {n:?}"
            ));
        }
    }

    FixtureReport {
        name: fx.name.clone(),
        failures,
    }
}

/// Run every gold-case fixture.
pub fn run_all() -> Vec<FixtureReport> {
    gold_cases().iter().map(run_fixture).collect()
}
