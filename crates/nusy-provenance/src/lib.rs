//! # nusy-provenance — surface a derivation's proof as a citable chain
//!
//! **VOY-V18-5 / EX-4612.** When the provable gate answers a claim, it must return *why* —
//! the derivation that proves it. The proof tree (EX-4592, [`nusy_forward_chain::ProofTree`])
//! is the recursive justification; this crate **surfaces** it as a flat, dependency-ordered
//! [`Provenance`] record a gate response can attach: the grounding **axioms** (the seed facts
//! the answer rests on — its citations) and the ordered **derivation steps** (each rule
//! application), ending at the claim.
//!
//! ## The gate contract
//!
//! [`surface`] returns `Some(Provenance)` iff the claim is **provable** from the saturated
//! facts (it has a proof tree); `None` otherwise — the signal for a gate to **abstain** rather
//! than assert. So a surfaced answer always carries a self-contained, checkable derivation.
//!
//! ## Why flatten?
//!
//! `ProofTree::render` already pretty-prints the *tree*. A gate needs the **linearised** proof:
//! steps in dependency order (every premise justified before it is used — see
//! [`Provenance::is_grounded`]), de-duplicated (a lemma proved once), with the axiom set
//! surfaced as the citations. That is what attaches to an answer and what an auditor checks.
//!
//! ## Example
//!
//! ```
//! use nusy_forward_chain::{forward_chain, IdRule};
//! use nusy_unify::{Rule, Triple, TriplePattern};
//! use nusy_provenance::surface;
//!
//! // grandparent(?x,?z) :- parent(?x,?y), parent(?y,?z)
//! let gp = IdRule::new("grandparent", Rule::new(
//!     vec![TriplePattern::parse("?x", "parent", "?y"), TriplePattern::parse("?y", "parent", "?z")],
//!     vec![TriplePattern::parse("?x", "grandparent", "?z")],
//! ));
//! let sat = forward_chain(&[gp], vec![
//!     Triple::new("a", "parent", "b"), Triple::new("b", "parent", "c"),
//! ]);
//!
//! let claim = Triple::new("a", "grandparent", "c");
//! let prov = surface(&sat, &claim).expect("provable → provenance");
//! assert_eq!(prov.steps.len(), 1);                 // one rule application
//! assert_eq!(prov.steps[0].rule_id, "grandparent");
//! assert_eq!(prov.axioms.len(), 2);                // grounded in parent(a,b), parent(b,c)
//! assert!(prov.is_grounded());                     // every premise justified before use
//!
//! // An unprovable claim surfaces nothing → the gate abstains.
//! assert!(surface(&sat, &Triple::new("a", "grandparent", "z")).is_none());
//! ```

use std::collections::HashSet;

use nusy_forward_chain::{ArrowSaturation, ProofTree, Saturation};
use nusy_unify::Triple;

/// One rule application in a linearised proof: a conclusion justified by a rule firing over
/// premises that appear *earlier* in the [`Provenance`] (as axioms or prior step conclusions).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivationStep {
    /// The fact this step concludes.
    pub conclusion: Triple,
    /// The id of the rule that fired.
    pub rule_id: String,
    /// The immediate premises the rule consumed (each justified earlier in the chain).
    pub premises: Vec<Triple>,
}

/// A surfaced, citable proof of one claim: the grounding axioms and the ordered derivation
/// steps that lead to it. Built from a [`ProofTree`] by [`surface`] / [`Provenance::from_proof`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Provenance {
    /// The proven claim (the root conclusion).
    pub claim: Triple,
    /// The seed facts the proof rests on — the answer's citations (de-duplicated).
    pub axioms: Vec<Triple>,
    /// Rule applications in **dependency order**: every step's premises are justified by an
    /// earlier axiom or step. The last step concludes [`claim`](Provenance::claim).
    pub steps: Vec<DerivationStep>,
}

impl Provenance {
    /// Surface the provenance of a [`ProofTree`] (flatten to axioms + ordered steps).
    pub fn from_proof(tree: &ProofTree) -> Self {
        let mut axioms = Vec::new();
        let mut steps = Vec::new();
        let mut seen_ax: HashSet<Triple> = HashSet::new();
        let mut seen_step: HashSet<Triple> = HashSet::new();
        flatten(tree, &mut axioms, &mut steps, &mut seen_ax, &mut seen_step);
        Self { claim: tree.conclusion().clone(), axioms, steps }
    }

    /// The rule ids used, in derivation order (a "lemma trail" for the answer).
    pub fn rule_chain(&self) -> Vec<&str> {
        self.steps.iter().map(|s| s.rule_id.as_str()).collect()
    }

    /// Number of derivation steps (rule applications) backing the claim.
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Is the proof **closed**: is every premise of every step justified earlier in the chain
    /// (an axiom, or the conclusion of a preceding step)? A surfaced proof must satisfy this;
    /// it is the gate's guarantee that the attached derivation is self-contained and checkable.
    pub fn is_grounded(&self) -> bool {
        let mut known: HashSet<&Triple> = self.axioms.iter().collect();
        for step in &self.steps {
            if !step.premises.iter().all(|p| known.contains(p)) {
                return false;
            }
            known.insert(&step.conclusion);
        }
        // And the claim itself must be concluded by the chain (or be a bare axiom).
        known.contains(&self.claim) || self.axioms.contains(&self.claim)
    }

    /// Render a human-readable citation block: the given axioms, each derivation step, then the
    /// conclusion. Suitable for attaching to a gated answer or an audit log.
    pub fn render(&self) -> String {
        let mut out = String::new();
        for ax in &self.axioms {
            out.push_str(&format!("given:  {} {} {}\n", ax.subject, ax.predicate, ax.object));
        }
        for step in &self.steps {
            let prems: Vec<String> = step
                .premises
                .iter()
                .map(|p| format!("{} {} {}", p.subject, p.predicate, p.object))
                .collect();
            out.push_str(&format!(
                "rule {}:  {}  ⊢  {} {} {}\n",
                step.rule_id,
                prems.join(" , "),
                step.conclusion.subject,
                step.conclusion.predicate,
                step.conclusion.object,
            ));
        }
        out.push_str(&format!(
            "∴  {} {} {}",
            self.claim.subject, self.claim.predicate, self.claim.object
        ));
        out
    }
}

/// Surface the provenance of `claim` from a [`Saturation`].
///
/// `Some(Provenance)` iff `claim` is provable (has a proof tree); `None` otherwise — the
/// gate's signal to **abstain** instead of asserting. A bare axiom (a claim that *is* a seed
/// fact) surfaces with itself as the sole citation and no steps.
pub fn surface(sat: &Saturation, claim: &Triple) -> Option<Provenance> {
    sat.proof_of(claim).map(|tree| Provenance::from_proof(&tree))
}

/// Surface provenance directly from the engine's [`ArrowSaturation`] — the zero-copy
/// path (EX-4671). Identical result to [`surface`], but the proof is read off the
/// engine's Arrow batches with no `Vec<Triple>` saturation materialized first (the
/// shared-memory-space path, VY-4667).
///
/// **Retained boundary copy (documented):** the returned [`Provenance`] *owns* its
/// `Triple`s (its public API — `axioms`/`steps` are consumed by `cql`/`cpg` and the
/// fixtures, which must not borrow the engine's buffers). So the proof tree's terms are
/// cloned once at this boundary, exactly as [`surface`] does. The win is eliminating the
/// *upstream* full-saturation Vec materialization, not this API-level ownership copy.
pub fn surface_arrow(sat: &ArrowSaturation, claim: &Triple) -> Option<Provenance> {
    sat.proof_of(claim).map(|tree| Provenance::from_proof(&tree))
}

/// Post-order flatten: emit each subtree's premises before the step that consumes them, so the
/// resulting `steps` are in dependency order. De-duplicates axioms and steps by their fact.
fn flatten(
    tree: &ProofTree,
    axioms: &mut Vec<Triple>,
    steps: &mut Vec<DerivationStep>,
    seen_ax: &mut HashSet<Triple>,
    seen_step: &mut HashSet<Triple>,
) {
    match tree {
        ProofTree::Axiom(t) => {
            if seen_ax.insert(t.clone()) {
                axioms.push(t.clone());
            }
        }
        ProofTree::Derived { conclusion, rule_id, premises } => {
            for p in premises {
                flatten(p, axioms, steps, seen_ax, seen_step);
            }
            if seen_step.insert(conclusion.clone()) {
                steps.push(DerivationStep {
                    conclusion: conclusion.clone(),
                    rule_id: rule_id.clone(),
                    premises: premises.iter().map(|p| p.conclusion().clone()).collect(),
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nusy_forward_chain::{forward_chain, IdRule};
    use nusy_unify::{Rule, TriplePattern};

    fn t(s: &str, p: &str, o: &str) -> Triple {
        Triple::new(s, p, o)
    }
    fn rule(id: &str, body: Vec<TriplePattern>, head: TriplePattern) -> IdRule {
        IdRule::new(id, Rule::new(body, vec![head]))
    }

    /// ancestor closure over a→b→c→d, so proofs have multiple chained steps.
    fn ancestor_sat() -> Saturation {
        let base = rule(
            "anc-base",
            vec![TriplePattern::parse("?x", "parent", "?z")],
            TriplePattern::parse("?x", "ancestor", "?z"),
        );
        let rec = rule(
            "anc-rec",
            vec![
                TriplePattern::parse("?x", "parent", "?y"),
                TriplePattern::parse("?y", "ancestor", "?z"),
            ],
            TriplePattern::parse("?x", "ancestor", "?z"),
        );
        forward_chain(&[base, rec], vec![t("a", "parent", "b"), t("b", "parent", "c"), t("c", "parent", "d")])
    }

    #[test]
    fn surfaces_provenance_for_a_provable_claim() {
        let sat = ancestor_sat();
        let prov = surface(&sat, &t("a", "ancestor", "c")).expect("provable");
        assert_eq!(prov.claim, t("a", "ancestor", "c"));
        // grounded in seed parent facts only.
        assert!(prov.axioms.iter().all(|ax| ax.predicate == "parent"));
        // last step concludes the claim.
        assert_eq!(prov.steps.last().unwrap().conclusion, t("a", "ancestor", "c"));
        assert!(prov.step_count() >= 2, "a→ancestor→c needs ≥2 steps");
    }

    #[test]
    fn unprovable_claim_surfaces_none_so_gate_abstains() {
        let sat = ancestor_sat();
        assert!(surface(&sat, &t("d", "ancestor", "a")).is_none());
        assert!(surface(&sat, &t("a", "ancestor", "z")).is_none());
    }

    #[test]
    fn provenance_is_grounded_premises_justified_before_use() {
        let sat = ancestor_sat();
        let prov = surface(&sat, &t("a", "ancestor", "d")).expect("provable");
        // The closure guarantee: every step's premises appear earlier (axiom or prior step).
        assert!(prov.is_grounded());
        // Steps are in dependency order — the first step's premises are all axioms.
        let axset: HashSet<&Triple> = prov.axioms.iter().collect();
        assert!(prov.steps[0].premises.iter().all(|p| axset.contains(p)));
    }

    #[test]
    fn rule_chain_and_render_reflect_the_derivation() {
        let sat = ancestor_sat();
        let prov = surface(&sat, &t("a", "ancestor", "c")).unwrap();
        let chain = prov.rule_chain();
        assert!(chain.contains(&"anc-base"), "base case used");
        assert!(chain.contains(&"anc-rec"), "recursive case used");
        let rendered = prov.render();
        assert!(rendered.contains("given:"), "cites grounding axioms");
        assert!(rendered.contains("∴  a ancestor c"), "concludes the claim");
    }

    #[test]
    fn bare_axiom_surfaces_with_itself_as_citation() {
        // A seed fact is provable trivially: itself as the only axiom, no steps.
        let sat = ancestor_sat();
        let prov = surface(&sat, &t("a", "parent", "b")).expect("seed fact is provable");
        assert_eq!(prov.axioms, vec![t("a", "parent", "b")]);
        assert!(prov.steps.is_empty());
        assert!(prov.is_grounded());
    }

    #[test]
    fn shared_lemma_is_not_duplicated() {
        // A diamond: two rules whose conclusions both rest on the same lemma; the lemma
        // appears once across axioms/steps (dedup), and the proof stays grounded.
        let sat = ancestor_sat();
        let prov = surface(&sat, &t("a", "ancestor", "d")).unwrap();
        // No conclusion appears twice among the steps.
        let concls: Vec<&Triple> = prov.steps.iter().map(|s| &s.conclusion).collect();
        let unique: HashSet<&Triple> = concls.iter().copied().collect();
        assert_eq!(unique.len(), concls.len(), "each derived fact proved once");
    }
}
