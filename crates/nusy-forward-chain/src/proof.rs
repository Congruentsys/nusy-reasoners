//! # Proof trees over a [`Saturation`] (EX-4592, VOY-V18-1)
//!
//! [`forward_chain`](crate::forward_chain) records, for each derived fact, a single
//! [`Derivation`](crate::Derivation) step — the rule that fired and the *immediate*
//! ground premises it consumed. This module chains those steps into a full **proof
//! tree**: a derived premise is itself recursively expanded until every leaf is a
//! **seed axiom**. The result is the complete "why does this hold?" justification a
//! provable claim carries through the gate (VOY-V18-5 / EX-4612 provenance surfacing).
//!
//! ```
//! use nusy_forward_chain::{forward_chain, IdRule, ProofTree};
//! use nusy_unify::{Rule, Triple, TriplePattern};
//!
//! // grandparent(?x,?z) :- parent(?x,?y), parent(?y,?z)
//! let gp = IdRule::new("grandparent", Rule::new(
//!     vec![TriplePattern::parse("?x", "parent", "?y"),
//!          TriplePattern::parse("?y", "parent", "?z")],
//!     vec![TriplePattern::parse("?x", "grandparent", "?z")]));
//! // greatgrandparent(?x,?w) :- grandparent(?x,?z), parent(?z,?w)
//! let ggp = IdRule::new("greatgrandparent", Rule::new(
//!     vec![TriplePattern::parse("?x", "grandparent", "?z"),
//!          TriplePattern::parse("?z", "parent", "?w")],
//!     vec![TriplePattern::parse("?x", "greatgrandparent", "?w")]));
//!
//! let seed = vec![
//!     Triple::new("a", "parent", "b"),
//!     Triple::new("b", "parent", "c"),
//!     Triple::new("c", "parent", "d"),
//! ];
//! let sat = forward_chain(&[gp, ggp], seed);
//!
//! // greatgrandparent(a,d) is two derivation levels deep.
//! let proof = sat.proof_of(&Triple::new("a", "greatgrandparent", "d")).unwrap();
//! assert_eq!(proof.depth(), 2);                 // ggp ← grandparent ← parents
//! assert_eq!(proof.axioms().len(), 3);          // grounded in 3 seed facts
//! assert!(!proof.is_axiom());
//! ```

use std::collections::HashSet;

use nusy_unify::Triple;

use crate::Saturation;

/// A node in a proof tree: either a seed **axiom** (true by assertion) or a **derived**
/// fact justified by the rule that produced it and a proof of each of its premises.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProofTree {
    /// A seed fact — part of the input ground set, true by assertion, no derivation.
    Axiom(Triple),
    /// A fact derived by a rule firing over premises that are themselves proven.
    Derived {
        /// The proven fact.
        conclusion: Triple,
        /// The [`IdRule`](crate::IdRule) id of the rule that fired.
        rule_id: String,
        /// A proof of each ground premise the rule consumed (recursively expanded).
        premises: Vec<ProofTree>,
    },
}

impl ProofTree {
    /// The fact this (sub)proof concludes.
    pub fn conclusion(&self) -> &Triple {
        match self {
            ProofTree::Axiom(t) => t,
            ProofTree::Derived { conclusion, .. } => conclusion,
        }
    }

    /// Is this a seed axiom (a leaf)?
    pub fn is_axiom(&self) -> bool {
        matches!(self, ProofTree::Axiom(_))
    }

    /// Proof depth: an axiom is `0`; a derived node is `1 +` its deepest premise.
    pub fn depth(&self) -> usize {
        match self {
            ProofTree::Axiom(_) => 0,
            ProofTree::Derived { premises, .. } => {
                1 + premises.iter().map(ProofTree::depth).max().unwrap_or(0)
            }
        }
    }

    /// Total number of nodes in the tree (axioms + derived steps).
    pub fn node_count(&self) -> usize {
        match self {
            ProofTree::Axiom(_) => 1,
            ProofTree::Derived { premises, .. } => {
                1 + premises.iter().map(ProofTree::node_count).sum::<usize>()
            }
        }
    }

    /// The seed axioms this proof is grounded in (its leaves), in left-to-right order.
    /// Duplicates are preserved — a fact used by two premises appears twice.
    pub fn axioms(&self) -> Vec<&Triple> {
        let mut out = Vec::new();
        self.collect_axioms(&mut out);
        out
    }

    fn collect_axioms<'a>(&'a self, out: &mut Vec<&'a Triple>) {
        match self {
            ProofTree::Axiom(t) => out.push(t),
            ProofTree::Derived { premises, .. } => {
                for p in premises {
                    p.collect_axioms(out);
                }
            }
        }
    }

    /// Every rule id used anywhere in the proof (with duplicates), root-first.
    pub fn rule_ids(&self) -> Vec<&str> {
        let mut out = Vec::new();
        self.collect_rule_ids(&mut out);
        out
    }

    fn collect_rule_ids<'a>(&'a self, out: &mut Vec<&'a str>) {
        if let ProofTree::Derived {
            rule_id, premises, ..
        } = self
        {
            out.push(rule_id);
            for p in premises {
                p.collect_rule_ids(out);
            }
        }
    }

    /// Render the proof as an indented, human-readable tree — for surfacing the
    /// derivation behind a gated answer (EX-4612).
    pub fn render(&self) -> String {
        let mut s = String::new();
        self.render_into(&mut s, 0);
        s
    }

    fn render_into(&self, s: &mut String, indent: usize) {
        let pad = "  ".repeat(indent);
        match self {
            ProofTree::Axiom(t) => {
                s.push_str(&format!(
                    "{pad}[axiom] {} {} {}\n",
                    t.subject, t.predicate, t.object
                ));
            }
            ProofTree::Derived {
                conclusion,
                rule_id,
                premises,
            } => {
                s.push_str(&format!(
                    "{pad}{} {} {}  (by {})\n",
                    conclusion.subject, conclusion.predicate, conclusion.object, rule_id
                ));
                for p in premises {
                    p.render_into(s, indent + 1);
                }
            }
        }
    }
}

impl Saturation {
    /// Build the full proof tree for `target`, recursively expanding every derived
    /// premise down to seed axioms.
    ///
    /// Returns `None` if `target` is not in the saturated fact set (not provable from
    /// the seed and rules). Uses each fact's *first* recorded derivation (the one
    /// [`forward_chain`](crate::forward_chain) kept).
    ///
    /// Termination: [`forward_chain`](crate::forward_chain) derives a fact only from
    /// premises already present in an earlier round, so the derivation graph is acyclic
    /// and the recursion is finite. A visited-set guard defends against a hand-built,
    /// malformed [`Saturation`] whose derivations form a cycle (it yields `None` rather
    /// than recursing forever).
    pub fn proof_of(&self, target: &Triple) -> Option<ProofTree> {
        let mut on_path = HashSet::new();
        self.build_proof(target, &mut on_path)
    }

    fn build_proof(&self, target: &Triple, on_path: &mut HashSet<Triple>) -> Option<ProofTree> {
        if let Some(d) = self.derivation_of(target) {
            // Cycle guard: target already being expanded on this path → malformed,
            // no finite proof. Cannot occur for a Saturation from `forward_chain`.
            if !on_path.insert(target.clone()) {
                return None;
            }
            let mut premises = Vec::with_capacity(d.premises.len());
            for premise in &d.premises {
                premises.push(self.build_proof(premise, on_path)?);
            }
            on_path.remove(target);
            Some(ProofTree::Derived {
                conclusion: target.clone(),
                rule_id: d.rule_id.clone(),
                premises,
            })
        } else if self.contains(target) {
            // In the fact set but no derivation → a seed axiom.
            Some(ProofTree::Axiom(target.clone()))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{IdRule, forward_chain};
    use nusy_unify::{Rule, Triple, TriplePattern};

    use super::*;

    fn grandparent_rule() -> IdRule {
        IdRule::new(
            "grandparent",
            Rule::new(
                vec![
                    TriplePattern::parse("?x", "parent", "?y"),
                    TriplePattern::parse("?y", "parent", "?z"),
                ],
                vec![TriplePattern::parse("?x", "grandparent", "?z")],
            ),
        )
    }

    fn greatgrandparent_rule() -> IdRule {
        IdRule::new(
            "greatgrandparent",
            Rule::new(
                vec![
                    TriplePattern::parse("?x", "grandparent", "?z"),
                    TriplePattern::parse("?z", "parent", "?w"),
                ],
                vec![TriplePattern::parse("?x", "greatgrandparent", "?w")],
            ),
        )
    }

    #[test]
    fn seed_fact_proves_as_axiom() {
        let seed = vec![Triple::new("a", "parent", "b")];
        let sat = forward_chain(&[grandparent_rule()], seed);
        let proof = sat.proof_of(&Triple::new("a", "parent", "b")).unwrap();
        assert!(proof.is_axiom());
        assert_eq!(proof.depth(), 0);
        assert_eq!(proof.node_count(), 1);
        assert_eq!(proof.axioms(), vec![&Triple::new("a", "parent", "b")]);
        assert!(proof.rule_ids().is_empty());
    }

    #[test]
    fn one_level_derivation_proof() {
        let seed = vec![
            Triple::new("a", "parent", "b"),
            Triple::new("b", "parent", "c"),
        ];
        let sat = forward_chain(&[grandparent_rule()], seed);
        let proof = sat.proof_of(&Triple::new("a", "grandparent", "c")).unwrap();

        assert!(!proof.is_axiom());
        assert_eq!(proof.depth(), 1);
        assert_eq!(proof.rule_ids(), vec!["grandparent"]);
        // Grounded in exactly the two parent axioms, both leaves.
        let axioms = proof.axioms();
        assert_eq!(axioms.len(), 2);
        assert!(axioms.contains(&&Triple::new("a", "parent", "b")));
        assert!(axioms.contains(&&Triple::new("b", "parent", "c")));
        match &proof {
            ProofTree::Derived { premises, .. } => {
                assert!(premises.iter().all(ProofTree::is_axiom))
            }
            _ => panic!("expected a derived root"),
        }
    }

    #[test]
    fn two_level_derivation_recurses_into_derived_premise() {
        let seed = vec![
            Triple::new("a", "parent", "b"),
            Triple::new("b", "parent", "c"),
            Triple::new("c", "parent", "d"),
        ];
        let sat = forward_chain(&[grandparent_rule(), greatgrandparent_rule()], seed);

        let target = Triple::new("a", "greatgrandparent", "d");
        let proof = sat.proof_of(&target).unwrap();

        // ggp(a,d) ← [ grandparent(a,c) (DERIVED), parent(c,d) (axiom) ]
        assert_eq!(proof.depth(), 2);
        // ggp + grandparent + parent(a,b) + parent(b,c) + parent(c,d) = 5 nodes.
        assert_eq!(proof.node_count(), 5);
        // Leaves are the three seed parents (grandparent's two + the extra parent(c,d)).
        assert_eq!(proof.axioms().len(), 3);
        // Both rules appear; ggp is the root rule.
        let rules = proof.rule_ids();
        assert_eq!(rules[0], "greatgrandparent");
        assert!(rules.contains(&"grandparent"));

        match &proof {
            ProofTree::Derived { premises, .. } => {
                assert_eq!(premises.len(), 2);
                // One premise is itself a derived grandparent proof, the other a seed axiom.
                assert!(premises.iter().any(|p| !p.is_axiom()));
                assert!(premises.iter().any(ProofTree::is_axiom));
            }
            _ => panic!("expected derived root"),
        }
    }

    #[test]
    fn unprovable_fact_has_no_proof() {
        let seed = vec![Triple::new("a", "parent", "b")];
        let sat = forward_chain(&[grandparent_rule()], seed);
        assert!(
            sat.proof_of(&Triple::new("x", "grandparent", "y"))
                .is_none()
        );
    }

    #[test]
    fn render_shows_rule_and_axioms() {
        let seed = vec![
            Triple::new("a", "parent", "b"),
            Triple::new("b", "parent", "c"),
        ];
        let sat = forward_chain(&[grandparent_rule()], seed);
        let proof = sat.proof_of(&Triple::new("a", "grandparent", "c")).unwrap();
        let rendered = proof.render();
        assert!(rendered.contains("by grandparent"));
        assert!(rendered.contains("[axiom] a parent b"));
        assert!(rendered.contains("[axiom] b parent c"));
    }
}
