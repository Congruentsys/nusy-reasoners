//! # Conform the validated engines to the Reasoner contract (VY-D E2, EX-4747)
//!
//! Three adapters, one per validated engine — each answering through
//! [`nusy_reasoner::Reasoner`] with the provability its substrate honestly supports:
//!
//! | Adapter | Engine | Provability |
//! |---|---|---|
//! | [`DeductiveReasoner`] | `nusy-forward-chain` fixpoint | **Proven** (complete derivations) or Abstained |
//! | [`CausalReasoner`] | do-calculus identification | **Proven** (identifiable effect) or Abstained |
//! | [`CounterfactualReasoner`] | Pearl L3 estimate | **Heuristic** (calibrated confidence) or Abstained |
//!
//! The spread is the point: the contract's guarantee invariant means each adapter
//! *cannot overclaim* — the counterfactual estimator emits [`ProofTrace::Evidence`] and is
//! structurally unable to mint `Proven`, while the deductive adapter's `Proven` is computed
//! from a complete [`DerivationTrace`] adapted off the engine's own proof tree.
//!
//! **Adapters only:** no engine public API is touched, and this crate has no
//! `nusy-arrow-core` dependency — the whole set is Wave-1 FOSS-movable (VY-C). The
//! store-backed path (proofs citing store `triple_id`s) lives product-side in
//! `nusy-store-seam`, which wraps [`DeductiveReasoner`] over a projection.

use nusy_forward_chain::{ArrowSaturation, IdRule, ProofTree, forward_chain_arrow};
use nusy_reasoner::{
    Answer, CompetenceEnvelope, DerivationTrace, Guarantee, ProofTrace, Query, QueryShape,
    Reasoner, Substrate,
};
use nusy_reasoning_causal::counterfactual::counterfactual;
use nusy_reasoning_causal::identifiability::verify_identifiability;
use nusy_reasoning_causal::{CausalDag, IdentificationCriterion};
use nusy_unify::Triple;

/// Adapt the engine's [`ProofTree`] (EX-4592) into the contract's substrate-neutral
/// [`DerivationTrace`]. Lossless on shape: axioms stay axioms, derived nodes keep their
/// rule id and premise structure — so completeness (and therefore provability) is decided
/// by the contract from the same tree the engine proved.
pub fn to_derivation_trace(tree: &ProofTree) -> DerivationTrace {
    match tree {
        ProofTree::Axiom(t) => DerivationTrace::Axiom(t.clone()),
        ProofTree::Derived {
            conclusion,
            rule_id,
            premises,
        } => DerivationTrace::Derived {
            conclusion: conclusion.clone(),
            rule_id: rule_id.clone(),
            premises: premises.iter().map(to_derivation_trace).collect(),
        },
    }
}

// ── Deductive ────────────────────────────────────────────────────────────────────

/// The forward-chain engine behind the contract: saturate rules over facts (plus any
/// query context), answer goal membership with the engine's own proof adapted into the
/// contract's trace. A complete proof computes to `Proven`; an unknown goal abstains.
pub struct DeductiveReasoner {
    rules: Vec<IdRule>,
    facts: Vec<Triple>,
    envelope: CompetenceEnvelope,
}

impl DeductiveReasoner {
    /// A deductive reasoner over `rules` + `facts`. The envelope defaults to the
    /// wildcard shape (ground-pattern entailment over any predicate); narrow it with
    /// [`with_envelope`](Self::with_envelope) when registering with a router.
    pub fn new(rules: Vec<IdRule>, facts: Vec<Triple>) -> Self {
        Self {
            rules,
            facts,
            envelope: CompetenceEnvelope {
                shapes: vec![QueryShape {
                    name: "ground-pattern entailment".into(),
                    predicates: vec![],
                }],
            },
        }
    }

    /// Replace the competence envelope (e.g. restrict to the predicates the rule set
    /// actually concludes).
    pub fn with_envelope(mut self, envelope: CompetenceEnvelope) -> Self {
        self.envelope = envelope;
        self
    }

    /// Saturate facts + query context. Exposed so differential tests can compare the
    /// adapter against the engine on the *same* saturation.
    pub fn saturate(&self, query: &Query) -> ArrowSaturation {
        let mut seed = self.facts.clone();
        seed.extend(query.context.iter().cloned());
        forward_chain_arrow(&self.rules, seed)
    }
}

impl Reasoner for DeductiveReasoner {
    fn answer(&self, query: &Query) -> Answer {
        if !self.envelope.covers(query) {
            return Answer::abstained();
        }
        let sat = self.saturate(query);
        let Some(tree) = sat.proof_of(&query.goal) else {
            return Answer::abstained(); // not entailed — loud abstention, never a guess
        };
        let trace = to_derivation_trace(&tree);
        // Provenance: the seed axioms this proof is grounded in (the engine's leaves).
        // Product-side store citation (triple_ids) is layered on by nusy-store-seam.
        let provenance = tree
            .axioms()
            .iter()
            .map(|a| format!("axiom:({} {} {})", a.subject, a.predicate, a.object))
            .collect();
        Answer {
            value: Some(query.goal.clone()),
            proof: ProofTrace::Derivation(trace),
            provenance,
        }
    }

    fn competence_envelope(&self) -> &CompetenceEnvelope {
        &self.envelope
    }

    fn substrate(&self) -> Substrate {
        Substrate::Symbolic
    }

    fn guarantee(&self) -> Guarantee {
        Guarantee {
            sound: true,
            complete: true, // within its envelope: fixpoint closure finds every entailment
            probabilistic: false,
        }
    }
}

// ── Causal (interventional, L2) ──────────────────────────────────────────────────

/// The goal predicate the causal adapter answers: `(treatment, causally_affects, outcome)`.
pub const CAUSALLY_AFFECTS: &str = "causally_affects";

/// Do-calculus identification behind the contract: a goal
/// `(treatment, causally_affects, outcome)` is answered `Proven` iff the effect is
/// **formally identifiable** on the DAG (backdoor / frontdoor / direct), with the
/// derivation rendered from the criterion and the causal-path edges; a non-identifiable
/// effect **abstains** (the engine's zero-false-positive refusal discipline, H-4118).
pub struct CausalReasoner {
    dag: CausalDag,
    envelope: CompetenceEnvelope,
}

impl CausalReasoner {
    /// A causal reasoner over `dag`.
    pub fn new(dag: CausalDag) -> Self {
        Self {
            dag,
            envelope: CompetenceEnvelope {
                shapes: vec![QueryShape {
                    name: "do-calculus identification".into(),
                    predicates: vec![CAUSALLY_AFFECTS.into()],
                }],
            },
        }
    }

    /// One directed causal path treatment→outcome (BFS), as the edge triples the
    /// derivation cites. Empty when no path exists.
    fn causal_path_edges(&self, treatment: &str, outcome: &str) -> Vec<Triple> {
        use std::collections::{HashMap, VecDeque};
        let mut parent: HashMap<String, (String, String)> = HashMap::new(); // node → (pred, via predicate)
        let mut queue = VecDeque::from([treatment.to_string()]);
        while let Some(node) = queue.pop_front() {
            if node == outcome {
                break;
            }
            for (child, predicate) in self.dag.children_of(&node) {
                if child != treatment && !parent.contains_key(child) {
                    parent.insert(child.clone(), (node.clone(), predicate.clone()));
                    queue.push_back(child.clone());
                }
            }
        }
        let mut edges = Vec::new();
        let mut cur = outcome.to_string();
        while let Some((prev, pred)) = parent.get(&cur) {
            edges.push(Triple::new(prev.clone(), pred.clone(), cur.clone()));
            cur = prev.clone();
        }
        edges.reverse();
        edges
    }
}

impl Reasoner for CausalReasoner {
    fn answer(&self, query: &Query) -> Answer {
        if !self.envelope.covers(query) {
            return Answer::abstained();
        }
        let treatment = query.goal.subject.as_str();
        let outcome = query.goal.object.as_str();
        let Ok(verification) = verify_identifiability(&self.dag, treatment, outcome) else {
            return Answer::abstained(); // unknown nodes / engine error → loud abstention
        };
        if !verification.identifiable {
            return Answer::abstained(); // not identifiable → refuse, never estimate (H-4118)
        }
        let criterion = match verification.criterion {
            Some(IdentificationCriterion::Backdoor) => "do-calculus:backdoor",
            Some(IdentificationCriterion::Frontdoor) => "do-calculus:frontdoor",
            Some(IdentificationCriterion::DirectEffect) => "do-calculus:direct-effect",
            None => return Answer::abstained(), // identifiable must carry its criterion
        };
        let path = self.causal_path_edges(treatment, outcome);
        if path.is_empty() {
            return Answer::abstained(); // identifiable but pathless would be vacuous
        }
        let trace = DerivationTrace::Derived {
            conclusion: query.goal.clone(),
            rule_id: criterion.to_string(),
            premises: path.iter().cloned().map(DerivationTrace::Axiom).collect(),
        };
        let mut provenance: Vec<String> = path
            .iter()
            .map(|e| format!("edge:({} {} {})", e.subject, e.predicate, e.object))
            .collect();
        provenance.push(format!("identification:{}", verification.explanation));
        Answer {
            value: Some(query.goal.clone()),
            proof: ProofTrace::Derivation(trace),
            provenance,
        }
    }

    fn competence_envelope(&self) -> &CompetenceEnvelope {
        &self.envelope
    }

    fn substrate(&self) -> Substrate {
        Substrate::Symbolic
    }

    fn guarantee(&self) -> Guarantee {
        Guarantee {
            sound: true,     // identification is formal; refusal on non-identifiable
            complete: false, // refuses rather than estimates — deliberately incomplete
            probabilistic: false,
        }
    }
}

// ── Counterfactual (L3) ──────────────────────────────────────────────────────────

/// The goal predicate the counterfactual adapter answers:
/// `(treatment, counterfactually_affects, outcome)` — "would the outcome differ had the
/// treatment not occurred?"
pub const COUNTERFACTUALLY_AFFECTS: &str = "counterfactually_affects";

/// Pearl L3 behind the contract — and the honest mapping: counterfactual results are
/// **estimates with calibrated confidence**, so the adapter emits
/// [`ProofTrace::Evidence`] and is *structurally unable* to claim `Proven` (the
/// guarantee invariant doing its job). The engine's certifiability gate (refusal on
/// very-low confidence) maps to abstention.
pub struct CounterfactualReasoner {
    dag: CausalDag,
    envelope: CompetenceEnvelope,
}

impl CounterfactualReasoner {
    /// A counterfactual reasoner over `dag`.
    pub fn new(dag: CausalDag) -> Self {
        Self {
            dag,
            envelope: CompetenceEnvelope {
                shapes: vec![QueryShape {
                    name: "counterfactual estimate".into(),
                    predicates: vec![COUNTERFACTUALLY_AFFECTS.into()],
                }],
            },
        }
    }
}

impl Reasoner for CounterfactualReasoner {
    fn answer(&self, query: &Query) -> Answer {
        if !self.envelope.covers(query) {
            return Answer::abstained();
        }
        let treatment = query.goal.subject.as_str();
        let outcome = query.goal.object.as_str();
        // The engine's certifiability gate returns Err on uncertifiable queries → abstain.
        let Ok(result) = counterfactual(&self.dag, treatment, outcome, None) else {
            return Answer::abstained();
        };
        let why = vec![
            format!("causal_chain:{}", result.causal_chain.join(" -> ")),
            format!(
                "counterfactual_outcome:{}",
                result
                    .counterfactual_outcome
                    .as_deref()
                    .unwrap_or("unknown")
            ),
        ];
        Answer {
            value: Some(query.goal.clone()),
            proof: ProofTrace::Evidence {
                confidence: result.confidence,
                why,
            },
            provenance: result.causal_chain.clone(),
        }
    }

    fn competence_envelope(&self) -> &CompetenceEnvelope {
        &self.envelope
    }

    fn substrate(&self) -> Substrate {
        Substrate::Symbolic
    }

    fn guarantee(&self) -> Guarantee {
        Guarantee {
            sound: false, // estimates — never asserted as certain
            complete: false,
            probabilistic: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nusy_reasoner::Provability;
    use nusy_unify::{Rule, TriplePattern};

    fn t(s: &str, p: &str, o: &str) -> Triple {
        Triple::new(s, p, o)
    }

    fn kinship_rules() -> Vec<IdRule> {
        vec![
            IdRule::new(
                "ancestor-base",
                Rule::new(
                    vec![TriplePattern::parse("?x", "parent", "?y")],
                    vec![TriplePattern::parse("?x", "ancestor", "?y")],
                ),
            ),
            IdRule::new(
                "ancestor-rec",
                Rule::new(
                    vec![
                        TriplePattern::parse("?x", "parent", "?y"),
                        TriplePattern::parse("?y", "ancestor", "?z"),
                    ],
                    vec![TriplePattern::parse("?x", "ancestor", "?z")],
                ),
            ),
        ]
    }

    fn chain_facts(n: usize) -> Vec<Triple> {
        (0..n - 1)
            .map(|i| t(&format!("p{i}"), "parent", &format!("p{}", i + 1)))
            .collect()
    }

    #[test]
    fn deductive_entailed_goal_is_proven_with_engine_proof() {
        let r = DeductiveReasoner::new(kinship_rules(), chain_facts(4));
        let a = r.answer(&Query::new(t("p0", "ancestor", "p3")));
        assert_eq!(a.provability(), Provability::Proven);
        assert_eq!(a.value, Some(t("p0", "ancestor", "p3")));
        // Grounded in the parent-chain axioms.
        assert!(a.provenance.iter().all(|p| p.starts_with("axiom:")));
        assert!(!a.provenance.is_empty());
    }

    #[test]
    fn deductive_unknown_goal_abstains_loudly() {
        let r = DeductiveReasoner::new(kinship_rules(), chain_facts(4));
        let a = r.answer(&Query::new(t("p3", "ancestor", "p0"))); // inverted — not entailed
        assert_eq!(a.provability(), Provability::Abstained);
        assert!(a.value.is_none());
    }

    #[test]
    fn deductive_query_context_extends_the_seed() {
        let r = DeductiveReasoner::new(kinship_rules(), chain_facts(3));
        let mut q = Query::new(t("p0", "ancestor", "p5"));
        assert_eq!(r.answer(&q).provability(), Provability::Abstained);
        // Supply the missing links as query context → now entailed.
        q.context = vec![t("p2", "parent", "p4"), t("p4", "parent", "p5")];
        assert_eq!(r.answer(&q).provability(), Provability::Proven);
    }

    fn confounded_dag() -> CausalDag {
        // confounder → treatment, confounder → outcome, treatment → outcome:
        // identifiable via backdoor adjustment on the (observed) confounder.
        let mut dag = CausalDag::new();
        dag.add_edge("confounder", "treatment", "causes");
        dag.add_edge("confounder", "outcome", "causes");
        dag.add_edge("treatment", "outcome", "causes");
        dag
    }

    #[test]
    fn causal_identifiable_effect_is_proven_with_criterion_rule() {
        let r = CausalReasoner::new(confounded_dag());
        let a = r.answer(&Query::new(t("treatment", CAUSALLY_AFFECTS, "outcome")));
        assert_eq!(a.provability(), Provability::Proven);
        let ProofTrace::Derivation(d) = &a.proof else {
            panic!("causal proof must be a derivation");
        };
        assert!(d.rule_ids()[0].starts_with("do-calculus:"));
        assert!(
            a.provenance
                .iter()
                .any(|p| p.starts_with("identification:"))
        );
    }

    #[test]
    fn causal_unknown_nodes_abstain() {
        let r = CausalReasoner::new(confounded_dag());
        let a = r.answer(&Query::new(t("nope", CAUSALLY_AFFECTS, "outcome")));
        assert_eq!(a.provability(), Provability::Abstained);
    }

    #[test]
    fn causal_wrong_predicate_is_outside_the_envelope() {
        let r = CausalReasoner::new(confounded_dag());
        let a = r.answer(&Query::new(t("treatment", "ancestor", "outcome")));
        assert_eq!(a.provability(), Provability::Abstained);
    }

    #[test]
    fn counterfactual_is_heuristic_never_proven() {
        // Rich-enough DAG to pass the certifiability gate.
        let mut dag = CausalDag::new();
        dag.add_edge("confounder", "treatment", "causes");
        dag.add_edge("confounder", "outcome", "causes");
        dag.add_edge("treatment", "mediator", "causes");
        dag.add_edge("mediator", "outcome", "causes");
        dag.add_edge("other", "outcome", "causes");
        let r = CounterfactualReasoner::new(dag);
        let a = r.answer(&Query::new(t(
            "treatment",
            COUNTERFACTUALLY_AFFECTS,
            "outcome",
        )));
        // THE point of the contract: an estimate cannot be Proven, no matter how confident.
        assert_ne!(a.provability(), Provability::Proven);
        if a.value.is_some() {
            assert_eq!(a.provability(), Provability::Heuristic);
            assert!(matches!(a.proof, ProofTrace::Evidence { .. }));
        }
    }

    #[test]
    fn the_family_routes_by_envelope_behind_dyn() {
        let family: Vec<Box<dyn Reasoner>> = vec![
            Box::new(
                DeductiveReasoner::new(kinship_rules(), chain_facts(4)).with_envelope(
                    CompetenceEnvelope {
                        shapes: vec![QueryShape {
                            name: "kinship".into(),
                            predicates: vec!["ancestor".into()],
                        }],
                    },
                ),
            ),
            Box::new(CausalReasoner::new(confounded_dag())),
        ];
        let deductive_q = Query::new(t("p0", "ancestor", "p2"));
        let causal_q = Query::new(t("treatment", CAUSALLY_AFFECTS, "outcome"));
        let pick = |q: &Query| {
            family
                .iter()
                .find(|r| r.competence_envelope().covers(q))
                .expect("covered")
        };
        assert_eq!(
            pick(&deductive_q).answer(&deductive_q).provability(),
            Provability::Proven
        );
        assert_eq!(
            pick(&causal_q).answer(&causal_q).provability(),
            Provability::Proven
        );
    }
}
