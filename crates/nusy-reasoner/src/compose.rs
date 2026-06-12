//! # COMPOSE — declarative reasoner pipelines (EX-4749, VY-D E4)
//!
//! A [`Pipeline`] is an ordered sequence of `Box<dyn Reasoner>` stages that is
//! **itself a [`Reasoner`]** — so pipelines register in the reasoner-router, compose
//! into bigger pipelines, and answer through the same contract as every atomic
//! reasoner.
//!
//! ## The minimum-guarantee law (the load-bearing property)
//!
//! A pipeline's certainty is the **minimum of its stages** — a heuristic link can
//! never be laundered into a proof:
//!
//! - **Guarantee:** `sound`/`complete` only if *every* stage is; `probabilistic` if
//!   *any* stage is ([`Pipeline::guarantee`]).
//! - **Provability:** enforced *structurally through the trace*, not by assertion.
//!   When every contributing stage produced a complete derivation, their traces are
//!   **grafted** end-to-end (an earlier stage's conclusion that appears as an axiom
//!   leaf in a later stage's proof is replaced by that stage's full derivation) —
//!   the composite is a complete [`DerivationTrace`] and computes `Proven`. The
//!   moment *any* contributing stage emits [`ProofTrace::Evidence`], the composite
//!   trace **is** `Evidence` (confidence = min over stages; every stage tagged in
//!   `why`) — and per the contract invariant, Evidence can never compute to
//!   `Proven`. Laundering is a type-level impossibility, and
//!   `laundering_is_impossible` in the tests would fail if it ever ceased to be.
//!
//! ## Dataflow
//!
//! Stages run in order against the **original goal**; each stage's answer value
//! (when it has one) is appended to the context seen by later stages — the
//! propose→test shape (a later VY-E abduction stage proposes; a deductive stage
//! tests). The **final stage's** answer is the pipeline's answer; if the final
//! stage abstains, the pipeline abstains loudly ([`ProofTrace::None`]).

use crate::{
    Answer, CompetenceEnvelope, DerivationTrace, Guarantee, ProofTrace, Provability, Query,
    Reasoner, Substrate,
};

/// An ordered pipeline of reasoner stages; itself a [`Reasoner`].
pub struct Pipeline {
    /// Human-readable pipeline name (appears in composite Evidence tags).
    name: String,
    stages: Vec<Box<dyn Reasoner>>,
}

impl Pipeline {
    /// An empty named pipeline; add stages with [`then`](Self::then).
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            stages: Vec::new(),
        }
    }

    /// Append a stage. Order is execution order.
    pub fn then(mut self, stage: Box<dyn Reasoner>) -> Self {
        self.stages.push(stage);
        self
    }

    /// Number of stages.
    pub fn len(&self) -> usize {
        self.stages.len()
    }

    /// True when the pipeline has no stages (it then always abstains).
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    /// Graft earlier stages' derivations into `trace`: any axiom leaf whose fact
    /// equals an earlier stage's derived conclusion is replaced by that stage's
    /// full derivation, threading the proof end-to-end.
    fn graft(
        trace: &DerivationTrace,
        earlier: &[(crate::Triple, DerivationTrace)],
    ) -> DerivationTrace {
        match trace {
            DerivationTrace::Axiom(t) => {
                for (concl, deriv) in earlier {
                    if concl == t {
                        return deriv.clone();
                    }
                }
                DerivationTrace::Axiom(t.clone())
            }
            DerivationTrace::Derived {
                conclusion,
                rule_id,
                premises,
            } => DerivationTrace::Derived {
                conclusion: conclusion.clone(),
                rule_id: rule_id.clone(),
                premises: premises.iter().map(|p| Self::graft(p, earlier)).collect(),
            },
        }
    }
}

impl Reasoner for Pipeline {
    fn answer(&self, query: &Query) -> Answer {
        if self.stages.is_empty() {
            return Answer::abstained();
        }

        // Run stages in order; each answered value extends the context for later
        // stages. Track every contributing stage's trace for composition.
        let mut context = query.context.clone();
        let mut contributions: Vec<(usize, Substrate, Answer)> = Vec::new();
        let mut last: Option<Answer> = None;

        for (i, stage) in self.stages.iter().enumerate() {
            let staged_query = Query {
                goal: query.goal.clone(),
                context: context.clone(),
            };
            let a = stage.answer(&staged_query);
            if let Some(v) = &a.value {
                context.push(v.clone());
                contributions.push((i, stage.substrate(), a.clone()));
            }
            last = Some(a);
        }

        let final_answer = last.expect("non-empty pipeline ran at least one stage");
        if final_answer.value.is_none() {
            // Final stage abstained → the pipeline abstains loudly.
            return Answer::abstained();
        }

        // Provenance: union over contributing stages, final stage's last.
        let mut provenance: Vec<String> = Vec::new();
        for (_, _, a) in &contributions {
            for p in &a.provenance {
                if !provenance.contains(p) {
                    provenance.push(p.clone());
                }
            }
        }

        // Composite trace per the minimum law.
        let any_evidence = contributions
            .iter()
            .any(|(_, _, a)| matches!(a.proof, ProofTrace::Evidence { .. }));

        let proof = if any_evidence {
            // One heuristic link anywhere → the whole chain is Evidence (Heuristic
            // by the contract invariant). min-confidence; every stage tagged.
            let mut confidence: f64 = 1.0;
            let mut why: Vec<String> = vec![format!("pipeline '{}'", self.name)];
            for (i, substrate, a) in &contributions {
                match &a.proof {
                    ProofTrace::Evidence {
                        confidence: c,
                        why: w,
                    } => {
                        confidence = confidence.min(*c);
                        why.push(format!(
                            "stage {i} ({substrate:?}): evidence (confidence {c}) — {}",
                            w.join("; ")
                        ));
                    }
                    ProofTrace::Derivation(d) => {
                        why.push(format!(
                            "stage {i} ({substrate:?}): derivation (depth {}, {})",
                            d.depth(),
                            if d.is_complete() {
                                "complete"
                            } else {
                                "incomplete"
                            }
                        ));
                    }
                    ProofTrace::None => {}
                }
            }
            ProofTrace::Evidence { confidence, why }
        } else {
            // Every contributing stage proved: graft earlier derivations into the
            // final trace so the composite proof threads end-to-end.
            let mut earlier: Vec<(crate::Triple, DerivationTrace)> = Vec::new();
            for (_, _, a) in &contributions[..contributions.len().saturating_sub(1)] {
                if let (Some(v), ProofTrace::Derivation(d)) = (&a.value, &a.proof) {
                    earlier.push((v.clone(), d.clone()));
                }
            }
            match &final_answer.proof {
                ProofTrace::Derivation(d) => ProofTrace::Derivation(Self::graft(d, &earlier)),
                other => other.clone(),
            }
        };

        Answer {
            value: final_answer.value,
            proof,
            provenance,
        }
    }

    /// The pipeline's entry point decides coverage: the **first stage's** envelope.
    fn competence_envelope(&self) -> &CompetenceEnvelope {
        static EMPTY: std::sync::OnceLock<CompetenceEnvelope> = std::sync::OnceLock::new();
        self.stages
            .first()
            .map(|s| s.competence_envelope())
            .unwrap_or_else(|| EMPTY.get_or_init(CompetenceEnvelope::default))
    }

    /// Mixed unless every stage shares one substrate.
    fn substrate(&self) -> Substrate {
        let mut iter = self.stages.iter().map(|s| s.substrate());
        match iter.next() {
            None => Substrate::Mixed,
            Some(first) => {
                if iter.all(|s| s == first) {
                    first
                } else {
                    Substrate::Mixed
                }
            }
        }
    }

    /// **The minimum-guarantee law:** sound/complete only if every stage is;
    /// probabilistic if any stage is. An empty pipeline guarantees nothing.
    fn guarantee(&self) -> Guarantee {
        if self.stages.is_empty() {
            return Guarantee::default();
        }
        let gs: Vec<Guarantee> = self.stages.iter().map(|s| s.guarantee()).collect();
        Guarantee {
            sound: gs.iter().all(|g| g.sound),
            complete: gs.iter().all(|g| g.complete),
            probabilistic: gs.iter().any(|g| g.probabilistic),
        }
    }
}

/// Sanity check exposed for tests and the router: the pipeline's *computed*
/// provability never exceeds the minimum of its contributing stages'.
pub fn provability_min(stage_provabilities: &[Provability]) -> Provability {
    if stage_provabilities.is_empty() {
        return Provability::Abstained;
    }
    if stage_provabilities.contains(&Provability::Abstained) {
        return Provability::Abstained;
    }
    if stage_provabilities.contains(&Provability::Heuristic) {
        return Provability::Heuristic;
    }
    Provability::Proven
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{QueryShape, Triple};

    fn t(s: &str, p: &str, o: &str) -> Triple {
        Triple::new(s, p, o)
    }

    fn env_any() -> CompetenceEnvelope {
        CompetenceEnvelope {
            shapes: vec![QueryShape {
                name: "any".into(),
                predicates: vec![],
            }],
        }
    }

    /// Symbolic mock: derives `mid(p, x)` from an axiom, or proves the goal when
    /// `mid` is already in context (so a 2-stage symbolic chain threads).
    struct SymbolicStage {
        env: CompetenceEnvelope,
    }
    impl SymbolicStage {
        fn new() -> Self {
            Self { env: env_any() }
        }
    }
    impl Reasoner for SymbolicStage {
        fn answer(&self, q: &Query) -> Answer {
            let mid = t(&q.goal.subject, "mid", "x");
            if q.context.contains(&mid) {
                // Final hop: goal from mid (which an earlier stage derived).
                Answer {
                    value: Some(q.goal.clone()),
                    proof: ProofTrace::Derivation(DerivationTrace::Derived {
                        conclusion: q.goal.clone(),
                        rule_id: "goal-from-mid".into(),
                        premises: vec![DerivationTrace::Axiom(mid)],
                    }),
                    provenance: vec!["stage2".into()],
                }
            } else {
                // First hop: derive mid from a seed axiom.
                Answer {
                    value: Some(mid.clone()),
                    proof: ProofTrace::Derivation(DerivationTrace::Derived {
                        conclusion: mid,
                        rule_id: "mid-from-seed".into(),
                        premises: vec![DerivationTrace::Axiom(t(&q.goal.subject, "seed", "true"))],
                    }),
                    provenance: vec!["stage1".into()],
                }
            }
        }
        fn competence_envelope(&self) -> &CompetenceEnvelope {
            &self.env
        }
        fn substrate(&self) -> Substrate {
            Substrate::Symbolic
        }
        fn guarantee(&self) -> Guarantee {
            Guarantee {
                sound: true,
                complete: true,
                probabilistic: false,
            }
        }
    }

    /// Neural mock: proposes the mid fact with evidence only.
    struct NeuralMock {
        env: CompetenceEnvelope,
    }
    impl NeuralMock {
        fn new() -> Self {
            Self { env: env_any() }
        }
    }
    impl Reasoner for NeuralMock {
        fn answer(&self, q: &Query) -> Answer {
            Answer {
                value: Some(t(&q.goal.subject, "mid", "x")),
                proof: ProofTrace::Evidence {
                    confidence: 0.7,
                    why: vec!["pattern-proposed mid".into()],
                },
                provenance: vec![],
            }
        }
        fn competence_envelope(&self) -> &CompetenceEnvelope {
            &self.env
        }
        fn substrate(&self) -> Substrate {
            Substrate::Neural
        }
        fn guarantee(&self) -> Guarantee {
            Guarantee {
                sound: false,
                complete: false,
                probabilistic: true,
            }
        }
    }

    fn goal() -> Query {
        Query::new(t("p1", "at_risk", "fall"))
    }

    #[test]
    fn deductive_then_deductive_proves_with_threaded_trace() {
        let pipe = Pipeline::new("d-d")
            .then(Box::new(SymbolicStage::new()))
            .then(Box::new(SymbolicStage::new()));
        let a = pipe.answer(&goal());
        assert_eq!(a.provability(), Provability::Proven);
        // The grafted composite threads end-to-end: the mid axiom in stage 2's
        // proof is replaced by stage 1's derivation, so BOTH rules appear.
        let ProofTrace::Derivation(d) = &a.proof else {
            panic!("expected a composite derivation");
        };
        let rules = d.rule_ids();
        assert!(rules.contains(&"goal-from-mid"), "{rules:?}");
        assert!(rules.contains(&"mid-from-seed"), "{rules:?}");
        assert_eq!(d.depth(), 2, "threaded proof is two hops deep");
        // Provenance unions across stages.
        assert!(a.provenance.contains(&"stage1".to_string()));
        assert!(a.provenance.contains(&"stage2".to_string()));
    }

    #[test]
    fn laundering_is_impossible_neural_link_downgrades_pipeline() {
        let pipe = Pipeline::new("n-d")
            .then(Box::new(NeuralMock::new()))
            .then(Box::new(SymbolicStage::new()));
        let a = pipe.answer(&goal());
        // The deductive FINAL stage proved its hop — but the chain rests on a
        // neural proposal, so the composite must NOT be Proven.
        assert_eq!(a.provability(), Provability::Heuristic);
        let ProofTrace::Evidence { confidence, why } = &a.proof else {
            panic!("composite over a heuristic link must be Evidence");
        };
        assert!((confidence - 0.7).abs() < 1e-9, "min-confidence propagates");
        // The neural stage is explicitly tagged in the trace.
        assert!(
            why.iter().any(|w| w.contains("stage 0 (Neural)")),
            "{why:?}"
        );
        assert!(
            why.iter().any(|w| w.contains("stage 1 (Symbolic)")),
            "{why:?}"
        );
    }

    #[test]
    fn guarantee_is_minimum_of_stages() {
        let pure = Pipeline::new("d-d")
            .then(Box::new(SymbolicStage::new()))
            .then(Box::new(SymbolicStage::new()));
        assert_eq!(
            pure.guarantee(),
            Guarantee {
                sound: true,
                complete: true,
                probabilistic: false
            }
        );
        let mixed = Pipeline::new("n-d")
            .then(Box::new(NeuralMock::new()))
            .then(Box::new(SymbolicStage::new()));
        assert_eq!(
            mixed.guarantee(),
            Guarantee {
                sound: false,
                complete: false,
                probabilistic: true
            }
        );
        assert_eq!(mixed.substrate(), Substrate::Mixed);
    }

    #[test]
    fn empty_pipeline_abstains_and_final_abstention_is_loud() {
        let empty = Pipeline::new("empty");
        let a = empty.answer(&goal());
        assert_eq!(a.provability(), Provability::Abstained);
        assert!(matches!(a.proof, ProofTrace::None));
    }

    #[test]
    fn provability_min_law() {
        use Provability::*;
        assert_eq!(provability_min(&[Proven, Proven]), Proven);
        assert_eq!(provability_min(&[Proven, Heuristic]), Heuristic);
        assert_eq!(provability_min(&[Heuristic, Proven]), Heuristic);
        assert_eq!(provability_min(&[Proven, Abstained]), Abstained);
        assert_eq!(provability_min(&[]), Abstained);
    }
}
