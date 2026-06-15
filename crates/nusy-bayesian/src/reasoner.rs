//! Reasoner-contract conformance (VY-Bayes E2, EX-4817).
//!
//! Wraps the [`BayesianUpdate`](crate::BayesianUpdate) engine (E1) as a first-class
//! [`Reasoner`] so the router and [`Pipeline`](nusy_reasoner::compose::Pipeline) can hold it
//! beside the symbolic reasoners. **The entire point of this layer is that a probabilistic
//! answer is NEVER laundered into a proof:**
//!
//! - [`answer`](BayesianReasoner::answer) emits a [`ProofTrace::Evidence`] trace (a calibrated
//!   posterior confidence + the evidence behind it), so [`Answer::provability`] computes
//!   [`Provability::Heuristic`](nusy_reasoner::Provability::Heuristic) — **never `Proven`**,
//!   by the guarantee invariant. There is no code path that returns a `Derivation` here.
//! - [`guarantee`](BayesianReasoner::guarantee) reports `probabilistic: true, sound: false`,
//!   so any [`Pipeline`](nusy_reasoner::compose::Pipeline) containing a Bayesian stage is
//!   downgraded to the **minimum** guarantee (the VY-D minimum-guarantee law — we reuse that
//!   composition machinery, we do not re-implement it). A heuristic link makes the whole chain
//!   heuristic.
//!
//! ## Substrate
//! Bayesian updating is a deterministic *symbolic/mathematical* computation (no neural net),
//! so [`Substrate::Symbolic`] is the honest substrate — but, exactly like the abductive
//! [`Abducer`](https://docs.rs/nusy-abduction-rank), provability is governed by the **proof
//! trace and the guarantee, never by the substrate tag**. A `Symbolic` substrate does not make
//! a posterior provable; the `Evidence` trace keeps it `Heuristic`.
//!
//! ## Query model
//! The reasoner is constructed with a hypothesis model + the evidence to apply; `answer`
//! returns the **MAP hypothesis** (most-probable, given the evidence) framed into the query
//! goal's predicate/object, with the full posterior carried in the evidence trace. (A richer
//! query→evidence extraction is a later refinement; E2's deliverable is the contract
//! conformance and the never-laundered provability tag.)

use nusy_reasoner::{
    Answer, CompetenceEnvelope, Guarantee, ProofTrace, Query, Reasoner, Substrate,
};
use nusy_unify::Triple;

use crate::{BayesianUpdate, Likelihood};

/// A [`BayesianUpdate`] model exposed through the [`Reasoner`] contract. Answers carry a
/// posterior as `Evidence` (Heuristic), never a derivation.
#[derive(Debug, Clone)]
pub struct BayesianReasoner {
    model: BayesianUpdate,
    /// Evidence applied (in order) to the prior before reading the posterior.
    evidence: Vec<Likelihood>,
    envelope: CompetenceEnvelope,
}

impl BayesianReasoner {
    /// Build a reasoner over a hypothesis model and the evidence to condition on. `envelope`
    /// declares which query shapes this reasoner is asked to answer.
    pub fn new(
        model: BayesianUpdate,
        evidence: Vec<Likelihood>,
        envelope: CompetenceEnvelope,
    ) -> Self {
        Self {
            model,
            evidence,
            envelope,
        }
    }
}

impl Reasoner for BayesianReasoner {
    fn answer(&self, query: &Query) -> Answer {
        // Condition the prior on all evidence → posterior.
        let posterior = self.model.observe(&self.evidence);
        let Some(map) = posterior.map_hypothesis() else {
            // Empty hypothesis space — nothing to rank. Honest abstention.
            return Answer::abstained();
        };
        let confidence = posterior.prob(map);

        // The answer value: the MAP hypothesis framed into the query goal's predicate/object
        // (the query asks "what best satisfies <predicate> <object>?"; the MAP is the answer).
        let value = Some(Triple::new(map, &query.goal.predicate, &query.goal.object));

        // Evidence trace: the full ranked posterior + the evidence + entropy. This is what
        // makes the answer Heuristic — there is deliberately NO Derivation branch.
        let mut why: Vec<String> = Vec::new();
        why.push(format!("MAP={map} p={confidence:.4}"));
        why.push(format!(
            "posterior: {}",
            posterior
                .ranked()
                .iter()
                .map(|(h, p)| format!("{h}={p:.4}"))
                .collect::<Vec<_>>()
                .join(", ")
        ));
        why.push(format!("evidence: {}", posterior.evidence().join(",")));
        why.push(format!("entropy_bits={:.4}", posterior.entropy_bits()));
        why.push("probabilistic — calibrated confidence, NOT a proof".to_string());

        Answer {
            value,
            // Evidence (not Derivation) → provability() is Heuristic. The Bayesian reasoner
            // cannot mint `Proven`; this is the never-laundered contract.
            proof: ProofTrace::Evidence { confidence, why },
            provenance: posterior.evidence().to_vec(),
        }
    }

    fn competence_envelope(&self) -> &CompetenceEnvelope {
        &self.envelope
    }

    fn substrate(&self) -> Substrate {
        // Symbolic/mathematical computation (not neural); provability is governed by the
        // Evidence trace + guarantee, never by this tag (see module docs).
        Substrate::Symbolic
    }

    fn guarantee(&self) -> Guarantee {
        // A posterior is calibrated, not certain: not sound (a high-posterior answer can be
        // wrong), not complete, and probabilistic — the flag that downgrades any Pipeline it
        // joins to the minimum guarantee.
        Guarantee {
            sound: false,
            complete: false,
            probabilistic: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Hypothesis;
    use nusy_reasoner::compose::{Pipeline, provability_min};
    use nusy_reasoner::{CompetenceEnvelope, DerivationTrace, Provability, QueryShape, Reasoner};

    fn env() -> CompetenceEnvelope {
        CompetenceEnvelope {
            shapes: vec![QueryShape {
                name: "diagnosis".into(),
                predicates: vec![],
            }],
        }
    }

    fn disease_reasoner() -> BayesianReasoner {
        // 1% prevalence; a positive test (sens .99, fp .05).
        let model = BayesianUpdate::new([
            Hypothesis::new("disease", 0.01),
            Hypothesis::new("healthy", 0.99),
        ])
        .unwrap();
        let positive = Likelihood::new("test+")
            .with("disease", 0.99)
            .with("healthy", 0.05);
        BayesianReasoner::new(model, vec![positive], env())
    }

    /// The headline contract: a probabilistic answer is Heuristic, NEVER Proven — even when
    /// the posterior is sharp. The proof is Evidence, and there is no Derivation path.
    #[test]
    fn bayesian_answer_is_heuristic_never_proven() {
        let r = disease_reasoner();
        let ans = r.answer(&Query::new(Triple::new("?", "diagnosis_of", "patient")));
        assert_eq!(
            ans.provability(),
            Provability::Heuristic,
            "a posterior must never be tagged Proven"
        );
        // MAP is 'healthy' (base-rate dominates a single positive) — framed into the goal.
        assert_eq!(
            ans.value,
            Some(Triple::new("healthy", "diagnosis_of", "patient"))
        );
        assert!(matches!(ans.proof, ProofTrace::Evidence { .. }));
        // Belt-and-braces: the trace is NOT a derivation under any reading.
        assert!(!matches!(ans.proof, ProofTrace::Derivation(_)));
    }

    /// Even a near-certain posterior stays Heuristic (the laundering guard at the extreme).
    #[test]
    fn near_certain_posterior_is_still_heuristic() {
        let model = BayesianUpdate::uniform(["a", "b"]).unwrap();
        let crusher = Likelihood::new("e").with("a", 0.999).with("b", 0.001);
        let r = BayesianReasoner::new(model, vec![crusher], env());
        let ans = r.answer(&Query::new(Triple::new("?", "is", "a")));
        if let ProofTrace::Evidence { confidence, .. } = &ans.proof {
            assert!(*confidence > 0.99, "posterior is sharp");
        } else {
            panic!("expected Evidence");
        }
        assert_eq!(ans.provability(), Provability::Heuristic);
    }

    /// Guarantee: probabilistic + not-sound — the flag the Pipeline reads to downgrade.
    #[test]
    fn guarantee_is_probabilistic_not_sound() {
        let g = disease_reasoner().guarantee();
        assert!(g.probabilistic && !g.sound && !g.complete);
        assert_eq!(disease_reasoner().substrate(), Substrate::Symbolic);
    }

    /// A sound, Proven-capable mock stage — so the pipeline test has something to downgrade.
    struct AlwaysProven {
        env: CompetenceEnvelope,
    }
    impl Reasoner for AlwaysProven {
        fn answer(&self, query: &Query) -> Answer {
            // A complete one-step derivation (axiom leaf) → Proven.
            Answer {
                value: Some(query.goal.clone()),
                proof: ProofTrace::Derivation(DerivationTrace::Derived {
                    conclusion: query.goal.clone(),
                    rule_id: "axiom".into(),
                    premises: vec![DerivationTrace::Axiom(query.goal.clone())],
                }),
                provenance: vec!["axiom".into()],
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

    /// REUSE the VY-D Pipeline (don't fork it): a pipeline with a Proven stage + a Bayesian
    /// stage reports the MINIMUM guarantee — sound collapses to false, probabilistic to true,
    /// and the composed answer is Heuristic. The Bayesian link launders nothing.
    #[test]
    fn pipeline_with_bayes_stage_reports_minimum_guarantee() {
        let proven = AlwaysProven { env: env() };
        let bayes = disease_reasoner();
        // Sanity: alone, the proven stage IS Proven.
        let solo = proven.answer(&Query::new(Triple::new("g", "p", "o")));
        assert_eq!(solo.provability(), Provability::Proven);

        let pipe = Pipeline::new("proven-then-bayes")
            .then(Box::new(AlwaysProven { env: env() }))
            .then(Box::new(bayes));

        // Guarantee downgraded by the Bayesian stage.
        let g = pipe.guarantee();
        assert!(!g.sound, "one probabilistic stage makes the chain unsound");
        assert!(g.probabilistic, "the chain is probabilistic");

        // And the composed answer is Heuristic — never laundered up to Proven.
        let ans = pipe.answer(&Query::new(Triple::new("?", "diagnosis_of", "patient")));
        assert_eq!(ans.provability(), Provability::Heuristic);
    }

    /// The min-guarantee law the pipeline relies on: Proven ∧ Heuristic = Heuristic.
    #[test]
    fn provability_min_downgrades_to_heuristic() {
        assert_eq!(
            provability_min(&[Provability::Proven, Provability::Heuristic]),
            Provability::Heuristic
        );
    }
}
