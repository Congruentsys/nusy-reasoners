//! # nusy-abduction-test — symbolic test stage for abductive candidates (EX-4776, VY-E E2)
//!
//! Abduction ([`nusy_abduction`], E1) *generates* candidate explanations; it does not say which
//! ones hold. This stage **tests each candidate symbolically**: assume the hypothesis `H`, forward
//! chain over the rules, and keep `H` only if
//!
//! 1. the observation `E` is **re-derivable** from `H` (`H ⊢ E`), and
//! 2. **no explicit negative** (`contraindicates`, EX-4692) fires against `H` — the defeater screen.
//!
//! A survivor carries its `H ⊢ E` [`DerivationTrace`] as evidence, so the eventual explanation is
//! **auditable** (this is what turns an abductive guess into a checkable claim). This stage does
//! **no ranking** (that is E3) — it only separates *survives* from *rejected* and says *why*.
//!
//! It uses the Vec-based [`forward_chain`] (the same monotonic deductive core the
//! `DeductiveReasoner` adapter wraps) directly, so the whole abduction line stays **arrow-free /
//! FOSS-movable**. Full re-fire per candidate is fine at fixture scale (the expedition's constraint).
//!
//! ```
//! use nusy_abduction::{CandidateSource, GraphCandidates};
//! use nusy_abduction_test::{TestStage, TestOutcome};
//! use nusy_forward_chain::IdRule;
//! use nusy_unify::{Rule, Triple, TriplePattern};
//!
//! let rule = IdRule::new("risk-from-smoking", Rule::new(
//!     vec![TriplePattern::parse("?p", "smokes", "true")],
//!     vec![TriplePattern::parse("?p", "at_risk", "cancer")],
//! ));
//! let obs = Triple::new("p1", "at_risk", "cancer");
//! let candidates = GraphCandidates::new(vec![rule.clone()], 1).enumerate(&obs);
//! // Test stage over the same rule set, with an empty graph slice (E excluded).
//! let stage = TestStage::new(vec![rule], vec![]);
//! let survivors = stage.survivors(&obs, &candidates);
//! assert_eq!(survivors.len(), 1); // "p1 smokes" survives: assuming it re-derives the observation.
//! ```

use nusy_abduction::Hypothesis;
use nusy_forward_chain::{IdRule, Saturation, forward_chain};
use nusy_reasoner::DerivationTrace;
use nusy_unify::Triple;

/// The EX-4692 explicit-negative predicate: `(_, "contraindicates", x)`. If such a fact targets one
/// of `H`'s assumed atoms, `H` is self-defeating and is screened out.
pub const CONTRAINDICATES: &str = "contraindicates";

/// Why a candidate was rejected by the test stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Rejection {
    /// Assuming `H` did not make the observation derivable (`H ⊬ E`).
    ExplainsNothing,
    /// An explicit negative fired against one of `H`'s assumed atoms (the defeater screen).
    Contraindicated {
        /// The firing `(_, "contraindicates", x)` fact, cited as the reason.
        negative: Triple,
    },
    /// The candidate still carries a free (existential) variable, so it cannot be asserted as
    /// concrete facts and tested at fixture scale.
    NotGround,
}

/// A candidate that passed the test stage, with its auditable `H ⊢ E` derivation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Survivor {
    /// The surviving hypothesis.
    pub hypothesis: Hypothesis,
    /// The derivation of the observation from the assumed hypothesis — the survivor's evidence.
    pub evidence: DerivationTrace,
}

/// The outcome of testing one candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TestOutcome {
    /// Passed both gates.
    Survived(Survivor),
    /// Failed a gate.
    Rejected {
        /// The rejected hypothesis.
        hypothesis: Hypothesis,
        /// Why it failed.
        reason: Rejection,
    },
}

/// Symbolically tests abductive candidates against a rule set and a graph slice.
#[derive(Debug, Clone)]
pub struct TestStage {
    rules: Vec<IdRule>,
    /// The graph slice the hypothesis is assumed *on top of*. It should **not** already assert the
    /// observation, or the `H ⊢ E` test is trivially true regardless of `H`.
    base_facts: Vec<Triple>,
}

impl TestStage {
    /// Build a test stage over a rule set and a base graph slice.
    pub fn new(rules: Vec<IdRule>, base_facts: Vec<Triple>) -> Self {
        Self { rules, base_facts }
    }

    /// Test one candidate: assume `H` (its ground atoms), forward-chain, and keep it iff the
    /// observation is re-derived and no explicit negative fires against `H`.
    pub fn test(&self, observation: &Triple, candidate: &Hypothesis) -> TestOutcome {
        let Some(h_facts) = candidate.ground() else {
            return TestOutcome::Rejected {
                hypothesis: candidate.clone(),
                reason: Rejection::NotGround,
            };
        };

        // seed = graph slice + H.
        let mut seed = self.base_facts.clone();
        seed.extend(h_facts.iter().cloned());
        let sat = forward_chain(&self.rules, seed);

        // Gate 1 — H ⊢ E: the observation must be re-derivable (and actually *derived*, not just a
        // seed leaf — H has to do the work).
        let derived = sat.derivation_of(observation).is_some();
        if !sat.contains(observation) || !derived {
            return TestOutcome::Rejected {
                hypothesis: candidate.clone(),
                reason: Rejection::ExplainsNothing,
            };
        }

        // Gate 2 — defeater screen: reject if an explicit negative targets one of H's assumed atoms.
        if let Some(negative) = self.firing_negative(&sat, &h_facts) {
            return TestOutcome::Rejected {
                hypothesis: candidate.clone(),
                reason: Rejection::Contraindicated { negative },
            };
        }

        // Survivor: attach the H ⊢ E derivation as evidence.
        let evidence = build_trace(&sat, observation, &mut Vec::new());
        TestOutcome::Survived(Survivor {
            hypothesis: candidate.clone(),
            evidence,
        })
    }

    /// Test every candidate, returning one outcome each (order preserved).
    pub fn test_all(&self, observation: &Triple, candidates: &[Hypothesis]) -> Vec<TestOutcome> {
        candidates
            .iter()
            .map(|c| self.test(observation, c))
            .collect()
    }

    /// The survivors only — the explanations that passed both gates, each with its evidence.
    pub fn survivors(&self, observation: &Triple, candidates: &[Hypothesis]) -> Vec<Survivor> {
        self.test_all(observation, candidates)
            .into_iter()
            .filter_map(|o| match o {
                TestOutcome::Survived(s) => Some(s),
                TestOutcome::Rejected { .. } => None,
            })
            .collect()
    }

    /// Find an explicit-negative fact in the saturation that targets one of `H`'s assumed atoms.
    /// A `(_, "contraindicates", x)` fires against `H` when `x` appears as the subject or object of
    /// any assumed atom — i.e. assuming `H` brings a contraindicated entity into play.
    fn firing_negative(&self, sat: &Saturation, h_facts: &[Triple]) -> Option<Triple> {
        let targets: Vec<&str> = h_facts
            .iter()
            .flat_map(|t| [t.subject.as_str(), t.object.as_str()])
            .collect();
        sat.facts
            .iter()
            .find(|t| t.predicate == CONTRAINDICATES && targets.contains(&t.object.as_str()))
            .cloned()
    }
}

/// Build the monotonic derivation trace of `target` from a saturation (a seed fact is an
/// [`Axiom`](DerivationTrace::Axiom); a derived fact a [`Derived`](DerivationTrace::Derived) over its
/// premises' traces). `visiting` guards against a cyclic premise chain (engine-impossible, defensive).
fn build_trace(sat: &Saturation, target: &Triple, visiting: &mut Vec<Triple>) -> DerivationTrace {
    match sat.derivation_of(target) {
        None => DerivationTrace::Axiom(target.clone()),
        Some(d) => {
            if visiting.contains(target) {
                return DerivationTrace::Axiom(target.clone());
            }
            visiting.push(target.clone());
            let premises = d
                .premises
                .iter()
                .map(|p| build_trace(sat, p, visiting))
                .collect();
            visiting.pop();
            DerivationTrace::Derived {
                conclusion: target.clone(),
                rule_id: d.rule_id.clone(),
                premises,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nusy_abduction::{CandidateSource, GraphCandidates};
    use nusy_reasoner::Substrate;
    use nusy_unify::{Rule, TriplePattern};

    fn idrule(id: &str, body: Vec<TriplePattern>, head: Vec<TriplePattern>) -> IdRule {
        IdRule::new(id, Rule::new(body, head))
    }

    fn hyp(atoms: &[(&str, &str, &str)]) -> Hypothesis {
        Hypothesis {
            explanation: atoms
                .iter()
                .map(|(s, p, o)| {
                    use nusy_unify::Term;
                    TriplePattern::new(
                        Term::Const(s.to_string()),
                        Term::Const(p.to_string()),
                        Term::Const(o.to_string()),
                    )
                })
                .collect(),
            provenance: vec!["test".to_string()],
            substrate: Substrate::Symbolic,
        }
    }

    /// A gold explanation re-derives the observation and survives, carrying a complete H ⊢ E trace.
    #[test]
    fn gold_explanation_survives_with_evidence() {
        let rule = idrule(
            "risk-from-smoking",
            vec![TriplePattern::parse("?p", "smokes", "true")],
            vec![TriplePattern::parse("?p", "at_risk", "cancer")],
        );
        let obs = Triple::new("p1", "at_risk", "cancer");
        let stage = TestStage::new(vec![rule], vec![]);

        let candidate = hyp(&[("p1", "smokes", "true")]);
        match stage.test(&obs, &candidate) {
            TestOutcome::Survived(s) => {
                assert!(
                    s.evidence.is_complete(),
                    "evidence must be a complete H⊢E derivation"
                );
                // The trace concludes the observation via the reversed rule.
                if let DerivationTrace::Derived {
                    conclusion,
                    rule_id,
                    ..
                } = &s.evidence
                {
                    assert_eq!(conclusion, &obs);
                    assert_eq!(rule_id, "risk-from-smoking");
                } else {
                    panic!("expected a derived trace");
                }
            }
            other => panic!("expected Survived, got {other:?}"),
        }
    }

    /// A candidate whose assumption triggers an explicit negative is rejected, with the firing
    /// `contraindicates` fact cited.
    #[test]
    fn contraindicated_candidate_is_rejected_with_negative_cited() {
        // smokes -> at_risk(cancer);  smokes -> contraindicates(some_drug) [the explicit negative].
        let risk = idrule(
            "risk-from-smoking",
            vec![TriplePattern::parse("?p", "smokes", "true")],
            vec![TriplePattern::parse("?p", "at_risk", "cancer")],
        );
        let contra = idrule(
            "smoking-contra",
            vec![TriplePattern::parse("?p", "smokes", "true")],
            vec![TriplePattern::parse("plan", "contraindicates", "?p")],
        );
        let obs = Triple::new("p1", "at_risk", "cancer");
        let stage = TestStage::new(vec![risk, contra], vec![]);

        let candidate = hyp(&[("p1", "smokes", "true")]);
        match stage.test(&obs, &candidate) {
            TestOutcome::Rejected {
                reason: Rejection::Contraindicated { negative },
                ..
            } => {
                assert_eq!(negative, Triple::new("plan", "contraindicates", "p1"));
            }
            other => panic!("expected Contraindicated rejection, got {other:?}"),
        }
    }

    /// A candidate that does not re-derive the observation is dropped.
    #[test]
    fn candidate_explaining_nothing_is_dropped() {
        let rule = idrule(
            "risk-from-smoking",
            vec![TriplePattern::parse("?p", "smokes", "true")],
            vec![TriplePattern::parse("?p", "at_risk", "cancer")],
        );
        let obs = Triple::new("p1", "at_risk", "cancer");
        let stage = TestStage::new(vec![rule], vec![]);

        // "p1 likes tea" assumes an irrelevant fact — the observation is not re-derived.
        let candidate = hyp(&[("p1", "likes", "tea")]);
        assert_eq!(
            stage.test(&obs, &candidate),
            TestOutcome::Rejected {
                hypothesis: candidate,
                reason: Rejection::ExplainsNothing,
            }
        );
    }

    /// An ungroundable (existential) candidate cannot be tested at fixture scale.
    #[test]
    fn ungroundable_candidate_is_not_ground() {
        let rule = idrule(
            "frail-from-some-condition",
            vec![TriplePattern::parse("?p", "has_condition", "?c")],
            vec![TriplePattern::parse("?p", "frail", "true")],
        );
        let obs = Triple::new("p1", "frail", "true");
        let candidates = GraphCandidates::new(vec![rule.clone()], 1).enumerate(&obs);
        // The single candidate carries an unbound ?c.
        let stage = TestStage::new(vec![rule], vec![]);
        let outcome = stage.test(&obs, &candidates[0]);
        assert!(matches!(
            outcome,
            TestOutcome::Rejected {
                reason: Rejection::NotGround,
                ..
            }
        ));
    }

    /// The asserted property: across a mixed batch, **zero contraindicated candidates survive**.
    #[test]
    fn zero_contraindicated_survivors() {
        let risk = idrule(
            "risk-from-smoking",
            vec![TriplePattern::parse("?p", "smokes", "true")],
            vec![TriplePattern::parse("?p", "at_risk", "cancer")],
        );
        let alt = idrule(
            "risk-from-asbestos",
            vec![TriplePattern::parse("?p", "exposed", "asbestos")],
            vec![TriplePattern::parse("?p", "at_risk", "cancer")],
        );
        let contra = idrule(
            "smoking-contra",
            vec![TriplePattern::parse("?p", "smokes", "true")],
            vec![TriplePattern::parse("plan", "contraindicates", "?p")],
        );
        let obs = Triple::new("p1", "at_risk", "cancer");
        let stage = TestStage::new(vec![risk, alt, contra], vec![]);

        let candidates = vec![
            hyp(&[("p1", "smokes", "true")]),      // contraindicated → rejected
            hyp(&[("p1", "exposed", "asbestos")]), // clean → survives
            hyp(&[("p1", "likes", "tea")]),        // explains nothing → dropped
        ];
        let survivors = stage.survivors(&obs, &candidates);

        assert_eq!(survivors.len(), 1);
        assert_eq!(
            survivors[0].hypothesis.ground().unwrap(),
            vec![Triple::new("p1", "exposed", "asbestos")]
        );
        // No survivor's assumed atoms are contraindicated in its own derivation (the safety property).
        for s in &survivors {
            let h = s.hypothesis.ground().unwrap();
            let mut seed = h.clone();
            seed.extend(vec![]);
            let sat = forward_chain(stage.rules.as_slice(), seed);
            assert!(
                stage.firing_negative(&sat, &h).is_none(),
                "a survivor must never be contraindicated"
            );
        }
    }
}
