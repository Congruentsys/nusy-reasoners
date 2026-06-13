//! # Reasoner-router — envelope dispatch over the Reasoner contract (EX-4748, VY-D E3)
//!
//! Generalizes the V18 gate: the provable branch becomes **any sound reasoner whose
//! [`CompetenceEnvelope`] covers the query**. The router holds a heterogeneous
//! `Vec<Box<dyn Reasoner>>`, dispatches a query to every covering reasoner, and
//! classifies the outcome with the V15-grounded taxonomy:
//!
//! | Class | Meaning |
//! |---|---|
//! | [`AnswerClass::Symbolic`] | answered **Proven** — a complete derivation from a sound reasoner |
//! | [`AnswerClass::Neural`]   | best heuristic answer — explicitly tagged, never asserted as proven |
//! | [`AnswerClass::CoVote`]   | symbolic and neural substrates **agree** on the value — tagged `co-vote`; provability is still *computed from the winning proof* (never upgraded by agreement) |
//!
//! **Loud abstention is preserved:** when no covering reasoner answers, the router
//! returns [`RouteOutcome::Abstained`] carrying the query, the reason, and how many
//! reasoners were consulted — it never silently drops a claim and never asserts one
//! it cannot back (the gate's zero-hallucination-by-construction guarantee, now over
//! an open reasoner family).
//!
//! **The gate is wrapped, not replaced** — [`nusy_gate::ProvableClaimGate`]'s
//! Proven/Unproven contract is untouched; a gate-backed engine joins the family via
//! its adapter (`nusy-reasoner-adapters::DeductiveReasoner`).
//!
//! ## PAR — the standing regression guard
//!
//! [`ReasonerRouter::par`] computes the **Provable Answer Rate** over a panel of
//! `(query, should_prove)` expectations and counts **false proofs** (a Proven answer
//! on a claim that must NOT be provable). The V18 invariant, generalized:
//! `false_proofs == 0` always, and `PAR == 1.0` on the entailed set of the clinical
//! gold panel (`tests/par_regression.rs` asserts both on every `cargo test`). This is
//! also the integration surface EX-4613 (coverage) and EX-4614 (calibration) drive.

use nusy_reasoner::{Answer, Provability, Query, Reasoner, Substrate};

/// The routed answer's class under the SYMBOLIC / NEURAL / CO-VOTE taxonomy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnswerClass {
    /// Proven by a symbolic (or mixed-acting-symbolic) reasoner — certifiable.
    Symbolic,
    /// Best-effort heuristic answer — tagged, never presented as proven.
    Neural,
    /// Two different substrates independently produced the **same value**. Tagged
    /// co-vote; provability remains whatever the winning proof computes to —
    /// agreement never upgrades a heuristic answer to Proven.
    CoVote,
}

/// A routed, classified answer: the winning [`Answer`], its class, and which
/// reasoner (by router index) produced it.
#[derive(Debug, Clone)]
pub struct RoutedVerdict {
    /// The winning answer (provability is computed from its proof, per the contract).
    pub answer: Answer,
    /// SYMBOLIC / NEURAL / CO-VOTE.
    pub class: AnswerClass,
    /// Index of the producing reasoner in the router's registration order.
    pub reasoner_index: usize,
    /// How many covering reasoners answered with the same value (1 = uncontested;
    /// ≥ 2 across substrates is what makes a CO-VOTE).
    pub agreeing: usize,
}

/// The router's outcome for one query — an answer, or a **loud** abstention.
#[derive(Debug, Clone)]
pub enum RouteOutcome {
    /// A covering reasoner answered; see the verdict's class and provability.
    Answered(RoutedVerdict),
    /// **Loud abstention** — surfaced, never silent: no covering reasoner produced
    /// an answer (or none covered the query at all). The claim is *flagged*, and
    /// the caller decides policy (route onward, ask the Captain, log).
    Abstained {
        /// The query nothing could answer.
        query: Query,
        /// Why: `no reasoner covers ...` or `all N covering reasoners abstained`.
        reason: String,
        /// How many registered reasoners covered (and were consulted on) the query.
        consulted: usize,
    },
}

impl RouteOutcome {
    /// Did the route end in a **Proven** symbolic answer?
    pub fn is_proven(&self) -> bool {
        matches!(
            self,
            RouteOutcome::Answered(v) if v.answer.provability() == Provability::Proven
        )
    }

    /// The verdict, if any reasoner answered.
    pub fn verdict(&self) -> Option<&RoutedVerdict> {
        match self {
            RouteOutcome::Answered(v) => Some(v),
            RouteOutcome::Abstained { .. } => None,
        }
    }
}

/// Envelope-dispatching router over an open family of [`Reasoner`]s.
///
/// Dispatch policy (the Reasoner-trait router contract, VY-D E1 doc):
/// 1. consult every reasoner whose envelope covers the query;
/// 2. the **first Proven** answer wins (registration order breaks ties) → SYMBOLIC;
/// 3. no proof → the highest-confidence heuristic answer wins → NEURAL;
/// 4. substrate-crossing agreement on the winning value → CO-VOTE (tag only);
/// 5. nothing answered → **loud** [`RouteOutcome::Abstained`].
#[derive(Default)]
pub struct ReasonerRouter {
    reasoners: Vec<Box<dyn Reasoner>>,
}

impl ReasonerRouter {
    /// An empty router; add reasoners with [`push`](Self::push).
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a reasoner. Registration order is the Proven tie-break order, so
    /// register preferred (e.g. store-backed) reasoners first.
    pub fn push(&mut self, reasoner: Box<dyn Reasoner>) -> &mut Self {
        self.reasoners.push(reasoner);
        self
    }

    /// Number of registered reasoners.
    pub fn len(&self) -> usize {
        self.reasoners.len()
    }

    /// True when no reasoners are registered.
    pub fn is_empty(&self) -> bool {
        self.reasoners.is_empty()
    }

    /// Route one query per the dispatch policy. Never panics, never silently drops:
    /// every path ends in `Answered` or a reasoned `Abstained`.
    pub fn route(&self, query: &Query) -> RouteOutcome {
        // (index, answer, substrate) for every covering reasoner that gave a value.
        let mut answered: Vec<(usize, Answer, Substrate)> = Vec::new();
        let mut consulted = 0usize;

        for (i, r) in self.reasoners.iter().enumerate() {
            if !r.competence_envelope().covers(query) {
                continue;
            }
            consulted += 1;
            let a = r.answer(query);
            if a.value.is_some() {
                answered.push((i, a, r.substrate()));
            }
        }

        if consulted == 0 {
            return RouteOutcome::Abstained {
                query: query.clone(),
                reason: format!(
                    "no reasoner covers predicate '{}' ({} registered)",
                    query.goal.predicate,
                    self.reasoners.len()
                ),
                consulted: 0,
            };
        }
        if answered.is_empty() {
            return RouteOutcome::Abstained {
                query: query.clone(),
                reason: format!("all {consulted} covering reasoners abstained"),
                consulted,
            };
        }

        // First Proven wins (registration order); else best heuristic by confidence.
        let winner_pos = answered
            .iter()
            .position(|(_, a, _)| a.provability() == Provability::Proven)
            .unwrap_or_else(|| {
                // No proof anywhere: highest Evidence confidence wins; answers whose
                // trace carries no confidence rank lowest. Stable, so registration
                // order breaks ties.
                let mut best = 0usize;
                let mut best_conf = f64::MIN;
                for (pos, (_, a, _)) in answered.iter().enumerate() {
                    let conf = match &a.proof {
                        nusy_reasoner::ProofTrace::Evidence { confidence, .. } => *confidence,
                        _ => f64::MIN,
                    };
                    if conf > best_conf {
                        best_conf = conf;
                        best = pos;
                    }
                }
                best
            });

        let (winner_index, winner_answer, winner_substrate) = answered[winner_pos].clone();

        // Agreement: same value from another reasoner; CO-VOTE iff a *different
        // substrate* agrees (symbolic+neural saying the same thing is the signal).
        let agreeing = answered
            .iter()
            .filter(|(_, a, _)| a.value == winner_answer.value)
            .count();
        let cross_substrate_agreement = answered.iter().any(|(i, a, s)| {
            *i != winner_index && a.value == winner_answer.value && *s != winner_substrate
        });

        let class = if cross_substrate_agreement {
            AnswerClass::CoVote
        } else if winner_answer.provability() == Provability::Proven {
            AnswerClass::Symbolic
        } else {
            AnswerClass::Neural
        };

        RouteOutcome::Answered(RoutedVerdict {
            answer: winner_answer,
            class,
            reasoner_index: winner_index,
            agreeing,
        })
    }

    /// **PAR battery** — the runnable surface for EX-4613 (coverage) / EX-4614
    /// (calibration) and the standing regression guard. Routes every panel claim and
    /// tallies the Provable Answer Rate over the should-prove set, plus the two
    /// failure counters that must stay zero:
    /// - `false_proofs` — Proven on a claim expected unprovable (the hallucination
    ///   direction; structurally impossible per the contract, asserted anyway);
    /// - `silent_drops` — always 0 by construction (every route returns an outcome),
    ///   tallied to keep the loud-abstention property observable in reports.
    pub fn par(&self, panel: &[(Query, bool)]) -> ParReport {
        let mut report = ParReport::default();
        for (query, should_prove) in panel {
            let outcome = self.route(query);
            let proven = outcome.is_proven();
            match (should_prove, proven) {
                (true, true) => report.proven_expected += 1,
                (true, false) => report.missed += 1,
                (false, true) => report.false_proofs += 1,
                (false, false) => report.correctly_unproven += 1,
            }
            if outcome.verdict().is_none() && proven {
                // Unreachable by types; kept as the observable silent-drop tally.
                report.silent_drops += 1;
            }
        }
        report
    }
}

/// The PAR battery's tallies. `par()` is the headline rate; the two failure
/// counters must be zero on every run.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ParReport {
    /// Should-prove claims that routed to a Proven answer.
    pub proven_expected: usize,
    /// Should-prove claims that did **not** prove (coverage misses).
    pub missed: usize,
    /// Must-not-prove claims that nevertheless proved — **must be 0** (the
    /// generalized zero-hallucination invariant).
    pub false_proofs: usize,
    /// Must-not-prove claims correctly left unproven (abstained or heuristic).
    pub correctly_unproven: usize,
    /// Claims that vanished without an outcome — **always 0 by construction**.
    pub silent_drops: usize,
}

impl ParReport {
    /// Provable Answer Rate over the should-prove set (`0/0 = 1.0`, vacuously perfect).
    pub fn par(&self) -> f64 {
        let expected = self.proven_expected + self.missed;
        if expected == 0 {
            1.0
        } else {
            self.proven_expected as f64 / expected as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nusy_reasoner::{
        Answer, CompetenceEnvelope, DerivationTrace, Guarantee, ProofTrace, QueryShape,
    };
    use nusy_unify::Triple;

    fn t(s: &str, p: &str, o: &str) -> Triple {
        Triple::new(s, p, o)
    }

    fn envelope(pred: &str) -> CompetenceEnvelope {
        CompetenceEnvelope {
            shapes: vec![QueryShape {
                name: format!("{pred} shape"),
                predicates: vec![pred.to_string()],
            }],
        }
    }

    /// Toy symbolic reasoner: proves `at_risk(*, fall)` with a complete derivation,
    /// abstains otherwise.
    struct ToySymbolic;
    impl Reasoner for ToySymbolic {
        fn answer(&self, q: &Query) -> Answer {
            if q.goal.object == "fall" {
                Answer {
                    value: Some(q.goal.clone()),
                    proof: ProofTrace::Derivation(DerivationTrace::Derived {
                        conclusion: q.goal.clone(),
                        rule_id: "at-risk-fall".into(),
                        premises: vec![DerivationTrace::Axiom(t(
                            &q.goal.subject,
                            "has_condition",
                            "osteoporosis",
                        ))],
                    }),
                    provenance: vec!["chunk-1".into()],
                }
            } else {
                Answer::abstained()
            }
        }
        fn competence_envelope(&self) -> &CompetenceEnvelope {
            static ENV: std::sync::OnceLock<CompetenceEnvelope> = std::sync::OnceLock::new();
            ENV.get_or_init(|| envelope("at_risk"))
        }
        fn substrate(&self) -> Substrate {
            Substrate::Symbolic
        }
        fn guarantee(&self) -> Guarantee {
            Guarantee {
                sound: true,
                complete: false,
                probabilistic: false,
            }
        }
    }

    /// Toy neural reasoner: always answers `at_risk` queries with evidence.
    struct ToyNeural {
        agrees: bool,
    }
    impl Reasoner for ToyNeural {
        fn answer(&self, q: &Query) -> Answer {
            let value = if self.agrees {
                q.goal.clone()
            } else {
                t(&q.goal.subject, &q.goal.predicate, "stroke")
            };
            Answer {
                value: Some(value),
                proof: ProofTrace::Evidence {
                    confidence: 0.8,
                    why: vec!["pattern".into()],
                },
                provenance: vec![],
            }
        }
        fn competence_envelope(&self) -> &CompetenceEnvelope {
            static ENV: std::sync::OnceLock<CompetenceEnvelope> = std::sync::OnceLock::new();
            ENV.get_or_init(|| envelope("at_risk"))
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

    fn q(s: &str, p: &str, o: &str) -> Query {
        Query::new(t(s, p, o))
    }

    #[test]
    fn proven_symbolic_wins_and_is_classified_symbolic() {
        let mut router = ReasonerRouter::new();
        router.push(Box::new(ToySymbolic));
        let out = router.route(&q("p1", "at_risk", "fall"));
        assert!(out.is_proven());
        let v = out.verdict().unwrap();
        assert_eq!(v.class, AnswerClass::Symbolic);
        assert_eq!(v.reasoner_index, 0);
    }

    #[test]
    fn heuristic_answer_is_classified_neural_never_proven() {
        let mut router = ReasonerRouter::new();
        router.push(Box::new(ToyNeural { agrees: true }));
        let out = router.route(&q("p1", "at_risk", "fall"));
        assert!(!out.is_proven(), "evidence can never compute to Proven");
        assert_eq!(out.verdict().unwrap().class, AnswerClass::Neural);
    }

    #[test]
    fn cross_substrate_agreement_is_co_vote_and_provability_unchanged() {
        let mut router = ReasonerRouter::new();
        router.push(Box::new(ToySymbolic));
        router.push(Box::new(ToyNeural { agrees: true }));
        let out = router.route(&q("p1", "at_risk", "fall"));
        let v = out.verdict().unwrap();
        assert_eq!(v.class, AnswerClass::CoVote);
        assert_eq!(v.agreeing, 2);
        // Agreement tags, never upgrades: still Proven only because the symbolic side proved.
        assert_eq!(v.answer.provability(), Provability::Proven);

        // And when the symbolic side CANNOT prove (object != fall ⇒ it abstains),
        // a co-vote-less neural answer stays Heuristic.
        let out2 = router.route(&q("p1", "at_risk", "stroke"));
        let v2 = out2.verdict().unwrap();
        assert_ne!(v2.answer.provability(), Provability::Proven);
    }

    #[test]
    fn disagreeing_substrates_do_not_co_vote() {
        let mut router = ReasonerRouter::new();
        router.push(Box::new(ToySymbolic));
        router.push(Box::new(ToyNeural { agrees: false }));
        let out = router.route(&q("p1", "at_risk", "fall"));
        let v = out.verdict().unwrap();
        assert_eq!(v.class, AnswerClass::Symbolic);
        assert_eq!(v.agreeing, 1);
    }

    #[test]
    fn abstention_is_loud_with_reason_and_counts() {
        let mut router = ReasonerRouter::new();
        router.push(Box::new(ToySymbolic));

        // Covered predicate, but unprovable claim and no neural fallback → all abstained.
        match router.route(&q("p1", "at_risk", "stroke")) {
            RouteOutcome::Abstained {
                reason, consulted, ..
            } => {
                assert_eq!(consulted, 1);
                assert!(reason.contains("abstained"), "{reason}");
            }
            other => panic!("expected loud abstention, got {other:?}"),
        }

        // Foreign predicate → nothing covers it; still loud, still reasoned.
        match router.route(&q("p1", "weather_today", "sunny")) {
            RouteOutcome::Abstained {
                reason, consulted, ..
            } => {
                assert_eq!(consulted, 0);
                assert!(reason.contains("no reasoner covers"), "{reason}");
            }
            other => panic!("expected loud abstention, got {other:?}"),
        }
    }

    #[test]
    fn par_battery_tallies_and_zero_false_proofs() {
        let mut router = ReasonerRouter::new();
        router.push(Box::new(ToySymbolic));
        let panel = vec![
            (q("p1", "at_risk", "fall"), true),         // provable, expected
            (q("p2", "at_risk", "fall"), true),         // provable, expected
            (q("p1", "at_risk", "stroke"), false),      // must NOT prove
            (q("p1", "weather_today", "sunny"), false), // uncovered, must not prove
        ];
        let report = router.par(&panel);
        assert_eq!(report.proven_expected, 2);
        assert_eq!(report.missed, 0);
        assert_eq!(report.false_proofs, 0);
        assert_eq!(report.correctly_unproven, 2);
        assert_eq!(report.silent_drops, 0);
        assert!((report.par() - 1.0).abs() < 1e-9);
    }
}
