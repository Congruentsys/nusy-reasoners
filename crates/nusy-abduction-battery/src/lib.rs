//! # nusy-abduction-battery — VY-E acceptance battery (EX-4778, VY-E E4)
//!
//! The abductive stack's acceptance test. It assembles the full VY-E pipeline —
//! **generate** ([`GraphCandidates`](nusy_abduction::GraphCandidates), E1) → **test**
//! ([`TestStage`](nusy_abduction_test::TestStage), E2) → **rank** ([`Abducer`] over
//! [`Ranker`](nusy_abduction_rank::Ranker), E3) — and runs a fixture set that exercises
//! the four behaviours abduction must get right:
//!
//! - **clear winner** — one gold explanation re-derives the observation and is the answer;
//! - **tie broken by parsimony** — two valid explanations of the *same* observation, the
//!   one assuming fewer facts ranks first (Occam, the E3 contract);
//! - **contraindicated distractor** — a rival explanation that *would* re-derive the
//!   observation is killed by the E2 defeater screen (an explicit `contraindicates`
//!   negative fires against its assumed atom), so the surviving gold wins;
//! - **no valid explanation → abstain** — an observation no rule reverses yields **no**
//!   survivor, and the Abducer abstains *loudly* ([`Provability::Abstained`]) rather than
//!   inventing a guess (the abduction analogue of the gate's zero-hallucination invariant).
//!
//! Every non-abstaining answer carries a **composite trace that threads all three stages**
//! — the proposed explanation (generate), its `H ⊢ E` completeness (test), and its
//! assumed-atom count (rank) all appear in the answer's evidence. The Abducer reports
//! [`Provability::Heuristic`] on every answer, never `Proven`: abduction infers the best
//! explanation, it does not prove the explanation true (see `nusy-abduction-rank`).
//!
//! **Fully symbolic in CI** (EX-4778 constraint): the candidate source is the symbolic
//! [`GraphCandidates`] rule-reverser — no neural proposer, no LLM, no GPU — so the battery
//! is a deterministic `cargo test` and a CI-permanent guard (the `jnc8_equivalence`
//! pattern). The H-item it feeds is recorded, **not** auto-closed (Captain guardrail #6).

use nusy_abduction::GraphCandidates;
use nusy_abduction_rank::{Abducer, Ranker};
use nusy_abduction_test::TestStage;
use nusy_forward_chain::IdRule;
use nusy_reasoner::{CompetenceEnvelope, QueryShape};
use nusy_unify::{Rule, Triple, TriplePattern};

/// Backward-chaining depth for the symbolic candidate source — 2 covers the multi-hop
/// explanations in the fixtures while staying bounded.
const ABDUCE_DEPTH: usize = 2;
/// The prior used when the store has no confidence signal (battery runs parsimony-only).
const DEFAULT_PRIOR: f64 = 0.5;

fn idrule(id: &str, body: Vec<TriplePattern>, head: Vec<TriplePattern>) -> IdRule {
    IdRule::new(id, Rule::new(body, head))
}

fn wildcard_envelope() -> CompetenceEnvelope {
    CompetenceEnvelope {
        shapes: vec![QueryShape {
            name: "abduction".into(),
            predicates: vec![], // any predicate — abduction is asked to explain anything
        }],
    }
}

/// One acceptance case: an observation, the rules + base graph it is abduced against, and
/// the gold outcome.
#[derive(Debug, Clone)]
pub struct Case {
    /// Stable case id.
    pub id: &'static str,
    /// The observation to explain.
    pub observation: Triple,
    /// The rule set the candidate source reverses and the test stage forward-chains.
    pub rules: Vec<IdRule>,
    /// The graph slice the hypothesis is assumed on top of (kept minimal so `H` does the work).
    pub base_facts: Vec<Triple>,
    /// Gold: the **principal atom** of the explanation that should rank first, or `None`
    /// when the case must abstain (no valid explanation).
    pub gold_principal: Option<Triple>,
}

/// Assemble the Abducer for a case: generate (symbolic) → test → rank (parsimony + prior).
pub fn abducer_for(case: &Case) -> Abducer<GraphCandidates> {
    let source = GraphCandidates::new(case.rules.clone(), ABDUCE_DEPTH);
    let test = TestStage::new(case.rules.clone(), case.base_facts.clone());
    let ranker = Ranker::uniform(&case.base_facts, DEFAULT_PRIOR);
    Abducer::new(source, test, ranker, wildcard_envelope())
}

/// The acceptance fixture set (≥4): the behaviours abduction must get right.
pub fn cases() -> Vec<Case> {
    vec![
        // 1. CLEAR WINNER — exactly one rule reverses the observation; its body is the
        //    gold explanation and the answer.
        Case {
            id: "clear-winner",
            observation: Triple::new("alarm", "state", "ringing"),
            rules: vec![idrule(
                "ring-if-smoke",
                vec![TriplePattern::parse("?x", "detects", "smoke")],
                vec![TriplePattern::parse("?x", "state", "ringing")],
            )],
            base_facts: vec![],
            gold_principal: Some(Triple::new("alarm", "detects", "smoke")),
        },
        // 2. TIE BROKEN BY PARSIMONY — two rules explain the SAME observation: a 1-atom
        //    explanation and a 2-atom one. The parsimonious (1-atom) gold ranks first.
        Case {
            id: "parsimony-tiebreak",
            observation: Triple::new("lawn", "is", "wet"),
            rules: vec![
                // 1 assumed atom: it rained.
                idrule(
                    "wet-if-rain",
                    vec![TriplePattern::parse("?x", "had", "rain")],
                    vec![TriplePattern::parse("?x", "is", "wet")],
                ),
                // 2 assumed atoms: the sprinkler was on AND the valve was open — less parsimonious.
                idrule(
                    "wet-if-sprinkler",
                    vec![
                        TriplePattern::parse("?x", "had", "sprinkler"),
                        TriplePattern::parse("?x", "had", "open_valve"),
                    ],
                    vec![TriplePattern::parse("?x", "is", "wet")],
                ),
            ],
            base_facts: vec![],
            gold_principal: Some(Triple::new("lawn", "had", "rain")),
        },
        // 3. CONTRAINDICATED DISTRACTOR — two rules explain the observation, but the rival's
        //    assumed atom triggers an explicit `contraindicates` negative (E2 defeater screen),
        //    so it is rejected and the surviving gold wins.
        Case {
            id: "contraindicated-distractor",
            observation: Triple::new("patient", "shows", "fever"),
            rules: vec![
                // Gold: an infection explains the fever.
                idrule(
                    "fever-if-infection",
                    vec![TriplePattern::parse("?p", "has", "infection")],
                    vec![TriplePattern::parse("?p", "shows", "fever")],
                ),
                // Distractor: a drug reaction would also explain it…
                idrule(
                    "fever-if-drug",
                    vec![TriplePattern::parse("?p", "took", "baddrug")],
                    vec![TriplePattern::parse("?p", "shows", "fever")],
                ),
                // …but assuming the drug fires an explicit negative against `baddrug` (the
                // object of the distractor's assumed atom) — the defeater screen kills it.
                idrule(
                    "drug-contra",
                    vec![TriplePattern::parse("?p", "took", "baddrug")],
                    vec![TriplePattern::parse(
                        "allergy_record",
                        "contraindicates",
                        "baddrug",
                    )],
                ),
            ],
            base_facts: vec![],
            gold_principal: Some(Triple::new("patient", "has", "infection")),
        },
        // 4. NO VALID EXPLANATION → ABSTAIN — no rule concludes the observation, so no
        //    candidate survives and the Abducer abstains loudly (no invented guess).
        Case {
            id: "no-explanation-abstains",
            observation: Triple::new("rock", "is", "igneous"),
            rules: vec![idrule(
                "ring-if-smoke",
                vec![TriplePattern::parse("?x", "detects", "smoke")],
                vec![TriplePattern::parse("?x", "state", "ringing")],
            )],
            base_facts: vec![],
            gold_principal: None,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use nusy_reasoner::{ProofTrace, Provability, Query, Reasoner};

    /// On every case with a gold explanation, the most parsimonious surviving explanation
    /// ranks first and IS the gold — and the answer reports Heuristic (never Proven).
    #[test]
    fn gold_ranks_first_on_every_case() {
        for case in cases() {
            let abducer = abducer_for(&case);
            let answer = abducer.answer(&Query::new(case.observation.clone()));
            match &case.gold_principal {
                Some(gold) => {
                    // The top-ranked explanation's principal atom is the gold.
                    let ranked = abducer.explain(&case.observation);
                    assert!(
                        !ranked.is_empty(),
                        "case {}: expected a surviving explanation",
                        case.id
                    );
                    let top = ranked[0]
                        .hypothesis
                        .ground()
                        .expect("survivors are ground")
                        .first()
                        .cloned()
                        .expect("explanation has a principal atom");
                    assert_eq!(&top, gold, "case {}: wrong top explanation", case.id);
                    assert_eq!(
                        answer.value.as_ref(),
                        Some(gold),
                        "case {}: answer value is the gold explanation",
                        case.id
                    );
                    assert_eq!(
                        answer.provability(),
                        Provability::Heuristic,
                        "case {}: abduction is Heuristic, never Proven",
                        case.id
                    );
                }
                None => { /* abstention asserted in its own test */ }
            }
        }
    }

    /// The no-explanation case abstains LOUDLY: no value, Abstained provability — never a
    /// fabricated explanation. (The abductive analogue of zero-hallucination.)
    #[test]
    fn no_explanation_case_abstains_loudly() {
        let case = cases()
            .into_iter()
            .find(|c| c.id == "no-explanation-abstains")
            .unwrap();
        let abducer = abducer_for(&case);
        let answer = abducer.answer(&Query::new(case.observation.clone()));
        assert_eq!(answer.value, None, "no explanation → no value");
        assert_eq!(answer.provability(), Provability::Abstained);
        assert!(
            abducer.explain(&case.observation).is_empty(),
            "no candidate survives"
        );
    }

    /// The contraindicated distractor is screened out: the rival explanation that would
    /// also re-derive the observation does NOT win — the gold (uncontraindicated) does.
    #[test]
    fn contraindicated_distractor_is_screened_out() {
        let case = cases()
            .into_iter()
            .find(|c| c.id == "contraindicated-distractor")
            .unwrap();
        let abducer = abducer_for(&case);
        let ranked = abducer.explain(&case.observation);
        // Exactly the gold survives; the drug-reaction distractor is rejected by the screen.
        assert_eq!(
            ranked.len(),
            1,
            "only the uncontraindicated explanation survives"
        );
        let survivor_atoms = ranked[0].hypothesis.ground().unwrap();
        assert!(
            survivor_atoms.contains(&Triple::new("patient", "has", "infection")),
            "the surviving explanation is the infection, not the drug"
        );
        assert!(
            !survivor_atoms.contains(&Triple::new("patient", "took", "baddrug")),
            "the contraindicated drug explanation must not survive"
        );
    }

    /// Every non-abstaining answer's composite trace threads generate→test→rank: the
    /// proposed rule chain (generate), the H⊢E completeness (test), and the assumed-atom
    /// count (rank) all appear in the evidence.
    #[test]
    fn composite_trace_threads_propose_test_rank() {
        for case in cases() {
            if case.gold_principal.is_none() {
                continue;
            }
            let abducer = abducer_for(&case);
            let answer = abducer.answer(&Query::new(case.observation.clone()));
            let ProofTrace::Evidence { why, .. } = &answer.proof else {
                panic!("case {}: expected an Evidence trace", case.id);
            };
            let joined = why.join(" | ");
            assert!(
                joined.contains("rule_chain="),
                "case {}: generate-stage tag (rule_chain) missing",
                case.id
            );
            assert!(
                joined.contains("H⊢E="),
                "case {}: test-stage tag (H⊢E) missing",
                case.id
            );
            assert!(
                joined.contains("assumed_count="),
                "case {}: rank-stage tag (assumed_count) missing",
                case.id
            );
        }
    }

    /// PAR-style guard: across the whole battery, the number of fabricated explanations is
    /// zero — every case either returns its gold or abstains, never a non-gold value.
    #[test]
    fn zero_fabricated_explanations() {
        let mut fabricated = 0;
        for case in cases() {
            let abducer = abducer_for(&case);
            let answer = abducer.answer(&Query::new(case.observation.clone()));
            match (&case.gold_principal, &answer.value) {
                (Some(gold), Some(v)) if v == gold => {}
                (None, None) => {}
                _ => fabricated += 1,
            }
        }
        assert_eq!(fabricated, 0, "no case may return a non-gold explanation");
    }
}
