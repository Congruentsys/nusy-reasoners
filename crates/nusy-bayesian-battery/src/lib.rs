//! # nusy-bayesian-battery — VY-Bayes acceptance battery (EX-4819, VY-Bayes E4)
//!
//! The measured acceptance for the Bayesian stack (E1 engine, E2 Reasoner conformance, E3
//! abduction-rank upgrade). Four checks, all against **hand-computed** golds (engine-generated
//! golds would be circular — the EX-4819 constraint):
//!
//! 1. **GRADE-aggregation fidelity** — worked GRADE examples (base + components → overall level)
//!    computed by hand; the [`RatingSystem::grade`] engine must match each.
//! 2. **Bayesian posterior fidelity** — textbook posteriors (e.g. the base-rate fallacy:
//!    1% prevalence + a 99%/5% test → P(disease|+) = 1/6) computed by hand; the
//!    [`BayesianUpdate`] engine must match within tolerance.
//! 3. **Abduction top-1 accuracy (Bayesian vs parsimony)** — diagnostic scenarios with a known
//!    gold explanation, ranked by both the parsimony [`Ranker`] and the [`BayesianRanker`];
//!    we record each ranker's top-1 accuracy and the delta (the E3 upgrade must not regress).
//! 4. **Proof-purity** — *zero probabilistic links in proven traces*: every Bayesian/abductive
//!    answer is `Heuristic` (or `Abstained`), **never `Proven`**. A probability is never
//!    laundered into a proof — the load-bearing invariant of the whole reasoner contract.
//!
//! Fully symbolic / deterministic — a CI-permanent `cargo test`; the [`run`] report feeds the
//! Bayesian EXPR + eval JSON. **Fully symbolic; no LLM/GPU.**

use nusy_abduction::GraphCandidates;
use nusy_abduction_rank::{Abducer, BayesianRanker, Rank, Ranker};
use nusy_abduction_test::TestStage;
use nusy_bayesian::{BayesianUpdate, Hypothesis, Likelihood, RatingSystem};
use nusy_forward_chain::IdRule;
use nusy_reasoner::{CompetenceEnvelope, Provability, Query, QueryShape, Reasoner};
use nusy_unify::{Rule, Triple, TriplePattern};

const TOL: f64 = 1e-3;

// ── 1. GRADE-aggregation fidelity (hand-computed) ────────────────────────────────────────────

/// One GRADE worked example: base certainty + components → the hand-computed overall level.
pub struct GradeCase {
    pub id: &'static str,
    pub base: &'static str,
    pub components: Vec<(&'static str, &'static str)>, // (component id, rating)
    pub expected_level: &'static str,                  // HAND-computed, not engine-generated
}

/// GRADE worked examples with levels worked out by hand from the standard step rule
/// (serious = −1 step, very-serious = −2, large-effect = +1; clamp to the 4-level scale).
pub fn grade_cases() -> Vec<GradeCase> {
    vec![
        GradeCase {
            id: "two-serious-downgrades",
            base: "high",
            components: vec![("rob", "serious"), ("imp", "serious")],
            expected_level: "low", // high −2 = low
        },
        GradeCase {
            id: "clamp-at-worst",
            base: "high",
            components: vec![("a", "very-serious"), ("b", "very-serious")],
            expected_level: "very-low", // high −4, clamped to very-low
        },
        GradeCase {
            id: "observational-upgrade",
            base: "low",
            components: vec![("eff", "large-effect")],
            expected_level: "moderate", // low +1 = moderate
        },
        GradeCase {
            id: "no-change",
            base: "moderate",
            components: vec![("x", "none"), ("y", "not-serious")],
            expected_level: "moderate", // 0 net steps
        },
    ]
}

// ── 2. Bayesian posterior fidelity (hand-computed) ───────────────────────────────────────────

/// One posterior worked example: a model + evidence + the hand-computed posterior for one
/// hypothesis.
pub struct PosteriorCase {
    pub id: &'static str,
    pub hypotheses: Vec<(&'static str, f64)>, // (id, prior weight)
    pub evidence: Vec<(&'static str, Vec<(&'static str, f64)>)>, // (evidence id, [(hyp, P(E|H))])
    pub target: &'static str,
    pub expected_posterior: f64, // HAND-computed
}

pub fn posterior_cases() -> Vec<PosteriorCase> {
    vec![
        // Base-rate fallacy: 1% prevalence, sens 0.99, fp 0.05.
        // P(d|+) = .01*.99 / (.01*.99 + .99*.05) = .0099/.0594 = 1/6 ≈ 0.16667.
        PosteriorCase {
            id: "base-rate-disease-test",
            hypotheses: vec![("disease", 0.01), ("healthy", 0.99)],
            evidence: vec![("test+", vec![("disease", 0.99), ("healthy", 0.05)])],
            target: "disease",
            expected_posterior: 1.0 / 6.0,
        },
        // Two independent positive tests sharpen it: after one, P(d)=1/6; second test
        // P(d|++) = (1/6)*.99 / ((1/6)*.99 + (5/6)*.05) = .165/.20667 = 0.79839.
        PosteriorCase {
            id: "two-positive-tests",
            hypotheses: vec![("disease", 0.01), ("healthy", 0.99)],
            evidence: vec![
                ("t1", vec![("disease", 0.99), ("healthy", 0.05)]),
                ("t2", vec![("disease", 0.99), ("healthy", 0.05)]),
            ],
            target: "disease",
            expected_posterior: 0.798_387,
        },
        // Uniform 3-way prior, one discriminating finding: P(h)=1/3 each;
        // likelihoods .1/.6/.3 → posterior_h2 = .6/(.1+.6+.3) = 0.6.
        PosteriorCase {
            id: "three-way-discriminating",
            hypotheses: vec![("h1", 1.0), ("h2", 1.0), ("h3", 1.0)],
            evidence: vec![("finding", vec![("h1", 0.1), ("h2", 0.6), ("h3", 0.3)])],
            target: "h2",
            expected_posterior: 0.6,
        },
    ]
}

fn run_posterior(case: &PosteriorCase) -> f64 {
    let model = BayesianUpdate::new(
        case.hypotheses
            .iter()
            .map(|(id, w)| Hypothesis::new(*id, *w)),
    )
    .unwrap();
    let evidence: Vec<Likelihood> = case
        .evidence
        .iter()
        .map(|(eid, ls)| {
            let mut l = Likelihood::new(*eid);
            for (h, p) in ls {
                l = l.with(*h, *p);
            }
            l
        })
        .collect();
    model.observe(&evidence).prob(case.target)
}

// ── 3. Abduction top-1 accuracy (Bayesian vs parsimony) ──────────────────────────────────────

/// A diagnostic abduction scenario with a known gold explanation (principal atom).
pub struct AbductionCase {
    pub id: &'static str,
    pub observation: Triple,
    pub rules: Vec<IdRule>,
    pub gold_principal: Triple,
}

fn idrule(id: &str, body: Vec<TriplePattern>, head: Vec<TriplePattern>) -> IdRule {
    IdRule::new(id, Rule::new(body, head))
}

pub fn abduction_cases() -> Vec<AbductionCase> {
    vec![
        // Clear winner: one rule reverses fever → infection.
        AbductionCase {
            id: "clear-infection",
            observation: Triple::new("patient", "shows", "fever"),
            rules: vec![idrule(
                "fever-if-infection",
                vec![TriplePattern::parse("?p", "has", "infection")],
                vec![TriplePattern::parse("?p", "shows", "fever")],
            )],
            gold_principal: Triple::new("patient", "has", "infection"),
        },
        // Parsimony winner: 1-atom (rain) beats 2-atom (sprinkler+valve) for wet.
        AbductionCase {
            id: "parsimony-rain",
            observation: Triple::new("lawn", "is", "wet"),
            rules: vec![
                idrule(
                    "wet-if-rain",
                    vec![TriplePattern::parse("?x", "had", "rain")],
                    vec![TriplePattern::parse("?x", "is", "wet")],
                ),
                idrule(
                    "wet-if-sprinkler",
                    vec![
                        TriplePattern::parse("?x", "had", "sprinkler"),
                        TriplePattern::parse("?x", "had", "valve_open"),
                    ],
                    vec![TriplePattern::parse("?x", "is", "wet")],
                ),
            ],
            gold_principal: Triple::new("lawn", "had", "rain"),
        },
    ]
}

fn env() -> CompetenceEnvelope {
    CompetenceEnvelope {
        shapes: vec![QueryShape {
            name: "abduction".into(),
            predicates: vec![],
        }],
    }
}

/// Top-1 hit for a case under a given ranker: does the Abducer's answer value equal the gold?
fn top1_hit<R: Rank>(case: &AbductionCase, ranker: R) -> bool {
    let abducer = Abducer::new(
        GraphCandidates::new(case.rules.clone(), 2),
        TestStage::new(case.rules.clone(), vec![]),
        ranker,
        env(),
    );
    abducer
        .answer(&Query::new(case.observation.clone()))
        .value
        .as_ref()
        == Some(&case.gold_principal)
}

// ── The report ───────────────────────────────────────────────────────────────────────────────

/// The acceptance result: fidelity counts + abduction accuracy + the proof-purity invariant.
#[derive(Debug, Clone, PartialEq)]
pub struct Report {
    pub grade_total: usize,
    pub grade_matched: usize,
    pub posterior_total: usize,
    pub posterior_matched: usize,
    pub abduction_total: usize,
    pub parsimony_top1: usize,
    pub bayesian_top1: usize,
    /// Number of answers that are simultaneously `Proven` AND probabilistic — must be 0.
    pub probabilistic_proven_links: usize,
}

impl Report {
    pub fn all_pass(&self) -> bool {
        self.grade_matched == self.grade_total
            && self.posterior_matched == self.posterior_total
            && self.bayesian_top1 == self.abduction_total
            && self.parsimony_top1 == self.abduction_total
            && self.probabilistic_proven_links == 0
    }
}

/// Run the whole battery and return the report.
pub fn run() -> Report {
    let grade = RatingSystem::grade();
    let grade_cases = grade_cases();
    let grade_matched = grade_cases
        .iter()
        .filter(|c| {
            grade
                .aggregate(c.base, c.components.iter().copied())
                .map(|a| a.level == c.expected_level)
                .unwrap_or(false)
        })
        .count();

    let posterior_cases = posterior_cases();
    let posterior_matched = posterior_cases
        .iter()
        .filter(|c| (run_posterior(c) - c.expected_posterior).abs() < TOL)
        .count();

    let abduction_cases = abduction_cases();
    let parsimony_top1 = abduction_cases
        .iter()
        .filter(|c| top1_hit(c, Ranker::uniform(&[], 0.5)))
        .count();
    let bayesian_top1 = abduction_cases
        .iter()
        .filter(|c| top1_hit(c, BayesianRanker::uniform(&[], 0.5)))
        .count();

    // Proof-purity: every abductive answer (Bayesian-ranked) must be Heuristic, never Proven.
    let probabilistic_proven_links = abduction_cases
        .iter()
        .filter(|c| {
            let abducer = Abducer::new(
                GraphCandidates::new(c.rules.clone(), 2),
                TestStage::new(c.rules.clone(), vec![]),
                BayesianRanker::uniform(&[], 0.5),
                env(),
            );
            abducer
                .answer(&Query::new(c.observation.clone()))
                .provability()
                == Provability::Proven
        })
        .count();

    Report {
        grade_total: grade_cases.len(),
        grade_matched,
        posterior_total: posterior_cases.len(),
        posterior_matched,
        abduction_total: abduction_cases.len(),
        parsimony_top1,
        bayesian_top1,
        probabilistic_proven_links,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grade_aggregation_matches_hand_computed() {
        let g = RatingSystem::grade();
        for c in grade_cases() {
            let got = g
                .aggregate(c.base, c.components.iter().copied())
                .unwrap()
                .level;
            assert_eq!(got, c.expected_level, "GRADE case {}", c.id);
        }
    }

    #[test]
    fn posteriors_match_hand_computed() {
        for c in posterior_cases() {
            let got = run_posterior(&c);
            assert!(
                (got - c.expected_posterior).abs() < TOL,
                "posterior case {}: got {got}, expected {}",
                c.id,
                c.expected_posterior
            );
        }
    }

    #[test]
    fn abduction_top1_bayesian_matches_parsimony_and_hits_gold() {
        // Both rankers select the gold on every case — the E3 Bayesian upgrade does not regress.
        for c in abduction_cases() {
            assert!(
                top1_hit(&c, Ranker::uniform(&[], 0.5)),
                "parsimony miss on {}",
                c.id
            );
            assert!(
                top1_hit(&c, BayesianRanker::uniform(&[], 0.5)),
                "bayesian miss on {}",
                c.id
            );
        }
    }

    #[test]
    fn proof_purity_no_probabilistic_answer_is_proven() {
        // Every Bayesian-ranked abductive answer is Heuristic — never laundered to Proven.
        for c in abduction_cases() {
            let abducer = Abducer::new(
                GraphCandidates::new(c.rules.clone(), 2),
                TestStage::new(c.rules.clone(), vec![]),
                BayesianRanker::uniform(&[], 0.5),
                env(),
            );
            let prov = abducer
                .answer(&Query::new(c.observation.clone()))
                .provability();
            assert_eq!(
                prov,
                Provability::Heuristic,
                "case {} must be Heuristic",
                c.id
            );
        }
    }

    #[test]
    fn report_all_pass() {
        let r = run();
        assert!(r.all_pass(), "battery report did not all-pass: {r:?}");
        assert_eq!(r.probabilistic_proven_links, 0);
    }
}
