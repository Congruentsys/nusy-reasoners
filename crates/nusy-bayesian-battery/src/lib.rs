//! # nusy-bayesian-battery — VY-Bayes acceptance battery (EX-4819, VY-Bayes E4)
//!
//! The measured acceptance for the Bayesian stack (E1 engine, E2 Reasoner conformance, E3
//! abduction-rank upgrade). Five checks, all against **hand-computed** golds (engine-generated
//! golds would be circular — the EX-4819 constraint):
//!
//! 1. **GRADE-aggregation fidelity** — worked GRADE examples (base + components → overall level)
//!    computed by hand; the [`RatingSystem::grade`] engine must match each.
//! 2. **Bayesian posterior fidelity** — textbook posteriors (e.g. the base-rate fallacy:
//!    1% prevalence + a 99%/5% test → P(disease|+) = 1/6) computed by hand; the
//!    [`BayesianUpdate`] engine must match within tolerance.
//! 3. **Abduction non-regression** — easy single-winner scenarios where both the parsimony
//!    [`Ranker`] and the [`BayesianRanker`] must hit the gold (the E3 upgrade must not regress).
//! 4. **Abduction improvement (CH-4831)** — *discriminating* scenarios engineered so the rankers
//!    diverge: a high-confidence multi-atom explanation that parsimony's atom-count misses but the
//!    Bayesian posterior (`mean_prior × 2^(−assumed_count)`) selects. Bayesian must HIT, parsimony
//!    must MISS — the empirical content of H-4823's "improves over parsimony" clause (delta > 0).
//! 5. **Proof-purity** — *zero probabilistic links in proven traces*: every Bayesian/abductive
//!    answer is `Heuristic` (or `Abstained`), **never `Proven`**. A probability is never
//!    laundered into a proof — the load-bearing invariant of the whole reasoner contract.
//!
//! Fully symbolic / deterministic — a CI-permanent `cargo test`; the [`run`] report feeds the
//! Bayesian EXPR + eval JSON. **Fully symbolic; no LLM/GPU.**

use std::collections::HashMap;

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

// ── 4. Discriminating case: Bayesian BEATS parsimony (CH-4831) ───────────────────────────────

/// The prior-map key the rankers use — mirrors `nusy_abduction_rank`'s (private) `triple_key`,
/// the documented seam for the caller's prior table (`"{s}|{p}|{o}"`). (A `pub triple_key` or a
/// `PriorTable` builder upstream would remove this duplication — noted as a small follow-up.)
fn prior_key(t: &Triple) -> String {
    format!("{}|{}|{}", t.subject, t.predicate, t.object)
}

fn priors_map(prs: &[(Triple, f64)]) -> HashMap<String, f64> {
    prs.iter().map(|(t, p)| (prior_key(t), *p)).collect()
}

/// A case engineered so parsimony and Bayesian **diverge**, so the battery can demonstrate
/// H-4823's *improves* clause — not merely non-regression. Parsimony is lexicographic (atom
/// count strictly dominates), so it always takes the fewest-atom explanation. The Bayesian
/// ranker scores `mean_prior × 2^(−assumed_count)`, so a high-confidence multi-atom explanation
/// can overcome one extra atom. When the high-confidence explanation is the gold, Bayesian hits
/// and parsimony misses.
pub struct DiscriminatingCase {
    pub id: &'static str,
    pub observation: Triple,
    pub rules: Vec<IdRule>,
    /// Per-atom prior confidence (read product-side off the store's confidence column).
    pub priors: Vec<(Triple, f64)>,
    pub default_prior: f64,
    /// The atom parsimony (wrongly) selects — the fewest-atom explanation's principal.
    pub parsimony_pick: Triple,
    /// The gold explanation's atoms; the Bayesian principal must be one of these (robust to the
    /// candidate generator's within-body atom ordering).
    pub bayesian_gold: Vec<Triple>,
}

pub fn discriminating_cases() -> Vec<DiscriminatingCase> {
    vec![
        // Jaundice: hemolysis is a 1-atom explanation but unlikely in this patient (prior 0.05);
        // obstruction is a 2-atom explanation (gallstone + blocked duct) both highly likely (0.95).
        // Parsimony picks the single atom (hemolysis); Bayesian's posterior (0.95·¼=0.2375 vs
        // 0.05·½=0.025) picks the obstruction — the gold.
        DiscriminatingCase {
            id: "jaundice-obstruction-beats-hemolysis",
            observation: Triple::new("patient", "shows", "jaundice"),
            rules: vec![
                idrule(
                    "jaundice-if-hemolysis",
                    vec![TriplePattern::parse("?p", "has", "hemolysis")],
                    vec![TriplePattern::parse("?p", "shows", "jaundice")],
                ),
                idrule(
                    "jaundice-if-obstruction",
                    vec![
                        TriplePattern::parse("?p", "has", "gallstone"),
                        TriplePattern::parse("?p", "has", "bile_duct_blocked"),
                    ],
                    vec![TriplePattern::parse("?p", "shows", "jaundice")],
                ),
            ],
            priors: vec![
                (Triple::new("patient", "has", "hemolysis"), 0.05),
                (Triple::new("patient", "has", "gallstone"), 0.95),
                (Triple::new("patient", "has", "bile_duct_blocked"), 0.95),
            ],
            default_prior: 0.5,
            parsimony_pick: Triple::new("patient", "has", "hemolysis"),
            bayesian_gold: vec![
                Triple::new("patient", "has", "gallstone"),
                Triple::new("patient", "has", "bile_duct_blocked"),
            ],
        },
    ]
}

/// The principal atom the Abducer returns under `ranker` for `observation`, with the given
/// priors wired into the ranker (the product-side confidence signal).
fn principal_with<R: Rank>(case: &DiscriminatingCase, ranker: R) -> Option<Triple> {
    Abducer::new(
        GraphCandidates::new(case.rules.clone(), 2),
        TestStage::new(case.rules.clone(), vec![]),
        ranker,
        env(),
    )
    .answer(&Query::new(case.observation.clone()))
    .value
}

/// (parsimony hit gold?, bayesian hit gold?) for a discriminating case. The discriminating
/// property is: parsimony MISSES (picks its low-prior single atom) and Bayesian HITS (picks an
/// atom of the high-confidence gold explanation).
fn discriminating_outcome(case: &DiscriminatingCase) -> (bool, bool) {
    let priors = priors_map(&case.priors);
    let parsimony = principal_with(case, Ranker::new(&[], priors.clone(), case.default_prior));
    let bayesian = principal_with(case, BayesianRanker::new(&[], priors, case.default_prior));
    let parsimony_hit = parsimony
        .as_ref()
        .is_some_and(|t| case.bayesian_gold.contains(t));
    let bayesian_hit = bayesian
        .as_ref()
        .is_some_and(|t| case.bayesian_gold.contains(t));
    (parsimony_hit, bayesian_hit)
}

// ── The report ───────────────────────────────────────────────────────────────────────────────

/// The acceptance result: fidelity counts + abduction accuracy + the proof-purity invariant.
#[derive(Debug, Clone, PartialEq)]
pub struct Report {
    pub grade_total: usize,
    pub grade_matched: usize,
    pub posterior_total: usize,
    pub posterior_matched: usize,
    // Non-regression (easy single-winner cases): BOTH rankers must hit gold.
    pub abduction_total: usize,
    pub parsimony_top1: usize,
    pub bayesian_top1: usize,
    // Discriminating cases (CH-4831): Bayesian must hit gold, parsimony must MISS — this is the
    // empirical content of H-4823's "improves over parsimony" clause (delta > 0).
    pub discrim_total: usize,
    pub discrim_bayesian_hits: usize,
    pub discrim_parsimony_hits: usize,
    /// Number of answers that are simultaneously `Proven` AND probabilistic — must be 0.
    pub probabilistic_proven_links: usize,
}

impl Report {
    pub fn all_pass(&self) -> bool {
        self.grade_matched == self.grade_total
            && self.posterior_matched == self.posterior_total
            // Non-regression: both rankers hit the easy cases.
            && self.bayesian_top1 == self.abduction_total
            && self.parsimony_top1 == self.abduction_total
            // Improvement: Bayesian hits every discriminating gold, parsimony misses every one.
            && self.discrim_total > 0
            && self.discrim_bayesian_hits == self.discrim_total
            && self.discrim_parsimony_hits == 0
            && self.probabilistic_proven_links == 0
    }

    /// The improvement delta this battery demonstrates: Bayesian top-1 hits minus parsimony's,
    /// over the discriminating cases. > 0 means the E3 Bayesian upgrade strictly improves ranking.
    pub fn discrim_delta(&self) -> i64 {
        self.discrim_bayesian_hits as i64 - self.discrim_parsimony_hits as i64
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

    // Discriminating cases (CH-4831): Bayesian should hit, parsimony should miss.
    let discrim = discriminating_cases();
    let (mut discrim_parsimony_hits, mut discrim_bayesian_hits) = (0usize, 0usize);
    for c in &discrim {
        let (p_hit, b_hit) = discriminating_outcome(c);
        discrim_parsimony_hits += p_hit as usize;
        discrim_bayesian_hits += b_hit as usize;
    }

    Report {
        grade_total: grade_cases.len(),
        grade_matched,
        posterior_total: posterior_cases.len(),
        posterior_matched,
        abduction_total: abduction_cases.len(),
        parsimony_top1,
        bayesian_top1,
        discrim_total: discrim.len(),
        discrim_bayesian_hits,
        discrim_parsimony_hits,
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
    fn bayesian_beats_parsimony_on_discriminating_case() {
        // H-4823 "improves" clause: on an engineered divergence the Bayesian posterior picks the
        // high-confidence gold explanation that parsimony's atom-count misses.
        for c in discriminating_cases() {
            let (p_hit, b_hit) = discriminating_outcome(&c);
            assert!(
                !p_hit,
                "parsimony should MISS the discriminating gold on {}",
                c.id
            );
            assert!(
                b_hit,
                "bayesian should HIT the discriminating gold on {}",
                c.id
            );
        }
    }

    #[test]
    fn discriminating_winner_is_still_heuristic() {
        // Even when the Bayesian posterior wins a divergence, the answer is never Proven.
        for c in discriminating_cases() {
            let priors = priors_map(&c.priors);
            let abducer = Abducer::new(
                GraphCandidates::new(c.rules.clone(), 2),
                TestStage::new(c.rules.clone(), vec![]),
                BayesianRanker::new(&[], priors, c.default_prior),
                env(),
            );
            assert_eq!(
                abducer
                    .answer(&Query::new(c.observation.clone()))
                    .provability(),
                Provability::Heuristic,
                "case {} must stay Heuristic",
                c.id
            );
        }
    }

    #[test]
    fn report_all_pass() {
        let r = run();
        assert!(r.all_pass(), "battery report did not all-pass: {r:?}");
        assert_eq!(r.probabilistic_proven_links, 0);
        // The improvement is real, not just non-regression: Bayesian strictly out-hits parsimony
        // on the discriminating cases.
        assert!(
            r.discrim_delta() > 0,
            "expected a positive Bayesian-vs-parsimony delta, got {}",
            r.discrim_delta()
        );
    }
}
