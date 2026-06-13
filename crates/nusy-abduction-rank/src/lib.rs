//! # Abductive ranking + the composed Abducer (VY-E E3)
//!
//! E1 (`nusy-abduction`) *generates* candidate explanations for an observation; E2
//! (`nusy-abduction-test`) *tests* them, keeping only those that re-derive the
//! observation and survive the defeater screen, each carrying a complete `H ⊢ E`
//! derivation. E3 is the last stage: **rank the survivors** and **compose the three
//! stages into a single [`Reasoner`]** — the [`Abducer`].
//!
//! ## Ranking (Phase 1)
//!
//! [`Ranker`] orders survivors **best-first** by two terms, in order:
//!
//! 1. **Parsimony** — fewer *assumed* atoms wins. An atom already in the base graph
//!    is not an assumption; only atoms the explanation has to *add* count. (Occam:
//!    the explanation that posits least, wins.)
//! 2. **Prior confidence** — higher mean prior over the assumed atoms wins. Priors
//!    come from the caller (product-side, read off the store's `confidence` column);
//!    unknown atoms take a configurable default. This crate stays arrow-free, so it
//!    never reads the column itself — it takes a `prior` map.
//!
//! Ties (equal parsimony *and* equal prior) break on a **canonical string key** so
//! the ordering is **total and deterministic** — the same survivors always rank in
//! the same order, run to run. Bayesian scoring (likelihoods, marginalisation) is
//! deferred to Wave 2; parsimony + prior is the E3 contract.
//!
//! ## The Abducer (Phase 2)
//!
//! [`Abducer`] implements [`Reasoner`]: `answer(query)` runs
//! generate → test → rank and returns the **top explanation** as the answer.
//!
//! **Provability is `Heuristic` by construction, and that is the point.** Abduction
//! is inference-to-the-best-explanation: even when a survivor carries a *complete*
//! `H ⊢ E` derivation (so it is *proven that H entails E*), the **assumption of H
//! itself is not proven** — a better explanation could displace it. The honest
//! provability of the abductive conclusion is therefore
//! `provability_min(Proven-that-H⊢E, Heuristic-that-H-holds) = Heuristic`. The
//! Abducer encodes this by emitting a [`ProofTrace::Evidence`] trace, which the
//! guarantee invariant ([`Answer::provability`]) computes as
//! [`Provability::Heuristic`] — never `Proven`. (The one case where an abductive
//! atom *could* be `Proven` is when `H` is independently derivable from the base
//! graph — but then it is *deduction*, answered by the symbolic reasoner, not
//! abduction. The Abducer never mints `Proven`.)

use std::collections::{HashMap, HashSet};

use nusy_abduction::{CandidateSource, Hypothesis};
use nusy_abduction_test::{Survivor, TestStage};
use nusy_reasoner::{
    Answer, CompetenceEnvelope, DerivationTrace, Guarantee, ProofTrace, Provability, Query,
    Reasoner, Substrate, compose::provability_min,
};
use nusy_unify::Triple;

/// Canonical string key for a triple — the seam between the arrow-free engine types
/// and the caller's prior map, and the deterministic tie-break key.
fn triple_key(t: &Triple) -> String {
    format!("{}|{}|{}", t.subject, t.predicate, t.object)
}

/// The score breakdown for one ranked explanation. Public so callers (and tests) can
/// inspect *why* an explanation ranked where it did — the ranking is auditable, not a
/// black box.
#[derive(Debug, Clone, PartialEq)]
pub struct RankScore {
    /// Number of explanation atoms **not** already in the base graph — the count of
    /// genuine assumptions. Lower is more parsimonious (primary sort key, ascending).
    pub assumed_count: usize,
    /// Mean prior confidence over the assumed atoms, in `[0, 1]` (secondary sort key,
    /// descending). Atoms with no known prior take the ranker's `default_prior`.
    pub prior: f64,
    /// Canonical string of the explanation's atoms — the deterministic tie-break
    /// (ascending) when parsimony and prior are equal.
    pub tiebreak: String,
    /// EX-4818: the Bayesian **posterior** probability over the surviving explanations,
    /// set by [`BayesianRanker`]; `None` for the parsimony+prior [`Ranker`]. When present it
    /// is the calibrated rank-stage confidence the [`Abducer`] reports (preferred over
    /// `prior`).
    pub posterior: Option<f64>,
}

/// A survivor with its computed score. [`Ranker::rank`] returns these best-first.
#[derive(Debug, Clone, PartialEq)]
pub struct RankedExplanation {
    /// The surviving hypothesis.
    pub hypothesis: Hypothesis,
    /// Its `H ⊢ E` derivation, carried through from the test stage.
    pub evidence: DerivationTrace,
    /// The score breakdown that placed it.
    pub score: RankScore,
}

/// Ranks abductive survivors by parsimony then prior, with a deterministic tie-break.
#[derive(Debug, Clone)]
pub struct Ranker {
    /// The base graph slice, as canonical keys — an atom already here is not assumed.
    base_keys: HashSet<String>,
    /// Per-atom prior confidence, keyed canonically. Supplied by the caller (read off
    /// the store `confidence` column product-side); this crate stays arrow-free.
    priors: HashMap<String, f64>,
    /// Prior for an atom with no entry in `priors` (a never-seen assumption).
    default_prior: f64,
}

impl Ranker {
    /// Build a ranker over a base graph slice and a prior table.
    ///
    /// `base_facts` should be the **same slice the [`TestStage`] assumed `H` on top
    /// of**, so "assumed" means the same thing in both stages. `priors` maps a
    /// triple's [`triple_key`] to its prior confidence; `default_prior` is used for
    /// atoms absent from the map.
    pub fn new(base_facts: &[Triple], priors: HashMap<String, f64>, default_prior: f64) -> Self {
        Self {
            base_keys: base_facts.iter().map(triple_key).collect(),
            priors,
            default_prior,
        }
    }

    /// A ranker with no priors: every assumed atom takes `default_prior`, so ranking
    /// is parsimony-only with a deterministic tie-break. Useful when the store has no
    /// confidence signal yet.
    pub fn uniform(base_facts: &[Triple], default_prior: f64) -> Self {
        Self::new(base_facts, HashMap::new(), default_prior)
    }

    /// Score one survivor. Existential (unground) atoms cannot be checked against the
    /// base graph or the prior table, so they always count as assumed and take the
    /// default prior — an honest penalty for a less concrete guess. (In practice the
    /// test stage only emits ground survivors, but scoring is total regardless.)
    fn score(&self, survivor: &Survivor) -> RankScore {
        let atoms = &survivor.hypothesis.explanation;
        let ground = survivor.hypothesis.ground(); // Some(..) iff every atom is ground.

        let mut assumed_count = 0usize;
        let mut prior_sum = 0.0f64;
        let mut prior_n = 0usize;

        match &ground {
            Some(triples) => {
                for t in triples {
                    let key = triple_key(t);
                    if !self.base_keys.contains(&key) {
                        assumed_count += 1;
                        prior_sum += *self.priors.get(&key).unwrap_or(&self.default_prior);
                        prior_n += 1;
                    }
                }
            }
            None => {
                // Unground: each atom is an assumption at the default prior.
                assumed_count = atoms.len();
                prior_sum = self.default_prior * atoms.len() as f64;
                prior_n = atoms.len();
            }
        }

        let prior = if prior_n == 0 {
            // Every atom was already a base fact — nothing assumed. Maximally
            // parsimonious; give it the top prior so it sorts first.
            1.0
        } else {
            prior_sum / prior_n as f64
        };

        // Tie-break: the explanation's atoms in canonical form, plus the generation
        // provenance, so two structurally distinct explanations never collide.
        let mut parts: Vec<String> = match &ground {
            Some(triples) => triples.iter().map(triple_key).collect(),
            None => atoms
                .iter()
                .map(|p| format!("{:?}", p)) // pattern Debug is stable for a fixed build
                .collect(),
        };
        parts.extend(survivor.hypothesis.provenance.iter().cloned());
        let tiebreak = parts.join(";");

        RankScore {
            assumed_count,
            prior,
            tiebreak,
            posterior: None, // parsimony+prior ranker carries no posterior
        }
    }

    /// Rank survivors **best-first**: ascending `assumed_count`, then descending
    /// `prior`, then ascending `tiebreak`. The ordering is total and deterministic.
    pub fn rank(&self, survivors: Vec<Survivor>) -> Vec<RankedExplanation> {
        let mut ranked: Vec<RankedExplanation> = survivors
            .into_iter()
            .map(|s| {
                let score = self.score(&s);
                RankedExplanation {
                    hypothesis: s.hypothesis,
                    evidence: s.evidence,
                    score,
                }
            })
            .collect();
        ranked.sort_by(|a, b| {
            a.score
                .assumed_count
                .cmp(&b.score.assumed_count)
                // Higher prior is better → reverse on the float compare.
                .then(b.score.prior.total_cmp(&a.score.prior))
                .then(a.score.tiebreak.cmp(&b.score.tiebreak))
        });
        ranked
    }
}

/// **The rank-stage contract (EX-4818).** Both ranking strategies implement this, so the
/// [`Abducer`]'s rank stage is *swappable behind the contract*: the parsimony+prior
/// [`Ranker`] (VY-E E3) and the [`BayesianRanker`] (VY-Bayes E3) are interchangeable.
pub trait Rank {
    /// Order survivors best-first, attaching a [`RankScore`] to each.
    fn rank_survivors(&self, survivors: Vec<Survivor>) -> Vec<RankedExplanation>;
}

impl Rank for Ranker {
    fn rank_survivors(&self, survivors: Vec<Survivor>) -> Vec<RankedExplanation> {
        self.rank(survivors)
    }
}

/// **Bayesian posterior ranking (VY-Bayes E3, EX-4818).** Replaces the parsimony+prior
/// *lexicographic* ordering with a calibrated **posterior distribution** over the surviving
/// explanations, computed by the [`nusy_bayesian`] engine.
///
/// Each survivor becomes a hypothesis whose Bayesian **prior** encodes the same two signals
/// the parsimony ranker used — Occam (fewer assumed atoms) and prior confidence — as
/// `mean_prior_confidence × 2^(-assumed_count)` (an exponential parsimony penalty). The
/// engine normalizes these to a posterior. Because every survivor *entails* the observation
/// (the E2 test stage guarantees `H ⊢ E` for each), the observation's likelihood is uniform
/// across them, so the posterior is the normalized parsimony-prior — same *ordering* as the
/// parsimony ranker (no battery regression), now as a **probability** (`RankScore::posterior`).
/// Discriminating secondary evidence is a later refinement; the contract swap is the deliverable.
///
/// Still `Heuristic` by construction: the [`Abducer`] emits an `Evidence` trace whichever
/// ranker it holds — a posterior is a calibrated confidence, never a proof.
#[derive(Debug, Clone)]
pub struct BayesianRanker {
    base_keys: HashSet<String>,
    priors: HashMap<String, f64>,
    default_prior: f64,
}

impl BayesianRanker {
    /// Build over the same base graph + prior table a [`Ranker`] uses (so "assumed" and
    /// "prior confidence" mean the same thing across both rankers).
    pub fn new(base_facts: &[Triple], priors: HashMap<String, f64>, default_prior: f64) -> Self {
        Self {
            base_keys: base_facts.iter().map(triple_key).collect(),
            priors,
            default_prior,
        }
    }

    /// A ranker with no prior table (every assumed atom takes `default_prior`).
    pub fn uniform(base_facts: &[Triple], default_prior: f64) -> Self {
        Self::new(base_facts, HashMap::new(), default_prior)
    }

    /// (assumed_count, mean prior confidence over assumed atoms) for a survivor — the two
    /// signals folded into the Bayesian prior weight.
    fn parsimony_and_prior(&self, survivor: &Survivor) -> (usize, f64) {
        let Some(ground) = survivor.hypothesis.ground() else {
            // Unground (existential): every atom assumed at the default prior.
            let n = survivor.hypothesis.explanation.len();
            return (n, self.default_prior);
        };
        let mut assumed = 0usize;
        let mut sum = 0.0f64;
        for t in &ground {
            let key = triple_key(t);
            if !self.base_keys.contains(&key) {
                assumed += 1;
                sum += *self.priors.get(&key).unwrap_or(&self.default_prior);
            }
        }
        let mean = if assumed == 0 {
            1.0
        } else {
            sum / assumed as f64
        };
        (assumed, mean)
    }
}

impl Rank for BayesianRanker {
    fn rank_survivors(&self, survivors: Vec<Survivor>) -> Vec<RankedExplanation> {
        if survivors.is_empty() {
            return Vec::new();
        }
        // Per-survivor (assumed_count, mean_prior) + the Bayesian prior weight.
        // ids are index-prefixed so two structurally-identical explanations never collide.
        let mut meta: Vec<(String, usize, f64, String)> = Vec::new(); // (id, assumed, mean_prior, tiebreak)
        let mut hypotheses: Vec<nusy_bayesian::Hypothesis> = Vec::new();
        for (i, s) in survivors.iter().enumerate() {
            let (assumed, mean) = self.parsimony_and_prior(s);
            let tiebreak = {
                let parts: Vec<String> = match s.hypothesis.ground() {
                    Some(g) => g.iter().map(triple_key).collect(),
                    None => s
                        .hypothesis
                        .explanation
                        .iter()
                        .map(|p| format!("{p:?}"))
                        .collect(),
                };
                let mut parts = parts;
                parts.extend(s.hypothesis.provenance.iter().cloned());
                parts.join(";")
            };
            let id = format!("{i}|{tiebreak}");
            // Occam as an exponential parsimony prior, scaled by mean confidence.
            let weight = mean * 2f64.powi(-(assumed as i32));
            hypotheses.push(nusy_bayesian::Hypothesis::new(id.clone(), weight));
            meta.push((id, assumed, mean, tiebreak));
        }

        // Build the posterior. With uniform likelihoods (every survivor entails E), the
        // posterior is the normalized prior. A non-normalizable set (all weights 0) falls
        // back to a uniform posterior so ranking still proceeds deterministically.
        let posterior = match nusy_bayesian::BayesianUpdate::new(hypotheses.clone()) {
            Ok(model) => model.prior(),
            Err(_) => nusy_bayesian::BayesianUpdate::uniform(meta.iter().map(|m| m.0.as_str()))
                .expect("non-empty")
                .prior(),
        };

        let mut ranked: Vec<RankedExplanation> = survivors
            .into_iter()
            .zip(meta)
            .map(|(s, (id, assumed, mean, tiebreak))| {
                let p = posterior.prob(&id);
                RankedExplanation {
                    hypothesis: s.hypothesis,
                    evidence: s.evidence,
                    score: RankScore {
                        assumed_count: assumed,
                        prior: mean,
                        tiebreak,
                        posterior: Some(p),
                    },
                }
            })
            .collect();
        // Best-first by posterior (desc), deterministic tie-break on the canonical key.
        ranked.sort_by(|a, b| {
            b.score
                .posterior
                .unwrap_or(0.0)
                .total_cmp(&a.score.posterior.unwrap_or(0.0))
                .then(a.score.tiebreak.cmp(&b.score.tiebreak))
        });
        ranked
    }
}

/// **Belief-revision decision (VY-Bayes E3 P2, EX-4818).** The arrow-free *decision* of
/// whether a better-explained hypothesis should displace the incumbent on the store's
/// epistemic axis. The actual write is the governed
/// [`displace_belief`](https://docs.rs/nusy-store-seam) (EX-4777, product-side) — this is
/// the engine-side choice that feeds it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BeliefRevision {
    /// The incumbent is still the best explanation (or there is no challenger) — no change.
    Keep,
    /// No incumbent yet — adopt the best explanation.
    Adopt { challenger: Triple },
    /// A strictly better explanation displaces the incumbent.
    Displace {
        incumbent: Triple,
        challenger: Triple,
    },
}

/// Decide belief revision from a best-first `ranked` list: the top explanation's principal
/// atom is the new best belief. If it differs from `incumbent`, it displaces; if it equals
/// the incumbent, keep; with no incumbent, adopt. Empty `ranked` → keep.
pub fn revise_belief(incumbent: Option<&Triple>, ranked: &[RankedExplanation]) -> BeliefRevision {
    let Some(challenger) = ranked
        .first()
        .and_then(|r| r.hypothesis.ground())
        .and_then(|g| g.into_iter().next())
    else {
        return BeliefRevision::Keep;
    };
    match incumbent {
        None => BeliefRevision::Adopt { challenger },
        Some(inc) if *inc == challenger => BeliefRevision::Keep,
        Some(inc) => BeliefRevision::Displace {
            incumbent: inc.clone(),
            challenger,
        },
    }
}

/// The composed abductive reasoner: generate → test → rank, answering with the top
/// explanation. See the crate docs for why its provability is always `Heuristic`. The rank
/// stage `R` is swappable (EX-4818): defaults to the parsimony+prior [`Ranker`]; pass a
/// [`BayesianRanker`] for the posterior upgrade.
#[derive(Debug, Clone)]
pub struct Abducer<S: CandidateSource, R: Rank = Ranker> {
    source: S,
    test: TestStage,
    ranker: R,
    envelope: CompetenceEnvelope,
    substrate: Substrate,
}

impl<S: CandidateSource, R: Rank> Abducer<S, R> {
    /// Wire the three stages into a reasoner. `substrate` is taken from the candidate
    /// `source` (symbolic rule-reversal → [`Substrate::Symbolic`]; a neural proposer
    /// → [`Substrate::Neural`]); a mixed source should report [`Substrate::Mixed`].
    /// `envelope` declares which query shapes this abducer is asked to explain. `ranker` is
    /// any [`Rank`] strategy (parsimony or Bayesian).
    pub fn new(source: S, test: TestStage, ranker: R, envelope: CompetenceEnvelope) -> Self {
        let substrate = source.substrate();
        Self {
            source,
            test,
            ranker,
            envelope,
            substrate,
        }
    }

    /// Run the pipeline and return the ranked survivors (best-first) without wrapping
    /// in an [`Answer`] — for callers (e.g. the displacement hook) that want the full
    /// score breakdown, not just the top.
    pub fn explain(&self, observation: &Triple) -> Vec<RankedExplanation> {
        let candidates = self.source.enumerate(observation);
        let survivors = self.test.survivors(observation, &candidates);
        self.ranker.rank_survivors(survivors)
    }
}

impl<S: CandidateSource, R: Rank> Reasoner for Abducer<S, R> {
    fn answer(&self, query: &Query) -> Answer {
        let ranked = self.explain(&query.goal);
        let Some(top) = ranked.first() else {
            // No explanation survived — honest abstention.
            return Answer::abstained();
        };

        // Survivors are ground (the test stage screens `NotGround`), so `ground()` is
        // Some; the principal atom represents the explanation as the answer value, and
        // the full atom set is carried in the evidence.
        let ground = top.hypothesis.ground().unwrap_or_default();
        let value = ground.first().cloned();

        // The composition's honest provability: the test evidence proves `H ⊢ E`
        // (complete derivation), but assuming `H` is only heuristic. min() = Heuristic.
        let h_entails_e = ProofTrace::Derivation(top.evidence.clone()).provability();
        let combined = provability_min(&[h_entails_e, Provability::Heuristic]);
        debug_assert_eq!(combined, Provability::Heuristic);

        // CH-4801 (credit: Air, from the closed-duplicate PROP-2740): stage-tag each
        // trace line in pipeline order so propose → test → rank is auditable from the
        // answer alone. The bare substrings ("rule_chain=", "H⊢E=", "assumed_count=")
        // are preserved as suffixes, so downstream consumers (the EX-4778 battery's
        // stage-threading assertions) keep matching.
        let mut why: Vec<String> = ground.iter().map(triple_key).collect();
        why.push(format!(
            "propose: rule_chain={}",
            top.hypothesis.provenance.join("->")
        ));
        why.push(format!(
            "test: H⊢E={:?}; abductive_provability={:?}",
            h_entails_e, combined
        ));
        // The rank-stage confidence: the Bayesian posterior when the ranker is Bayesian
        // (EX-4818), else the parsimony ranker's mean prior. Either way it rides in an
        // Evidence trace → Heuristic; the neural/assumed stage is never laundered to Proven.
        let rank_confidence = top.score.posterior.unwrap_or(top.score.prior);
        why.push(format!(
            "rank: assumed_count={} prior={:.3}{}",
            top.score.assumed_count,
            top.score.prior,
            match top.score.posterior {
                Some(p) => format!(" posterior={p:.3}"),
                None => String::new(),
            }
        ));

        // Evidence trace (not a Derivation) → provability() computes Heuristic. The
        // Abducer cannot mint Proven for an assumed explanation.
        Answer {
            value,
            proof: ProofTrace::Evidence {
                confidence: rank_confidence,
                why,
            },
            provenance: top.hypothesis.provenance.clone(),
        }
    }

    fn competence_envelope(&self) -> &CompetenceEnvelope {
        &self.envelope
    }

    fn substrate(&self) -> Substrate {
        self.substrate
    }

    fn guarantee(&self) -> Guarantee {
        // Abduction is neither sound (it can assert a wrong H) nor complete; it is
        // probabilistic — it carries a calibrated prior, never a certainty.
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
    use nusy_abduction::GraphCandidates;
    use nusy_forward_chain::IdRule;
    use nusy_reasoner::{QueryShape, Substrate};
    use nusy_unify::{Rule, Term, TriplePattern};

    fn idrule(id: &str, body: Vec<TriplePattern>, head: Vec<TriplePattern>) -> IdRule {
        IdRule::new(id, Rule::new(body, head))
    }

    fn hyp(atoms: &[(&str, &str, &str)], provenance: &[&str]) -> Hypothesis {
        Hypothesis {
            explanation: atoms
                .iter()
                .map(|(s, p, o)| {
                    TriplePattern::new(
                        Term::Const(s.to_string()),
                        Term::Const(p.to_string()),
                        Term::Const(o.to_string()),
                    )
                })
                .collect(),
            provenance: provenance.iter().map(|s| s.to_string()).collect(),
            substrate: Substrate::Symbolic,
        }
    }

    /// A survivor wrapping a hand-built hypothesis with a trivial axiom evidence — the
    /// ranker scores on the hypothesis + base graph, not on the evidence shape.
    fn survivor(atoms: &[(&str, &str, &str)], provenance: &[&str]) -> Survivor {
        let h = hyp(atoms, provenance);
        // A placeholder complete trace; ranking does not inspect it.
        let evidence = DerivationTrace::Axiom(Triple::new("x", "y", "z"));
        Survivor {
            hypothesis: h,
            evidence,
        }
    }

    // ---- Phase 1: Ranker ----

    /// Fewer assumed atoms ranks first; the order is identical across repeated runs.
    #[test]
    fn ranking_is_parsimony_ordered_and_deterministic() {
        let two = survivor(&[("p", "smokes", "true"), ("p", "obese", "true")], &["r2"]);
        let one = survivor(&[("p", "smokes", "true")], &["r1"]);
        let ranker = Ranker::uniform(&[], 0.5);

        let first = ranker.rank(vec![two.clone(), one.clone()]);
        assert_eq!(
            first[0].score.assumed_count, 1,
            "the 1-atom explanation wins"
        );
        assert_eq!(first[1].score.assumed_count, 2);

        // Same input in the opposite order → same ranking (determinism).
        let second = ranker.rank(vec![one, two]);
        assert_eq!(
            first
                .iter()
                .map(|r| r.score.tiebreak.clone())
                .collect::<Vec<_>>(),
            second
                .iter()
                .map(|r| r.score.tiebreak.clone())
                .collect::<Vec<_>>(),
        );
    }

    /// At equal parsimony, the higher prior wins.
    #[test]
    fn prior_breaks_equal_parsimony() {
        let a = survivor(&[("p", "smokes", "true")], &["a"]);
        let b = survivor(&[("p", "obese", "true")], &["b"]);
        let mut priors = HashMap::new();
        priors.insert("p|smokes|true".to_string(), 0.9);
        priors.insert("p|obese|true".to_string(), 0.2);
        let ranker = Ranker::new(&[], priors, 0.5);

        let ranked = ranker.rank(vec![b, a]);
        assert_eq!(ranked[0].hypothesis.provenance, vec!["a".to_string()]);
        assert!(ranked[0].score.prior > ranked[1].score.prior);
    }

    /// An atom already in the base graph is not assumed — it lowers the assumed count.
    #[test]
    fn base_facts_are_not_counted_as_assumptions() {
        let base = vec![Triple::new("p", "smokes", "true")];
        let ranker = Ranker::uniform(&base, 0.5);
        let s = survivor(&[("p", "smokes", "true"), ("p", "obese", "true")], &["r"]);
        let ranked = ranker.rank(vec![s]);
        assert_eq!(
            ranked[0].score.assumed_count, 1,
            "only the non-base atom counts"
        );
    }

    /// Ties on parsimony AND prior fall to the canonical tie-break, total and stable.
    #[test]
    fn tiebreak_is_total_and_stable() {
        let a = survivor(&[("p", "aaa", "true")], &["x"]);
        let b = survivor(&[("p", "bbb", "true")], &["x"]);
        let ranker = Ranker::uniform(&[], 0.5); // equal parsimony + equal prior
        let r1 = ranker.rank(vec![b.clone(), a.clone()]);
        let r2 = ranker.rank(vec![a, b]);
        assert_eq!(r1[0].score.tiebreak, "p|aaa|true;x");
        assert_eq!(
            r1.iter()
                .map(|r| r.score.tiebreak.clone())
                .collect::<Vec<_>>(),
            r2.iter()
                .map(|r| r.score.tiebreak.clone())
                .collect::<Vec<_>>(),
        );
    }

    // ---- Phase 2: Abducer ----

    fn envelope() -> CompetenceEnvelope {
        CompetenceEnvelope {
            shapes: vec![QueryShape {
                name: "abduction".into(),
                predicates: vec![], // wildcard
            }],
        }
    }

    /// The end-to-end pipeline answers with the top explanation and — crucially —
    /// reports **Heuristic** even though the survivor's `H ⊢ E` derivation is complete.
    #[test]
    fn abducer_answers_heuristic_with_top_explanation() {
        // bird(X) -> flies(X); penguin(X) -> bird(X). Observe flies(tweety).
        // Shortest explanation: bird(tweety) (1 hop). Deeper: penguin(tweety) (2 hops).
        let flies = idrule(
            "flies-if-bird",
            vec![TriplePattern::parse("?x", "is", "bird")],
            vec![TriplePattern::parse("?x", "can", "fly")],
        );
        let bird = idrule(
            "bird-if-penguin",
            vec![TriplePattern::parse("?x", "is", "penguin")],
            vec![TriplePattern::parse("?x", "is", "bird")],
        );
        let obs = Triple::new("tweety", "can", "fly");
        let source = GraphCandidates::new(vec![flies.clone(), bird.clone()], 2);
        let test = TestStage::new(vec![flies, bird], vec![]);
        let ranker = Ranker::uniform(&[], 0.5);
        let abducer = Abducer::new(source, test, ranker, envelope());

        let ans = abducer.answer(&Query::new(obs.clone()));
        assert!(ans.value.is_some(), "an explanation was found");
        assert_eq!(
            ans.provability(),
            Provability::Heuristic,
            "abduction infers, it never proves — even with a complete H⊢E trace"
        );
        // The most parsimonious explanation (bird, 1 hop) should be the answer.
        assert_eq!(ans.value.unwrap(), Triple::new("tweety", "is", "bird"));
    }

    /// No reversing rule for the observation → the abducer abstains (not a false guess).
    #[test]
    fn abducer_abstains_when_nothing_explains() {
        let rule = idrule(
            "flies-if-bird",
            vec![TriplePattern::parse("?x", "is", "bird")],
            vec![TriplePattern::parse("?x", "can", "fly")],
        );
        let obs = Triple::new("rock", "is", "igneous"); // no rule concludes this
        let source = GraphCandidates::new(vec![rule.clone()], 2);
        let test = TestStage::new(vec![rule], vec![]);
        let abducer = Abducer::new(source, test, Ranker::uniform(&[], 0.5), envelope());

        let ans = abducer.answer(&Query::new(obs));
        assert_eq!(ans.value, None);
        assert_eq!(ans.provability(), Provability::Abstained);
    }

    /// The Abducer's guarantee is unsound + incomplete + probabilistic — the router
    /// must never route a `Proven`-required claim to it.
    #[test]
    fn abducer_guarantee_is_probabilistic_not_sound() {
        let rule = idrule(
            "flies-if-bird",
            vec![TriplePattern::parse("?x", "is", "bird")],
            vec![TriplePattern::parse("?x", "can", "fly")],
        );
        let abducer = Abducer::new(
            GraphCandidates::new(vec![rule.clone()], 1),
            TestStage::new(vec![rule], vec![]),
            Ranker::uniform(&[], 0.5),
            envelope(),
        );
        let g = abducer.guarantee();
        assert!(!g.sound && !g.complete && g.probabilistic);
        assert_eq!(abducer.substrate(), Substrate::Symbolic);
    }

    // ---- EX-4818: Bayesian posterior ranking + belief revision ----

    /// The Bayesian ranker emits a real posterior: probabilities in (0,1] that sum to 1,
    /// best-first, with the more-parsimonious explanation getting the larger posterior.
    #[test]
    fn bayesian_ranker_posterior_sums_to_one_and_favors_parsimony() {
        let one = survivor(&[("p", "smokes", "true")], &["r1"]);
        let two = survivor(&[("p", "obese", "true"), ("p", "old", "true")], &["r2"]);
        let ranked = BayesianRanker::uniform(&[], 0.5).rank_survivors(vec![two, one]);
        // posteriors present and normalized.
        let sum: f64 = ranked.iter().map(|r| r.score.posterior.unwrap()).sum();
        assert!((sum - 1.0).abs() < 1e-9, "posterior sums to 1, got {sum}");
        // the 1-atom explanation ranks first with the larger posterior.
        assert_eq!(ranked[0].score.assumed_count, 1);
        assert!(ranked[0].score.posterior.unwrap() > ranked[1].score.posterior.unwrap());
    }

    /// Swappability without regression: on the same survivors the Bayesian ranker yields the
    /// SAME order as the parsimony ranker (the posterior is monotone in the parsimony prior),
    /// so swapping the rank stage behind the contract does not regress the VY-E battery.
    #[test]
    fn bayesian_and_parsimony_agree_on_order() {
        let a = survivor(&[("p", "aaa", "true")], &["x"]);
        let b = survivor(&[("p", "bbb", "true"), ("p", "ccc", "true")], &["y"]);
        let parsimony = Ranker::uniform(&[], 0.5).rank_survivors(vec![a.clone(), b.clone()]);
        let bayes = BayesianRanker::uniform(&[], 0.5).rank_survivors(vec![a, b]);
        let p_order: Vec<_> = parsimony.iter().map(|r| r.score.tiebreak.clone()).collect();
        let b_order: Vec<_> = bayes.iter().map(|r| r.score.tiebreak.clone()).collect();
        assert_eq!(
            p_order, b_order,
            "Bayesian and parsimony rankers agree on order"
        );
    }

    /// The Abducer with the Bayesian rank stage still answers Heuristic (never Proven), and
    /// reports the posterior as its confidence.
    #[test]
    fn abducer_with_bayesian_ranker_is_heuristic() {
        let flies = idrule(
            "flies-if-bird",
            vec![TriplePattern::parse("?x", "is", "bird")],
            vec![TriplePattern::parse("?x", "can", "fly")],
        );
        let obs = Triple::new("tweety", "can", "fly");
        let abducer = Abducer::new(
            GraphCandidates::new(vec![flies.clone()], 2),
            TestStage::new(vec![flies], vec![]),
            BayesianRanker::uniform(&[], 0.5),
            envelope(),
        );
        let ans = abducer.answer(&Query::new(obs));
        assert_eq!(ans.provability(), Provability::Heuristic);
        assert_eq!(ans.value.unwrap(), Triple::new("tweety", "is", "bird"));
        // confidence is the posterior (single survivor → 1.0).
        if let ProofTrace::Evidence { confidence, why } = ans.proof {
            assert!((confidence - 1.0).abs() < 1e-9);
            assert!(why.iter().any(|w| w.contains("posterior=")));
        } else {
            panic!("expected Evidence");
        }
    }

    /// Belief revision: adopt with no incumbent; keep when the incumbent is the top; displace
    /// when a different explanation ranks first.
    #[test]
    fn revise_belief_adopt_keep_displace() {
        let ranked = BayesianRanker::uniform(&[], 0.5)
            .rank_survivors(vec![survivor(&[("p", "has", "flu")], &["r"])]);
        let top = Triple::new("p", "has", "flu");

        assert_eq!(
            revise_belief(None, &ranked),
            BeliefRevision::Adopt {
                challenger: top.clone()
            }
        );
        assert_eq!(revise_belief(Some(&top), &ranked), BeliefRevision::Keep);
        let incumbent = Triple::new("p", "has", "cold");
        assert_eq!(
            revise_belief(Some(&incumbent), &ranked),
            BeliefRevision::Displace {
                incumbent,
                challenger: top
            }
        );
        // Empty ranked → keep (nothing to revise to).
        assert_eq!(revise_belief(None, &[]), BeliefRevision::Keep);
    }
}
