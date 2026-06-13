//! Bayesian belief update (VY-Bayes E1, phase 1).
//!
//! A discrete hypothesis space with priors, updated by evidence likelihoods to a normalized
//! posterior — `P(H|E) ∝ P(E|H)·P(H)`. The [`Posterior`] retains the evidence that shaped it
//! (provenance), and exposes the MAP hypothesis for ranking. This is the engine VY-Bayes E3
//! swaps in for abduction's parsimony+prior ranker.
//!
//! **Heuristic by construction.** A posterior is a *ranking* under uncertainty, never a
//! proof — the Reasoner-contract conformance (E2) tags every answer here un-provable so the
//! minimum-guarantee rule can never launder a probability into a proof.

use std::collections::BTreeMap;

/// An error constructing a Bayesian model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BayesError {
    /// No hypotheses were supplied.
    Empty,
    /// A prior was negative, or all priors summed to zero (nothing to normalize).
    NonNormalizablePrior,
    /// Two hypotheses shared an id.
    DuplicateHypothesis(String),
}

impl std::fmt::Display for BayesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BayesError::Empty => write!(f, "no hypotheses supplied"),
            BayesError::NonNormalizablePrior => {
                write!(f, "priors are negative or sum to zero — cannot normalize")
            }
            BayesError::DuplicateHypothesis(id) => write!(f, "duplicate hypothesis id: {id}"),
        }
    }
}

impl std::error::Error for BayesError {}

/// A discrete hypothesis with a prior probability (unnormalized weights are fine — the
/// model normalizes priors at construction).
#[derive(Debug, Clone, PartialEq)]
pub struct Hypothesis {
    pub id: String,
    pub prior: f64,
}

impl Hypothesis {
    pub fn new(id: impl Into<String>, prior: f64) -> Self {
        Self {
            id: id.into(),
            prior,
        }
    }
}

/// One piece of evidence: the likelihood `P(E|H)` for each hypothesis. A hypothesis absent
/// from the map is treated as **uninformative** for this evidence (likelihood 1.0) rather
/// than impossible (0.0) — a missing observation must not silently zero a hypothesis out.
#[derive(Debug, Clone, PartialEq)]
pub struct Likelihood {
    pub evidence_id: String,
    likelihoods: BTreeMap<String, f64>,
}

impl Likelihood {
    pub fn new(evidence_id: impl Into<String>) -> Self {
        Self {
            evidence_id: evidence_id.into(),
            likelihoods: BTreeMap::new(),
        }
    }

    /// Set `P(E|H)` for hypothesis `hyp`. Builder-style.
    pub fn with(mut self, hyp: impl Into<String>, p_e_given_h: f64) -> Self {
        self.likelihoods.insert(hyp.into(), p_e_given_h);
        self
    }

    /// `P(E|H)` for `hyp` — 1.0 (uninformative) if unspecified.
    pub fn for_hypothesis(&self, hyp: &str) -> f64 {
        self.likelihoods.get(hyp).copied().unwrap_or(1.0)
    }
}

/// A normalized posterior over a hypothesis space, plus the evidence applied (provenance).
#[derive(Debug, Clone, PartialEq)]
pub struct Posterior {
    /// hypothesis id → probability, summing to 1 (within float tolerance).
    probs: BTreeMap<String, f64>,
    /// evidence ids applied so far, in application order.
    evidence: Vec<String>,
}

impl Posterior {
    /// `P(H)` for `hyp` (0.0 if not in the space).
    pub fn prob(&self, hyp: &str) -> f64 {
        self.probs.get(hyp).copied().unwrap_or(0.0)
    }

    /// The maximum-a-posteriori hypothesis (highest probability). Ties break by id order
    /// (deterministic). `None` only for an empty space.
    pub fn map_hypothesis(&self) -> Option<&str> {
        self.probs
            .iter()
            .max_by(|a, b| {
                a.1.partial_cmp(b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    // BTreeMap iterates id-ascending; max_by keeps the LAST max, so to break
                    // ties toward the smaller id we reverse the id tiebreak.
                    .then_with(|| b.0.cmp(a.0))
            })
            .map(|(id, _)| id.as_str())
    }

    /// (id, probability) pairs, ordered most- to least-probable (ties by id).
    pub fn ranked(&self) -> Vec<(&str, f64)> {
        let mut v: Vec<(&str, f64)> = self.probs.iter().map(|(k, p)| (k.as_str(), *p)).collect();
        v.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(b.0))
        });
        v
    }

    /// The evidence ids that shaped this posterior, in application order.
    pub fn evidence(&self) -> &[String] {
        &self.evidence
    }

    /// Shannon entropy (bits) of the posterior — a calibrated "how undecided am I" signal
    /// the abstention/confidence machinery (E2/E3) can threshold on. 0 = certain.
    pub fn entropy_bits(&self) -> f64 {
        -self
            .probs
            .values()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.log2())
            .sum::<f64>()
    }
}

/// A Bayesian belief updater over a fixed hypothesis space with normalized priors.
#[derive(Debug, Clone)]
pub struct BayesianUpdate {
    priors: BTreeMap<String, f64>,
}

impl BayesianUpdate {
    /// Build from hypotheses; priors are normalized to sum 1. Errors on an empty space,
    /// duplicate ids, or non-normalizable (negative / all-zero) priors.
    pub fn new(hypotheses: impl IntoIterator<Item = Hypothesis>) -> Result<Self, BayesError> {
        let mut priors = BTreeMap::new();
        let mut sum = 0.0;
        for h in hypotheses {
            if h.prior < 0.0 {
                return Err(BayesError::NonNormalizablePrior);
            }
            if priors.insert(h.id.clone(), h.prior).is_some() {
                return Err(BayesError::DuplicateHypothesis(h.id));
            }
            sum += h.prior;
        }
        if priors.is_empty() {
            return Err(BayesError::Empty);
        }
        if sum <= 0.0 {
            return Err(BayesError::NonNormalizablePrior);
        }
        for p in priors.values_mut() {
            *p /= sum;
        }
        Ok(Self { priors })
    }

    /// A uniform prior over the named hypotheses.
    pub fn uniform<'a>(ids: impl IntoIterator<Item = &'a str>) -> Result<Self, BayesError> {
        let hs: Vec<Hypothesis> = ids.into_iter().map(|id| Hypothesis::new(id, 1.0)).collect();
        Self::new(hs)
    }

    /// The prior as a [`Posterior`] (no evidence applied yet).
    pub fn prior(&self) -> Posterior {
        Posterior {
            probs: self.priors.clone(),
            evidence: Vec::new(),
        }
    }

    /// Apply one evidence item to a posterior: multiply each hypothesis's probability by
    /// its likelihood and renormalize. If every product is zero (the evidence is
    /// impossible under all *surviving* hypotheses) the prior distribution is returned
    /// unchanged with the evidence still recorded — a contradictory observation does not
    /// produce a meaningless all-zero belief.
    pub fn update(&self, current: &Posterior, ev: &Likelihood) -> Posterior {
        let mut next: BTreeMap<String, f64> = BTreeMap::new();
        let mut sum = 0.0;
        for (id, &p) in &current.probs {
            let post = p * ev.for_hypothesis(id);
            next.insert(id.clone(), post);
            sum += post;
        }
        let mut evidence = current.evidence.clone();
        evidence.push(ev.evidence_id.clone());

        if sum <= 0.0 {
            // Degenerate: keep the incoming distribution, record the evidence anyway.
            return Posterior {
                probs: current.probs.clone(),
                evidence,
            };
        }
        for p in next.values_mut() {
            *p /= sum;
        }
        Posterior {
            probs: next,
            evidence,
        }
    }

    /// Sequentially apply many evidence items starting from the prior.
    pub fn observe(&self, evidence: &[Likelihood]) -> Posterior {
        let mut post = self.prior();
        for ev in evidence {
            post = self.update(&post, ev);
        }
        post
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Textbook base-rate example: disease prevalence 1%, test sensitivity 99%, false-
    /// positive 5%. A positive test → P(disease|+) ≈ 0.1667. Verifies the posterior matches
    /// the hand-computed Bayes result (catching the base-rate fallacy correctly).
    #[test]
    fn disease_test_matches_hand_computed_bayes() {
        let model = BayesianUpdate::new([
            Hypothesis::new("disease", 0.01),
            Hypothesis::new("healthy", 0.99),
        ])
        .unwrap();
        let positive = Likelihood::new("test+")
            .with("disease", 0.99) // P(+|disease) sensitivity
            .with("healthy", 0.05); // P(+|healthy) false-positive
        let post = model.update(&model.prior(), &positive);
        // 0.01*0.99 / (0.01*0.99 + 0.99*0.05) = 0.0099 / 0.0594 = 0.16667
        assert!(
            (post.prob("disease") - 0.166_67).abs() < 1e-4,
            "got {}",
            post.prob("disease")
        );
        assert!((post.prob("healthy") - 0.833_33).abs() < 1e-4);
        assert_eq!(post.map_hypothesis(), Some("healthy"));
        assert_eq!(post.evidence(), &["test+".to_string()]);
    }

    #[test]
    fn priors_are_normalized_and_uniform_is_flat() {
        let m = BayesianUpdate::uniform(["a", "b", "c", "d"]).unwrap();
        let prior = m.prior();
        for h in ["a", "b", "c", "d"] {
            assert!((prior.prob(h) - 0.25).abs() < 1e-12);
        }
        // unnormalized weights normalize.
        let m2 =
            BayesianUpdate::new([Hypothesis::new("x", 3.0), Hypothesis::new("y", 1.0)]).unwrap();
        assert!((m2.prior().prob("x") - 0.75).abs() < 1e-12);
    }

    #[test]
    fn sequential_updates_compose_and_are_order_independent() {
        let model = BayesianUpdate::uniform(["h1", "h2"]).unwrap();
        let e1 = Likelihood::new("e1").with("h1", 0.8).with("h2", 0.2);
        let e2 = Likelihood::new("e2").with("h1", 0.5).with("h2", 0.9);
        let ab = model.observe(&[e1.clone(), e2.clone()]);
        let ba = model.observe(&[e2, e1]);
        // Bayes is order-independent given conditional independence.
        assert!((ab.prob("h1") - ba.prob("h1")).abs() < 1e-12);
        assert_eq!(ab.evidence().len(), 2);
    }

    #[test]
    fn missing_likelihood_is_uninformative_not_zero() {
        let model = BayesianUpdate::uniform(["h1", "h2"]).unwrap();
        // Evidence only specifies h1; h2 defaults to 1.0 (uninformative), so this is
        // equivalent to scaling only h1.
        let ev = Likelihood::new("partial").with("h1", 0.5);
        let post = model.update(&model.prior(), &ev);
        // 0.5*0.5 / (0.5*0.5 + 0.5*1.0) = 0.25/0.75 = 1/3
        assert!((post.prob("h1") - 1.0 / 3.0).abs() < 1e-12);
        assert!((post.prob("h2") - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn contradictory_evidence_does_not_zero_the_belief() {
        let model = BayesianUpdate::uniform(["h1", "h2"]).unwrap();
        let impossible = Likelihood::new("impossible")
            .with("h1", 0.0)
            .with("h2", 0.0);
        let post = model.update(&model.prior(), &impossible);
        // sum would be 0 → keep the incoming distribution, still record the evidence.
        assert!((post.prob("h1") - 0.5).abs() < 1e-12);
        assert_eq!(post.evidence(), &["impossible".to_string()]);
    }

    #[test]
    fn entropy_drops_as_evidence_sharpens_belief() {
        let model = BayesianUpdate::uniform(["h1", "h2"]).unwrap();
        let flat = model.prior().entropy_bits(); // 1 bit for a uniform binary
        assert!((flat - 1.0).abs() < 1e-9);
        let sharp = model.update(
            &model.prior(),
            &Likelihood::new("strong").with("h1", 0.99).with("h2", 0.01),
        );
        assert!(sharp.entropy_bits() < flat);
    }

    #[test]
    fn ranked_orders_by_probability_then_id() {
        let model = BayesianUpdate::new([
            Hypothesis::new("a", 0.2),
            Hypothesis::new("b", 0.5),
            Hypothesis::new("c", 0.3),
        ])
        .unwrap();
        let prior = model.prior();
        let ranked = prior.ranked();
        assert_eq!(ranked[0].0, "b");
        assert_eq!(ranked[1].0, "c");
        assert_eq!(ranked[2].0, "a");
    }

    #[test]
    fn construction_errors_are_explicit() {
        assert_eq!(
            BayesianUpdate::new(std::iter::empty()).unwrap_err(),
            BayesError::Empty
        );
        assert_eq!(
            BayesianUpdate::new([Hypothesis::new("a", 0.0)]).unwrap_err(),
            BayesError::NonNormalizablePrior
        );
        assert_eq!(
            BayesianUpdate::new([Hypothesis::new("a", -1.0), Hypothesis::new("b", 2.0)])
                .unwrap_err(),
            BayesError::NonNormalizablePrior
        );
        assert_eq!(
            BayesianUpdate::new([Hypothesis::new("a", 0.5), Hypothesis::new("a", 0.5)])
                .unwrap_err(),
            BayesError::DuplicateHypothesis("a".into())
        );
    }
}
