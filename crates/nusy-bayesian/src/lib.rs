//! # nusy-bayesian — VY-Bayes E1: probabilistic inference + certainty aggregation
//!
//! The first Wave-2 reasoner engine (V19-REASONER-ROADMAP §2). Two cores:
//!
//! 1. [`bayes`] — a **Bayesian belief update**: a discrete hypothesis space with priors,
//!    updated by evidence likelihoods to a normalized [`Posterior`] (`P(H|E) ∝ P(E|H)·P(H)`),
//!    retaining the evidence that shaped it. This is the engine VY-Bayes E3 swaps in for
//!    abduction's parsimony+prior ranker.
//! 2. [`grade`] — **certainty aggregation**, GRADE generalized: a recursive evidence model
//!    (sub-components → overall) aggregated to an overall level under a named [`RatingSystem`]
//!    (GRADE is one instance — the scale and step rule are data, not enums).
//!
//! The [`prior_from_certainty`] bridge connects them: aggregated certainty levels become the
//! Bayesian prior, so "prior + evidence → updated belief over the confidence column" is one
//! pipeline (VY-Bayes E1's goal).
//!
//! ## Genericity (why this crate is Arrow-free)
//! Like its sibling reasoner crates (nusy-abduction-rank et al.), nusy-bayesian depends on
//! no product crate, so it stays extractable into the nusy-reasoners FOSS suite. It operates
//! over the generic [`EvidenceRating`] — the product adapts its Arrow `Certainty` rows into
//! these; there is no parallel certainty *store*, only an in-memory DTO.
//!
//! ## Heuristic by construction
//! A posterior is a **ranking under uncertainty, never a proof**. VY-Bayes E2 conforms this
//! engine to the `Reasoner` contract with its provability tag fixed to un-provable, so the
//! minimum-guarantee rule can never launder a probability into a proof.
//!
//! ```
//! use nusy_bayesian::{BayesianUpdate, Hypothesis, Likelihood, RatingSystem, prior_from_certainty};
//!
//! // Certainty → prior: two explanations, one high-certainty, one low.
//! let grade = RatingSystem::grade();
//! let model = prior_from_certainty(
//!     &grade,
//!     &[
//!         ("pneumonia".into(), "high".into(), Some("exact".into())),
//!         ("reflux".into(), "low".into(), None),
//!     ],
//! )
//! .unwrap();
//!
//! // Evidence: a finding far more expected under pneumonia.
//! let fever = Likelihood::new("fever").with("pneumonia", 0.9).with("reflux", 0.1);
//! let post = model.update(&model.prior(), &fever);
//! assert_eq!(post.map_hypothesis(), Some("pneumonia"));
//! ```

mod bayes;
mod grade;

pub use bayes::{BayesError, BayesianUpdate, Hypothesis, Likelihood, Posterior};
pub use grade::{Aggregate, EvidenceRating, RatingSystem, directness_factor};

/// Build a Bayesian prior over candidate hypotheses from their aggregated certainty.
///
/// Each candidate's prior weight is `system.weight(level) × directness_factor(directness)` —
/// higher-certainty, more-direct evidence gets a larger prior. The weights are handed to
/// [`BayesianUpdate::new`], which normalizes them to a distribution. This is the bridge from
/// the certainty/evidence axis to the Bayesian update (and the VY-Bayes E3 abduction-ranking
/// upgrade: candidate explanations enter ranked by their evidence certainty, then evidence
/// likelihoods refine the order).
///
/// Errors if a candidate's `level` is not in the system's scale (via a zero weight that
/// makes the priors non-normalizable only when *all* are unknown), or per
/// [`BayesianUpdate::new`]'s rules (empty / duplicate / non-normalizable).
pub fn prior_from_certainty(
    system: &RatingSystem,
    candidates: &[(String, String, Option<String>)], // (hypothesis id, level, directness)
) -> Result<BayesianUpdate, BayesError> {
    let hypotheses = candidates.iter().map(|(id, level, directness)| {
        let w = system.weight(level).unwrap_or(0.0) * directness_factor(directness.as_deref());
        Hypothesis::new(id.clone(), w)
    });
    BayesianUpdate::new(hypotheses)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn certainty_sets_the_prior_ordering() {
        let grade = RatingSystem::grade();
        let model = prior_from_certainty(
            &grade,
            &[
                ("strong".into(), "high".into(), Some("exact".into())),
                ("weak".into(), "very-low".into(), Some("low".into())),
            ],
        )
        .unwrap();
        let prior = model.prior();
        // high+exact must outweigh very-low+low before any evidence.
        assert!(prior.prob("strong") > prior.prob("weak"));
        assert_eq!(prior.map_hypothesis(), Some("strong"));
    }

    #[test]
    fn full_pipeline_certainty_prior_then_evidence() {
        // GRADE aggregation → prior; then a likelihood flips toward the better-explained H.
        let grade = RatingSystem::grade();
        let model = prior_from_certainty(
            &grade,
            &[
                ("h_common".into(), "moderate".into(), None),
                ("h_rare".into(), "moderate".into(), None),
            ],
        )
        .unwrap();
        // Equal certainty → equal prior; evidence decides.
        assert!((model.prior().prob("h_common") - model.prior().prob("h_rare")).abs() < 1e-12);
        let finding = Likelihood::new("pathognomonic")
            .with("h_common", 0.2)
            .with("h_rare", 0.95);
        let post = model.update(&model.prior(), &finding);
        assert_eq!(post.map_hypothesis(), Some("h_rare"));
    }

    #[test]
    fn aggregate_then_prior_end_to_end() {
        let grade = RatingSystem::grade();
        // Build each candidate's overall level by aggregating its components, then prior.
        let h1 = grade.aggregate("high", [("rob", "serious")]).unwrap().level; // → moderate
        let h2 = grade.aggregate("high", [("rob", "none")]).unwrap().level; // → high
        assert_eq!(h1, "moderate");
        assert_eq!(h2, "high");
        let model = prior_from_certainty(
            &grade,
            &[("downgraded".into(), h1, None), ("clean".into(), h2, None)],
        )
        .unwrap();
        assert!(model.prior().prob("clean") > model.prior().prob("downgraded"));
    }
}
