//! Certainty aggregation — GRADE generalized (VY-Bayes E1, phase 2).
//!
//! A [`RatingSystem`] is an ordered scale of certainty levels (best → worst) plus a map
//! from a component rating to a signed number of **steps** (positive = downgrade, negative
//! = upgrade). GRADE is *one* instance built by [`RatingSystem::grade`] — the scale and the
//! step rule are **data**, so a second system sits beside GRADE with no code change (the
//! same "rating systems are data, not enums" rule the product's certainty store follows).
//!
//! [`EvidenceRating`] is the Arrow-free analog of the product's `Certainty` row: the product
//! adapts its rows into these (no parallel store). Recursive via `parent`: sub-component
//! ratings point at the overall rating they decompose, exactly like `parent_certainty_id`.

use std::collections::BTreeMap;

/// A generic evidence rating — the Arrow-free analog of the product's `Certainty` row.
/// Recursive: a sub-component (`parent = Some(overall_id)`) decomposes a top-level
/// (`parent = None`) overall rating.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceRating {
    pub id: String,
    /// e.g. `Overall` | `RiskOfBias` | `Inconsistency` | `Imprecision` | …
    pub certainty_type: String,
    /// The rating value in the system's scale (a level for `Overall`, a severity such as
    /// `none` | `serious` | `very-serious` for a downgrade component).
    pub rating: String,
    /// e.g. `GRADE` — a value, not an enum.
    pub rating_system: String,
    /// COG context-match axis: `low` | `moderate` | `high` | `exact`.
    pub directness: Option<String>,
    /// Parent (overall) rating this is a sub-component of — `None` for a top-level rating.
    pub parent: Option<String>,
}

impl EvidenceRating {
    /// A top-level rating (no parent) — i.e. an overall grade rather than a component.
    pub fn is_top_level(&self) -> bool {
        self.parent.is_none()
    }
}

/// The result of aggregating a recursive evidence set into an overall certainty level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Aggregate {
    /// The aggregated overall level (a member of the system's scale).
    pub level: String,
    /// Net steps applied from the base (positive = net downgrade, negative = net upgrade,
    /// before clamping to the scale's bounds).
    pub net_steps: i64,
    /// The component rating ids that contributed — provenance of the aggregate.
    pub from: Vec<String>,
}

/// An ordered certainty scale plus the step contribution of each component rating.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RatingSystem {
    pub name: String,
    /// Levels best → worst, e.g. `["high", "moderate", "low", "very-low"]`.
    levels: Vec<String>,
    /// Component rating → signed steps (positive downgrades, negative upgrades).
    steps: BTreeMap<String, i64>,
}

impl RatingSystem {
    /// Build a system from an explicit best→worst scale and a component-step map.
    pub fn new(
        name: impl Into<String>,
        levels: impl IntoIterator<Item = impl Into<String>>,
        steps: impl IntoIterator<Item = (impl Into<String>, i64)>,
    ) -> Self {
        Self {
            name: name.into(),
            levels: levels.into_iter().map(Into::into).collect(),
            steps: steps.into_iter().map(|(k, v)| (k.into(), v)).collect(),
        }
    }

    /// The canonical GRADE instance: four levels and the standard downgrade/upgrade steps.
    /// Risk-of-bias / inconsistency / indirectness / imprecision / publication-bias each
    /// downgrade (`serious` = 1, `very-serious` = 2); large-effect / dose-response upgrade.
    pub fn grade() -> Self {
        Self::new(
            "GRADE",
            ["high", "moderate", "low", "very-low"],
            [
                ("none", 0),
                ("not-serious", 0),
                ("serious", 1),
                ("very-serious", 2),
                // upgrades (apply to observational evidence in GRADE)
                ("large-effect", -1),
                ("very-large-effect", -2),
                ("dose-response", -1),
            ],
        )
    }

    /// The scale, best → worst.
    pub fn levels(&self) -> &[String] {
        &self.levels
    }

    /// The index of `level` in the scale (0 = best), or `None` if not a member.
    pub fn level_index(&self, level: &str) -> Option<usize> {
        self.levels.iter().position(|l| l == level)
    }

    /// The signed steps a component `rating` contributes. Unknown ratings contribute 0
    /// (a rating the system doesn't recognize neither downgrades nor upgrades — it is
    /// ignored rather than silently treated as a downgrade).
    pub fn steps_for(&self, rating: &str) -> i64 {
        self.steps.get(rating).copied().unwrap_or(0)
    }

    /// Clamp a (possibly out-of-range) level index to the scale's bounds.
    fn clamp(&self, idx: i64) -> usize {
        idx.clamp(0, self.levels.len() as i64 - 1) as usize
    }

    /// Aggregate component ratings into an overall level: start at `base`, apply each
    /// component's signed steps, clamp to the scale. Returns `None` if `base` is not a
    /// member of the scale.
    pub fn aggregate<'a>(
        &self,
        base: &str,
        components: impl IntoIterator<Item = (&'a str, &'a str)>, // (component_id, rating)
    ) -> Option<Aggregate> {
        let base_idx = self.level_index(base)? as i64;
        let mut net = 0i64;
        let mut from = Vec::new();
        for (id, rating) in components {
            let s = self.steps_for(rating);
            if s != 0 {
                from.push(id.to_string());
            }
            net += s;
        }
        let idx = self.clamp(base_idx + net);
        Some(Aggregate {
            level: self.levels[idx].clone(),
            net_steps: net,
            from,
        })
    }

    /// Aggregate the recursive ratings for one overall rating drawn from a flat set:
    /// pick the top-level rating (no parent) under this system, collect its direct
    /// children, and aggregate from `base`. Returns `None` if no such top-level rating
    /// exists. The top-level rating's own `rating` is *not* used as the base — `base`
    /// is the evidence design's starting confidence (e.g. `high` for trials, `low` for
    /// observational), per GRADE.
    pub fn aggregate_recursive(&self, ratings: &[EvidenceRating], base: &str) -> Option<Aggregate> {
        let overall = ratings
            .iter()
            .find(|r| r.is_top_level() && r.rating_system == self.name)?;
        let components: Vec<(&str, &str)> = ratings
            .iter()
            .filter(|r| r.parent.as_deref() == Some(overall.id.as_str()))
            .map(|r| (r.id.as_str(), r.rating.as_str()))
            .collect();
        self.aggregate(base, components)
    }

    /// Map a level to a `[0, 1]` confidence weight — best level → 1.0, worst → a small
    /// floor, linearly spaced across the scale. This is the bridge from a certainty level
    /// to a prior/likelihood scale for the Bayesian update (phase 1).
    pub fn weight(&self, level: &str) -> Option<f64> {
        let idx = self.level_index(level)?;
        let n = self.levels.len();
        if n <= 1 {
            return Some(1.0);
        }
        // best (idx 0) → 1.0, worst (idx n-1) → 1/n (a floor, never 0 — abstention, not denial).
        let span = 1.0 - 1.0 / n as f64;
        Some(1.0 - span * (idx as f64 / (n - 1) as f64))
    }
}

/// Discount a confidence weight by a COG directness / context-match label. `exact` keeps
/// the full weight; less-direct evidence is discounted (the directness axis of the
/// certainty store, used for COG transfer scoring).
pub fn directness_factor(directness: Option<&str>) -> f64 {
    match directness {
        Some("exact") | None => 1.0,
        Some("high") => 0.9,
        Some("moderate") => 0.75,
        Some("low") => 0.5,
        Some(_) => 1.0, // unknown label → no discount (don't silently penalize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn comp(id: &str, ty: &str, rating: &str) -> EvidenceRating {
        EvidenceRating {
            id: id.into(),
            certainty_type: ty.into(),
            rating: rating.into(),
            rating_system: "GRADE".into(),
            directness: None,
            parent: Some("overall".into()),
        }
    }
    fn overall(rating: &str) -> EvidenceRating {
        EvidenceRating {
            id: "overall".into(),
            certainty_type: "Overall".into(),
            rating: rating.into(),
            rating_system: "GRADE".into(),
            directness: Some("exact".into()),
            parent: None,
        }
    }

    #[test]
    fn grade_downgrades_from_base() {
        let g = RatingSystem::grade();
        // high, two serious downgrades → low.
        let agg = g
            .aggregate("high", [("rob", "serious"), ("imp", "serious")])
            .unwrap();
        assert_eq!(agg.level, "low");
        assert_eq!(agg.net_steps, 2);
        assert_eq!(agg.from, vec!["rob".to_string(), "imp".to_string()]);
    }

    #[test]
    fn grade_clamps_at_worst() {
        let g = RatingSystem::grade();
        // high minus 2 + 2 = 4 steps → clamp to very-low (index 3).
        let agg = g
            .aggregate("high", [("a", "very-serious"), ("b", "very-serious")])
            .unwrap();
        assert_eq!(agg.level, "very-low");
        assert_eq!(agg.net_steps, 4);
    }

    #[test]
    fn grade_upgrades_observational() {
        let g = RatingSystem::grade();
        // low base, large effect upgrades one step → moderate.
        let agg = g.aggregate("low", [("eff", "large-effect")]).unwrap();
        assert_eq!(agg.level, "moderate");
        assert_eq!(agg.net_steps, -1);
    }

    #[test]
    fn unknown_component_rating_is_ignored_not_a_downgrade() {
        let g = RatingSystem::grade();
        let agg = g.aggregate("high", [("x", "totally-unknown")]).unwrap();
        assert_eq!(agg.level, "high");
        assert_eq!(agg.net_steps, 0);
        assert!(agg.from.is_empty());
    }

    #[test]
    fn recursive_aggregation_uses_children_of_the_overall() {
        let g = RatingSystem::grade();
        let ratings = vec![
            overall("high"),
            comp("rob", "RiskOfBias", "serious"),
            comp("inc", "Inconsistency", "none"),
            comp("imp", "Imprecision", "serious"),
        ];
        // high, two serious → low.
        let agg = g.aggregate_recursive(&ratings, "high").unwrap();
        assert_eq!(agg.level, "low");
        assert_eq!(agg.from, vec!["rob".to_string(), "imp".to_string()]);
    }

    #[test]
    fn unknown_base_level_yields_none() {
        let g = RatingSystem::grade();
        assert!(g.aggregate("excellent", [("a", "serious")]).is_none());
    }

    #[test]
    fn second_rating_system_coexists_with_grade() {
        // A synthetic 3-level system, data only — no code change.
        let oxford = RatingSystem::new("OxfordCEBM", ["1", "2", "3"], [("minor", 1), ("major", 2)]);
        let agg = oxford.aggregate("1", [("c", "major")]).unwrap();
        assert_eq!(agg.level, "3");
        // GRADE still works independently.
        assert_eq!(RatingSystem::grade().level_index("moderate"), Some(1));
    }

    #[test]
    fn weight_is_monotonic_best_to_worst() {
        let g = RatingSystem::grade();
        let w_high = g.weight("high").unwrap();
        let w_mod = g.weight("moderate").unwrap();
        let w_vlow = g.weight("very-low").unwrap();
        assert!((w_high - 1.0).abs() < 1e-9);
        assert!(w_high > w_mod && w_mod > w_vlow);
        assert!(
            w_vlow > 0.0,
            "worst level floors above 0 — abstention, not denial"
        );
    }

    #[test]
    fn directness_discounts_less_direct_evidence() {
        assert_eq!(directness_factor(Some("exact")), 1.0);
        assert_eq!(directness_factor(None), 1.0);
        assert!(directness_factor(Some("low")) < directness_factor(Some("high")));
    }
}
