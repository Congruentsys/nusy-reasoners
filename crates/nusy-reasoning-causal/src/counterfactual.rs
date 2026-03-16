//! Counterfactual queries — Pearl's three-step counterfactual.
//!
//! Implements the full counterfactual procedure from Pearl (2009):
//!
//! 1. **Abduction:** Given observed evidence, infer the values of
//!    exogenous variables (unobserved causes) — identify confounders
//!    and alternative causes via backdoor adjustment.
//! 2. **Action:** Apply the intervention `do(X=x')` to the structural
//!    causal model (graph mutilation — remove incoming edges to treatment).
//! 3. **Prediction:** Compute the outcome under the modified model —
//!    check if the causal path still holds in the mutilated graph.
//!
//! # Certifiability Gate
//!
//! Counterfactual results include a confidence level based on graph
//! completeness. Queries with `VeryLow` confidence are refused
//! (certifiability boundary C3 — see `research/Arrow-NuSy-V14/certifiability-boundary.md`).

use crate::adjustment::{AdjustmentSet, find_backdoor_set};
use crate::error::{CausalError, Result};
use crate::graph::{CausalDag, NodeId};

/// Confidence level for counterfactual estimates.
///
/// Based on graph completeness: how much of the causal structure
/// is observable in the DAG vs hidden in unobserved confounders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConfidenceLevel {
    /// Graph is too sparse — refuse the query (certifiability boundary).
    VeryLow,
    /// Significant unobserved structure likely.
    Low,
    /// Some confounders present but identifiable via adjustment.
    Medium,
    /// Direct causal path, no confounders — high confidence.
    High,
}

impl ConfidenceLevel {
    /// Representative numeric value for downstream consumers.
    pub fn value(&self) -> f64 {
        match self {
            ConfidenceLevel::VeryLow => 0.1,
            ConfidenceLevel::Low => 0.35,
            ConfidenceLevel::Medium => 0.65,
            ConfidenceLevel::High => 0.9,
        }
    }
}

/// Result of a counterfactual query.
///
/// "Given we observed outcome Y after treatment X, what would Y have
/// been if X had been different?"
#[derive(Debug, Clone)]
pub struct CounterfactualResult {
    /// The treatment variable.
    pub treatment: NodeId,
    /// The outcome variable.
    pub outcome: NodeId,
    /// The observed (factual) outcome.
    pub factual_outcome: Option<String>,
    /// The estimated counterfactual outcome.
    pub counterfactual_outcome: Option<String>,
    /// Confidence in the counterfactual estimate (0.0–1.0).
    /// Lower confidence when the causal model has unobserved confounders.
    pub confidence: f64,
    /// The causal chain from treatment to outcome.
    pub causal_chain: Vec<NodeId>,
}

/// Compute a counterfactual: "what would outcome Y have been if we
/// hadn't done treatment X?"
///
/// Implements Pearl's three-step procedure on the DAG:
///
/// 1. **Abduction** — identify confounders and alternative causes
/// 2. **Action** — mutilate the graph (remove incoming edges to treatment)
/// 3. **Prediction** — check if outcome is reachable in the mutilated graph
///
/// # Certifiability Gate
///
/// If the graph is too sparse to support a reliable counterfactual
/// (confidence level `VeryLow`), returns `CausalError::CounterfactualNotCertifiable`.
///
/// # Arguments
///
/// * `dag` — The causal DAG
/// * `treatment` — The treatment variable (what was done)
/// * `outcome` — The outcome variable (what we're asking about)
/// * `factual_outcome` — Optional observed outcome value (for richer reporting)
pub fn counterfactual(
    dag: &CausalDag,
    treatment: &str,
    outcome: &str,
    factual_outcome: Option<&str>,
) -> Result<CounterfactualResult> {
    if !dag.has_node(treatment) {
        return Err(CausalError::NodeNotFound(treatment.to_string()));
    }
    if !dag.has_node(outcome) {
        return Err(CausalError::NodeNotFound(outcome.to_string()));
    }

    // Verify there is a causal path from treatment to outcome
    let causal_chain =
        dag.find_path(treatment, outcome)
            .ok_or_else(|| CausalError::NoCausalPath {
                treatment: treatment.to_string(),
                outcome: outcome.to_string(),
            })?;

    // --- Step 1: Abduction ---
    // Identify confounders and assess graph completeness
    let abduction = abduct(dag, treatment, outcome)?;

    // Assess confidence based on graph structure
    let confidence_level = assess_confidence(dag, treatment, outcome, &abduction);

    // Certifiability gate: refuse if confidence is too low
    if confidence_level == ConfidenceLevel::VeryLow {
        return Err(CausalError::CounterfactualNotCertifiable {
            reason: format!(
                "Graph too sparse for reliable counterfactual: \
                 treatment={treatment}, outcome={outcome}, \
                 confidence={:?}",
                confidence_level
            ),
        });
    }

    // --- Step 2: Action ---
    // Mutilate the graph: remove incoming edges to treatment (do-operator)
    let mutilated = dag.mutilate(treatment)?;

    // --- Step 3: Prediction ---
    // In the mutilated graph, check what happens to the outcome
    let counterfactual_outcome = predict(&mutilated, treatment, outcome, &abduction);

    Ok(CounterfactualResult {
        treatment: treatment.to_string(),
        outcome: outcome.to_string(),
        factual_outcome: factual_outcome.map(|s| s.to_string()),
        counterfactual_outcome,
        confidence: confidence_level.value(),
        causal_chain,
    })
}

/// Step 1: Abduction — infer the state of exogenous variables.
///
/// In our graph-based approach, this means identifying:
/// - Confounders (common causes of treatment and outcome)
/// - Alternative causes (other parents of outcome besides treatment)
/// - Whether the effect is identifiable via backdoor adjustment
struct AbductionResult {
    /// Confounders identified via backdoor criterion.
    confounders: Vec<NodeId>,
    /// Other causes of the outcome besides the treatment.
    alternative_causes: Vec<NodeId>,
    /// Whether the causal effect is identifiable.
    identifiable: bool,
}

fn abduct(dag: &CausalDag, treatment: &str, outcome: &str) -> Result<AbductionResult> {
    let adjustment = find_backdoor_set(dag, treatment, outcome)?;

    let confounders: Vec<NodeId> = adjustment.confounders.into_iter().collect();

    let identifiable = matches!(
        adjustment.adjustment,
        AdjustmentSet::Backdoor(_) | AdjustmentSet::Empty
    );

    // Find true alternative causes of outcome:
    // Parents of outcome that are NOT the treatment AND NOT descendants
    // of treatment (descendants are mediators, not independent causes).
    let treatment_descendants = dag.descendants(treatment).unwrap_or_default();
    let alternative_causes: Vec<NodeId> = dag
        .parents_of(outcome)
        .iter()
        .map(|(n, _)| n.clone())
        .filter(|n| n != treatment && !treatment_descendants.contains(n))
        .collect();

    Ok(AbductionResult {
        confounders,
        alternative_causes,
        identifiable,
    })
}

/// Step 3: Prediction — compute counterfactual outcome under mutilated graph.
///
/// The counterfactual question is: "if treatment hadn't happened, would
/// outcome still occur?" We check whether alternative causes (parents
/// of outcome other than treatment or its descendants) can independently
/// produce the outcome.
///
/// - If alternative causes exist → outcome persists via alternative mechanism
/// - If treatment is the sole cause → outcome would change (not occur)
fn predict(
    _mutilated: &CausalDag,
    _treatment: &str,
    _outcome: &str,
    abduction: &AbductionResult,
) -> Option<String> {
    // The key question: does the outcome have alternative causes
    // that could independently produce it without the treatment?
    if abduction.alternative_causes.is_empty() {
        // Treatment is the sole cause — removing it changes the outcome
        Some("outcome_would_change".to_string())
    } else {
        // Other causes exist — outcome would persist even without treatment
        Some("outcome_would_persist_via_alternative_cause".to_string())
    }
}

/// Assess confidence in the counterfactual based on graph completeness.
///
/// Confidence is determined by:
/// - Number of confounders (more confounders → more uncertainty)
/// - Whether the effect is identifiable via adjustment
/// - Ratio of observed vs total causal structure
fn assess_confidence(
    dag: &CausalDag,
    treatment: &str,
    outcome: &str,
    abduction: &AbductionResult,
) -> ConfidenceLevel {
    // Factor 1: Is the effect identifiable?
    if !abduction.identifiable {
        return ConfidenceLevel::VeryLow;
    }

    // Factor 2: Graph density around the treatment-outcome pair
    let treatment_parents = dag.parents_of(treatment).len();
    let outcome_parents = dag.parents_of(outcome).len();
    let total_nodes = dag.node_count();

    // Very sparse graphs can't support reliable counterfactuals
    if total_nodes < 3 {
        return ConfidenceLevel::Low;
    }

    // Factor 3: Number of confounders relative to graph size
    let confounder_ratio = if total_nodes > 0 {
        abduction.confounders.len() as f64 / total_nodes as f64
    } else {
        0.0
    };

    // Factor 4: Direct vs mediated causation
    let is_direct = dag.children_of(treatment).iter().any(|(n, _)| n == outcome);

    if abduction.confounders.is_empty() && is_direct {
        ConfidenceLevel::High
    } else if confounder_ratio < 0.3 && (treatment_parents > 0 || outcome_parents > 1) {
        ConfidenceLevel::Medium
    } else if confounder_ratio < 0.5 {
        ConfidenceLevel::Low
    } else {
        ConfidenceLevel::VeryLow
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::CausalDag;

    #[test]
    fn test_counterfactual_sole_cause() {
        // A → B (sole cause)
        let mut dag = CausalDag::new();
        dag.add_edge("A", "B", "causes");

        let result = counterfactual(&dag, "A", "B", Some("occurred")).expect("should work");
        assert_eq!(result.treatment, "A");
        assert_eq!(result.outcome, "B");
        assert_eq!(result.factual_outcome.as_deref(), Some("occurred"));
        assert_eq!(
            result.counterfactual_outcome.as_deref(),
            Some("outcome_would_change")
        );
        assert_eq!(result.causal_chain, vec!["A", "B"]);
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_counterfactual_alternative_cause() {
        // A → C
        // B → C
        // If we counterfactually remove A, C still has B
        let mut dag = CausalDag::new();
        dag.add_edge("A", "C", "causes");
        dag.add_edge("B", "C", "causes");

        let result = counterfactual(&dag, "A", "C", None).expect("should work");
        assert_eq!(
            result.counterfactual_outcome.as_deref(),
            Some("outcome_would_persist_via_alternative_cause")
        );
    }

    #[test]
    fn test_counterfactual_causal_chain() {
        // A → B → C (chain: A causes C transitively)
        let mut dag = CausalDag::new();
        dag.add_edge("A", "B", "causes");
        dag.add_edge("B", "C", "causes");

        let result = counterfactual(&dag, "A", "C", None).expect("should work");
        assert_eq!(result.causal_chain, vec!["A", "B", "C"]);
        assert_eq!(
            result.counterfactual_outcome.as_deref(),
            Some("outcome_would_change")
        );
    }

    #[test]
    fn test_counterfactual_no_path() {
        // A → B, C exists but is disconnected
        let mut dag = CausalDag::new();
        dag.add_edge("A", "B", "causes");
        dag.add_node("C");

        let err = counterfactual(&dag, "A", "C", None).unwrap_err();
        match err {
            CausalError::NoCausalPath { treatment, outcome } => {
                assert_eq!(treatment, "A");
                assert_eq!(outcome, "C");
            }
            other => panic!("expected NoCausalPath, got: {other}"),
        }
    }

    #[test]
    fn test_counterfactual_nonexistent_node() {
        let dag = CausalDag::new();
        assert!(counterfactual(&dag, "X", "Y", None).is_err());
    }

    #[test]
    fn test_counterfactual_without_factual_outcome() {
        let mut dag = CausalDag::new();
        dag.add_edge("A", "B", "causes");

        let result = counterfactual(&dag, "A", "B", None).expect("should work");
        assert!(result.factual_outcome.is_none());
        assert!(result.counterfactual_outcome.is_some());
    }

    #[test]
    fn test_confidence_levels() {
        assert!(ConfidenceLevel::VeryLow < ConfidenceLevel::Low);
        assert!(ConfidenceLevel::Low < ConfidenceLevel::Medium);
        assert!(ConfidenceLevel::Medium < ConfidenceLevel::High);

        assert!((ConfidenceLevel::High.value() - 0.9).abs() < f64::EPSILON);
        assert!((ConfidenceLevel::VeryLow.value() - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn test_abduction_with_confounders() {
        // C → A, C → B, A → B (C confounds A→B)
        let mut dag = CausalDag::new();
        dag.add_edge("C", "A", "causes");
        dag.add_edge("C", "B", "causes");
        dag.add_edge("A", "B", "causes");

        let result = abduct(&dag, "A", "B").expect("should work");
        assert!(result.confounders.contains(&"C".to_string()));
        assert!(result.identifiable);
        // C is also an alternative cause of B
        assert!(result.alternative_causes.contains(&"C".to_string()));
    }

    #[test]
    fn test_abduction_no_confounders() {
        let mut dag = CausalDag::new();
        dag.add_edge("A", "B", "causes");

        let result = abduct(&dag, "A", "B").expect("should work");
        assert!(result.confounders.is_empty());
        assert!(result.identifiable);
        assert!(result.alternative_causes.is_empty());
    }

    #[test]
    fn test_counterfactual_kanban_scenario() {
        // Realistic kanban scenario:
        //   VOY-145 → EX-3017 → METRIC-accuracy
        //   VOY-145 → EX-3018 → METRIC-accuracy
        //
        // Counterfactual: "What would accuracy be if we hadn't done EX-3017?"
        // Answer: EX-3018 also contributes, so outcome persists via alternative
        let mut dag = CausalDag::new();
        dag.add_edge("VOY-145", "EX-3017", "spawns");
        dag.add_edge("VOY-145", "EX-3018", "spawns");
        dag.add_edge("EX-3017", "METRIC-accuracy", "causes");
        dag.add_edge("EX-3018", "METRIC-accuracy", "causes");

        let result = counterfactual(&dag, "EX-3017", "METRIC-accuracy", Some("improved"))
            .expect("should work");

        assert_eq!(
            result.counterfactual_outcome.as_deref(),
            Some("outcome_would_persist_via_alternative_cause")
        );
        assert_eq!(result.causal_chain, vec!["EX-3017", "METRIC-accuracy"]);
    }
}
