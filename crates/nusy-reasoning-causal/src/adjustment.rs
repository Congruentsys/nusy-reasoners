//! Adjustment set identification — backdoor and frontdoor criteria.
//!
//! Given a treatment variable X and outcome variable Y in a causal DAG,
//! find a set of variables Z such that conditioning on Z blocks all
//! spurious (non-causal) paths from X to Y.
//!
//! # Backdoor Criterion (Pearl, 2009)
//!
//! A set Z satisfies the backdoor criterion relative to (X, Y) if:
//! 1. No node in Z is a descendant of X
//! 2. Z blocks every path from X to Y that starts with an arrow INTO X
//!    (i.e., Z d-separates X from Y in the mutilated graph G_X̄)
//!
//! In practice for our kanban DAG: the backdoor set is the set of
//! non-descendant nodes that are common ancestors of both X and Y
//! (confounders).

use crate::error::{CausalError, Result};
use crate::graph::{CausalDag, NodeId};
use std::collections::HashSet;

/// The type of adjustment used for causal identification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdjustmentSet {
    /// Backdoor adjustment: condition on these confounders.
    Backdoor(HashSet<NodeId>),
    /// Empty set — no confounders, direct causal effect is identifiable
    /// without adjustment (no backdoor paths exist).
    Empty,
}

/// Result of attempting to identify the causal effect.
#[derive(Debug, Clone)]
pub struct AdjustmentResult {
    pub treatment: NodeId,
    pub outcome: NodeId,
    pub adjustment: AdjustmentSet,
    /// Nodes that are confounders (common causes of treatment and outcome).
    pub confounders: HashSet<NodeId>,
}

/// Find the backdoor adjustment set for estimating the causal effect
/// of `treatment` on `outcome`.
///
/// The backdoor set Z is the minimal set of non-descendant nodes that
/// block all backdoor paths (paths from treatment to outcome that go
/// through parents of treatment).
///
/// # Algorithm
///
/// 1. Find ancestors of treatment
/// 2. Find ancestors of outcome
/// 3. Confounders = ancestors of treatment ∩ ancestors of outcome
///    (common causes that create spurious correlation)
/// 4. Remove any confounders that are descendants of treatment
///    (conditioning on descendants introduces collider bias)
/// 5. The remaining set satisfies the backdoor criterion
pub fn find_backdoor_set(
    dag: &CausalDag,
    treatment: &str,
    outcome: &str,
) -> Result<AdjustmentResult> {
    if !dag.has_node(treatment) {
        return Err(CausalError::NodeNotFound(treatment.to_string()));
    }
    if !dag.has_node(outcome) {
        return Err(CausalError::NodeNotFound(outcome.to_string()));
    }

    let treatment_ancestors = dag.ancestors(treatment)?;
    let outcome_ancestors = dag.ancestors(outcome)?;
    let treatment_descendants = dag.descendants(treatment)?;

    // Confounders: nodes that are ancestors of BOTH treatment and outcome
    let common_ancestors: HashSet<NodeId> = treatment_ancestors
        .intersection(&outcome_ancestors)
        .cloned()
        .collect();

    // Remove descendants of treatment from adjustment set
    // (conditioning on descendants introduces collider bias)
    let backdoor_set: HashSet<NodeId> = common_ancestors
        .iter()
        .filter(|node| !treatment_descendants.contains(*node))
        .cloned()
        .collect();

    let adjustment = if backdoor_set.is_empty() {
        AdjustmentSet::Empty
    } else {
        AdjustmentSet::Backdoor(backdoor_set.clone())
    };

    Ok(AdjustmentResult {
        treatment: treatment.to_string(),
        outcome: outcome.to_string(),
        adjustment,
        confounders: backdoor_set,
    })
}

/// Check whether the causal effect of `treatment` on `outcome` is
/// identifiable via the backdoor criterion.
///
/// Returns `true` if a valid adjustment set exists (including the empty set
/// when there are no confounders).
pub fn is_identifiable(dag: &CausalDag, treatment: &str, outcome: &str) -> Result<bool> {
    let result = find_backdoor_set(dag, treatment, outcome)?;
    // The effect is identifiable if we found a valid adjustment set
    // (Empty means no confounders — direct effect is identifiable)
    Ok(matches!(
        result.adjustment,
        AdjustmentSet::Backdoor(_) | AdjustmentSet::Empty
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::CausalDag;

    #[test]
    fn test_no_confounders() {
        // A -> B (direct cause, no confounders)
        let mut dag = CausalDag::new();
        dag.add_edge("A", "B", "causes");

        let result = find_backdoor_set(&dag, "A", "B").expect("should work");
        assert_eq!(result.adjustment, AdjustmentSet::Empty);
        assert!(result.confounders.is_empty());
    }

    #[test]
    fn test_single_confounder() {
        // Classic confounding:
        //   C → A
        //   C → B
        //   A → B
        // C confounds A→B
        let mut dag = CausalDag::new();
        dag.add_edge("C", "A", "causes");
        dag.add_edge("C", "B", "causes");
        dag.add_edge("A", "B", "causes");

        let result = find_backdoor_set(&dag, "A", "B").expect("should work");
        assert!(result.confounders.contains("C"));
        assert_eq!(result.confounders.len(), 1);
        match &result.adjustment {
            AdjustmentSet::Backdoor(set) => assert!(set.contains("C")),
            _ => panic!("expected backdoor adjustment"),
        }
    }

    #[test]
    fn test_chain_no_confounder() {
        // A → B → C (chain, no confounding)
        let mut dag = CausalDag::new();
        dag.add_edge("A", "B", "causes");
        dag.add_edge("B", "C", "causes");

        let result = find_backdoor_set(&dag, "A", "C").expect("should work");
        assert_eq!(result.adjustment, AdjustmentSet::Empty);
    }

    #[test]
    fn test_descendant_excluded() {
        // Confounder C, but also a mediator M (descendant of treatment):
        //   C → A
        //   C → B
        //   A → M → B
        // M is a descendant of A — must NOT be in adjustment set
        let mut dag = CausalDag::new();
        dag.add_edge("C", "A", "causes");
        dag.add_edge("C", "B", "causes");
        dag.add_edge("A", "M", "causes");
        dag.add_edge("M", "B", "causes");

        let result = find_backdoor_set(&dag, "A", "B").expect("should work");
        assert!(result.confounders.contains("C"));
        assert!(!result.confounders.contains("M"));
    }

    #[test]
    fn test_identifiable() {
        let mut dag = CausalDag::new();
        dag.add_edge("A", "B", "causes");

        assert!(is_identifiable(&dag, "A", "B").expect("should work"));
    }

    #[test]
    fn test_kanban_confounding_scenario() {
        // Realistic kanban scenario:
        //   VOY-145 (voyage) → EX-3017 (our expedition)
        //   VOY-145 → EX-3018 (another expedition)
        //   EX-3017 → METRIC-accuracy
        //   EX-3018 → METRIC-accuracy
        //
        // VOY-145 confounds EX-3017→METRIC-accuracy because it also
        // influences EX-3018 which also affects the metric.
        let mut dag = CausalDag::new();
        dag.add_edge("VOY-145", "EX-3017", "spawns");
        dag.add_edge("VOY-145", "EX-3018", "spawns");
        dag.add_edge("EX-3017", "METRIC-accuracy", "causes");
        dag.add_edge("EX-3018", "METRIC-accuracy", "causes");

        let result = find_backdoor_set(&dag, "EX-3017", "METRIC-accuracy").expect("should work");
        assert!(result.confounders.contains("VOY-145"));
    }

    #[test]
    fn test_nonexistent_node() {
        let dag = CausalDag::new();
        assert!(find_backdoor_set(&dag, "A", "B").is_err());
    }
}
