//! Interventional queries — formal do-operator for kanban operations.
//!
//! Implements `do_skip`, `do_reprioritize`, and `confounders` using
//! Pearl's graph mutilation + adjustment set computation.
//!
//! These replace the heuristic functions planned in Voyage 2's
//! specification (never implemented) with formal causal operations.

use crate::adjustment::{AdjustmentSet, find_backdoor_set};
use crate::error::{CausalError, Result};
use crate::graph::{CausalDag, NodeId};
use std::collections::HashSet;

/// The effect of an intervention on downstream items.
#[derive(Debug, Clone)]
pub struct InterventionEffect {
    /// Items directly affected (immediate children in mutilated graph).
    pub direct: HashSet<NodeId>,
    /// Items transitively affected (all reachable descendants).
    pub transitive: HashSet<NodeId>,
    /// Items that are blocked (depend on the treatment and have no
    /// alternative path to completion).
    pub blocked: HashSet<NodeId>,
}

/// Full result of an interventional query.
#[derive(Debug, Clone)]
pub struct InterventionResult {
    /// The item being intervened on.
    pub treatment: NodeId,
    /// The intervention type.
    pub intervention: InterventionType,
    /// Downstream effects.
    pub effect: InterventionEffect,
    /// Confounders that could bias the assessment.
    pub confounders: HashSet<NodeId>,
    /// Whether the causal effect is cleanly identifiable.
    pub identifiable: bool,
}

/// Types of intervention on kanban items.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InterventionType {
    /// Skip this item entirely — what downstream work is affected?
    Skip,
    /// Change priority — what changes in execution order?
    Reprioritize { new_priority: String },
}

/// `do(skip item_id)` — Model the effect of skipping (not completing) a kanban item.
///
/// Uses graph mutilation to remove the item's causal influence, then
/// identifies which downstream items are blocked or affected.
///
/// # Returns
///
/// An `InterventionResult` describing:
/// - Which items are directly dependent on the skipped item
/// - Which items are transitively affected
/// - Which items are fully blocked (no alternative path)
/// - What confounders exist
pub fn do_skip(dag: &CausalDag, item_id: &str) -> Result<InterventionResult> {
    if !dag.has_node(item_id) {
        return Err(CausalError::NodeNotFound(item_id.to_string()));
    }

    // Mutilate: remove item's causal influence on children
    // (skipping means its outputs don't propagate)
    let direct_children: HashSet<NodeId> = dag
        .children_of(item_id)
        .iter()
        .map(|(n, _)| n.clone())
        .collect();

    let all_descendants = dag.descendants(item_id)?;

    // Blocked items: descendants that have NO alternative path to their
    // goal (all their dependencies go through the skipped item)
    let blocked = find_blocked_items(dag, item_id, &all_descendants)?;

    // Find confounders for each affected outcome
    let mut all_confounders = HashSet::new();
    for descendant in &direct_children {
        if let Ok(adj) = find_backdoor_set(dag, item_id, descendant) {
            all_confounders.extend(adj.confounders);
        }
    }

    let identifiable = all_confounders.is_empty()
        || direct_children.iter().all(|child| {
            find_backdoor_set(dag, item_id, child)
                .map(|r| {
                    matches!(
                        r.adjustment,
                        AdjustmentSet::Backdoor(_) | AdjustmentSet::Empty
                    )
                })
                .unwrap_or(false)
        });

    Ok(InterventionResult {
        treatment: item_id.to_string(),
        intervention: InterventionType::Skip,
        effect: InterventionEffect {
            direct: direct_children,
            transitive: all_descendants,
            blocked,
        },
        confounders: all_confounders,
        identifiable,
    })
}

/// `do(reprioritize item_id priority)` — Model the effect of changing
/// an item's priority.
///
/// Priority changes affect execution order. Items that depend on the
/// reprioritized item may complete sooner or later. This is modeled
/// as an intervention on the item's outgoing edges (the causal effect
/// of completing it earlier or later).
pub fn do_reprioritize(
    dag: &CausalDag,
    item_id: &str,
    new_priority: &str,
) -> Result<InterventionResult> {
    if !dag.has_node(item_id) {
        return Err(CausalError::NodeNotFound(item_id.to_string()));
    }

    // Reprioritization affects the same downstream items as skip,
    // but none are fully blocked — they're just reordered.
    let direct_children: HashSet<NodeId> = dag
        .children_of(item_id)
        .iter()
        .map(|(n, _)| n.clone())
        .collect();

    let all_descendants = dag.descendants(item_id)?;

    let mut all_confounders = HashSet::new();
    for descendant in &direct_children {
        if let Ok(adj) = find_backdoor_set(dag, item_id, descendant) {
            all_confounders.extend(adj.confounders);
        }
    }

    Ok(InterventionResult {
        treatment: item_id.to_string(),
        intervention: InterventionType::Reprioritize {
            new_priority: new_priority.to_string(),
        },
        effect: InterventionEffect {
            direct: direct_children,
            transitive: all_descendants,
            blocked: HashSet::new(), // Reprioritize doesn't block, just reorders
        },
        confounders: all_confounders,
        identifiable: true, // Priority changes are always identifiable (no confounders for ordering)
    })
}

/// Find confounders for a treatment→outcome relationship.
///
/// Confounders are common causes of both treatment and outcome that
/// create spurious correlation. Conditioning on confounders isolates
/// the true causal effect.
pub fn confounders(dag: &CausalDag, treatment: &str, outcome: &str) -> Result<HashSet<NodeId>> {
    let result = find_backdoor_set(dag, treatment, outcome)?;
    Ok(result.confounders)
}

/// Identify items that are fully blocked when `skipped_item` is removed.
///
/// A downstream item is "blocked" if ALL paths from its parents to it
/// go through the skipped item (no alternative dependency path exists).
fn find_blocked_items(
    dag: &CausalDag,
    skipped_item: &str,
    descendants: &HashSet<NodeId>,
) -> Result<HashSet<NodeId>> {
    let mut blocked = HashSet::new();

    for descendant in descendants {
        // Check if this descendant has any parent that is NOT the skipped
        // item and NOT itself a descendant of the skipped item
        let parents: Vec<&str> = dag
            .parents_of(descendant)
            .iter()
            .map(|(n, _)| n.as_str())
            .collect();

        let has_alternative = parents
            .iter()
            .any(|parent| *parent != skipped_item && !descendants.contains(*parent));

        if !has_alternative && !parents.is_empty() {
            blocked.insert(descendant.clone());
        }
    }

    Ok(blocked)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::CausalDag;

    fn expedition_dag() -> CausalDag {
        // Realistic kanban dependency graph:
        //
        // VY-3016 (voyage)
        //   ├── EX-3017 (C1: taxonomy + crate)
        //   │     └── EX-3019 (C3: certifiability, depends on C1)
        //   │           └── EX-3019-P2 (counterfactual impl)
        //   ├── EX-3018 (C2: formal spec)
        //   │     └── EX-3019 (C3 also depends on C2)
        //   └── LIT-003 (literature, already done)
        //         └── EX-3018 (C2 depends on LIT-003)
        //
        let mut dag = CausalDag::new();
        dag.add_edge("VY-3016", "EX-3017", "spawns");
        dag.add_edge("VY-3016", "EX-3018", "spawns");
        dag.add_edge("VY-3016", "LIT-003", "spawns");
        dag.add_edge("LIT-003", "EX-3018", "depends_on");
        dag.add_edge("EX-3017", "EX-3019", "depends_on");
        dag.add_edge("EX-3018", "EX-3019", "depends_on");
        dag.add_edge("EX-3019", "EX-3019-P2", "depends_on");
        dag
    }

    #[test]
    fn test_do_skip_leaf() {
        let dag = expedition_dag();
        let result = do_skip(&dag, "EX-3019-P2").expect("should work");

        // Leaf node — no downstream effects
        assert!(result.effect.direct.is_empty());
        assert!(result.effect.transitive.is_empty());
        assert!(result.effect.blocked.is_empty());
    }

    #[test]
    fn test_do_skip_with_downstream() {
        let dag = expedition_dag();
        let result = do_skip(&dag, "EX-3017").expect("should work");

        // EX-3017 → EX-3019 → EX-3019-P2
        assert!(result.effect.direct.contains("EX-3019"));
        assert!(result.effect.transitive.contains("EX-3019"));
        assert!(result.effect.transitive.contains("EX-3019-P2"));
    }

    #[test]
    fn test_do_skip_blocking_analysis() {
        // A → C (only path)
        // B → C (alternative path)
        let mut dag = CausalDag::new();
        dag.add_edge("A", "C", "depends_on");
        dag.add_edge("B", "C", "depends_on");

        let result = do_skip(&dag, "A").expect("should work");
        // C is NOT blocked because B provides an alternative path
        assert!(!result.effect.blocked.contains("C"));

        // Now test with only one path:
        let mut dag2 = CausalDag::new();
        dag2.add_edge("A", "C", "depends_on");

        let result2 = do_skip(&dag2, "A").expect("should work");
        // C IS blocked — only path is through A
        assert!(result2.effect.blocked.contains("C"));
    }

    #[test]
    fn test_do_skip_voyage_root() {
        let dag = expedition_dag();
        let result = do_skip(&dag, "VY-3016").expect("should work");

        // Skipping the voyage affects everything
        assert!(result.effect.transitive.contains("EX-3017"));
        assert!(result.effect.transitive.contains("EX-3018"));
        assert!(result.effect.transitive.contains("EX-3019"));
        assert!(result.effect.transitive.contains("LIT-003"));
    }

    #[test]
    fn test_do_reprioritize() {
        let dag = expedition_dag();
        let result = do_reprioritize(&dag, "EX-3017", "critical").expect("should work");

        assert_eq!(
            result.intervention,
            InterventionType::Reprioritize {
                new_priority: "critical".to_string()
            }
        );
        // Same downstream items as skip
        assert!(result.effect.direct.contains("EX-3019"));
        // But nothing is blocked (just reordered)
        assert!(result.effect.blocked.is_empty());
    }

    #[test]
    fn test_confounders_shared_cause() {
        let dag = expedition_dag();
        // VY-3016 is a confounder for EX-3017 → EX-3019 because
        // VY-3016 also spawns EX-3018 which also affects EX-3019
        let conf = confounders(&dag, "EX-3017", "EX-3019").expect("should work");
        assert!(conf.contains("VY-3016"));
    }

    #[test]
    fn test_confounders_no_shared_cause() {
        let mut dag = CausalDag::new();
        dag.add_edge("A", "B", "causes");
        // Direct causation, no confounders
        let conf = confounders(&dag, "A", "B").expect("should work");
        assert!(conf.is_empty());
    }

    #[test]
    fn test_nonexistent_item() {
        let dag = CausalDag::new();
        assert!(do_skip(&dag, "NONEXISTENT").is_err());
        assert!(do_reprioritize(&dag, "NONEXISTENT", "high").is_err());
    }
}
