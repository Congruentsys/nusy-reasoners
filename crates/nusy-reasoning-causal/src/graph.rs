//! Causal DAG — directed acyclic graph built from Arrow relation data.
//!
//! The DAG is constructed from kanban `RelationsTable` batches or
//! knowledge-graph triples. Edges represent causal influence:
//! `depends_on`, `blocks`, `causes`, etc.
//!
//! Graph mutilation (Pearl's do-operator) removes all incoming edges
//! to a treatment node, simulating an external intervention.

use crate::error::{CausalError, Result};
use arrow::array::{BooleanArray, RecordBatch, StringArray};
use std::collections::{HashMap, HashSet, VecDeque};

/// Opaque node identifier (typically a kanban item ID like "EX-3017").
pub type NodeId = String;

/// A directed edge in the causal DAG.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CausalEdge {
    pub source: NodeId,
    pub target: NodeId,
    pub predicate: String,
}

/// A directed acyclic graph for causal reasoning.
///
/// Nodes are string IDs. Edges are directed: `source -> target` means
/// "source causally influences target" (e.g., `EX-A blocks EX-B` means
/// A's completion causally affects B's timeline).
#[derive(Debug, Clone)]
pub struct CausalDag {
    /// Adjacency list: node → set of (target, predicate).
    children: HashMap<NodeId, Vec<(NodeId, String)>>,
    /// Reverse adjacency: node → set of (source, predicate).
    parents: HashMap<NodeId, Vec<(NodeId, String)>>,
    /// All known nodes.
    nodes: HashSet<NodeId>,
}

impl CausalDag {
    /// Create an empty DAG.
    pub fn new() -> Self {
        CausalDag {
            children: HashMap::new(),
            parents: HashMap::new(),
            nodes: HashSet::new(),
        }
    }

    /// Build a DAG from Arrow relation batches.
    ///
    /// Expects batches with columns at the standard `rel_col` positions:
    /// - col 1: source_id (Utf8)
    /// - col 2: target_id (Utf8)
    /// - col 3: predicate (Utf8)
    /// - col 5: deleted (Boolean)
    ///
    /// Only active (non-deleted) relations with causal predicates are included.
    pub fn from_relation_batches(batches: &[RecordBatch]) -> Result<Self> {
        let mut dag = CausalDag::new();

        for batch in batches {
            let sources = batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("source_id column should be StringArray");
            let targets = batch
                .column(2)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("target_id column should be StringArray");
            let predicates = batch
                .column(3)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("predicate column should be StringArray");
            let deleted = batch
                .column(5)
                .as_any()
                .downcast_ref::<BooleanArray>()
                .expect("deleted column should be BooleanArray");

            for i in 0..batch.num_rows() {
                if deleted.value(i) {
                    continue;
                }

                let source = sources.value(i).to_string();
                let target = targets.value(i).to_string();
                let predicate = predicates.value(i).to_string();

                // Only include causal predicates (directional influence)
                if is_causal_predicate(&predicate) {
                    dag.add_edge(&source, &target, &predicate);
                }
            }
        }

        Ok(dag)
    }

    /// Add a directed edge: source → target.
    pub fn add_edge(&mut self, source: &str, target: &str, predicate: &str) {
        self.nodes.insert(source.to_string());
        self.nodes.insert(target.to_string());

        self.children
            .entry(source.to_string())
            .or_default()
            .push((target.to_string(), predicate.to_string()));

        self.parents
            .entry(target.to_string())
            .or_default()
            .push((source.to_string(), predicate.to_string()));
    }

    /// Add a node without edges.
    pub fn add_node(&mut self, node: &str) {
        self.nodes.insert(node.to_string());
    }

    /// Check if a node exists.
    pub fn has_node(&self, node: &str) -> bool {
        self.nodes.contains(node)
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.children.values().map(|v| v.len()).sum()
    }

    /// All nodes in the DAG.
    pub fn nodes(&self) -> &HashSet<NodeId> {
        &self.nodes
    }

    /// Get children (direct successors) of a node.
    pub fn children_of(&self, node: &str) -> Vec<&(NodeId, String)> {
        self.children
            .get(node)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// Get parents (direct predecessors) of a node.
    pub fn parents_of(&self, node: &str) -> Vec<&(NodeId, String)> {
        self.parents
            .get(node)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// Perform graph mutilation: remove all incoming edges to `treatment`.
    ///
    /// This is Pearl's do-operator: `do(X=x)` is modeled by removing all
    /// arrows pointing INTO X, making X exogenous (set by external intervention
    /// rather than caused by its parents).
    ///
    /// Returns a new DAG with the incoming edges removed.
    pub fn mutilate(&self, treatment: &str) -> Result<CausalDag> {
        if !self.has_node(treatment) {
            return Err(CausalError::NodeNotFound(treatment.to_string()));
        }

        let mut mutilated = self.clone();

        // Remove all incoming edges to treatment
        if let Some(parent_edges) = mutilated.parents.remove(treatment) {
            for (parent, predicate) in &parent_edges {
                if let Some(children) = mutilated.children.get_mut(parent) {
                    children.retain(|(t, p)| !(t == treatment && p == predicate));
                }
            }
        }

        // Ensure the parents entry exists but is empty
        mutilated.parents.insert(treatment.to_string(), Vec::new());

        Ok(mutilated)
    }

    /// Find all ancestors of a node (transitive closure of parents).
    pub fn ancestors(&self, node: &str) -> Result<HashSet<NodeId>> {
        if !self.has_node(node) {
            return Err(CausalError::NodeNotFound(node.to_string()));
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        // Seed with direct parents
        for (parent, _) in self.parents_of(node) {
            queue.push_back(parent.clone());
        }

        while let Some(current) = queue.pop_front() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());
            for (parent, _) in self.parents_of(&current) {
                if !visited.contains(parent) {
                    queue.push_back(parent.clone());
                }
            }
        }

        Ok(visited)
    }

    /// Find all descendants of a node (transitive closure of children).
    pub fn descendants(&self, node: &str) -> Result<HashSet<NodeId>> {
        if !self.has_node(node) {
            return Err(CausalError::NodeNotFound(node.to_string()));
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        for (child, _) in self.children_of(node) {
            queue.push_back(child.clone());
        }

        while let Some(current) = queue.pop_front() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());
            for (child, _) in self.children_of(&current) {
                if !visited.contains(child) {
                    queue.push_back(child.clone());
                }
            }
        }

        Ok(visited)
    }

    /// Check if there is a directed path from `source` to `target`.
    ///
    /// Uses early-termination BFS — stops as soon as `target` is found,
    /// avoiding the cost of computing the full descendant set.
    pub fn has_path(&self, source: &str, target: &str) -> bool {
        if source == target {
            return true;
        }
        if !self.has_node(source) || !self.has_node(target) {
            return false;
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        visited.insert(source.to_string());

        for (child, _) in self.children_of(source) {
            if child == target {
                return true;
            }
            if visited.insert(child.clone()) {
                queue.push_back(child);
            }
        }

        while let Some(current) = queue.pop_front() {
            for (child, _) in self.children_of(current) {
                if child == target {
                    return true;
                }
                if visited.insert(child.clone()) {
                    queue.push_back(child);
                }
            }
        }

        false
    }

    /// Find a directed path from `source` to `target` (BFS, returns first found).
    ///
    /// Returns the sequence of node IDs from source to target (inclusive),
    /// or `None` if no path exists.
    pub fn find_path(&self, source: &str, target: &str) -> Option<Vec<NodeId>> {
        if source == target {
            return Some(vec![source.to_string()]);
        }
        if !self.has_node(source) || !self.has_node(target) {
            return None;
        }

        let mut queue = VecDeque::new();
        let mut came_from: HashMap<NodeId, NodeId> = HashMap::new();
        let sentinel = String::new();

        queue.push_back(source.to_string());
        came_from.insert(source.to_string(), sentinel.clone());

        while let Some(current) = queue.pop_front() {
            for (child, _) in self.children_of(&current) {
                if came_from.contains_key(child) {
                    continue;
                }
                came_from.insert(child.clone(), current.clone());
                if child == target {
                    // Reconstruct path
                    let mut path = vec![target.to_string()];
                    let mut node = target.to_string();
                    while came_from[&node] != sentinel {
                        node = came_from[&node].clone();
                        path.push(node.clone());
                    }
                    path.reverse();
                    return Some(path);
                }
                queue.push_back(child.clone());
            }
        }

        None
    }

    /// Extract the subgraph relevant to a treatment→outcome query.
    ///
    /// Includes: ancestors of treatment, ancestors of outcome, and all
    /// nodes on directed paths between them.
    pub fn extract_relevant(&self, treatment: &str, outcome: &str) -> Result<CausalDag> {
        if !self.has_node(treatment) {
            return Err(CausalError::NodeNotFound(treatment.to_string()));
        }
        if !self.has_node(outcome) {
            return Err(CausalError::NodeNotFound(outcome.to_string()));
        }

        let treatment_ancestors = self.ancestors(treatment)?;
        let outcome_ancestors = self.ancestors(outcome)?;
        let treatment_descendants = self.descendants(treatment)?;

        // Relevant nodes: ancestors of both + descendants of treatment that
        // are ancestors of outcome
        let mut relevant: HashSet<NodeId> = HashSet::new();
        relevant.insert(treatment.to_string());
        relevant.insert(outcome.to_string());
        relevant.extend(treatment_ancestors.iter().cloned());
        relevant.extend(outcome_ancestors.iter().cloned());
        // Nodes on directed paths from treatment to outcome
        for desc in &treatment_descendants {
            if outcome_ancestors.contains(desc) || desc == outcome {
                relevant.insert(desc.clone());
            }
        }

        // Build subgraph with only relevant nodes and their edges
        let mut subgraph = CausalDag::new();
        for node in &relevant {
            subgraph.add_node(node);
        }
        for node in &relevant {
            for (child, pred) in self.children_of(node) {
                if relevant.contains(child) {
                    subgraph.add_edge(node, child, pred);
                }
            }
        }

        Ok(subgraph)
    }
}

impl Default for CausalDag {
    fn default() -> Self {
        Self::new()
    }
}

/// Determine if a relation predicate represents causal influence.
///
/// Causal predicates imply that the source *causes* or *influences* the target.
/// Non-causal predicates (e.g., `related_to`) are bidirectional associations.
fn is_causal_predicate(predicate: &str) -> bool {
    matches!(
        predicate,
        "depends_on"
            | "blocks"
            | "implements"
            | "spawns"
            | "causes"
            | "derived_from"
            | "caused_by"
            | "hyp:measuredBy"
            | "expr:hypothesis"
            | "expr:run_for"
            | "paper:cites"
            | "lit:references"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diamond_dag() -> CausalDag {
        //   A
        //  / \
        // B   C
        //  \ /
        //   D
        let mut dag = CausalDag::new();
        dag.add_edge("A", "B", "blocks");
        dag.add_edge("A", "C", "blocks");
        dag.add_edge("B", "D", "depends_on");
        dag.add_edge("C", "D", "depends_on");
        dag
    }

    fn chain_dag() -> CausalDag {
        // A -> B -> C -> D
        let mut dag = CausalDag::new();
        dag.add_edge("A", "B", "blocks");
        dag.add_edge("B", "C", "blocks");
        dag.add_edge("C", "D", "blocks");
        dag
    }

    #[test]
    fn test_empty_dag() {
        let dag = CausalDag::new();
        assert_eq!(dag.node_count(), 0);
        assert_eq!(dag.edge_count(), 0);
    }

    #[test]
    fn test_add_edges() {
        let dag = diamond_dag();
        assert_eq!(dag.node_count(), 4);
        assert_eq!(dag.edge_count(), 4);
    }

    #[test]
    fn test_children_and_parents() {
        let dag = diamond_dag();
        assert_eq!(dag.children_of("A").len(), 2);
        assert_eq!(dag.parents_of("D").len(), 2);
        assert_eq!(dag.parents_of("A").len(), 0);
        assert_eq!(dag.children_of("D").len(), 0);
    }

    #[test]
    fn test_ancestors() {
        let dag = diamond_dag();
        let ancestors_d = dag.ancestors("D").expect("should find ancestors");
        assert!(ancestors_d.contains("A"));
        assert!(ancestors_d.contains("B"));
        assert!(ancestors_d.contains("C"));
        assert_eq!(ancestors_d.len(), 3);

        let ancestors_a = dag.ancestors("A").expect("should find ancestors");
        assert!(ancestors_a.is_empty());
    }

    #[test]
    fn test_descendants() {
        let dag = diamond_dag();
        let desc_a = dag.descendants("A").expect("should find descendants");
        assert!(desc_a.contains("B"));
        assert!(desc_a.contains("C"));
        assert!(desc_a.contains("D"));
        assert_eq!(desc_a.len(), 3);
    }

    #[test]
    fn test_has_path() {
        let dag = diamond_dag();
        assert!(dag.has_path("A", "D"));
        assert!(dag.has_path("A", "B"));
        assert!(dag.has_path("B", "D"));
        assert!(!dag.has_path("D", "A"));
        assert!(!dag.has_path("B", "C"));
    }

    #[test]
    fn test_mutilate() {
        let dag = diamond_dag();
        let mutilated = dag.mutilate("B").expect("should mutilate");

        // B should have no parents after mutilation
        assert_eq!(mutilated.parents_of("B").len(), 0);
        // A should no longer have B as a child
        let a_children: Vec<&str> = mutilated
            .children_of("A")
            .iter()
            .map(|(n, _)| n.as_str())
            .collect();
        assert!(!a_children.contains(&"B"));
        assert!(a_children.contains(&"C"));
        // B should still have D as a child
        assert_eq!(mutilated.children_of("B").len(), 1);
        // Total nodes unchanged
        assert_eq!(mutilated.node_count(), 4);
    }

    #[test]
    fn test_mutilate_preserves_downstream() {
        let dag = chain_dag();
        let mutilated = dag.mutilate("B").expect("should mutilate");

        // B has no parents
        assert!(mutilated.parents_of("B").is_empty());
        // B -> C still exists
        assert!(mutilated.has_path("B", "D"));
        // A -> B is severed
        assert!(!mutilated.has_path("A", "B"));
        assert!(!mutilated.has_path("A", "D"));
    }

    #[test]
    fn test_mutilate_nonexistent_node() {
        let dag = diamond_dag();
        let result = dag.mutilate("Z");
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_relevant() {
        //   A → B → D
        //   A → C → D
        //   E → F (unrelated)
        let mut dag = CausalDag::new();
        dag.add_edge("A", "B", "blocks");
        dag.add_edge("A", "C", "blocks");
        dag.add_edge("B", "D", "depends_on");
        dag.add_edge("C", "D", "depends_on");
        dag.add_edge("E", "F", "blocks");

        let sub = dag.extract_relevant("A", "D").expect("should extract");
        assert!(sub.has_node("A"));
        assert!(sub.has_node("B"));
        assert!(sub.has_node("C"));
        assert!(sub.has_node("D"));
        // E and F are not relevant to A→D
        assert!(!sub.has_node("E"));
        assert!(!sub.has_node("F"));
    }

    #[test]
    fn test_is_causal_predicate() {
        assert!(is_causal_predicate("depends_on"));
        assert!(is_causal_predicate("blocks"));
        assert!(is_causal_predicate("implements"));
        assert!(is_causal_predicate("causes"));
        assert!(!is_causal_predicate("related_to"));
        assert!(!is_causal_predicate("tagged_with"));
    }

    // ── Early termination tests (EX-4069) ──────────────────────────────────────

    /// Build a wide DAG with `n` nodes: one root, `fan` direct children,
    /// each with a chain of `depth` descendants. Total nodes = 1 + fan × depth.
    fn wide_dag(fan: usize, depth: usize) -> CausalDag {
        let mut dag = CausalDag::new();
        for i in 0..fan {
            let child = format!("child-{i}");
            dag.add_edge("root", &child, "blocks");
            let mut prev = child;
            for d in 1..depth {
                let node = format!("child-{i}-depth-{d}");
                dag.add_edge(&prev, &node, "depends_on");
                prev = node;
            }
        }
        dag
    }

    #[test]
    fn test_has_path_early_termination_positive() {
        // 500-node DAG: root → 10 children, each with chain of 49 descendants
        let dag = wide_dag(10, 50);
        assert_eq!(dag.node_count(), 501);

        // Near target: direct child of root → found immediately
        assert!(dag.has_path("root", "child-0"));

        // Mid-depth target: should find without full traversal
        assert!(dag.has_path("root", "child-5-depth-25"));

        // Deepest target
        assert!(dag.has_path("root", "child-9-depth-49"));
    }

    #[test]
    fn test_has_path_early_termination_negative() {
        let dag = wide_dag(10, 50);

        // No path from leaf to root
        assert!(!dag.has_path("child-0-depth-49", "root"));

        // No path between sibling branches
        assert!(!dag.has_path("child-0", "child-1-depth-49"));
        assert!(!dag.has_path("child-3-depth-10", "child-7-depth-20"));

        // Nonexistent nodes
        assert!(!dag.has_path("root", "nonexistent"));
        assert!(!dag.has_path("nonexistent", "root"));
    }

    #[test]
    fn test_has_path_correctness_matches_descendants() {
        // Verify early-termination has_path agrees with descendants() on all pairs.
        let dag = diamond_dag();
        let nodes = vec!["A", "B", "C", "D"];
        for source in &nodes {
            let desc = dag.descendants(source).unwrap_or_default();
            for target in &nodes {
                let expected = *source == *target || desc.contains(*target);
                assert_eq!(
                    dag.has_path(source, target),
                    expected,
                    "has_path({source}, {target}) mismatch"
                );
            }
        }
    }

    #[test]
    fn test_has_path_correctness_on_wide_dag() {
        let dag = wide_dag(5, 10);
        let nodes: Vec<String> = dag.nodes.iter().cloned().collect();

        // Sample 50+ pairs and verify has_path matches descendants
        let mut checked = 0;
        for source in nodes.iter().take(10) {
            let desc = dag.descendants(source).unwrap_or_default();
            for target in nodes.iter().take(10) {
                let expected = source == target || desc.contains(target);
                assert_eq!(
                    dag.has_path(source, target),
                    expected,
                    "has_path({source}, {target}) mismatch on wide DAG"
                );
                checked += 1;
            }
        }
        assert!(
            checked >= 50,
            "should check at least 50 pairs, got {checked}"
        );
    }

    #[test]
    fn test_has_path_benchmark_500_node_dag() {
        use std::time::Instant;

        let dag = wide_dag(10, 50);
        assert!(dag.node_count() >= 500);

        // Positive queries: root → deep targets
        let positive_targets = [
            "child-0-depth-49",
            "child-5-depth-25",
            "child-9-depth-1",
            "child-3",
        ];

        let start = Instant::now();
        let iterations = 1000;
        for _ in 0..iterations {
            for target in &positive_targets {
                assert!(dag.has_path("root", target));
            }
        }
        let positive_duration = start.elapsed();
        let positive_per_query = positive_duration / (iterations * positive_targets.len() as u32);

        // Negative queries: cross-branch
        let negative_pairs = [
            ("child-0-depth-49", "child-9-depth-49"),
            ("child-0", "child-5-depth-25"),
            ("child-3-depth-10", "child-7-depth-20"),
        ];

        let start = Instant::now();
        for _ in 0..iterations {
            for (s, t) in &negative_pairs {
                assert!(!dag.has_path(s, t));
            }
        }
        let negative_duration = start.elapsed();
        let negative_per_query = negative_duration / (iterations * negative_pairs.len() as u32);

        eprintln!(
            "[EX-4069 Benchmark] 500-node DAG, {iterations} iterations\n\
             Positive queries: {positive_per_query:?}/query (total {positive_duration:?})\n\
             Negative queries: {negative_per_query:?}/query (total {negative_duration:?})"
        );

        // Target: positive queries should be fast (early termination).
        // We don't assert a hard time limit (CI varies), but the test proves
        // the benchmark runs and the eprintln shows timing for review.
    }
}
