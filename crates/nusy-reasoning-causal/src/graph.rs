//! Causal DAG — directed acyclic graph built from Arrow relation data.
//!
//! The DAG is constructed from kanban `RelationsTable` batches or
//! knowledge-graph triples. Edges represent causal influence:
//! `depends_on`, `blocks`, `causes`, etc.
//!
//! Graph mutilation (Pearl's do-operator) uses Arrow-style bitmask masking:
//! instead of cloning the entire DAG, incoming edges to the treatment node
//! are masked via a lightweight `HashSet`. The heavy graph data is shared
//! through `Arc`, making mutilation O(1) instead of O(V + E).

use crate::error::{CausalError, Result};
use arrow::array::{BooleanArray, RecordBatch, StringArray};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

/// Opaque node identifier (typically a kanban item ID like "EX-3017").
pub type NodeId = String;

/// A directed edge in the causal DAG.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CausalEdge {
    pub source: NodeId,
    pub target: NodeId,
    pub predicate: String,
}

/// Lazily-built integer index and reachability cache (EX-4071).
///
/// `None` means the cache is stale or never built. The cache is rebuilt
/// on first access after any mutation (add_edge, add_node).
#[derive(Debug, Clone, Default)]
#[allow(dead_code)] // adj/radj used during build; kept for future COO integration
struct ReachabilityCache {
    /// String → integer index mapping.
    node_to_idx: HashMap<String, usize>,
    /// Integer index → string mapping.
    idx_to_node: Vec<String>,
    /// Adjacency list by index: adj[i] = child indices of node i.
    adj: Vec<Vec<usize>>,
    /// Reverse adjacency by index: radj[i] = parent indices of node i.
    radj: Vec<Vec<usize>>,
    /// reachable[i] = set of all j where node i can reach node j (transitively).
    reachable: Vec<HashSet<usize>>,
}

/// Immutable base data shared between original and mutilated DAGs.
///
/// Shared via `Arc` so that mutilation (and repeated mutilation) costs
/// O(1) for the graph structure — only the mask is copied.
#[derive(Debug, Clone, Default)]
struct DagBase {
    /// Adjacency list: node → set of (target, predicate).
    children: HashMap<NodeId, Vec<(NodeId, String)>>,
    /// Reverse adjacency: node → set of (source, predicate).
    parents: HashMap<NodeId, Vec<(NodeId, String)>>,
    /// All known nodes.
    nodes: HashSet<NodeId>,
}

/// A directed acyclic graph for causal reasoning.
///
/// Nodes are string IDs. Edges are directed: `source -> target` means
/// "source causally influences target" (e.g., `EX-A blocks EX-B` means
/// A's completion causally affects B's timeline).
///
/// Graph mutilation uses bitmask masking: the heavy graph data is shared
/// via `Arc<DagBase>`, and a `masked_targets` set records which nodes
/// have had their incoming edges severed. Traversal methods check the
/// mask, returning the same results as a cloned+mutated DAG but without
/// the O(V + E) clone cost.
///
/// Internally maintains a lazily-built reachability cache for O(1)
/// `has_path` lookups (EX-4071).
#[derive(Debug, Clone)]
pub struct CausalDag {
    /// Shared immutable graph structure. Cloned via Arc (O(1) ref-count bump).
    base: Arc<DagBase>,
    /// Mutilation mask: nodes whose incoming parent edges are inactive.
    /// Empty for the original (unmutilated) graph.
    masked_targets: HashSet<NodeId>,
    /// Lazily-built integer index + reachability cache (EX-4071).
    cache: RefCell<Option<ReachabilityCache>>,
}

impl CausalDag {
    /// Create an empty DAG.
    pub fn new() -> Self {
        CausalDag {
            base: Arc::new(DagBase::default()),
            masked_targets: HashSet::new(),
            cache: RefCell::new(None),
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
        let mut base = DagBase::default();

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
                    base.nodes.insert(source.clone());
                    base.nodes.insert(target.clone());
                    base.children
                        .entry(source)
                        .or_default()
                        .push((target, predicate));
                    // Note: we push (source, predicate) for parents below
                }
            }
        }

        // Build parents index from children
        for (source, edges) in &base.children {
            for (target, predicate) in edges {
                base.parents
                    .entry(target.clone())
                    .or_default()
                    .push((source.clone(), predicate.clone()));
            }
        }

        Ok(CausalDag {
            base: Arc::new(base),
            masked_targets: HashSet::new(),
            cache: RefCell::new(None),
        })
    }

    /// Build a DAG from Y-graph knowledge triples (EX-4638, VOY-V18-1).
    ///
    /// The derivation front-end for V18: lets the existing Pearl do-calculus
    /// engine run over Y1/Y2 knowledge triples instead of kanban relations.
    /// Unlike [`from_relation_batches`](Self::from_relation_batches) (kanban
    /// `RelationsTable` schema: source/target/predicate at cols 1/2/3), this
    /// reads the **Y-graph triple schema** — subject/predicate/object at the
    /// `nusy_arrow_core::schema::col` positions (1/2/3) — and turns each
    /// causal-predicate triple into a directed edge `subject → object`.
    ///
    /// Only active triples should be passed in; the caller (e.g. a dual-store
    /// `query`) is responsible for excluding tombstoned rows. Non-causal
    /// predicates (labels, types, …) are skipped, mirroring
    /// `DualStore::build_causal_dag`.
    pub fn from_y_graph(batches: &[RecordBatch]) -> Result<Self> {
        // Y-graph triple schema column positions (nusy_arrow_core::schema::col).
        const SUBJECT_COL: usize = 1;
        const PREDICATE_COL: usize = 2;
        const OBJECT_COL: usize = 3;

        let mut dag = CausalDag::new();

        for batch in batches {
            let subjects = batch
                .column(SUBJECT_COL)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("subject column should be StringArray");
            let predicates = batch
                .column(PREDICATE_COL)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("predicate column should be StringArray");
            let objects = batch
                .column(OBJECT_COL)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("object column should be StringArray");

            for i in 0..batch.num_rows() {
                let predicate = predicates.value(i);
                // Only causal predicates become directional edges.
                if is_causal_predicate(predicate) {
                    dag.add_edge(subjects.value(i), objects.value(i), predicate);
                }
            }
        }

        Ok(dag)
    }

    /// Add a directed edge: source → target.
    ///
    /// Uses `Arc::make_mut` for copy-on-write: if the base data is shared
    /// (e.g., after mutilation), it is cloned before mutation. If unique,
    /// mutation is in-place (zero cost).
    pub fn add_edge(&mut self, source: &str, target: &str, predicate: &str) {
        let base = Arc::make_mut(&mut self.base);
        base.nodes.insert(source.to_string());
        base.nodes.insert(target.to_string());

        base.children
            .entry(source.to_string())
            .or_default()
            .push((target.to_string(), predicate.to_string()));

        base.parents
            .entry(target.to_string())
            .or_default()
            .push((source.to_string(), predicate.to_string()));

        *self.cache.borrow_mut() = None;
    }

    /// Add a node without edges.
    pub fn add_node(&mut self, node: &str) {
        Arc::make_mut(&mut self.base).nodes.insert(node.to_string());
        *self.cache.borrow_mut() = None;
    }

    /// Check if a node exists.
    pub fn has_node(&self, node: &str) -> bool {
        self.base.nodes.contains(node)
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.base.nodes.len()
    }

    /// Number of active edges (respecting mutilation mask).
    pub fn edge_count(&self) -> usize {
        if self.masked_targets.is_empty() {
            return self.base.children.values().map(|v| v.len()).sum();
        }
        self.base
            .children
            .values()
            .map(|v| {
                v.iter()
                    .filter(|(target, _)| !self.masked_targets.contains(target))
                    .count()
            })
            .sum()
    }

    /// All nodes in the DAG (including masked ones).
    pub fn nodes(&self) -> &HashSet<NodeId> {
        &self.base.nodes
    }

    /// Get children (direct successors) of a node, excluding edges
    /// targeting masked (mutilated) nodes.
    pub fn children_of(&self, node: &str) -> Vec<&(NodeId, String)> {
        self.base
            .children
            .get(node)
            .map(|v| {
                v.iter()
                    .filter(|(target, _)| !self.masked_targets.contains(target))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get parents (direct predecessors) of a node.
    ///
    /// Returns empty for masked (mutilated) nodes — their incoming edges
    /// are considered severed by the mutilation mask.
    pub fn parents_of(&self, node: &str) -> Vec<&(NodeId, String)> {
        if self.masked_targets.contains(node) {
            return Vec::new();
        }
        self.base
            .parents
            .get(node)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// Perform graph mutilation: remove all incoming edges to `treatment`.
    ///
    /// This is Pearl's do-operator: `do(X=x)` is modeled by masking all
    /// arrows pointing INTO X, making X exogenous (set by external intervention
    /// rather than caused by its parents).
    ///
    /// Instead of cloning the entire DAG (O(V + E)), this creates a new view
    /// sharing the same `Arc<DagBase>` and adds the treatment to the mask.
    /// The cost is O(M) where M = number of previously masked targets (typically 0).
    ///
    /// Traversal methods (`children_of`, `parents_of`, etc.) automatically
    /// respect the mask, returning the same results as a cloned+mutated DAG.
    pub fn mutilate(&self, treatment: &str) -> Result<CausalDag> {
        if !self.has_node(treatment) {
            return Err(CausalError::NodeNotFound(treatment.to_string()));
        }

        let mut masked = self.masked_targets.clone();
        masked.insert(treatment.to_string());

        Ok(CausalDag {
            base: Arc::clone(&self.base),
            masked_targets: masked,
            cache: RefCell::new(None),
        })
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
    /// Uses a lazily-built reachability cache for O(1) lookups after the
    /// first query (EX-4071). The cache is invalidated whenever edges or
    /// nodes are added.
    pub fn has_path(&self, source: &str, target: &str) -> bool {
        if source == target {
            return true;
        }
        self.ensure_cache();
        let cache = self.cache.borrow();
        let c = cache.as_ref().expect("cache should be built");
        let Some(&src_idx) = c.node_to_idx.get(source) else {
            return false;
        };
        let Some(&tgt_idx) = c.node_to_idx.get(target) else {
            return false;
        };
        c.reachable[src_idx].contains(&tgt_idx)
    }

    /// Build the integer node index and reachability cache.
    ///
    /// Assigns contiguous `0..n` indices, populates integer adjacency lists,
    /// then runs BFS from each node to compute transitive reachability.
    /// Respects the mutilation mask when traversing edges.
    fn build_cache(&self) -> ReachabilityCache {
        let n = self.base.nodes.len();
        let mut node_to_idx = HashMap::with_capacity(n);
        let mut idx_to_node = Vec::with_capacity(n);

        for node in &self.base.nodes {
            let idx = idx_to_node.len();
            node_to_idx.insert(node.clone(), idx);
            idx_to_node.push(node.clone());
        }

        let mut adj = vec![Vec::new(); n];
        let mut radj = vec![Vec::new(); n];

        for (node, children) in &self.base.children {
            let src = node_to_idx[node];
            for (child, _pred) in children {
                if self.masked_targets.contains(child) {
                    continue;
                }
                if let Some(&tgt) = node_to_idx.get(child) {
                    adj[src].push(tgt);
                    radj[tgt].push(src);
                }
            }
        }

        let mut reachable = vec![HashSet::new(); n];
        for start in 0..n {
            let mut visited = HashSet::new();
            let mut queue = VecDeque::new();

            for &child in &adj[start] {
                if visited.insert(child) {
                    queue.push_back(child);
                }
            }

            while let Some(current) = queue.pop_front() {
                for &child in &adj[current] {
                    if visited.insert(child) {
                        queue.push_back(child);
                    }
                }
            }

            reachable[start] = visited;
        }

        ReachabilityCache {
            node_to_idx,
            idx_to_node,
            adj,
            radj,
            reachable,
        }
    }

    /// Ensure the reachability cache is built and valid.
    fn ensure_cache(&self) {
        if self.cache.borrow().is_some() {
            return;
        }
        let cache = self.build_cache();
        *self.cache.borrow_mut() = Some(cache);
    }

    /// Return the reachability matrix in COO (coordinate) sparse format.
    ///
    /// Returns `(n_nodes, row_indices, col_indices)` where each pair
    /// `(row_indices[k], col_indices[k])` represents a reachable pair
    /// with implicit value 1.0. Compatible with `CooMatrix` from
    /// `kg_attention.rs`.
    ///
    /// The cache is built lazily if not already valid.
    pub fn reachability_matrix(&self) -> (usize, Vec<usize>, Vec<usize>) {
        self.ensure_cache();
        let cache = self.cache.borrow();
        let c = cache.as_ref().expect("cache should be built");
        let n = c.idx_to_node.len();
        let mut rows = Vec::new();
        let mut cols = Vec::new();

        for i in 0..n {
            for &j in &c.reachable[i] {
                rows.push(i);
                cols.push(j);
            }
        }

        (n, rows, cols)
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
///
/// The accepted set spans:
/// - Project / kanban edges: `depends_on`, `blocks`, `implements`, `spawns`
/// - Generic causal: `causes`, `caused_by`, `derived_from`
/// - Research provenance: `hyp:measuredBy`, `expr:hypothesis`, `expr:run_for`,
///   `paper:cites`, `lit:references`
/// - Formal-reasoning (CH-4433): `proves`, `is_proved_by`, `axiom_of`,
///   `follows_from`, `entails`, `derives`, `precedes_in_proof`,
///   `concludes_proof`, `is_definition_of`, `defines`
///
/// L1 arithmetic predicates (`equals`, `sum_of`, `divides`, `is_prime`, etc.)
/// are intentionally REJECTED — they encode mathematical fact, not causal
/// influence. Phase 6's plate refiner walks inferential L2 edges (proves /
/// entails / derives), not arithmetic content edges. See
/// `research/audits/predicate-math-audit.md` §3 for the Pearl-classification
/// rationale.
fn is_causal_predicate(predicate: &str) -> bool {
    matches!(
        predicate,
        // Project / kanban edges
        "depends_on"
            | "blocks"
            | "implements"
            | "spawns"
            // Generic causal
            | "causes"
            | "derived_from"
            | "caused_by"
            // Research provenance
            | "hyp:measuredBy"
            | "expr:hypothesis"
            | "expr:run_for"
            | "paper:cites"
            | "lit:references"
            // Formal reasoning (CH-4433 — VY-4412 Phase 6 prerequisite)
            | "proves"
            | "is_proved_by"
            | "axiom_of"
            | "follows_from"
            | "entails"
            | "derives"
            | "precedes_in_proof"
            | "concludes_proof"
            | "is_definition_of"
            | "defines"
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
    fn test_mutilate_edge_count() {
        //   A
        //  / \
        // B   C
        //  \ /
        //   D
        let dag = diamond_dag();
        assert_eq!(dag.edge_count(), 4);

        let mutilated = dag.mutilate("B").expect("should mutilate");
        // A→B is severed; A→C, B→D, C→D remain
        assert_eq!(mutilated.edge_count(), 3);
    }

    #[test]
    fn test_mutilate_shared_base() {
        // Verify that mutilation shares the base data (Arc ref count).
        let dag = diamond_dag();
        let mutilated = dag.mutilate("B").expect("should mutilate");

        // Both should reference the same DagBase — Arc pointers are equal
        assert!(Arc::ptr_eq(&dag.base, &mutilated.base));
    }

    #[test]
    fn test_mutilate_cascading() {
        // Mutilate multiple nodes — mask accumulates.
        //   A → B → C
        let dag = chain_dag();
        let m1 = dag.mutilate("B").expect("first mutilation");
        assert_eq!(m1.masked_targets.len(), 1);

        let m2 = m1.mutilate("C").expect("second mutilation");
        assert_eq!(m2.masked_targets.len(), 2);

        // C has no parents (masked), B has no parents (masked)
        assert!(m2.parents_of("B").is_empty());
        assert!(m2.parents_of("C").is_empty());
        // A→B severed, B→C severed
        assert!(!m2.has_path("A", "C"));
        assert!(!m2.has_path("B", "C"));
    }

    #[test]
    fn test_mutilate_mask_correctness_vs_full_dag() {
        // Verify masked traversal matches the original clone-based semantics
        // across multiple DAG shapes and mutilation targets.
        let dag = diamond_dag();

        // Mutilate each non-root node and verify consistency
        for target in ["B", "C", "D"] {
            let mutilated = dag.mutilate(target).expect("should mutilate");

            // Target has no parents
            assert!(
                mutilated.parents_of(target).is_empty(),
                "{target} should have no parents after mutilation"
            );

            // All nodes still present
            assert_eq!(mutilated.node_count(), dag.node_count());

            // Traversal from root: no path to target if target's parents were
            // the only incoming edges (B, C have only A as parent; D has B, C).
            if target == "D" {
                // D's parents (B, C) are still reachable from A, but D's
                // incoming edges are masked — however has_path checks outgoing,
                // so A→B→D path: B→D edge is filtered (D is masked in children_of(B))
                assert!(!mutilated.has_path("A", "D"));
                assert!(!mutilated.has_path("B", "D"));
            } else {
                // B or C: A→target edge is masked
                assert!(!mutilated.has_path("A", target));
            }
        }
    }

    #[test]
    fn test_mutilate_benchmark_mask_vs_clone_cost() {
        use std::time::Instant;

        let dag = wide_dag(50, 100);
        assert!(dag.node_count() >= 5000);

        let iterations = 1000;

        // Measure mutilation cost (Arc clone + HashSet insert)
        let start = Instant::now();
        let mut mutilated_dags = Vec::with_capacity(iterations);
        for i in 0..iterations {
            // Mutilate different nodes each time
            let target = format!("child-{}-depth-1", i % 50);
            mutilated_dags.push(dag.mutilate(&target).expect("should mutilate"));
        }
        let mask_duration = start.elapsed();

        // Verify all mutilated DAGs share the same base
        for mutilated in &mutilated_dags {
            assert!(Arc::ptr_eq(&dag.base, &mutilated.base));
        }

        let node_count = dag.node_count();
        let per_mutilation = mask_duration / iterations as u32;
        eprintln!(
            "[EX-4070 Benchmark] {node_count}-node DAG, {iterations} mutilations via mask\n\
             Total: {mask_duration:?}\n\
             Per-mutilation: {per_mutilation:?}"
        );

        // The key invariant: all mutilated views share one DagBase
        // Memory cost per mutilation = sizeof(HashSet<NodeId>) + 1 entry
        // vs. old approach = sizeof(HashMap<NodeId, Vec<...>>) * 2 + HashSet
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
        // Project / kanban edges
        assert!(is_causal_predicate("depends_on"));
        assert!(is_causal_predicate("blocks"));
        assert!(is_causal_predicate("implements"));
        assert!(is_causal_predicate("spawns"));
        // Generic causal
        assert!(is_causal_predicate("causes"));
        assert!(is_causal_predicate("caused_by"));
        assert!(is_causal_predicate("derived_from"));
        // Research provenance
        assert!(is_causal_predicate("hyp:measuredBy"));
        assert!(is_causal_predicate("expr:hypothesis"));
        assert!(is_causal_predicate("expr:run_for"));
        assert!(is_causal_predicate("paper:cites"));
        assert!(is_causal_predicate("lit:references"));
        // Non-causal observational
        assert!(!is_causal_predicate("related_to"));
        assert!(!is_causal_predicate("tagged_with"));
    }

    #[test]
    fn test_is_causal_predicate_formal_reasoning() {
        // CH-4433 — L2 formal-reasoning predicates ARE inferentially
        // directional. Phase 6's plate refiner walks chains via these.
        assert!(is_causal_predicate("proves"));
        assert!(is_causal_predicate("is_proved_by"));
        assert!(is_causal_predicate("axiom_of"));
        assert!(is_causal_predicate("follows_from"));
        assert!(is_causal_predicate("entails"));
        assert!(is_causal_predicate("derives"));
        assert!(is_causal_predicate("precedes_in_proof"));
        assert!(is_causal_predicate("concludes_proof"));
        assert!(is_causal_predicate("is_definition_of"));
        assert!(is_causal_predicate("defines"));
    }

    #[test]
    fn test_is_causal_predicate_rejects_l1_math() {
        // CH-4433 — L1 arithmetic predicates encode mathematical fact, not
        // causal influence. They MUST stay rejected so that CausalDag walks
        // do not traverse arithmetic content edges as if they were causal.
        // See research/audits/predicate-math-audit.md §3.
        assert!(!is_causal_predicate("equals"));
        assert!(!is_causal_predicate("equal_to"));
        assert!(!is_causal_predicate("sum_of"));
        assert!(!is_causal_predicate("product_of"));
        assert!(!is_causal_predicate("difference_of"));
        assert!(!is_causal_predicate("quotient_of"));
        assert!(!is_causal_predicate("divides"));
        assert!(!is_causal_predicate("divisible_by"));
        assert!(!is_causal_predicate("is_prime"));
        assert!(!is_causal_predicate("is_composite"));
        assert!(!is_causal_predicate("is_even"));
        assert!(!is_causal_predicate("is_odd"));
        assert!(!is_causal_predicate("greater_than"));
        assert!(!is_causal_predicate("less_than"));
        assert!(!is_causal_predicate("coefficient_of"));
        assert!(!is_causal_predicate("term_of"));
        assert!(!is_causal_predicate("expression_of"));
    }

    #[test]
    fn test_is_causal_predicate_rejects_l0_narrative() {
        // CH-4440 — L0 narrative predicates describe story content, not
        // causal influence. They MUST stay rejected from CausalDag walks
        // (per CH-4433 precedent: observational predicates would pollute
        // Phase 6's causal-chain walker).
        // See research/audits/predicate-l0-narrative-audit.md.

        // Spatial/locative
        assert!(!is_causal_predicate("lives_in"));
        assert!(!is_causal_predicate("habitat"));
        assert!(!is_causal_predicate("located_in"));
        assert!(!is_causal_predicate("located_at"));
        // Presence
        assert!(!is_causal_predicate("appears_in"));
        assert!(!is_causal_predicate("stationed_at"));
        assert!(!is_causal_predicate("active_at"));
        // Functional/role
        assert!(!is_causal_predicate("used_for"));
        assert!(!is_causal_predicate("used_by"));
        assert!(!is_causal_predicate("serves_as"));
        assert!(!is_causal_predicate("functions_as"));
        // Definitional/linguistic
        assert!(!is_causal_predicate("means"));
        assert!(!is_causal_predicate("stands_for"));
        assert!(!is_causal_predicate("synonym_of"));
        assert!(!is_causal_predicate("translates_to"));
        assert!(!is_causal_predicate("pronounced_as"));
        assert!(!is_causal_predicate("named_after"));
        // Symbolic/narrative
        assert!(!is_causal_predicate("symbolizes"));
        assert!(!is_causal_predicate("represents"));
        assert!(!is_causal_predicate("depicts"));
        assert!(!is_causal_predicate("illustrates"));
        assert!(!is_causal_predicate("narrates_about"));
        assert!(!is_causal_predicate("companion_of"));
        // Attribution
        assert!(!is_causal_predicate("has_property"));
        assert!(!is_causal_predicate("displays"));
        assert!(!is_causal_predicate("makes_sound"));
        // Action-role
        assert!(!is_causal_predicate("hunts_with"));
        assert!(!is_causal_predicate("shoots"));
        assert!(!is_causal_predicate("tethered_by"));
        assert!(!is_causal_predicate("lifted_by"));
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
        let nodes: Vec<String> = dag.nodes().iter().cloned().collect();

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

    // ── EX-4071: Integer Indexing + Reachability Cache Tests ──────────────────

    #[test]
    fn test_integer_indexing_built() {
        let dag = diamond_dag();
        // has_path triggers ensure_cache which builds the index
        dag.has_path("A", "D");

        let cache = dag.cache.borrow();
        let c = cache
            .as_ref()
            .expect("cache should be built after has_path");

        assert_eq!(c.node_to_idx.len(), 4, "all 4 nodes should be indexed");
        assert_eq!(c.idx_to_node.len(), 4);
        // Every node should have a unique index
        let indices: std::collections::HashSet<usize> = c.node_to_idx.values().copied().collect();
        assert_eq!(indices.len(), 4, "indices should be unique");

        // Verify bidirectional mapping
        for node in dag.nodes() {
            let idx = c.node_to_idx[node];
            assert_eq!(&c.idx_to_node[idx], node);
        }
    }

    #[test]
    fn test_reachability_cache_correctness() {
        // Compare cached has_path against descendants() on diamond DAG
        let dag = diamond_dag();

        let nodes = vec!["A", "B", "C", "D"];
        for src in &nodes {
            let desc: HashSet<NodeId> = if dag.has_node(src) {
                dag.descendants(src).unwrap()
            } else {
                HashSet::new()
            };
            for tgt in &nodes {
                let cached = dag.has_path(src, tgt);
                let via_desc = *src == *tgt || desc.contains(*tgt);
                assert_eq!(
                    cached, via_desc,
                    "has_path({src}, {tgt}) = {cached}, but descendants says {via_desc}"
                );
            }
        }
    }

    #[test]
    fn test_reachability_cache_wide_dag() {
        // Build a 500-node DAG: node 0 → 1 → 2 → ... → 499
        let mut dag = CausalDag::new();
        for i in 0..499u32 {
            dag.add_edge(&format!("N{i}"), &format!("N{}", i + 1), "depends_on");
        }
        assert_eq!(dag.node_count(), 500);

        // Trigger cache build
        assert!(dag.has_path("N0", "N499"));
        assert!(!dag.has_path("N499", "N0"));
        assert!(dag.has_path("N100", "N200"));

        // Spot-check all edges reachable
        for i in 0u32..499 {
            assert!(
                dag.has_path(&format!("N{i}"), &format!("N{}", i + 1)),
                "N{i} → N{} should be reachable",
                i + 1
            );
        }

        // Reverse should be false
        for i in (1..500u32).rev() {
            assert!(
                !dag.has_path(&format!("N{i}"), &format!("N{}", i - 1)),
                "N{i} → N{} should NOT be reachable",
                i - 1
            );
        }
    }

    #[test]
    fn test_reachability_cache_invalidated_on_add_edge() {
        let mut dag = CausalDag::new();
        dag.add_edge("A", "B", "blocks");

        // Build cache
        assert!(dag.has_path("A", "B"));
        assert!(!dag.has_path("A", "C"));

        // Add new edge — cache should be invalidated
        dag.add_edge("B", "C", "depends_on");

        // has_path should now see A→C via A→B→C
        assert!(dag.has_path("A", "C"));
        assert!(dag.has_path("B", "C"));
        assert!(!dag.has_path("C", "A"));
    }

    #[test]
    fn test_reachability_matrix_coo_format() {
        // Chain: A → B → C
        let mut dag = CausalDag::new();
        dag.add_edge("A", "B", "blocks");
        dag.add_edge("B", "C", "depends_on");

        let (n, rows, cols) = dag.reachability_matrix();
        assert_eq!(n, 3);

        // Pairs: A→B, A→C, B→C (3 reachable pairs)
        assert_eq!(rows.len(), 3, "should have 3 reachable pairs");

        // Convert to set of (row, col) for easier checking
        let pairs: HashSet<(usize, usize)> = rows
            .iter()
            .zip(cols.iter())
            .map(|(&r, &c)| (r, c))
            .collect();

        // Get indices from the built cache
        let cache = dag.cache.borrow();
        let c = cache.as_ref().expect("cache should be built");
        let a = c.node_to_idx["A"];
        let b = c.node_to_idx["B"];
        let c_idx = c.node_to_idx["C"];

        assert!(pairs.contains(&(a, b)), "A→B should be in COO");
        assert!(
            pairs.contains(&(a, c_idx)),
            "A→C should be in COO (transitive)"
        );
        assert!(pairs.contains(&(b, c_idx)), "B→C should be in COO");
        assert!(!pairs.contains(&(c_idx, a)), "C→A should NOT be in COO");
        assert!(!pairs.contains(&(b, a)), "B→A should NOT be in COO");
    }

    #[test]
    fn test_has_path_o1_performance() {
        // Build a wide-ish DAG: 500 nodes, each node i connects to i+1 and i+2
        let mut dag = CausalDag::new();
        for i in 0..498u32 {
            dag.add_edge(&format!("N{i}"), &format!("N{}", i + 1), "blocks");
            dag.add_edge(&format!("N{i}"), &format!("N{}", i + 2), "blocks");
        }
        // Add the last edge
        dag.add_edge("N498", "N499", "blocks");

        assert_eq!(dag.node_count(), 500);

        // First has_path call builds the cache (amortized cost)
        assert!(dag.has_path("N0", "N499"));

        // Subsequent 10,000 lookups should be O(1) — just hash set lookups
        let start = std::time::Instant::now();
        for _ in 0..10_000 {
            assert!(dag.has_path("N0", "N499"));
            assert!(!dag.has_path("N499", "N0"));
            assert!(dag.has_path("N250", "N499"));
        }
        let elapsed = start.elapsed();
        // 30,000 cached lookups should complete in well under 1 second
        assert!(
            elapsed.as_millis() < 1000,
            "30,000 cached lookups took {elapsed:?} — should be < 1s"
        );
    }

    #[test]
    fn test_reachability_cache_respects_mutilation_mask() {
        // Build: A → B → C → D
        let dag = chain_dag();

        // Without mutilation: A can reach everything
        assert!(dag.has_path("A", "D"));
        assert!(dag.has_path("A", "C"));

        // Mutilate B: A→B severed
        let mutilated = dag.mutilate("B").expect("mutilate");

        // Mutilated DAG has its own cache (starts empty)
        assert!(!mutilated.has_path("A", "B"));
        assert!(!mutilated.has_path("A", "D"));
        // B→C→D still works
        assert!(mutilated.has_path("B", "D"));
    }

    // ── from_y_graph (EX-4638): build the causal DAG from Y-graph triples ──

    /// Build a Y-graph triple-schema batch: col0=id (unused), col1=subject,
    /// col2=predicate, col3=object — matching `nusy_arrow_core::schema::col`.
    fn y_triple_batch(rows: &[(&str, &str, &str)]) -> RecordBatch {
        use arrow::array::ArrayRef;
        use arrow::datatypes::{DataType, Field, Schema};

        let subjects: Vec<&str> = rows.iter().map(|(s, _, _)| *s).collect();
        let predicates: Vec<&str> = rows.iter().map(|(_, p, _)| *p).collect();
        let objects: Vec<&str> = rows.iter().map(|(_, _, o)| *o).collect();
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("subject", DataType::Utf8, false),
            Field::new("predicate", DataType::Utf8, false),
            Field::new("object", DataType::Utf8, false),
        ]));
        let cols: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(vec!["row"; rows.len()])),
            Arc::new(StringArray::from(subjects)),
            Arc::new(StringArray::from(predicates)),
            Arc::new(StringArray::from(objects)),
        ];
        RecordBatch::try_new(schema, cols).expect("valid triple batch")
    }

    #[test]
    fn test_from_y_graph_builds_causal_edges() {
        // A causes B ; B depends_on C ; A hasLabel "foo" (non-causal, skipped).
        let batch = y_triple_batch(&[
            ("A", "causes", "B"),
            ("B", "depends_on", "C"),
            ("A", "hasLabel", "foo"),
        ]);
        let dag = CausalDag::from_y_graph(&[batch]).expect("build from y-graph");

        // Only causal-edge endpoints become nodes — "foo" is excluded.
        assert_eq!(dag.node_count(), 3, "A, B, C — not the label object 'foo'");
        assert_eq!(
            dag.edge_count(),
            2,
            "two causal edges, label triple skipped"
        );
        assert_eq!(
            dag.children_of("A").len(),
            1,
            "A→B only (label edge skipped)"
        );
        assert_eq!(dag.parents_of("C").len(), 1, "C has parent B");
        // The derivation front-end supports transitive closure the engine needs.
        assert!(dag.has_path("A", "C"), "A → B → C is derivable");
        assert!(!dag.has_path("C", "A"), "edges are directional");
    }

    #[test]
    fn test_from_y_graph_skips_non_causal_predicates() {
        // No causal predicates → empty DAG.
        let batch = y_triple_batch(&[("A", "hasLabel", "foo"), ("A", "rdf:type", "Patient")]);
        let dag = CausalDag::from_y_graph(&[batch]).expect("build from y-graph");
        assert_eq!(dag.node_count(), 0);
        assert_eq!(dag.edge_count(), 0);
    }

    #[test]
    fn test_from_y_graph_empty() {
        let dag = CausalDag::from_y_graph(&[]).expect("build from empty");
        assert_eq!(dag.node_count(), 0);
        assert_eq!(dag.edge_count(), 0);
    }

    #[test]
    fn test_from_y_graph_combines_multiple_batches() {
        // Edges split across batches must combine into one reachable chain.
        let b1 = y_triple_batch(&[("A", "causes", "B")]);
        let b2 = y_triple_batch(&[("B", "causes", "C")]);
        let dag = CausalDag::from_y_graph(&[b1, b2]).expect("build from y-graph");
        assert_eq!(dag.node_count(), 3);
        assert_eq!(dag.edge_count(), 2);
        assert!(dag.has_path("A", "C"), "chain spans both batches");
    }
}
