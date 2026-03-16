//! Pearl's do-calculus for NuSy — graph mutilation, adjustment sets,
//! interventional queries, and counterfactual reasoning.
//!
//! This crate provides the formal causal reasoning engine for NuSy's
//! Arrow-native substrate. It operates on directed acyclic graphs (DAGs)
//! built from kanban relations or knowledge-graph triples.
//!
//! # Pearl's Three Levels
//!
//! 1. **Structural (L1)** — graph topology, handled by existing traversal code
//! 2. **Interventional (L2)** — `do(X=x)`: mutilate graph, compute causal effect
//! 3. **Counterfactual (L3)** — "what would have been?": abduction + action + prediction
//!
//! All three levels are implemented. Counterfactual queries (L3) include a
//! certifiability gate that refuses queries when graph completeness is too
//! low for reliable inference (see EX-3019 certifiability boundary).

pub mod adjustment;
pub mod counterfactual;
pub mod error;
pub mod graph;
pub mod intervention;

pub use adjustment::{AdjustmentResult, AdjustmentSet};
pub use counterfactual::{ConfidenceLevel, CounterfactualResult};
pub use error::{CausalError, Result};
pub use graph::{CausalDag, CausalEdge, NodeId};
pub use intervention::{InterventionEffect, InterventionResult};
