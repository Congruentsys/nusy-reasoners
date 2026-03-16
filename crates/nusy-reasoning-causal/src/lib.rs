//! Pearl's do-calculus for NuSy — graph mutilation, adjustment sets,
//! interventional queries, and counterfactual stubs.
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
//! This crate implements L2 and stubs L3 (full counterfactuals require the
//! formal loss specification from EX-3018 and certifiability boundary from EX-3019).

pub mod adjustment;
pub mod counterfactual;
pub mod error;
pub mod graph;
pub mod intervention;

pub use adjustment::{AdjustmentResult, AdjustmentSet};
pub use counterfactual::CounterfactualResult;
pub use error::{CausalError, Result};
pub use graph::{CausalDag, CausalEdge, NodeId};
pub use intervention::{InterventionEffect, InterventionResult};
