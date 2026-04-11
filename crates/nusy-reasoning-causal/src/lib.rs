//! Pearl's do-calculus for NuSy — graph mutilation, adjustment sets,
//! interventional queries, counterfactual reasoning, and safety routing.
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
//!
//! # Safety Routing (EX-4078)
//!
//! Safety-critical queries (clinical, medical, legal, financial domains and
//! counterfactual queries) are routed through DoWhy identifiability verification.
//! Unidentifiable queries are refused with zero false positives (H-4118).
//! Non-critical queries use the fast path (CAC or symbolic pipeline).

pub mod adjustment;
pub mod counterfactual;
pub mod error;
pub mod graph;
pub mod identifiability;
pub mod intervention;
pub mod safety_routing;

pub use adjustment::{AdjustmentResult, AdjustmentSet};
pub use counterfactual::{ConfidenceLevel, CounterfactualResult};
pub use error::{CausalError, Result};
pub use graph::{CausalDag, CausalEdge, NodeId};
pub use identifiability::{IdentifiabilityVerification, IdentificationCriterion};
pub use intervention::{InterventionEffect, InterventionResult};
pub use safety_routing::{
    CausalQuery, PearlLevel, ProvenanceGateResult, RoutingPath, SafetyClassification,
    SafetyRoutingResult,
};
