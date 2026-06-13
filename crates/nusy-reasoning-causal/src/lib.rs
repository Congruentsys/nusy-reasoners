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
//! # Safety routing — `clinical-policy` feature (EX-4078, CH-4752 split)
//!
//! The generic core above carries **no domain policy**. Behind the
//! **`clinical-policy`** feature (on by default; off in the FOSS extraction) sits an
//! optional safety layer: it routes queries whose domain is in a configurable
//! safety-critical set — or any counterfactual — through identifiability
//! verification, refusing unidentifiable queries with zero false positives (H-4118),
//! and applies a configurable provenance floor. The policy is **data**
//! (`SafetyPolicy`), not hardcoded — see `safety_routing`. Build the pure generic
//! core with `--no-default-features`.

// ── Generic do-calculus core (always built; arrow-only; zero domain policy) ──
pub mod adjustment;
pub mod counterfactual;
pub mod error;
pub mod graph;
pub mod identifiability;
pub mod intervention;

// Clinical safety policy (clinical_gate / safety_routing) stays product-side per the
// V19-VOYAGES §3 disposition — the FOSS extraction is the pure generic do-calculus core.

pub use adjustment::{AdjustmentResult, AdjustmentSet};
pub use counterfactual::{ConfidenceLevel, CounterfactualResult};
pub use error::{CausalError, Result};
pub use graph::{CausalDag, CausalEdge, NodeId};
pub use identifiability::{IdentifiabilityVerification, IdentificationCriterion};
pub use intervention::{InterventionEffect, InterventionResult};
