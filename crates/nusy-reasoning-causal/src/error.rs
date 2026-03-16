//! Error types for causal reasoning operations.

/// Errors from causal operations.
#[derive(Debug, thiserror::Error)]
pub enum CausalError {
    #[error("Node not found in DAG: {0}")]
    NodeNotFound(String),

    #[error("Cycle detected in DAG involving node: {0}")]
    CycleDetected(String),

    #[error(
        "Causal effect not identifiable: no valid adjustment set exists for {treatment} -> {outcome}"
    )]
    NotIdentifiable { treatment: String, outcome: String },

    #[error("Counterfactual not certifiable: {reason}")]
    CounterfactualNotCertifiable { reason: String },

    #[error("No causal path from {treatment} to {outcome}")]
    NoCausalPath { treatment: String, outcome: String },

    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
}

pub type Result<T> = std::result::Result<T, CausalError>;
