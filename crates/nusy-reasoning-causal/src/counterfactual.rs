//! Counterfactual queries — Pearl's three-step counterfactual.
//!
//! **STUB:** Full counterfactual implementation requires:
//! - EX-3018 (formal causal asymmetry loss specification)
//! - EX-3019 Phase 2 (counterfactual query implementation)
//!
//! This module defines the types and provides a stub that returns
//! `CausalError::CounterfactualNotImplemented`. The full implementation
//! will follow Pearl's three-step procedure:
//!
//! 1. **Abduction:** Given observed evidence, infer the values of
//!    exogenous variables (unobserved causes).
//! 2. **Action:** Apply the intervention `do(X=x')` to the structural
//!    causal model (graph mutilation).
//! 3. **Prediction:** Compute the outcome under the modified model.

use crate::error::{CausalError, Result};
use crate::graph::NodeId;

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
/// # Current Status: STUB
///
/// Returns `CausalError::CounterfactualNotImplemented`. The full
/// implementation will be delivered in EX-3019 Phase 2 after the
/// formal causal specification (EX-3018) and certifiability boundary
/// (EX-3019 Phase 1) are complete.
///
/// # Future Implementation (Pearl's Three Steps)
///
/// ```text
/// 1. Abduction: P(U | X=x_observed, Y=y_observed)
///    - Infer exogenous variables from observed evidence
///
/// 2. Action: do(X = x_counterfactual)
///    - Mutilate graph, set treatment to hypothetical value
///
/// 3. Prediction: P(Y_x' | U)
///    - Compute outcome under modified model + inferred exogenous
/// ```
pub fn counterfactual(_treatment: &str, _outcome: &str) -> Result<CounterfactualResult> {
    Err(CausalError::CounterfactualNotImplemented)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counterfactual_is_stub() {
        let result = counterfactual("EX-3017", "accuracy");
        assert!(result.is_err());
        match result.unwrap_err() {
            CausalError::CounterfactualNotImplemented => {} // expected
            other => panic!("expected CounterfactualNotImplemented, got: {other}"),
        }
    }
}
