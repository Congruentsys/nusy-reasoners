//! Safety-critical query classification and routing gate.
//!
//! Implements the DoWhy verification routing from EX-4078:
//! 1. Classify causal queries as safety-critical or non-critical
//! 2. Route safety-critical queries through identifiability verification
//! 3. REFUSE unidentifiable safety-critical queries (zero false positives)
//! 4. Non-critical queries use fast path (no verification overhead)
//! 5. Provenance gate: provenance_validity ≥ 95% for clinical domains
//!
//! # Safety Classification Rules
//!
//! A query is **safety-critical** if ANY of:
//! - Domain is safety-critical (clinical, medical, legal, financial)
//! - Pearl level is L3 (counterfactual) — hypothetical reasoning about what
//!   would have happened
//! - Query involves Y2 (reasoning) or Y5 (procedural) knowledge with
//!   interventional predicates
//!
//! A query is **non-critical** if:
//! - Domain is non-safety (general, engineering, research)
//! - Pearl level is L1 (observational) — no causal claim
//! - Query involves Y0 (prose) or Y3 (experience) — being's own data

use crate::error::{CausalError, Result};
use crate::graph::CausalDag;
use crate::identifiability::{self, IdentifiabilityVerification};

/// Provenance validity threshold for clinical domains (CONCERN-6).
pub const CLINICAL_PROVENANCE_THRESHOLD: f64 = 0.95;

/// Domains that are safety-critical — causal queries in these domains
/// require formal identifiability verification. The clinical default for
/// [`SafetyPolicy::clinical`]; routing logic reads the policy, not this const.
pub const SAFETY_CRITICAL_DOMAINS: &[&str] = &[
    "clinical",
    "medical",
    "legal",
    "financial",
    "pharmaceutical",
    "diagnostic",
];

/// The safety policy as **data** (CH-4752): which domains are safety-critical and the
/// provenance floor they require. Extracted out of bare consts so a non-clinical
/// deployment can supply its own set — the routing logic is generic over the policy,
/// and the clinical values are merely the default ([`SafetyPolicy::clinical`]).
#[derive(Debug, Clone, PartialEq)]
pub struct SafetyPolicy {
    /// Domains whose causal queries require formal identifiability verification.
    pub critical_domains: Vec<String>,
    /// Provenance-validity floor `[0,1]` applied to safety-critical domains.
    pub provenance_threshold: f64,
}

impl SafetyPolicy {
    /// The clinical default policy — the historical CONCERN-6 values
    /// ([`SAFETY_CRITICAL_DOMAINS`] + [`CLINICAL_PROVENANCE_THRESHOLD`]).
    pub fn clinical() -> Self {
        Self {
            critical_domains: SAFETY_CRITICAL_DOMAINS
                .iter()
                .map(|d| d.to_string())
                .collect(),
            provenance_threshold: CLINICAL_PROVENANCE_THRESHOLD,
        }
    }

    /// Is `domain` safety-critical under this policy? (Case-insensitive.)
    pub fn is_critical_domain(&self, domain: &str) -> bool {
        self.critical_domains
            .iter()
            .any(|d| d.eq_ignore_ascii_case(domain))
    }
}

impl Default for SafetyPolicy {
    fn default() -> Self {
        Self::clinical()
    }
}

/// Pearl's causal hierarchy levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PearlLevel {
    /// Level 1: Observational — P(Y|X), statistical associations
    Observational,
    /// Level 2: Interventional — P(Y|do(X)), causal claims
    Interventional,
    /// Level 3: Counterfactual — P(Y_x|X',Y'), hypothetical reasoning
    Counterfactual,
}

/// Classification of a causal query's safety criticality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyClassification {
    /// Safety-critical: must route through DoWhy identifiability verification.
    /// Refuses unidentifiable queries with zero false positives.
    SafetyCritical,
    /// Non-critical: uses fast path (CAC or symbolic pipeline).
    NonCritical,
}

/// A causal query with full context for safety routing.
#[derive(Debug, Clone)]
pub struct CausalQuery {
    /// Treatment variable (the cause being queried).
    pub treatment: String,
    /// Outcome variable (the effect being measured).
    pub outcome: String,
    /// Domain of the query (e.g., "medical", "general").
    pub domain: String,
    /// Pearl causal hierarchy level of the query.
    pub pearl_level: PearlLevel,
    /// Provenance validity score [0.0, 1.0] — measures how complete
    /// the provenance chain is from source documents to the query result.
    pub provenance_validity: f64,
}

/// Result of safety routing — either verified or refused.
#[derive(Debug, Clone)]
pub struct SafetyRoutingResult {
    /// The original query.
    pub query: CausalQuery,
    /// Safety classification applied.
    pub classification: SafetyClassification,
    /// Routing path taken.
    pub path: RoutingPath,
    /// Identifiability verification (if safety-critical path was taken).
    pub verification: Option<IdentifiabilityVerification>,
    /// Provenance gate result (if checked).
    pub provenance_gate: Option<ProvenanceGateResult>,
}

/// Which routing path the query took.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingPath {
    /// Safety-critical: routed through DoWhy identifiability verification.
    DoWhyVerification,
    /// Non-critical fast path: CAC (Causal Answer Chain) or symbolic pipeline.
    FastPath,
    /// Refused: identifiability verification failed.
    Refused,
    /// Refused: provenance gate failed (below threshold for clinical domain).
    RefusedProvenance,
}

/// Result of the provenance gate check.
#[derive(Debug, Clone)]
pub struct ProvenanceGateResult {
    /// The provenance validity score measured.
    pub provenance_validity: f64,
    /// The threshold that was applied.
    pub threshold: f64,
    /// Whether the gate passed.
    pub passed: bool,
    /// Domain that triggered this gate (if any).
    pub domain: String,
}

/// Classify a causal query's safety criticality.
///
/// A query is safety-critical if ANY of:
/// 1. Domain is in [`SAFETY_CRITICAL_DOMAINS`]
/// 2. Pearl level is L3 (counterfactual)
/// 3. Pearl level is L2 (interventional) AND domain is safety-critical
///
/// All other queries are non-critical.
pub fn classify_query(query: &CausalQuery) -> SafetyClassification {
    classify_query_with(query, &SafetyPolicy::clinical())
}

/// Classify a query under an explicit [`SafetyPolicy`] (data-driven; CH-4752).
pub fn classify_query_with(query: &CausalQuery, policy: &SafetyPolicy) -> SafetyClassification {
    // Rule 1: Counterfactual queries are always safety-critical
    if query.pearl_level == PearlLevel::Counterfactual {
        return SafetyClassification::SafetyCritical;
    }

    // Rule 2: Safety-critical domains require verification for interventional queries
    if query.pearl_level == PearlLevel::Interventional && policy.is_critical_domain(&query.domain) {
        return SafetyClassification::SafetyCritical;
    }

    // Rule 3: Observational queries are never safety-critical (no causal claim)
    // Rule 4: Non-critical domains with interventional queries use fast path
    SafetyClassification::NonCritical
}

/// Check if a domain name is safety-critical under the clinical default policy.
pub fn is_safety_critical_domain(domain: &str) -> bool {
    SafetyPolicy::clinical().is_critical_domain(domain)
}

/// Check the provenance gate for a query.
///
/// For clinical/safety-critical domains, provenance_validity must be ≥ 95%.
/// For other domains, the gate always passes.
pub fn check_provenance_gate(query: &CausalQuery) -> ProvenanceGateResult {
    check_provenance_gate_with(query, &SafetyPolicy::clinical())
}

/// Check the provenance gate under an explicit [`SafetyPolicy`] (data-driven; CH-4752).
pub fn check_provenance_gate_with(
    query: &CausalQuery,
    policy: &SafetyPolicy,
) -> ProvenanceGateResult {
    let threshold = if policy.is_critical_domain(&query.domain) {
        policy.provenance_threshold
    } else {
        0.0 // No provenance requirement for non-critical domains
    };

    let passed = query.provenance_validity >= threshold;

    ProvenanceGateResult {
        provenance_validity: query.provenance_validity,
        threshold,
        passed,
        domain: query.domain.clone(),
    }
}

/// Route a causal query through the safety system.
///
/// This is the main entry point for safety routing (EX-4078 Task 2):
///
/// 1. Classify the query as safety-critical or non-critical
/// 2. For safety-critical queries:
///    a. Check provenance gate (≥95% for clinical domains)
///    b. Run DoWhy identifiability verification
///    c. REFUSE if unidentifiable (zero false positives)
/// 3. For non-critical queries: use fast path (no verification)
///
/// # Returns
///
/// - `Ok(SafetyRoutingResult)` — query is approved (either verified or fast-pathed)
/// - `Err(CausalError::NotIdentifiable)` — safety-critical query is unidentifiable
/// - `Err(CausalError::ProvenanceGateFailed)` — provenance below threshold
pub fn route_query(dag: &CausalDag, query: &CausalQuery) -> Result<SafetyRoutingResult> {
    route_query_with(dag, query, &SafetyPolicy::clinical())
}

/// Route a query under an explicit [`SafetyPolicy`] (data-driven; CH-4752). The clinical
/// behaviour is exactly `route_query_with(dag, query, &SafetyPolicy::clinical())`.
pub fn route_query_with(
    dag: &CausalDag,
    query: &CausalQuery,
    policy: &SafetyPolicy,
) -> Result<SafetyRoutingResult> {
    let classification = classify_query_with(query, policy);

    match classification {
        SafetyClassification::SafetyCritical => route_safety_critical(dag, query, policy),
        SafetyClassification::NonCritical => Ok(SafetyRoutingResult {
            query: query.clone(),
            classification,
            path: RoutingPath::FastPath,
            verification: None,
            provenance_gate: None,
        }),
    }
}

/// Route a safety-critical query through the full verification pipeline.
///
/// Step 1: Provenance gate (provenance_validity ≥ 95% for clinical domains)
/// Step 2: DoWhy identifiability verification (refuses unidentifiable)
fn route_safety_critical(
    dag: &CausalDag,
    query: &CausalQuery,
    policy: &SafetyPolicy,
) -> Result<SafetyRoutingResult> {
    // Step 1: Provenance gate
    let prov_gate = check_provenance_gate_with(query, policy);
    if !prov_gate.passed {
        return Err(CausalError::ProvenanceGateFailed {
            domain: query.domain.clone(),
            validity: query.provenance_validity,
            threshold: prov_gate.threshold,
        });
    }

    // Step 2: DoWhy identifiability verification — REFUSE if not identifiable
    let verification = identifiability::verify_and_refuse(dag, &query.treatment, &query.outcome)?;

    Ok(SafetyRoutingResult {
        query: query.clone(),
        classification: SafetyClassification::SafetyCritical,
        path: RoutingPath::DoWhyVerification,
        verification: Some(verification),
        provenance_gate: Some(prov_gate),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::CausalDag;

    // ── Classification Tests (Task 1) ──

    #[test]
    fn test_classify_medical_interventional_is_critical() {
        let query = CausalQuery {
            treatment: "drug_A".to_string(),
            outcome: "recovery".to_string(),
            domain: "medical".to_string(),
            pearl_level: PearlLevel::Interventional,
            provenance_validity: 0.98,
        };
        assert_eq!(classify_query(&query), SafetyClassification::SafetyCritical);
    }

    #[test]
    fn test_classify_counterfactual_always_critical() {
        let query = CausalQuery {
            treatment: "X".to_string(),
            outcome: "Y".to_string(),
            domain: "general".to_string(),
            pearl_level: PearlLevel::Counterfactual,
            provenance_validity: 0.5,
        };
        assert_eq!(classify_query(&query), SafetyClassification::SafetyCritical);
    }

    #[test]
    fn test_classify_general_observational_is_non_critical() {
        let query = CausalQuery {
            treatment: "A".to_string(),
            outcome: "B".to_string(),
            domain: "general".to_string(),
            pearl_level: PearlLevel::Observational,
            provenance_validity: 0.5,
        };
        assert_eq!(classify_query(&query), SafetyClassification::NonCritical);
    }

    #[test]
    fn test_classify_general_interventional_is_non_critical() {
        let query = CausalQuery {
            treatment: "A".to_string(),
            outcome: "B".to_string(),
            domain: "engineering".to_string(),
            pearl_level: PearlLevel::Interventional,
            provenance_validity: 0.7,
        };
        assert_eq!(classify_query(&query), SafetyClassification::NonCritical);
    }

    #[test]
    fn test_classify_clinical_interventional_is_critical() {
        let query = CausalQuery {
            treatment: "therapy".to_string(),
            outcome: "outcome".to_string(),
            domain: "clinical".to_string(),
            pearl_level: PearlLevel::Interventional,
            provenance_validity: 0.99,
        };
        assert_eq!(classify_query(&query), SafetyClassification::SafetyCritical);
    }

    #[test]
    fn test_classify_legal_interventional_is_critical() {
        let query = CausalQuery {
            treatment: "policy_X".to_string(),
            outcome: "compliance".to_string(),
            domain: "legal".to_string(),
            pearl_level: PearlLevel::Interventional,
            provenance_validity: 0.97,
        };
        assert_eq!(classify_query(&query), SafetyClassification::SafetyCritical);
    }

    #[test]
    fn test_classify_financial_interventional_is_critical() {
        let query = CausalQuery {
            treatment: "rate_change".to_string(),
            outcome: "default_rate".to_string(),
            domain: "financial".to_string(),
            pearl_level: PearlLevel::Interventional,
            provenance_validity: 0.96,
        };
        assert_eq!(classify_query(&query), SafetyClassification::SafetyCritical);
    }

    #[test]
    fn test_classify_diagnostic_counterfactual_is_critical() {
        let query = CausalQuery {
            treatment: "test_A".to_string(),
            outcome: "diagnosis".to_string(),
            domain: "diagnostic".to_string(),
            pearl_level: PearlLevel::Counterfactual,
            provenance_validity: 0.98,
        };
        assert_eq!(classify_query(&query), SafetyClassification::SafetyCritical);
    }

    // ── Domain Detection ──

    #[test]
    fn test_safety_critical_domains() {
        assert!(is_safety_critical_domain("medical"));
        assert!(is_safety_critical_domain("clinical"));
        assert!(is_safety_critical_domain("legal"));
        assert!(is_safety_critical_domain("financial"));
        assert!(is_safety_critical_domain("pharmaceutical"));
        assert!(is_safety_critical_domain("diagnostic"));
        // Case insensitive
        assert!(is_safety_critical_domain("Medical"));
        assert!(is_safety_critical_domain("CLINICAL"));
    }

    #[test]
    fn test_non_critical_domains() {
        assert!(!is_safety_critical_domain("general"));
        assert!(!is_safety_critical_domain("engineering"));
        assert!(!is_safety_critical_domain("research"));
        assert!(!is_safety_critical_domain("education"));
        assert!(!is_safety_critical_domain("unknown"));
    }

    // ── Provenance Gate (Task 5) ──

    #[test]
    fn test_provenance_gate_clinical_passes() {
        let query = CausalQuery {
            treatment: "X".to_string(),
            outcome: "Y".to_string(),
            domain: "clinical".to_string(),
            pearl_level: PearlLevel::Interventional,
            provenance_validity: 0.95,
        };
        let gate = check_provenance_gate(&query);
        assert!(gate.passed);
        assert!((gate.threshold - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn test_provenance_gate_clinical_above_threshold() {
        let query = CausalQuery {
            treatment: "X".to_string(),
            outcome: "Y".to_string(),
            domain: "medical".to_string(),
            pearl_level: PearlLevel::Interventional,
            provenance_validity: 0.99,
        };
        let gate = check_provenance_gate(&query);
        assert!(gate.passed);
    }

    #[test]
    fn test_provenance_gate_clinical_fails_below_threshold() {
        let query = CausalQuery {
            treatment: "X".to_string(),
            outcome: "Y".to_string(),
            domain: "medical".to_string(),
            pearl_level: PearlLevel::Interventional,
            provenance_validity: 0.94,
        };
        let gate = check_provenance_gate(&query);
        assert!(!gate.passed);
        assert!((gate.threshold - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn test_provenance_gate_general_always_passes() {
        let query = CausalQuery {
            treatment: "X".to_string(),
            outcome: "Y".to_string(),
            domain: "general".to_string(),
            pearl_level: PearlLevel::Interventional,
            provenance_validity: 0.0, // Even zero passes for non-critical domains
        };
        let gate = check_provenance_gate(&query);
        assert!(gate.passed);
        assert!((gate.threshold - 0.0).abs() < f64::EPSILON);
    }

    // ── Full Routing (Tasks 2-4) ──

    #[test]
    fn test_route_non_critical_uses_fast_path() {
        let dag = CausalDag::new();
        let query = CausalQuery {
            treatment: "A".to_string(),
            outcome: "B".to_string(),
            domain: "engineering".to_string(),
            pearl_level: PearlLevel::Observational,
            provenance_validity: 0.5,
        };

        let result = route_query(&dag, &query).expect("should route");
        assert_eq!(result.path, RoutingPath::FastPath);
        assert_eq!(result.classification, SafetyClassification::NonCritical);
        assert!(result.verification.is_none());
    }

    #[test]
    fn test_route_safety_critical_with_identifiable_effect() {
        let mut dag = CausalDag::new();
        dag.add_edge("drug", "recovery", "causes");

        let query = CausalQuery {
            treatment: "drug".to_string(),
            outcome: "recovery".to_string(),
            domain: "medical".to_string(),
            pearl_level: PearlLevel::Interventional,
            provenance_validity: 0.98,
        };

        let result = route_query(&dag, &query).expect("should route");
        assert_eq!(result.path, RoutingPath::DoWhyVerification);
        assert_eq!(result.classification, SafetyClassification::SafetyCritical);
        assert!(result.verification.is_some());
        let v = result.verification.expect("verification");
        assert!(v.identifiable);
        assert!(result.provenance_gate.is_some());
        assert!(result.provenance_gate.expect("gate").passed);
    }

    #[test]
    fn test_route_safety_critical_refuses_unidentifiable() {
        // Empty DAG — treatment and outcome don't exist
        let dag = CausalDag::new();
        let query = CausalQuery {
            treatment: "drug_X".to_string(),
            outcome: "outcome_Y".to_string(),
            domain: "clinical".to_string(),
            pearl_level: PearlLevel::Interventional,
            provenance_validity: 0.98,
        };

        let result = route_query(&dag, &query);
        assert!(result.is_err());
        // Should be refused because nodes don't exist (not identifiable)
        match result.unwrap_err() {
            CausalError::NodeNotFound(node) => assert_eq!(node, "drug_X"),
            other => panic!("expected NodeNotFound for non-existent treatment, got: {other}"),
        }
    }

    #[test]
    fn test_route_safety_critical_refuses_low_provenance() {
        let mut dag = CausalDag::new();
        dag.add_edge("therapy", "outcome", "causes");

        let query = CausalQuery {
            treatment: "therapy".to_string(),
            outcome: "outcome".to_string(),
            domain: "medical".to_string(),
            pearl_level: PearlLevel::Interventional,
            provenance_validity: 0.80, // Below 95% threshold
        };

        let result = route_query(&dag, &query);
        assert!(result.is_err());
        match result.unwrap_err() {
            CausalError::ProvenanceGateFailed {
                domain,
                validity,
                threshold,
            } => {
                assert_eq!(domain, "medical");
                assert!((validity - 0.80).abs() < f64::EPSILON);
                assert!((threshold - 0.95).abs() < f64::EPSILON);
            }
            other => panic!("expected ProvenanceGateFailed, got: {other}"),
        }
    }

    #[test]
    fn test_route_counterfactual_any_domain_is_critical() {
        let mut dag = CausalDag::new();
        dag.add_edge("A", "B", "causes");

        let query = CausalQuery {
            treatment: "A".to_string(),
            outcome: "B".to_string(),
            domain: "general".to_string(), // Not safety-critical domain
            pearl_level: PearlLevel::Counterfactual, // But counterfactual
            provenance_validity: 0.99,
        };

        let result = route_query(&dag, &query).expect("should route");
        assert_eq!(result.classification, SafetyClassification::SafetyCritical);
        assert_eq!(result.path, RoutingPath::DoWhyVerification);
    }

    // ── Zero False Positives (H-4118) ──

    #[test]
    fn test_zero_false_positives_unidentifiable_medical() {
        // Build a DAG where the effect cannot be verified — nodes don't exist
        let dag = CausalDag::new();
        let query = CausalQuery {
            treatment: "treatment_A".to_string(),
            outcome: "outcome_B".to_string(),
            domain: "medical".to_string(),
            pearl_level: PearlLevel::Interventional,
            provenance_validity: 0.98,
        };

        let result = route_query(&dag, &query);
        assert!(
            result.is_err(),
            "Must refuse unidentifiable safety-critical query"
        );
    }

    #[test]
    fn test_zero_false_positives_counterfactual_nonexistent() {
        let dag = CausalDag::new();
        let query = CausalQuery {
            treatment: "X".to_string(),
            outcome: "Y".to_string(),
            domain: "general".to_string(),
            pearl_level: PearlLevel::Counterfactual,
            provenance_validity: 1.0,
        };

        let result = route_query(&dag, &query);
        assert!(
            result.is_err(),
            "Must refuse counterfactual with non-existent nodes"
        );
    }

    // ── Kanban Scenario ──

    #[test]
    fn test_route_kanban_research_query() {
        // Research query: non-critical domain, interventional
        let mut dag = CausalDag::new();
        dag.add_edge("EX-3017", "accuracy", "causes");

        let query = CausalQuery {
            treatment: "EX-3017".to_string(),
            outcome: "accuracy".to_string(),
            domain: "research".to_string(),
            pearl_level: PearlLevel::Interventional,
            provenance_validity: 0.7,
        };

        let result = route_query(&dag, &query).expect("should route");
        assert_eq!(result.path, RoutingPath::FastPath);
    }

    // ── Pearl Level Ordering ──

    #[test]
    fn test_pearl_level_ordering() {
        assert!(PearlLevel::Observational < PearlLevel::Interventional);
        assert!(PearlLevel::Interventional < PearlLevel::Counterfactual);
    }
}
