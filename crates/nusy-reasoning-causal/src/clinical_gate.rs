//! Clinical-causal sub-gate (EX-4643, VOY-V18-5).
//!
//! The **clinical-causal sub-gate** of V18's provable gate. It is *distinct* from the
//! general [`ProvableClaimGate`](https://docs.rs/nusy-gate) (EX-4610): that gate decides
//! whether *any* claim has a proof in the executable Y-graph; this sub-gate adds the
//! **stricter clinical layer** — a `provenance_validity ≥ 0.95` floor plus
//! **identifiability-based refusal** — for safety-critical causal claims in clinical /
//! medical / legal / financial / pharmaceutical / diagnostic domains.
//!
//! It does **not** reimplement the V17 safety router: it *wraps*
//! [`route_query`](crate::safety_routing::route_query), mapping its
//! `Result<SafetyRoutingResult>` onto a gate verdict that parallels EX-4610's
//! `GateResponse` but carries identifiability / provenance evidence instead of a proof tree.
//!
//! # Scope — the SAFETY_CRITICAL_DOMAINS path only
//!
//! A claim is **in scope** of this sub-gate iff its domain is in
//! [`SAFETY_CRITICAL_DOMAINS`](crate::safety_routing::SAFETY_CRITICAL_DOMAINS) **and** it
//! makes a causal claim ([`classify_query`](crate::safety_routing::classify_query) returns
//! [`SafetyCritical`](SafetyClassification::SafetyCritical) — i.e. interventional or
//! counterfactual). Everything else — non-clinical domains, and purely observational
//! (non-causal) claims — is [`NotInScope`](ClinicalGateVerdict::NotInScope) and deferred to
//! the general gate. Restricting scope this way keeps the `0.95` provenance floor uniformly
//! in force across every evaluated claim, so the gate's two measures are well-defined.
//!
//! # Verdicts
//!
//! - [`Admitted`](ClinicalGateVerdict::Admitted) — cleared **both** the `0.95` provenance
//!   gate and identifiability verification; carries the [`SafetyRoutingResult`] as evidence.
//! - [`Refused`](ClinicalGateVerdict::Refused) — turned away. For a safety-critical clinical
//!   claim, **refusal is the safe action** (not "route to neural"): either provenance fell
//!   below `0.95` ([`InsufficientProvenance`](ClinicalRefusalReason::InsufficientProvenance))
//!   or the effect is not identifiable from the causal DAG
//!   ([`NotIdentifiable`](ClinicalRefusalReason::NotIdentifiable)). The gate is **fail-closed**:
//!   any verification failure refuses rather than admits.
//! - [`NotInScope`](ClinicalGateVerdict::NotInScope) — outside the clinical path; the general
//!   [`ProvableClaimGate`](https://docs.rs/nusy-gate) decides it.
//!
//! ```
//! use nusy_reasoning_causal::{ClinicalCausalGate, CausalDag, CausalQuery, PearlLevel};
//!
//! let mut dag = CausalDag::new();
//! dag.add_edge("drug", "recovery", "causes");
//!
//! let gate = ClinicalCausalGate::new();
//!
//! // High-provenance, identifiable clinical claim → admitted with its verification.
//! let ok = CausalQuery {
//!     treatment: "drug".into(), outcome: "recovery".into(), domain: "clinical".into(),
//!     pearl_level: PearlLevel::Interventional, provenance_validity: 0.98,
//! };
//! assert!(gate.gate(&dag, &ok).is_admitted());
//!
//! // Same claim with thin provenance → refused (never silently admitted).
//! let thin = CausalQuery { provenance_validity: 0.80, ..ok.clone() };
//! assert!(gate.gate(&dag, &thin).is_refused());
//! ```

use crate::error::CausalError;
use crate::graph::CausalDag;
use crate::safety_routing::{
    CLINICAL_PROVENANCE_THRESHOLD, CausalQuery, SafetyClassification, SafetyRoutingResult,
    classify_query, is_safety_critical_domain, route_query,
};

/// Why a safety-critical clinical/causal claim was refused by the sub-gate.
#[derive(Debug, Clone, PartialEq)]
pub enum ClinicalRefusalReason {
    /// Provenance validity fell below [`CLINICAL_PROVENANCE_THRESHOLD`] (0.95).
    InsufficientProvenance {
        /// The measured provenance validity.
        validity: f64,
        /// The threshold that was applied (the clinical floor, 0.95).
        threshold: f64,
    },
    /// The causal effect is not identifiable from the DAG (do-calculus / DoWhy refusal),
    /// or verification could not be certified. The gate is fail-closed: every non-provenance
    /// verification failure lands here, with the underlying cause preserved in `detail`.
    NotIdentifiable {
        /// The underlying reason (unidentifiable, missing node, no causal path, …).
        detail: String,
    },
}

/// The clinical-causal sub-gate's verdict on a causal claim.
#[derive(Debug, Clone)]
pub enum ClinicalGateVerdict {
    /// Cleared the 0.95 provenance gate **and** identifiability verification — admitted,
    /// carrying the full [`SafetyRoutingResult`] (verification + provenance evidence).
    Admitted {
        /// The claim that was admitted.
        query: CausalQuery,
        /// The routing evidence justifying admission. Boxed — it is much larger than the
        /// other variants' payloads (it carries the identifiability verification + sets).
        routing: Box<SafetyRoutingResult>,
    },
    /// Turned away — provenance below 0.95, or the effect is not identifiable. For a
    /// safety-critical clinical claim, refusal is the safe action.
    Refused {
        /// The claim that was refused.
        query: CausalQuery,
        /// Why it was refused.
        reason: ClinicalRefusalReason,
    },
    /// Not a safety-critical clinical/causal claim — outside this sub-gate's remit.
    /// Deferred to the general [`ProvableClaimGate`](https://docs.rs/nusy-gate) (EX-4610).
    NotInScope {
        /// The claim deferred to the general gate.
        query: CausalQuery,
    },
}

impl ClinicalGateVerdict {
    /// Was the claim admitted (cleared provenance + identifiability)?
    pub fn is_admitted(&self) -> bool {
        matches!(self, ClinicalGateVerdict::Admitted { .. })
    }

    /// Was the claim actively refused (a safety refusal, distinct from out-of-scope)?
    pub fn is_refused(&self) -> bool {
        matches!(self, ClinicalGateVerdict::Refused { .. })
    }

    /// Was the claim within this sub-gate's remit (admitted or refused, not deferred)?
    pub fn in_scope(&self) -> bool {
        !matches!(self, ClinicalGateVerdict::NotInScope { .. })
    }

    /// The claim this verdict is about.
    pub fn query(&self) -> &CausalQuery {
        match self {
            ClinicalGateVerdict::Admitted { query, .. }
            | ClinicalGateVerdict::Refused { query, .. }
            | ClinicalGateVerdict::NotInScope { query } => query,
        }
    }

    /// The refusal reason, if the claim was refused.
    pub fn refusal_reason(&self) -> Option<&ClinicalRefusalReason> {
        match self {
            ClinicalGateVerdict::Refused { reason, .. } => Some(reason),
            _ => None,
        }
    }

    /// The routing evidence, if the claim was admitted.
    pub fn routing(&self) -> Option<&SafetyRoutingResult> {
        match self {
            ClinicalGateVerdict::Admitted { routing, .. } => Some(routing.as_ref()),
            _ => None,
        }
    }

    /// A human-readable rendering of the verdict.
    pub fn render(&self) -> String {
        match self {
            ClinicalGateVerdict::Admitted { query, .. } => format!(
                "ADMITTED (clinical-causal): {} -> {} [{}] — provenance {:.3} ≥ {:.2}, identifiable",
                query.treatment,
                query.outcome,
                query.domain,
                query.provenance_validity,
                CLINICAL_PROVENANCE_THRESHOLD,
            ),
            ClinicalGateVerdict::Refused {
                query,
                reason:
                    ClinicalRefusalReason::InsufficientProvenance {
                        validity,
                        threshold,
                    },
            } => format!(
                "REFUSED (insufficient provenance): {} -> {} [{}] — provenance {validity:.3} < {threshold:.2}",
                query.treatment, query.outcome, query.domain,
            ),
            ClinicalGateVerdict::Refused {
                query,
                reason: ClinicalRefusalReason::NotIdentifiable { detail },
            } => format!(
                "REFUSED (not identifiable): {} -> {} [{}] — {detail}",
                query.treatment, query.outcome, query.domain,
            ),
            ClinicalGateVerdict::NotInScope { query } => format!(
                "NOT IN SCOPE (non-safety-critical → general gate): {} -> {} [{}]",
                query.treatment, query.outcome, query.domain,
            ),
        }
    }
}

/// The clinical-causal sub-gate over [`crate::safety_routing`] (EX-4643).
///
/// Stateless — it wraps the pure [`route_query`] routing function. Construct with
/// [`new`](Self::new) (or [`Default`]) and call [`gate`](Self::gate) per claim.
#[derive(Debug, Clone, Copy, Default)]
pub struct ClinicalCausalGate;

impl ClinicalCausalGate {
    /// Build the clinical-causal sub-gate.
    pub fn new() -> Self {
        Self
    }

    /// Is this claim within the sub-gate's remit? True iff its domain is safety-critical
    /// **and** it makes a causal claim (interventional or counterfactual). Observational
    /// claims make no causal assertion; non-clinical domains are the general gate's job.
    pub fn in_scope(&self, query: &CausalQuery) -> bool {
        is_safety_critical_domain(&query.domain)
            && classify_query(query) == SafetyClassification::SafetyCritical
    }

    /// Gate one clinical/causal claim.
    ///
    /// In scope → run the wrapped [`route_query`] and map its outcome:
    /// `Ok` → [`Admitted`](ClinicalGateVerdict::Admitted); provenance failure →
    /// [`Refused`](ClinicalGateVerdict::Refused) with
    /// [`InsufficientProvenance`](ClinicalRefusalReason::InsufficientProvenance); any other
    /// verification failure → [`Refused`](ClinicalGateVerdict::Refused) with
    /// [`NotIdentifiable`](ClinicalRefusalReason::NotIdentifiable) (fail-closed). Out of scope
    /// → [`NotInScope`](ClinicalGateVerdict::NotInScope).
    pub fn gate(&self, dag: &CausalDag, query: &CausalQuery) -> ClinicalGateVerdict {
        if !self.in_scope(query) {
            return ClinicalGateVerdict::NotInScope {
                query: query.clone(),
            };
        }
        match route_query(dag, query) {
            Ok(routing) => ClinicalGateVerdict::Admitted {
                query: query.clone(),
                routing: Box::new(routing),
            },
            Err(CausalError::ProvenanceGateFailed {
                validity,
                threshold,
                ..
            }) => ClinicalGateVerdict::Refused {
                query: query.clone(),
                reason: ClinicalRefusalReason::InsufficientProvenance {
                    validity,
                    threshold,
                },
            },
            Err(other) => ClinicalGateVerdict::Refused {
                query: query.clone(),
                reason: ClinicalRefusalReason::NotIdentifiable {
                    detail: other.to_string(),
                },
            },
        }
    }

    /// Gate a batch of claims, preserving order.
    pub fn gate_all(&self, dag: &CausalDag, queries: &[CausalQuery]) -> Vec<ClinicalGateVerdict> {
        queries.iter().map(|q| self.gate(dag, q)).collect()
    }

    /// Gate a batch and tally the verdict mix into a [`ClinicalGateSummary`] (the source of
    /// the gate's two measures).
    pub fn summarize(&self, dag: &CausalDag, queries: &[CausalQuery]) -> ClinicalGateSummary {
        let mut s = ClinicalGateSummary::default();
        for q in queries {
            s.record(&self.gate(dag, q));
        }
        s
    }
}

/// Tally of a batch of clinical-causal verdicts, and the gate's two measures.
///
/// Every evaluated claim (admitted + refused) was in scope, so the `0.95` provenance floor
/// applied uniformly — which is what makes both rates below well-defined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ClinicalGateSummary {
    /// Cleared provenance + identifiability.
    pub admitted: usize,
    /// Refused: provenance below 0.95 (failed the provenance gate, step 1).
    pub refused_provenance: usize,
    /// Refused: not identifiable (passed provenance, failed identifiability, step 2).
    pub refused_identifiability: usize,
    /// Deferred — not a safety-critical clinical/causal claim.
    pub not_in_scope: usize,
}

impl ClinicalGateSummary {
    fn record(&mut self, v: &ClinicalGateVerdict) {
        match v {
            ClinicalGateVerdict::Admitted { .. } => self.admitted += 1,
            ClinicalGateVerdict::Refused {
                reason: ClinicalRefusalReason::InsufficientProvenance { .. },
                ..
            } => self.refused_provenance += 1,
            ClinicalGateVerdict::Refused {
                reason: ClinicalRefusalReason::NotIdentifiable { .. },
                ..
            } => self.refused_identifiability += 1,
            ClinicalGateVerdict::NotInScope { .. } => self.not_in_scope += 1,
        }
    }

    /// Total in-scope claims the gate evaluated (everything but the deferred ones).
    pub fn evaluated(&self) -> usize {
        self.admitted + self.refused_provenance + self.refused_identifiability
    }

    /// **Measure: clinical-provenance ≥0.95 pass rate.** Fraction of evaluated claims that
    /// cleared the provenance gate. `None` if nothing was evaluated.
    pub fn provenance_pass_rate(&self) -> Option<f64> {
        let n = self.evaluated();
        if n == 0 {
            return None;
        }
        Some((self.admitted + self.refused_identifiability) as f64 / n as f64)
    }

    /// **Measure: identifiability-refusal rate.** Of the claims that *cleared* provenance and
    /// reached identifiability verification, the fraction refused as not identifiable. `None`
    /// if none reached identifiability.
    pub fn identifiability_refusal_rate(&self) -> Option<f64> {
        let reached = self.admitted + self.refused_identifiability;
        if reached == 0 {
            return None;
        }
        Some(self.refused_identifiability as f64 / reached as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::CausalDag;
    use crate::safety_routing::{PearlLevel, RoutingPath};

    /// A clinical interventional query over the given treatment/outcome/provenance.
    fn clinical_q(treatment: &str, outcome: &str, provenance: f64) -> CausalQuery {
        CausalQuery {
            treatment: treatment.into(),
            outcome: outcome.into(),
            domain: "clinical".into(),
            pearl_level: PearlLevel::Interventional,
            provenance_validity: provenance,
        }
    }

    fn dag_with(treatment: &str, outcome: &str) -> CausalDag {
        let mut dag = CausalDag::new();
        dag.add_edge(treatment, outcome, "causes");
        dag
    }

    // ── In-scope: the three verdict paths ──

    #[test]
    fn admits_high_provenance_identifiable_clinical_claim() {
        let gate = ClinicalCausalGate::new();
        let dag = dag_with("drug", "recovery");
        let v = gate.gate(&dag, &clinical_q("drug", "recovery", 0.98));

        assert!(v.is_admitted());
        let routing = v.routing().expect("admitted carries routing evidence");
        // It went through the DoWhy verification path (not the fast path) and is identifiable.
        assert_eq!(routing.path, RoutingPath::DoWhyVerification);
        assert!(
            routing
                .verification
                .as_ref()
                .expect("verification")
                .identifiable
        );
        // The provenance gate it cleared used the clinical 0.95 floor.
        let pg = routing.provenance_gate.as_ref().expect("provenance gate");
        assert!(pg.passed);
        assert!((pg.threshold - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn refuses_thin_provenance_below_floor() {
        let gate = ClinicalCausalGate::new();
        let dag = dag_with("therapy", "outcome");
        let v = gate.gate(&dag, &clinical_q("therapy", "outcome", 0.80));

        assert!(v.is_refused());
        match v.refusal_reason().expect("refused has a reason") {
            ClinicalRefusalReason::InsufficientProvenance {
                validity,
                threshold,
            } => {
                assert!((validity - 0.80).abs() < f64::EPSILON);
                assert!((threshold - 0.95).abs() < f64::EPSILON);
            }
            other => panic!("expected InsufficientProvenance, got {other:?}"),
        }
    }

    #[test]
    fn refuses_unidentifiable_clinical_claim() {
        let gate = ClinicalCausalGate::new();
        // Empty DAG: high provenance clears step 1, but the effect is not identifiable.
        let dag = CausalDag::new();
        let v = gate.gate(&dag, &clinical_q("drug_X", "outcome_Y", 0.99));

        assert!(v.is_refused());
        assert!(matches!(
            v.refusal_reason(),
            Some(ClinicalRefusalReason::NotIdentifiable { .. })
        ));
    }

    // ── Provenance floor is exactly 0.95 (boundary) ──

    #[test]
    fn provenance_floor_is_exactly_095() {
        let gate = ClinicalCausalGate::new();
        let dag = dag_with("t", "o");
        // 0.95 is admitted (>=), 0.94 is refused (<) — identifiability held in both cases.
        assert!(gate.gate(&dag, &clinical_q("t", "o", 0.95)).is_admitted());
        let just_under = gate.gate(&dag, &clinical_q("t", "o", 0.94));
        assert!(matches!(
            just_under.refusal_reason(),
            Some(ClinicalRefusalReason::InsufficientProvenance { .. })
        ));
    }

    // ── Out of scope: deferred to the general gate ──

    #[test]
    fn observational_clinical_claim_is_out_of_scope() {
        // Clinical DOMAIN but observational → makes no causal claim → not this gate's job.
        let gate = ClinicalCausalGate::new();
        let dag = dag_with("a", "b");
        let q = CausalQuery {
            domain: "clinical".into(),
            pearl_level: PearlLevel::Observational,
            ..clinical_q("a", "b", 0.99)
        };
        let v = gate.gate(&dag, &q);
        assert!(!v.in_scope());
        assert!(matches!(v, ClinicalGateVerdict::NotInScope { .. }));
    }

    #[test]
    fn non_clinical_domain_is_out_of_scope() {
        // Engineering interventional is safety-relevant nowhere near the clinical floor.
        let gate = ClinicalCausalGate::new();
        let dag = dag_with("a", "b");
        let q = CausalQuery {
            domain: "engineering".into(),
            pearl_level: PearlLevel::Interventional,
            ..clinical_q("a", "b", 0.10)
        };
        assert!(matches!(
            gate.gate(&dag, &q),
            ClinicalGateVerdict::NotInScope { .. }
        ));
    }

    #[test]
    fn counterfactual_in_general_domain_defers_to_general_gate() {
        // Counterfactual is safety-critical per classify_query, but the DOMAIN is not
        // clinical, so the clinical provenance floor does not apply here — defer it.
        let gate = ClinicalCausalGate::new();
        let dag = dag_with("a", "b");
        let q = CausalQuery {
            domain: "general".into(),
            pearl_level: PearlLevel::Counterfactual,
            ..clinical_q("a", "b", 0.10)
        };
        assert!(matches!(
            gate.gate(&dag, &q),
            ClinicalGateVerdict::NotInScope { .. }
        ));
    }

    #[test]
    fn clinical_counterfactual_is_in_scope_and_gated() {
        // Clinical + counterfactual → in scope, full 0.95 + identifiability treatment.
        let gate = ClinicalCausalGate::new();
        let dag = dag_with("drug", "recovery");
        let q = CausalQuery {
            domain: "medical".into(),
            pearl_level: PearlLevel::Counterfactual,
            ..clinical_q("drug", "recovery", 0.99)
        };
        let v = gate.gate(&dag, &q);
        assert!(v.in_scope());
        assert!(v.is_admitted());
    }

    // ── Batch summary + the two measures ──

    #[test]
    fn summary_computes_both_measures() {
        let gate = ClinicalCausalGate::new();
        let dag = dag_with("drug", "recovery");

        let batch = vec![
            clinical_q("drug", "recovery", 0.98), // admitted
            clinical_q("drug", "recovery", 0.96), // admitted
            clinical_q("drug", "recovery", 0.90), // refused: provenance
            clinical_q("ghost", "nowhere", 0.99), // refused: not identifiable (not in dag)
            CausalQuery {
                // out of scope (observational)
                domain: "clinical".into(),
                pearl_level: PearlLevel::Observational,
                ..clinical_q("drug", "recovery", 0.99)
            },
        ];

        let s = gate.summarize(&dag, &batch);
        assert_eq!(s.admitted, 2);
        assert_eq!(s.refused_provenance, 1);
        assert_eq!(s.refused_identifiability, 1);
        assert_eq!(s.not_in_scope, 1);
        assert_eq!(s.evaluated(), 4);

        // provenance pass rate = (2 admitted + 1 identifiability-refused) / 4 evaluated = 0.75
        assert!((s.provenance_pass_rate().unwrap() - 0.75).abs() < 1e-9);
        // identifiability refusal rate = 1 / (2 admitted + 1 id-refused) = 1/3
        assert!((s.identifiability_refusal_rate().unwrap() - (1.0 / 3.0)).abs() < 1e-9);
    }

    #[test]
    fn measures_are_none_when_no_in_scope_claims() {
        let gate = ClinicalCausalGate::new();
        let dag = dag_with("a", "b");
        // Only out-of-scope claims evaluated.
        let s = gate.summarize(
            &dag,
            &[CausalQuery {
                domain: "general".into(),
                pearl_level: PearlLevel::Observational,
                ..clinical_q("a", "b", 0.5)
            }],
        );
        assert_eq!(s.evaluated(), 0);
        assert!(s.provenance_pass_rate().is_none());
        assert!(s.identifiability_refusal_rate().is_none());
    }

    #[test]
    fn gate_all_preserves_order() {
        let gate = ClinicalCausalGate::new();
        let dag = dag_with("drug", "recovery");
        let batch = vec![
            clinical_q("drug", "recovery", 0.98), // admitted
            clinical_q("drug", "recovery", 0.10), // refused provenance
        ];
        let verdicts = gate.gate_all(&dag, &batch);
        assert_eq!(verdicts.len(), 2);
        assert!(verdicts[0].is_admitted());
        assert!(verdicts[1].is_refused());
    }

    // ── render() smoke for each verdict shape ──

    #[test]
    fn render_describes_each_verdict() {
        let gate = ClinicalCausalGate::new();
        let dag = dag_with("drug", "recovery");

        assert!(
            gate.gate(&dag, &clinical_q("drug", "recovery", 0.98))
                .render()
                .starts_with("ADMITTED")
        );
        assert!(
            gate.gate(&dag, &clinical_q("drug", "recovery", 0.50))
                .render()
                .contains("insufficient provenance")
        );
        assert!(
            gate.gate(&CausalDag::new(), &clinical_q("x", "y", 0.99))
                .render()
                .contains("not identifiable")
        );
        let oos = CausalQuery {
            domain: "general".into(),
            pearl_level: PearlLevel::Observational,
            ..clinical_q("a", "b", 0.5)
        };
        assert!(gate.gate(&dag, &oos).render().starts_with("NOT IN SCOPE"));
    }
}
