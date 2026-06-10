//! # nusy-gate — the provable-claim gate (EX-4610, VOY-V18-5)
//!
//! The contract between V18's symbolic and neural layers, and the source of its
//! **zero-hallucination-on-provable-claims** guarantee. A claim is routed through the
//! executable reasoning engine ([`nusy_forward_chain`]):
//!
//! - **Provable** — the claim is a fact in the [`Saturation`] and a proof tree exists →
//!   [`GateResponse::Proven`], carrying the derivation that justifies it. The gate answers
//!   symbolically, with evidence.
//! - **Not provable** — no derivation → [`GateResponse::Unproven`]. The gate **never asserts**
//!   it; it is flagged and routed to the neural layer (the un-provable remainder). Abstaining
//!   or a flagged neural answer is the caller's policy — the gate's job is the verdict + proof.
//!
//! The gate is the *general* `ProvableClaimGate` over the VOY-1 proof API (§12.4). The
//! clinical-causal sub-gate (0.95 provenance + identifiability refusal, EX-4643) wraps
//! `safety_routing` separately; provenance *surfacing* of a returned proof is EX-4612
//! ([`nusy-provenance`](https://docs.rs/nusy-provenance)).
//!
//! ```
//! use nusy_gate::ProvableClaimGate;
//! use nusy_forward_chain::{forward_chain, IdRule};
//! use nusy_unify::{Rule, Triple, TriplePattern};
//!
//! // at_risk(?p,"fall") :- has_condition(?p,?c), increases_fall_risk(?c,"true")
//! let rule = IdRule::new("at-risk-fall", Rule::new(
//!     vec![TriplePattern::parse("?p", "has_condition", "?c"),
//!          TriplePattern::parse("?c", "increases_fall_risk", "true")],
//!     vec![TriplePattern::parse("?p", "at_risk", "fall")]));
//! let facts = vec![
//!     Triple::new("p1", "has_condition", "osteoporosis"),
//!     Triple::new("osteoporosis", "increases_fall_risk", "true"),
//! ];
//! let gate = ProvableClaimGate::new(forward_chain(&[rule], facts));
//!
//! // A provable claim is answered WITH its proof.
//! assert!(gate.gate(&Triple::new("p1", "at_risk", "fall")).is_proven());
//! // An unsupported claim is flagged, never asserted.
//! assert!(!gate.gate(&Triple::new("p1", "at_risk", "stroke")).is_proven());
//! ```

use nusy_forward_chain::{ProofTree, Saturation};
use nusy_unify::Triple;

/// The gate's verdict on a claim.
#[derive(Debug, Clone)]
pub enum GateResponse {
    /// The claim is **provable** — answered symbolically, carrying the proof that justifies it.
    Proven {
        /// The claim, now an established fact.
        claim: Triple,
        /// Its derivation tree (down to seed axioms).
        proof: ProofTree,
    },
    /// The claim is **not provable** from the knowledge graph — flagged for the neural layer.
    /// The gate never asserts it; the caller abstains or routes to a flagged neural answer.
    Unproven {
        /// The claim that could not be established.
        claim: Triple,
        /// Why it was flagged.
        reason: String,
    },
}

impl GateResponse {
    /// Was the claim provable (answered symbolically)?
    pub fn is_proven(&self) -> bool {
        matches!(self, GateResponse::Proven { .. })
    }

    /// The proof, if the claim was provable.
    pub fn proof(&self) -> Option<&ProofTree> {
        match self {
            GateResponse::Proven { proof, .. } => Some(proof),
            GateResponse::Unproven { .. } => None,
        }
    }

    /// The claim this verdict is about.
    pub fn claim(&self) -> &Triple {
        match self {
            GateResponse::Proven { claim, .. } | GateResponse::Unproven { claim, .. } => claim,
        }
    }

    /// A human-readable rendering: the answer + its proof, or the flagged reason.
    pub fn render(&self) -> String {
        match self {
            GateResponse::Proven { claim, proof } => format!(
                "PROVEN: {} {} {}\nproof:\n{}",
                claim.subject,
                claim.predicate,
                claim.object,
                proof.render()
            ),
            GateResponse::Unproven { claim, reason } => format!(
                "UNPROVEN (flagged → neural): {} {} {}  — {reason}",
                claim.subject, claim.predicate, claim.object
            ),
        }
    }
}

/// Summary of gating a batch of claims — how many were answered symbolically vs flagged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GateSummary {
    /// Claims answered with a proof.
    pub proven: usize,
    /// Claims flagged unproven (routed to neural).
    pub unproven: usize,
}

/// The provable-claim gate over a forward-chained [`Saturation`].
///
/// Construct it from the engine's saturation (the closed fact set + derivations); each
/// [`gate`](Self::gate) call asks the proof API whether a claim holds and returns the verdict.
#[derive(Debug, Clone)]
pub struct ProvableClaimGate {
    saturation: Saturation,
}

impl ProvableClaimGate {
    /// Build a gate over the engine's saturated knowledge.
    pub fn new(saturation: Saturation) -> Self {
        Self { saturation }
    }

    /// Gate one claim: provable → [`GateResponse::Proven`] with its proof; otherwise
    /// [`GateResponse::Unproven`] (flagged, never asserted).
    pub fn gate(&self, claim: &Triple) -> GateResponse {
        match self.saturation.proof_of(claim) {
            Some(proof) => GateResponse::Proven {
                claim: claim.clone(),
                proof,
            },
            None => GateResponse::Unproven {
                claim: claim.clone(),
                reason: "no derivation in the executable Y-graph".to_string(),
            },
        }
    }

    /// Gate a batch of claims, preserving order.
    pub fn gate_all(&self, claims: &[Triple]) -> Vec<GateResponse> {
        claims.iter().map(|c| self.gate(c)).collect()
    }

    /// Gate a batch and tally proven vs flagged.
    pub fn summarize(&self, claims: &[Triple]) -> GateSummary {
        let mut s = GateSummary::default();
        for c in claims {
            if self.gate(c).is_proven() {
                s.proven += 1;
            } else {
                s.unproven += 1;
            }
        }
        s
    }
}
