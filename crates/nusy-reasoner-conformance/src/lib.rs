//! # nusy-reasoner-conformance — Wave-3 conformance + router registration (VY-B E4, EX-4890)
//!
//! The three **Wave-3** reasoners — [`nusy_inductive`] (generalize rules from instances),
//! [`nusy_analogical`] (structure-mapping), [`nusy_case_based`] (retrieve-and-adapt) — each
//! implement the [`Reasoner`](nusy_reasoner::Reasoner) contract directly. This crate is the thin
//! **conformance layer**: it registers them on a [`ReasonerRouter`](nusy_router::ReasonerRouter)
//! via their competence envelopes, and its test battery proves the contract holds for the family —
//! every one is object-safe (`Box<dyn Reasoner>`), routes by predicate, **never launders a
//! Heuristic into a Proven**, and reuses the min-guarantee `Pipeline` without forking the contract.
//!
//! Wave-3 is the *learning / non-deductive* tier: all three are inherently `sound: false`,
//! `probabilistic: true`, and answer only [`Provability::Heuristic`](nusy_reasoner::Provability)
//! or [`Abstained`](nusy_reasoner::Provability::Abstained) — never `Proven`. That is exactly what
//! the router's PAR battery's `false_proofs == 0` invariant guards, generalized across the family.
//!
//! ```
//! use nusy_reasoner_conformance::register_wave3;
//! use nusy_router::ReasonerRouter;
//! use nusy_inductive::{InductionConfig, InductiveReasoner};
//! use nusy_analogical::{AnalogicalReasoner, AnalogyConfig};
//! use nusy_case_based::{CaseBasedReasoner, CbrConfig};
//!
//! let mut router = ReasonerRouter::new();
//! register_wave3(
//!     &mut router,
//!     InductiveReasoner::from_rules(vec![]),
//!     AnalogicalReasoner::new(vec![], AnalogyConfig::default()),
//!     CaseBasedReasoner::new(vec![], CbrConfig::default()),
//! );
//! assert_eq!(router.len(), 3);
//! let _ = InductionConfig::default(); // (re-exported config types compose normally)
//! ```

use nusy_analogical::AnalogicalReasoner;
use nusy_case_based::CaseBasedReasoner;
use nusy_inductive::InductiveReasoner;
use nusy_router::ReasonerRouter;

/// Register the three Wave-3 reasoners onto `router`, in the canonical order
/// **inductive → analogical → case-based**.
///
/// Registration order is the router's Proven tie-break order ([`ReasonerRouter::push`]); since every
/// Wave-3 reasoner is Heuristic-only, order only affects equal-confidence ties, never whether a
/// proof is minted. The caller constructs each reasoner with its own domain data (instances / cases)
/// — this helper only does the honest wiring, so the competence envelopes the router dispatches on
/// are exactly the ones each crate computed from that data.
pub fn register_wave3(
    router: &mut ReasonerRouter,
    inductive: InductiveReasoner,
    analogical: AnalogicalReasoner,
    case_based: CaseBasedReasoner,
) {
    router.push(Box::new(inductive));
    router.push(Box::new(analogical));
    router.push(Box::new(case_based));
}
