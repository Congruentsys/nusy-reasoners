//! Wave-3 reasoner conformance battery (VY-B E4, EX-4890).
//!
//! Proves the three learning-tier reasoners conform to the Reasoner contract as a family:
//! object-safe, route by competence envelope, NEVER launder a Heuristic into a Proven, and reuse
//! the min-guarantee `Pipeline`. The load-bearing assertion is the generalized zero-hallucination
//! invariant: across the whole Wave-3 family, the router's PAR battery reports `false_proofs == 0`.

use nusy_analogical::{AnalogicalReasoner, AnalogyConfig};
use nusy_case_based::{CaseBasedReasoner, CbrConfig};
use nusy_inductive::{InductionConfig, InductiveReasoner};
use nusy_reasoner::compose::Pipeline;
use nusy_reasoner::{Provability, Query, Reasoner};
use nusy_reasoner_conformance::register_wave3;
use nusy_router::ReasonerRouter;
use nusy_router::reasoner_router::RouteOutcome;
use nusy_unify::Triple;

fn t(s: &str, p: &str, o: &str) -> Triple {
    Triple::new(s, p, o)
}

// ── Fixtures: each reasoner gets a DISTINCT head predicate so routing is unambiguous ──
//   inductive  → predicate "can"   (induces is_a=bird ⇒ can=fly)
//   analogical → predicate "is"    (precedent: enforceable)
//   case-based → predicate "order" (clinical: order flu_test)

fn inductive() -> InductiveReasoner {
    let instances = vec![
        t("tweety", "is_a", "bird"),
        t("tweety", "can", "fly"),
        t("robin", "is_a", "bird"),
        t("robin", "can", "fly"),
        t("crow", "is_a", "bird"),
        t("crow", "can", "fly"),
    ];
    InductiveReasoner::from_instances(&instances, &InductionConfig::default())
}

fn analogical() -> AnalogicalReasoner {
    let precedent = nusy_analogical::Case::new(
        "precedent-1",
        vec![
            t("contract_a", "has_clause", "arbitration"),
            t("court", "ruled", "contract_a"),
            t("contract_a", "is", "enforceable"),
        ],
    );
    AnalogicalReasoner::new(vec![precedent], AnalogyConfig::default())
}

fn case_based() -> CaseBasedReasoner {
    let prior = nusy_case_based::Case::new(
        "flu-case",
        vec![("has_symptom", "fever"), ("has_symptom", "cough")],
        vec![("order", "flu_test")],
    );
    CaseBasedReasoner::new(vec![prior], CbrConfig::default())
}

// The queries each reasoner should answer (Heuristically).
fn inductive_query() -> Query {
    Query {
        goal: t("sparrow", "can", "fly"),
        context: vec![t("sparrow", "is_a", "bird")],
    }
}
fn analogical_query() -> Query {
    Query {
        goal: t("contract_b", "is", "enforceable"),
        context: vec![
            t("contract_b", "has_clause", "arbitration"),
            t("court", "ruled", "contract_b"),
        ],
    }
}
fn case_based_query() -> Query {
    Query {
        goal: t("patient_b", "order", "flu_test"),
        context: vec![
            t("patient_b", "has_symptom", "fever"),
            t("patient_b", "has_symptom", "cough"),
        ],
    }
}

// ── 1. Object safety ─────────────────────────────────────────────────────────

#[test]
fn wave3_are_object_safe_boxed_reasoners() {
    // If any of the three were not object-safe, this vec would not compile.
    let fleet: Vec<Box<dyn Reasoner>> = vec![
        Box::new(inductive()),
        Box::new(analogical()),
        Box::new(case_based()),
    ];
    assert_eq!(fleet.len(), 3);
    // Each honestly self-describes as unsound + probabilistic (never Proven-capable).
    for r in &fleet {
        let g = r.guarantee();
        assert!(!g.sound, "Wave-3 reasoner must be unsound (Heuristic-only)");
        assert!(g.probabilistic, "Wave-3 reasoner carries a confidence");
    }
}

// ── 2. Router registration ───────────────────────────────────────────────────

#[test]
fn register_wave3_populates_router() {
    let mut router = ReasonerRouter::new();
    assert!(router.is_empty());
    register_wave3(&mut router, inductive(), analogical(), case_based());
    assert_eq!(router.len(), 3);
}

// ── 3. Each routes by envelope → Heuristic, never Proven ──────────────────────

#[test]
fn router_routes_each_to_heuristic_never_proven() {
    let mut router = ReasonerRouter::new();
    register_wave3(&mut router, inductive(), analogical(), case_based());

    for (label, q) in [
        ("inductive", inductive_query()),
        ("analogical", analogical_query()),
        ("case-based", case_based_query()),
    ] {
        match router.route(&q) {
            RouteOutcome::Answered(v) => {
                assert_eq!(
                    v.answer.provability(),
                    Provability::Heuristic,
                    "{label} must answer Heuristic"
                );
                assert_ne!(
                    v.answer.provability(),
                    Provability::Proven,
                    "{label} must NEVER launder to Proven"
                );
            }
            RouteOutcome::Abstained { reason, .. } => {
                panic!("{label} query should have been answered, got abstain: {reason}")
            }
        }
    }
}

// ── 4. The generalized zero-hallucination invariant (load-bearing) ────────────

#[test]
fn wave3_par_battery_zero_false_proofs() {
    let mut router = ReasonerRouter::new();
    register_wave3(&mut router, inductive(), analogical(), case_based());

    // Panel: the three answerable claims are should_prove=FALSE (the family must NOT prove them —
    // they are heuristic generalizations/analogies/precedents, not derivations); plus one
    // should_prove=TRUE claim no Wave-3 reasoner can prove (→ a coverage miss, never a false proof).
    let panel = vec![
        (inductive_query(), false),
        (analogical_query(), false),
        (case_based_query(), false),
        (
            Query {
                goal: t("x", "deductively_entails", "y"),
                context: vec![t("x", "is_a", "bird")],
            },
            true,
        ),
    ];
    let report = router.par(&panel);

    // The invariant the whole contract exists to protect:
    assert_eq!(
        report.false_proofs, 0,
        "Wave-3 family must NEVER mint a false proof"
    );
    assert_eq!(
        report.silent_drops, 0,
        "every claim routes to an outcome (loud abstention)"
    );
    // The three heuristic claims were correctly left unproven; the deductive claim was a miss.
    assert_eq!(report.correctly_unproven, 3);
    assert_eq!(report.missed, 1);
    assert_eq!(report.proven_expected, 0);
}

// ── 5. Min-guarantee Pipeline reuse (no fork of the contract) ─────────────────

#[test]
fn wave3_reuses_min_guarantee_pipeline() {
    // A pipeline whose stages are Wave-3 reasoners stays Heuristic — composition can only weaken,
    // never strengthen, the guarantee. case-based abstains on the enforceability query (no matching
    // features), then analogical answers it: the composite is Heuristic, never Proven.
    let pipeline = Pipeline::new("wave3-min-guarantee")
        .then(Box::new(case_based()))
        .then(Box::new(analogical()));

    let a = pipeline.answer(&analogical_query());
    assert_eq!(a.value, Some(t("contract_b", "is", "enforceable")));
    assert_eq!(
        a.provability(),
        Provability::Heuristic,
        "a Wave-3 pipeline composes to Heuristic"
    );
    assert_ne!(
        a.provability(),
        Provability::Proven,
        "min-guarantee never launders a heuristic chain to Proven"
    );
}

// ── 6. Determinism through the router ─────────────────────────────────────────

#[test]
fn routing_is_deterministic() {
    let mut router = ReasonerRouter::new();
    register_wave3(&mut router, inductive(), analogical(), case_based());
    let q = case_based_query();
    let first = router.route(&q);
    let second = router.route(&q);
    match (first, second) {
        (RouteOutcome::Answered(a), RouteOutcome::Answered(b)) => {
            assert_eq!(a.answer.value, b.answer.value);
            assert_eq!(a.reasoner_index, b.reasoner_index);
        }
        _ => panic!("expected both routes to answer"),
    }
}
