//! The provable-claim gate's contract: a provable claim is answered WITH its proof; an
//! unprovable one is flagged (never asserted) — the zero-hallucination invariant.

use nusy_forward_chain::{IdRule, forward_chain};
use nusy_gate::{GateResponse, ProvableClaimGate};
use nusy_unify::{Rule, Triple, TriplePattern};

fn t(s: &str, p: &str, o: &str) -> Triple {
    Triple::new(s, p, o)
}

/// Two-step clinical guideline: at_risk ← condition+risk, then recommend ← at_risk+age.
fn fall_gate() -> ProvableClaimGate {
    let at_risk = IdRule::new(
        "at-risk-fall",
        Rule::new(
            vec![
                TriplePattern::parse("?p", "has_condition", "?c"),
                TriplePattern::parse("?c", "increases_fall_risk", "true"),
            ],
            vec![TriplePattern::parse("?p", "at_risk", "fall")],
        ),
    );
    let recommend = IdRule::new(
        "recommend-fall-assessment",
        Rule::new(
            vec![
                TriplePattern::parse("?p", "at_risk", "fall"),
                TriplePattern::parse("?p", "age_band", "over_65"),
            ],
            vec![TriplePattern::parse("?p", "recommend", "fall_assessment")],
        ),
    );
    let facts = vec![
        t("p1", "has_condition", "osteoporosis"),
        t("osteoporosis", "increases_fall_risk", "true"),
        t("p1", "age_band", "over_65"),
    ];
    ProvableClaimGate::new(forward_chain(&[at_risk, recommend], facts))
}

#[test]
fn provable_claim_is_answered_with_its_proof() {
    let gate = fall_gate();
    let resp = gate.gate(&t("p1", "recommend", "fall_assessment"));
    assert!(resp.is_proven());
    let proof = resp.proof().expect("proof attached");
    // Two-step derivation: recommend ← at_risk ← seed facts.
    assert_eq!(proof.depth(), 2);
    assert!(proof.rule_ids().contains(&"recommend-fall-assessment"));
    assert!(proof.rule_ids().contains(&"at-risk-fall"));
    // Grounded only in seed axioms.
    assert!(
        proof
            .axioms()
            .iter()
            .all(|ax| gate.gate(ax).proof().map(|p| p.is_axiom()).unwrap_or(true))
    );
}

#[test]
fn unprovable_claim_is_flagged_never_asserted() {
    let gate = fall_gate();
    // Nothing derives a stroke risk or a transplant for p1 — the gate must NOT assert them.
    for claim in [
        t("p1", "at_risk", "stroke"),
        t("p1", "recommend", "kidney_transplant"),
        t("p2", "recommend", "fall_assessment"), // unknown patient
    ] {
        let resp = gate.gate(&claim);
        assert!(!resp.is_proven(), "must not prove {claim:?}");
        assert!(resp.proof().is_none());
        match resp {
            GateResponse::Unproven { reason, .. } => assert!(reason.contains("no derivation")),
            GateResponse::Proven { .. } => panic!("hallucination: asserted an unprovable claim"),
        }
    }
}

#[test]
fn intermediate_and_seed_facts_are_both_provable() {
    let gate = fall_gate();
    // The intermediate derived fact is provable (1-step proof).
    let at_risk = gate.gate(&t("p1", "at_risk", "fall"));
    assert!(at_risk.is_proven());
    assert_eq!(at_risk.proof().unwrap().depth(), 1);
    // A seed fact is provable as an axiom (0-step proof).
    let seed = gate.gate(&t("p1", "age_band", "over_65"));
    assert!(seed.is_proven());
    assert_eq!(seed.proof().unwrap().depth(), 0);
}

#[test]
fn batch_gating_preserves_order_and_summarizes() {
    let gate = fall_gate();
    let claims = vec![
        t("p1", "recommend", "fall_assessment"), // proven
        t("p1", "at_risk", "stroke"),            // flagged
        t("p1", "at_risk", "fall"),              // proven
    ];
    let responses = gate.gate_all(&claims);
    assert_eq!(responses.len(), 3);
    assert!(responses[0].is_proven());
    assert!(!responses[1].is_proven());
    assert!(responses[2].is_proven());
    // Order preserved: each response is about its input claim.
    assert_eq!(responses[1].claim(), &t("p1", "at_risk", "stroke"));

    let summary = gate.summarize(&claims);
    assert_eq!(summary.proven, 2);
    assert_eq!(summary.unproven, 1);
}

#[test]
fn render_distinguishes_proven_from_flagged() {
    let gate = fall_gate();
    let proven = gate.gate(&t("p1", "at_risk", "fall")).render();
    assert!(proven.contains("PROVEN"));
    assert!(proven.contains("by at-risk-fall")); // the proof tree is rendered

    let flagged = gate.gate(&t("p1", "at_risk", "stroke")).render();
    assert!(flagged.contains("UNPROVEN"));
    assert!(flagged.contains("neural"));
}
