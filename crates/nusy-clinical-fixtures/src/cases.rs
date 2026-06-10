//! The gold-case clinical fixtures.
//!
//! Each models a small, real-guideline-flavoured scenario with multi-step derivation so the
//! proof path has depth. Conditions are categorical (value-set / finding membership);
//! numeric thresholds (age cut-offs, eGFR bands) are the upstream perception/`nusy-cql`
//! layer's job and arrive here already as findings — keeping the harness focused on
//! derivation, contraindication, proof, and abstention.

use crate::ClinicalFixture;
use nusy_unify::{Rule, Triple, TriplePattern};

use crate::IdRule;

fn t(s: &str, p: &str, o: &str) -> Triple {
    Triple::new(s, p, o)
}

/// `id: head :- body...`
fn rule(id: &str, body: &[(&str, &str, &str)], head: (&str, &str, &str)) -> IdRule {
    let lhs = body
        .iter()
        .map(|(s, p, o)| TriplePattern::parse(s, p, o))
        .collect();
    let rhs = vec![TriplePattern::parse(head.0, head.1, head.2)];
    IdRule::new(id, Rule::new(lhs, rhs))
}

/// Fall-risk assessment: a two-step guideline (at-risk, then recommend) that fires only for
/// an at-risk patient who is also in the over-65 age band.
fn fall_risk_fires() -> ClinicalFixture {
    let rules = vec![
        rule(
            "at-risk-fall",
            &[
                ("?p", "has_condition", "?c"),
                ("?c", "increases_fall_risk", "true"),
            ],
            ("?p", "at_risk", "fall"),
        ),
        rule(
            "recommend-fall-assessment",
            &[("?p", "at_risk", "fall"), ("?p", "age_band", "over_65")],
            ("?p", "recommend", "fall_assessment"),
        ),
    ];
    ClinicalFixture {
        name: "fall_risk_assessment_fires".to_string(),
        patient_facts: vec![
            t("patient1", "has_condition", "osteoporosis"),
            t("osteoporosis", "increases_fall_risk", "true"),
            t("patient1", "age_band", "over_65"),
        ],
        rules,
        expected_recommendations: vec![
            t("patient1", "at_risk", "fall"), // intermediate, also provable
            t("patient1", "recommend", "fall_assessment"),
        ],
        contraindicated: vec![],
        negative_controls: vec![
            // Nothing derives a transplant for this patient — the gate must abstain.
            t("patient1", "recommend", "kidney_transplant"),
        ],
    }
}

/// Same guideline, but the patient is NOT in the over-65 band: the recommendation must not
/// fire (the precondition is simply absent — provable-only, never assumed).
fn fall_risk_missing_age_does_not_fire() -> ClinicalFixture {
    let rules = vec![
        rule(
            "at-risk-fall",
            &[
                ("?p", "has_condition", "?c"),
                ("?c", "increases_fall_risk", "true"),
            ],
            ("?p", "at_risk", "fall"),
        ),
        rule(
            "recommend-fall-assessment",
            &[("?p", "at_risk", "fall"), ("?p", "age_band", "over_65")],
            ("?p", "recommend", "fall_assessment"),
        ),
    ];
    ClinicalFixture {
        name: "fall_risk_no_age_does_not_fire".to_string(),
        patient_facts: vec![
            t("patient2", "has_condition", "osteoporosis"),
            t("osteoporosis", "increases_fall_risk", "true"),
            // no age_band fact for patient2
        ],
        rules,
        // The intermediate is still correctly derivable...
        expected_recommendations: vec![t("patient2", "at_risk", "fall")],
        // ...but the recommendation must NOT fire without the age precondition.
        contraindicated: vec![t("patient2", "recommend", "fall_assessment")],
        negative_controls: vec![],
    }
}

/// Glycemic-therapy intensification gated on adequate renal status. One patient qualifies;
/// a second meets the indication but has impaired renal status — a contraindication that
/// must suppress the recommendation.
fn glycemic_intensify_with_renal_contraindication() -> ClinicalFixture {
    let rules = vec![
        rule(
            "intensify-candidate",
            &[
                ("?p", "has_condition", "type2_diabetes"),
                ("?p", "has_observation", "hba1c_elevated"),
            ],
            ("?p", "intensify_candidate", "glycemic"),
        ),
        rule(
            "recommend-intensify",
            &[
                ("?p", "intensify_candidate", "glycemic"),
                ("?p", "renal_status", "adequate"),
            ],
            ("?p", "recommend", "intensify_glycemic"),
        ),
    ];
    ClinicalFixture {
        name: "glycemic_intensify_renal_contraindication".to_string(),
        patient_facts: vec![
            // patient3 — indicated but renal-impaired → contraindicated.
            t("patient3", "has_condition", "type2_diabetes"),
            t("patient3", "has_observation", "hba1c_elevated"),
            t("patient3", "renal_status", "impaired"),
            // patient4 — indicated and renal-adequate → recommended.
            t("patient4", "has_condition", "type2_diabetes"),
            t("patient4", "has_observation", "hba1c_elevated"),
            t("patient4", "renal_status", "adequate"),
        ],
        rules,
        expected_recommendations: vec![t("patient4", "recommend", "intensify_glycemic")],
        contraindicated: vec![
            // Renal impairment must block intensification for patient3.
            t("patient3", "recommend", "intensify_glycemic"),
        ],
        negative_controls: vec![],
    }
}

/// The gold-case set the VOY-6 eval battery (EX-4617) scores against.
pub fn gold_cases() -> Vec<ClinicalFixture> {
    vec![
        fall_risk_fires(),
        fall_risk_missing_age_does_not_fire(),
        glycemic_intensify_with_renal_contraindication(),
    ]
}
