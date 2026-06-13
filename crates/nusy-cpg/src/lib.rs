//! # nusy-cpg — one real clinical practice guideline, imported into computable form
//!
//! **VOY-V18-2 / EX-4599.** This crate imports a single *real* clinical practice guideline —
//! **JNC 8** blood-pressure management (2014 Eighth Joint National Committee evidence-based
//! guideline for hypertension in adults) — into NuSy's computable representation: a
//! [`nusy_plandef::PlanDefinition`] whose applicability conditions are CQL-analog expressions
//! ([`nusy_cql`]). It is the proof that the VOY-2 representation (FHIR-style entities + CQL +
//! PlanDefinition `$apply`) can carry a guideline a clinician would recognise, end to end.
//!
//! ## What is encoded (JNC 8, the load-bearing recommendations)
//!
//! Source: James PA et al., *2014 Evidence-Based Guideline for the Management of High Blood
//! Pressure in Adults (JNC 8)*, JAMA 2014;311(5):507–520. The computable subset:
//!
//! | # | Guideline statement (paraphrased) | Encoded as |
//! |---|---|---|
//! | 1 | Age ≥ 60, no DM/CKD → BP target < 150/90 | `bp-target-150-90` |
//! | 2 | Age < 60 → BP target < 140/90 | `bp-target-140-90` |
//! | 3 | Diabetes **or** CKD (any age) → BP target < 140/90 | `bp-target-140-90-comorbid` |
//! | 8 | CKD present → include an ACEI **or** ARB (renal protection) | `init-acei-or-arb-ckd` |
//! | 9 | General non-black population → thiazide / CCB / ACEI / ARB first-line | `init-first-line-general` |
//!
//! **A contraindication carried as a *negative*:** JNC 8 (and every later guideline) says an
//! ACEI and an ARB must **not** be combined. There is therefore *no* `combine-acei-arb` action
//! in the graph — [`hypertension_jnc8`] can never propose it. Tests assert that invariant.
//!
//! ## Why a PlanDefinition (not free text)
//!
//! `$apply` ([`nusy_plandef::apply`]) instantiates the graph against a patient's FHIR-style
//! facts and returns grounded recommendations *with an evidence chain*, abstaining (never
//! asserting) when a guarding condition is unknown. A recommendation therefore fires only when
//! its conditions are **provably** true — the provable-gate contract, on a real guideline.
//!
//! ## Value sets
//!
//! Conditions reference named value sets (`HypertensionVS`, `DiabetesVS`, `CKDVS`) resolved by
//! the caller's [`nusy_cql::FactStore`]; [`JNC8_VALUE_SETS`] lists their member codes (SNOMED)
//! so a backend can implement membership. The codes are illustrative anchors, not a full
//! terminology import (that is the VOY-2 terminology expedition, EX-4595).

use nusy_plandef::{Action, PlanAction, PlanDefinition};

/// Stable identifier for the imported guideline.
pub const GUIDELINE_ID: &str = "jnc8-hypertension";

/// Citation for the imported guideline (provenance).
pub const GUIDELINE_CITATION: &str = "James PA et al. 2014 Evidence-Based Guideline for the Management of High Blood Pressure \
     in Adults (JNC 8). JAMA. 2014;311(5):507-520.";

/// The SNOMED codes each JNC 8 value set contains (illustrative anchors).
/// A [`nusy_cql::FactStore`] backs `in_value_set` from this table.
pub const JNC8_VALUE_SETS: &[(&str, &[&str])] = &[
    // Essential / secondary hypertension.
    ("HypertensionVS", &["38341003", "59621000"]),
    // Diabetes mellitus type 1 / type 2.
    ("DiabetesVS", &["46635009", "44054006"]),
    // Chronic kidney disease (CKD) and CKD stage 3.
    ("CKDVS", &["709044004", "433146000"]),
];

/// Build the JNC 8 hypertension-management guideline as a computable [`PlanDefinition`].
///
/// The whole plan is gated on the patient actually having hypertension; recommendations branch
/// on age band and comorbidity. Applied with [`nusy_plandef::apply`] over a patient's facts.
pub fn hypertension_jnc8() -> PlanDefinition {
    PlanDefinition::new(GUIDELINE_ID, "JNC 8 hypertension management").with_action(
        // Root: only applicable to patients with a hypertension diagnosis.
        PlanAction::when("Condition.code in \"HypertensionVS\"")
            // Rec 2 — age < 60 → target < 140/90.
            .with_sub(
                PlanAction::when("Patient.age < 60")
                    .recommend(Action::new("bp-target-140-90", "Treat to BP < 140/90 mmHg")),
            )
            // Rec 1 — age >= 60 (and, per the comorbidity override below, refined when DM/CKD).
            .with_sub(
                PlanAction::when("Patient.age >= 60")
                    .recommend(Action::new("bp-target-150-90", "Treat to BP < 150/90 mmHg")),
            )
            // Rec 3 — diabetes or CKD at any age → tighter target < 140/90 (overrides the
            // age-60 relaxation; both may be proposed, and the comorbid target is the governing one).
            .with_sub(
                PlanAction::when("Condition.code in \"DiabetesVS\" or Condition.code in \"CKDVS\"")
                    .recommend(Action::new(
                        "bp-target-140-90-comorbid",
                        "Diabetes/CKD: treat to BP < 140/90 mmHg",
                    )),
            )
            // Rec 8 — CKD present → include an ACEI or ARB for renal protection.
            .with_sub(
                PlanAction::when("Condition.code in \"CKDVS\"").recommend(Action::new(
                    "init-acei-or-arb-ckd",
                    "CKD: initial therapy should include an ACEI or ARB",
                )),
            )
            // Rec 9 — general non-black population → thiazide / CCB / ACEI / ARB first-line.
            .with_sub(
                PlanAction::when("not (Patient.race = \"black\")").recommend(Action::new(
                    "init-first-line-general",
                    "Initiate thiazide, CCB, ACEI, or ARB (general population)",
                )),
            ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use nusy_cql::{Code, FactStore, Value};
    use std::collections::HashMap;

    /// A patient-fact store backed by simple maps, with JNC 8 value sets wired from the table.
    #[derive(Default)]
    struct Patient {
        props: HashMap<String, Vec<Value>>,
    }

    impl Patient {
        fn with(mut self, key: &str, vals: Vec<Value>) -> Self {
            self.props.insert(key.to_string(), vals);
            self
        }
        fn code(c: &str) -> Value {
            Value::Code(Code::new("SNOMED", c))
        }
    }

    impl FactStore for Patient {
        fn get_property(&self, entity: &str, path: &[String]) -> Vec<Value> {
            let key = if path.is_empty() {
                entity.to_string()
            } else {
                format!("{entity}.{}", path.join("."))
            };
            self.props.get(&key).cloned().unwrap_or_default()
        }
        fn in_value_set(&self, code: &Code, valueset: &str) -> Option<bool> {
            // Resolve membership from the published JNC8 value-set table.
            JNC8_VALUE_SETS
                .iter()
                .find(|(name, _)| *name == valueset)
                .map(|(_, codes)| codes.contains(&code.code.as_str()))
        }
        fn subsumes(&self, _: &Code, _: &Code) -> Option<bool> {
            None
        }
    }

    fn proposed_ids(out: &nusy_plandef::ApplyOutcome) -> Vec<&str> {
        out.proposed.iter().map(|p| p.action.id.as_str()).collect()
    }

    #[test]
    fn guideline_imports_and_cites_its_source() {
        let plan = hypertension_jnc8();
        assert_eq!(plan.id, GUIDELINE_ID);
        assert!(GUIDELINE_CITATION.contains("JNC 8"));
        assert!(!plan.actions.is_empty());
    }

    #[test]
    fn elderly_hypertensive_gets_150_90_target() {
        // 72yo with hypertension, no DM/CKD, not black.
        let p = Patient::default()
            .with("Patient.age", vec![Value::Integer(72)])
            .with("Patient.race", vec![Value::Str("white".into())])
            .with("Condition.code", vec![Patient::code("38341003")]);
        let out = nusy_plandef::apply(&hypertension_jnc8(), &p).unwrap();
        let ids = proposed_ids(&out);
        assert!(ids.contains(&"bp-target-150-90"), "age>=60 → 150/90");
        assert!(!ids.contains(&"bp-target-140-90"), "not <60");
        assert!(!ids.contains(&"bp-target-140-90-comorbid"), "no DM/CKD");
        assert!(
            ids.contains(&"init-first-line-general"),
            "non-black general first-line"
        );
        // The recommendation carries its evidence chain (root + age condition).
        let target = out
            .proposed
            .iter()
            .find(|p| p.action.id == "bp-target-150-90")
            .unwrap();
        assert!(target.evidence.iter().any(|e| e.contains("HypertensionVS")));
        assert!(target.evidence.iter().any(|e| e.contains("age >= 60")));
    }

    #[test]
    fn young_hypertensive_gets_140_90_target() {
        let p = Patient::default()
            .with("Patient.age", vec![Value::Integer(45)])
            .with("Patient.race", vec![Value::Str("white".into())])
            .with("Condition.code", vec![Patient::code("38341003")]);
        let out = nusy_plandef::apply(&hypertension_jnc8(), &p).unwrap();
        let ids = proposed_ids(&out);
        assert!(ids.contains(&"bp-target-140-90"), "age<60 → 140/90");
        assert!(!ids.contains(&"bp-target-150-90"));
    }

    #[test]
    fn diabetic_hypertensive_gets_comorbid_target_and_acei_when_ckd() {
        // 68yo, hypertension + CKD → comorbid 140/90 target AND ACEI/ARB renal protection.
        let p = Patient::default()
            .with("Patient.age", vec![Value::Integer(68)])
            .with("Patient.race", vec![Value::Str("white".into())])
            .with(
                "Condition.code",
                vec![Patient::code("38341003"), Patient::code("709044004")], // HTN + CKD
            );
        let out = nusy_plandef::apply(&hypertension_jnc8(), &p).unwrap();
        let ids = proposed_ids(&out);
        assert!(
            ids.contains(&"bp-target-140-90-comorbid"),
            "CKD → tighter target"
        );
        assert!(ids.contains(&"init-acei-or-arb-ckd"), "CKD → ACEI/ARB");
        assert!(
            ids.contains(&"bp-target-150-90"),
            "age>=60 base rec also fires"
        );
    }

    #[test]
    fn non_hypertensive_gets_no_recommendations() {
        // Root condition false → entire guideline pruned (negative control).
        let p = Patient::default()
            .with("Patient.age", vec![Value::Integer(72)])
            .with("Condition.code", vec![Patient::code("00000000")]); // not in HypertensionVS
        let out = nusy_plandef::apply(&hypertension_jnc8(), &p).unwrap();
        assert!(
            out.proposed.is_empty(),
            "no hypertension → no recommendations"
        );
        assert!(
            out.abstained.is_empty(),
            "false root prunes, does not abstain"
        );
    }

    #[test]
    fn hypertensive_with_unknown_age_abstains_age_band_not_fires() {
        // Has hypertension but no recorded age → age-band recs abstain (unknown), never assert.
        let p = Patient::default()
            .with("Patient.race", vec![Value::Str("white".into())])
            .with("Condition.code", vec![Patient::code("38341003")]);
        // no Patient.age fact
        let out = nusy_plandef::apply(&hypertension_jnc8(), &p).unwrap();
        let ids = proposed_ids(&out);
        assert!(
            !ids.contains(&"bp-target-140-90"),
            "unknown age must not fire <60 target"
        );
        assert!(
            !ids.contains(&"bp-target-150-90"),
            "unknown age must not fire >=60 target"
        );
        // The age-band actions are surfaced as abstentions, not silently dropped.
        let abstained_ids: Vec<&str> = out.abstained.iter().map(|a| a.action.id.as_str()).collect();
        assert!(abstained_ids.contains(&"bp-target-140-90"));
        assert!(abstained_ids.contains(&"bp-target-150-90"));
        // The non-age-gated general first-line rec still fires (race known).
        assert!(ids.contains(&"init-first-line-general"));
    }

    #[test]
    fn contraindication_combine_acei_arb_is_never_proposed() {
        // The guideline must NEVER recommend combining ACEI + ARB — there is no such action.
        // Exercise a CKD patient (the arm most likely to touch ACEI/ARB) and assert absence.
        let p = Patient::default()
            .with("Patient.age", vec![Value::Integer(70)])
            .with("Patient.race", vec![Value::Str("white".into())])
            .with(
                "Condition.code",
                vec![Patient::code("38341003"), Patient::code("433146000")],
            );
        let out = nusy_plandef::apply(&hypertension_jnc8(), &p).unwrap();
        let ids = proposed_ids(&out);
        assert!(
            !ids.iter().any(|id| id.contains("combine")),
            "ACEI+ARB combination never proposed"
        );
        // Sanity: the CKD ACEI/ARB single-agent rec DID fire (so the arm was actually reached).
        assert!(ids.contains(&"init-acei-or-arb-ckd"));
    }
}
