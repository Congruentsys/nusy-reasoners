//! Integration test: a small clinical-guideline-style fixture exercised end-to-end
//! through the public API, with a realistic in-memory [`FactStore`].
//!
//! This mirrors the VOY-V18-2 intent — a CQL-analog rule (e.g. an eGFR dosing
//! threshold, a value-set condition match, a subsumption check) evaluated over
//! patient facts — without any dependency on the (still-landing) engine substrate.

use nusy_cql::{Code, FactStore, Value, eval, evaluate, parse};
use std::collections::HashMap;

struct Patient {
    props: HashMap<String, Vec<Value>>,
    hypertension_codes: Vec<String>,
    /// (ancestor, descendant) is-a pairs known to hold.
    isa: Vec<(String, String)>,
}

impl Patient {
    fn key(entity: &str, path: &[String]) -> String {
        if path.is_empty() {
            entity.to_string()
        } else {
            format!("{entity}.{}", path.join("."))
        }
    }
}

impl FactStore for Patient {
    fn get_property(&self, entity: &str, path: &[String]) -> Vec<Value> {
        self.props
            .get(&Self::key(entity, path))
            .cloned()
            .unwrap_or_default()
    }
    fn in_value_set(&self, code: &Code, valueset: &str) -> Option<bool> {
        match valueset {
            "Hypertension" => Some(self.hypertension_codes.contains(&code.code)),
            _ => None, // unknown value set -> Null
        }
    }
    fn subsumes(&self, ancestor: &Code, descendant: &Code) -> Option<bool> {
        Some(
            self.isa
                .iter()
                .any(|(a, d)| *a == ancestor.code && *d == descendant.code),
        )
    }
}

fn truth(v: Value) -> Option<bool> {
    match v {
        Value::Boolean(b) => Some(b),
        _ => None,
    }
}

fn elderly_hypertensive_with_normal_kidney() -> Patient {
    let mut props = HashMap::new();
    props.insert("Patient.age".to_string(), vec![Value::Integer(72)]);
    props.insert("Observation.eGFR".to_string(), vec![Value::Decimal(58.0)]);
    props.insert(
        "Condition.code".to_string(),
        vec![Value::Code(Code::new("SNOMED", "38341003"))],
    );
    Patient {
        props,
        hypertension_codes: vec!["38341003".to_string()],
        isa: vec![("73211009".to_string(), "44054006".to_string())],
    }
}

#[test]
fn guideline_recommendation_fires() {
    let p = elderly_hypertensive_with_normal_kidney();
    // Recommend antihypertensive review: age >= 65 AND hypertensive AND eGFR not critically low.
    let rule =
        "Patient.age >= 65 and Condition.code in \"Hypertension\" and Observation.eGFR >= 30";
    assert_eq!(truth(evaluate(rule, &p).unwrap()), Some(true));
}

#[test]
fn contraindication_blocks_when_egfr_low() {
    let mut p = elderly_hypertensive_with_normal_kidney();
    p.props
        .insert("Observation.eGFR".to_string(), vec![Value::Decimal(18.0)]);
    // doNotPerform-style: eGFR < 30 must contraindicate.
    let contra = "Observation.eGFR < 30";
    assert_eq!(truth(evaluate(contra, &p).unwrap()), Some(true));
}

#[test]
fn unknown_fact_yields_null_not_false() {
    let p = elderly_hypertensive_with_normal_kidney();
    // No smoking-status fact recorded -> the rule is *unknown*, not false.
    let rule = "Patient.smoker = true";
    assert!(matches!(evaluate(rule, &p).unwrap(), Value::Null));
}

#[test]
fn subsumption_drives_a_match() {
    let p = elderly_hypertensive_with_normal_kidney();
    // Diabetes mellitus subsumes type-2 diabetes.
    let rule = "Code('SNOMED','73211009') subsumes Code('SNOMED','44054006')";
    assert_eq!(truth(evaluate(rule, &p).unwrap()), Some(true));
    // Reverse does not hold.
    let rev = "Code('SNOMED','44054006') subsumes Code('SNOMED','73211009')";
    assert_eq!(truth(evaluate(rev, &p).unwrap()), Some(false));
}

#[test]
fn parse_once_eval_many() {
    // Compile the rule once; evaluate against two different patients.
    let ast = parse("Patient.age >= 65").unwrap();

    let old = elderly_hypertensive_with_normal_kidney();
    assert_eq!(truth(eval(&ast, &old).unwrap()), Some(true));

    let mut young_props = HashMap::new();
    young_props.insert("Patient.age".to_string(), vec![Value::Integer(30)]);
    let young = Patient {
        props: young_props,
        hypertension_codes: vec![],
        isa: vec![],
    };
    assert_eq!(truth(eval(&ast, &young).unwrap()), Some(false));
}
