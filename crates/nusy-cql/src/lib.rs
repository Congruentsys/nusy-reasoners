//! # nusy-cql — a CQL-analog expression language for the Y-graph
//!
//! `nusy-cql` is the **VOY-V18-2** computable expression layer: a small, self-contained
//! language for evaluating boolean / comparison / temporal / value-set / subsumption
//! logic over knowledge-graph facts. It is the Rust analog of HL7 **CQL**
//! (Clinical Quality Language) as described in the V12 prior art (Paper-128 /
//! `Architecture-for-CDS.md`) — the `IExpressionEngine` behavior, ported per the V14
//! Rust rules (port the *algorithm*, not the Python).
//!
//! ## Design: decoupled by a fact-access trait
//!
//! The language depends on **nothing** in the rest of the workspace. All graph access
//! goes through the [`FactStore`] trait, which a VOY-1/VOY-2 backend implements over the
//! Y1 entity store + value-set registry + ontology, and which tests implement in memory.
//! This keeps the expression language stable while the engine substrate (rdf-fusion /
//! native-Arrow, EX-4653) and the Y2 computable-rule representation are still being landed.
//!
//! ## Semantics
//!
//! Evaluation uses CQL **three-valued (Kleene) logic**: a missing fact is
//! [`Value::Null`] ("unknown"), and unknowns propagate (`null and true == null`,
//! `null or true == true`). This matters for the provable-gate use case downstream:
//! "unknown" is distinct from "false", so the gate can abstain rather than assert.
//!
//! ## Example
//!
//! ```
//! use nusy_cql::{evaluate, Code, FactStore, Value};
//!
//! struct Demo;
//! impl FactStore for Demo {
//!     fn get_property(&self, entity: &str, path: &[String]) -> Vec<Value> {
//!         match (entity, path.first().map(String::as_str)) {
//!             ("Patient", Some("age")) => vec![Value::Integer(72)],
//!             ("Condition", Some("code")) => vec![Value::Code(Code::new("SNOMED", "38341003"))],
//!             _ => vec![],
//!         }
//!     }
//!     fn in_value_set(&self, code: &Code, vs: &str) -> Option<bool> {
//!         Some(vs == "Hypertension" && code.code == "38341003")
//!     }
//!     fn subsumes(&self, _a: &Code, _d: &Code) -> Option<bool> { Some(false) }
//! }
//!
//! let v = evaluate("Patient.age >= 65 and Condition.code in \"Hypertension\"", &Demo).unwrap();
//! assert!(matches!(v, Value::Boolean(true)));
//! ```

mod ast;
mod error;
mod eval;
mod lexer;
mod parser;

pub use ast::{Code, CompOp, Expr, TemporalOp, Value};
pub use error::{EvalError, ParseError};
pub use eval::{eval, FactStore};
pub use parser::parse;

/// An error from [`evaluate`]: either parsing or evaluation failed.
#[derive(Debug, thiserror::Error, Clone, PartialEq)]
pub enum CqlError {
    /// The source failed to lex or parse.
    #[error("parse error: {0}")]
    Parse(#[from] ParseError),
    /// The parsed expression failed to evaluate.
    #[error("evaluation error: {0}")]
    Eval(#[from] EvalError),
}

/// Parse `src` and evaluate it against `store` in one call.
///
/// For repeated evaluation of the same expression, [`parse`] once and call [`eval`]
/// per fact set instead.
pub fn evaluate(src: &str, store: &dyn FactStore) -> Result<Value, CqlError> {
    let expr = parse(src)?;
    Ok(eval(&expr, store)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// In-memory store: property paths joined by '.', plus value-set and subsumption maps.
    #[derive(Default)]
    struct MapStore {
        props: HashMap<String, Vec<Value>>,
        value_sets: HashMap<(String, String), bool>, // (code, valueset) -> member
        subsumes: HashMap<(String, String), bool>,   // (ancestor, descendant) -> holds
    }

    impl MapStore {
        fn key(entity: &str, path: &[String]) -> String {
            if path.is_empty() {
                entity.to_string()
            } else {
                format!("{entity}.{}", path.join("."))
            }
        }
        fn with_prop(mut self, k: &str, v: Vec<Value>) -> Self {
            self.props.insert(k.to_string(), v);
            self
        }
    }

    impl FactStore for MapStore {
        fn get_property(&self, entity: &str, path: &[String]) -> Vec<Value> {
            self.props.get(&Self::key(entity, path)).cloned().unwrap_or_default()
        }
        fn in_value_set(&self, code: &Code, valueset: &str) -> Option<bool> {
            self.value_sets.get(&(code.code.clone(), valueset.to_string())).copied()
        }
        fn subsumes(&self, ancestor: &Code, descendant: &Code) -> Option<bool> {
            self.subsumes.get(&(ancestor.code.clone(), descendant.code.clone())).copied()
        }
    }

    fn b(v: Value) -> Option<bool> {
        match v {
            Value::Boolean(x) => Some(x),
            _ => None,
        }
    }

    #[test]
    fn literals_and_boolean_algebra() {
        let s = MapStore::default();
        assert_eq!(b(evaluate("true and false", &s).unwrap()), Some(false));
        assert_eq!(b(evaluate("true or false", &s).unwrap()), Some(true));
        assert_eq!(b(evaluate("not false", &s).unwrap()), Some(true));
        assert_eq!(b(evaluate("not (true and true)", &s).unwrap()), Some(false));
    }

    #[test]
    fn kleene_null_propagation() {
        let s = MapStore::default();
        // null and true == null ; null or true == true ; null and false == false
        assert!(matches!(evaluate("null and true", &s).unwrap(), Value::Null));
        assert_eq!(b(evaluate("null or true", &s).unwrap()), Some(true));
        assert_eq!(b(evaluate("null and false", &s).unwrap()), Some(false));
        assert!(matches!(evaluate("not null", &s).unwrap(), Value::Null));
    }

    #[test]
    fn numeric_comparison_int_decimal_mix() {
        let s = MapStore::default();
        assert_eq!(b(evaluate("3 < 4", &s).unwrap()), Some(true));
        assert_eq!(b(evaluate("3 >= 3", &s).unwrap()), Some(true));
        assert_eq!(b(evaluate("2.5 > 2", &s).unwrap()), Some(true));
        assert_eq!(b(evaluate("5 = 5.0", &s).unwrap()), Some(true));
        assert_eq!(b(evaluate("5 != 6", &s).unwrap()), Some(true));
    }

    #[test]
    fn property_lookup_and_absence() {
        let s = MapStore::default().with_prop("Patient.age", vec![Value::Integer(72)]);
        assert_eq!(b(evaluate("Patient.age >= 65", &s).unwrap()), Some(true));
        // Absent property -> Null -> comparison -> Null.
        assert!(matches!(evaluate("Patient.weight > 10", &s).unwrap(), Value::Null));
    }

    #[test]
    fn exists_present_and_absent() {
        let s = MapStore::default().with_prop("Condition.code", vec![Value::Code(Code::new("SNOMED", "38341003"))]);
        assert_eq!(b(evaluate("exists(Condition.code)", &s).unwrap()), Some(true));
        assert_eq!(b(evaluate("exists(Condition.absent)", &s).unwrap()), Some(false));
    }

    #[test]
    fn value_set_membership() {
        let mut s = MapStore::default().with_prop("Condition.code", vec![Value::Code(Code::new("SNOMED", "38341003"))]);
        s.value_sets.insert(("38341003".to_string(), "Hypertension".to_string()), true);
        assert_eq!(b(evaluate("Condition.code in \"Hypertension\"", &s).unwrap()), Some(true));
        // Unknown value set -> Null.
        assert!(matches!(evaluate("Condition.code in \"Unknown\"", &s).unwrap(), Value::Null));
    }

    #[test]
    fn list_membership() {
        let s = MapStore::default();
        assert_eq!(b(evaluate("2 in (1, 2, 3)", &s).unwrap()), Some(true));
        assert_eq!(b(evaluate("9 in (1, 2, 3)", &s).unwrap()), Some(false));
        // null element makes a non-match undecidable.
        assert!(matches!(evaluate("9 in (1, null, 3)", &s).unwrap(), Value::Null));
    }

    #[test]
    fn subsumption() {
        let mut s = MapStore::default();
        s.subsumes.insert(("73211009".to_string(), "44054006".to_string()), true);
        // Diabetes mellitus (parent) subsumes type-2 diabetes (child).
        let src = "Code('SNOMED','73211009') subsumes Code('SNOMED','44054006')";
        assert_eq!(b(evaluate(src, &s).unwrap()), Some(true));
        // Unknown pair -> Null.
        let unk = "Code('SNOMED','1') subsumes Code('SNOMED','2')";
        assert!(matches!(evaluate(unk, &s).unwrap(), Value::Null));
    }

    #[test]
    fn temporal_relations() {
        let s = MapStore::default();
        assert_eq!(b(evaluate("DateTime(1) before DateTime(5)", &s).unwrap()), Some(true));
        assert_eq!(b(evaluate("DateTime(9) after DateTime(5)", &s).unwrap()), Some(true));
        assert_eq!(b(evaluate("DateTime(3) during Interval(DateTime(1), DateTime(5))", &s).unwrap()), Some(true));
        assert_eq!(b(evaluate("DateTime(7) during Interval(DateTime(1), DateTime(5))", &s).unwrap()), Some(false));
        assert_eq!(
            b(evaluate("Interval(DateTime(1), DateTime(4)) overlaps Interval(DateTime(3), DateTime(9))", &s).unwrap()),
            Some(true)
        );
    }

    #[test]
    fn quantity_equality_respects_units() {
        let s = MapStore::default();
        assert_eq!(b(evaluate("Quantity(5, 'mg') = Quantity(5, 'mg')", &s).unwrap()), Some(true));
        assert_eq!(b(evaluate("Quantity(5, 'mg') = Quantity(5, 'g')", &s).unwrap()), Some(false));
    }

    #[test]
    fn parse_errors_are_reported() {
        assert!(matches!(parse("3 < "), Err(ParseError::UnexpectedEof { .. })));
        assert!(matches!(parse("3 @ 4"), Err(ParseError::UnexpectedChar { .. })));
        assert!(matches!(parse("(1, 2"), Err(_)));
    }

    #[test]
    fn type_errors_surface() {
        let s = MapStore::default();
        // boolean operator on a string
        assert!(matches!(evaluate("'x' and true", &s), Err(CqlError::Eval(EvalError::TypeError { .. }))));
        // ordering on codes
        let bad = "Code('S','1') < Code('S','2')";
        assert!(matches!(evaluate(bad, &s), Err(CqlError::Eval(EvalError::TypeError { .. }))));
    }

    #[test]
    fn compound_clinical_rule() {
        // "elderly hypertensive": age >= 65 AND condition code in Hypertension value set
        let mut s = MapStore::default()
            .with_prop("Patient.age", vec![Value::Integer(72)])
            .with_prop("Condition.code", vec![Value::Code(Code::new("SNOMED", "38341003"))]);
        s.value_sets.insert(("38341003".to_string(), "Hypertension".to_string()), true);
        let rule = "Patient.age >= 65 and Condition.code in \"Hypertension\"";
        assert_eq!(b(evaluate(rule, &s).unwrap()), Some(true));

        // Younger patient -> false (and short-circuits the unknown value set away).
        let s2 = MapStore::default().with_prop("Patient.age", vec![Value::Integer(40)]);
        let rule2 = "Patient.age >= 65 and Condition.code in \"Hypertension\"";
        assert_eq!(b(evaluate(rule2, &s2).unwrap()), Some(false));
    }
}
