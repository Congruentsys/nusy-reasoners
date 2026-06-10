//! `$apply` behaviour: grouped conditions accumulate into an evidence chain, provably-false
//! branches prune, and actions under an unknown condition abstain (never silently fire).

use nusy_cql::{Code, FactStore, Value};
use nusy_plandef::{Action, PlanAction, PlanDefinition, apply};
use std::collections::HashMap;

#[derive(Default)]
struct MapStore {
    props: HashMap<String, Vec<Value>>,
}

impl MapStore {
    fn set(mut self, key: &str, v: Value) -> Self {
        self.props.insert(key.to_string(), vec![v]);
        self
    }
}

impl FactStore for MapStore {
    fn get_property(&self, entity: &str, path: &[String]) -> Vec<Value> {
        let key = if path.is_empty() {
            entity.to_string()
        } else {
            format!("{entity}.{}", path.join("."))
        };
        self.props.get(&key).cloned().unwrap_or_default()
    }
    fn in_value_set(&self, _: &Code, _: &str) -> Option<bool> {
        None
    }
    fn subsumes(&self, _: &Code, _: &Code) -> Option<bool> {
        None
    }
}

/// A small two-level guideline: an elderly group, with a fall-assessment action gated on a
/// fall-risk finding.
fn fall_plan() -> PlanDefinition {
    PlanDefinition::new("fall-prevention", "Fall prevention").with_action(
        PlanAction::when("Patient.age >= 65").with_sub(
            PlanAction::when("Patient.fallRisk = true")
                .recommend(Action::new("assess-fall-risk", "Assess fall risk")),
        ),
    )
}

#[test]
fn grounded_proposal_carries_the_full_evidence_chain() {
    let store = MapStore::default()
        .set("Patient.age", Value::Integer(72))
        .set("Patient.fallRisk", Value::Boolean(true));
    let out = apply(&fall_plan(), &store).unwrap();

    assert_eq!(out.proposed.len(), 1);
    assert!(out.abstained.is_empty());
    let p = &out.proposed[0];
    assert_eq!(p.action.id, "assess-fall-risk");
    // Both the group condition and the leaf condition, root → action.
    assert_eq!(
        p.evidence,
        vec!["Patient.age >= 65", "Patient.fallRisk = true"]
    );
}

#[test]
fn provably_false_group_prunes_the_subtree() {
    // Younger patient: the group condition is false, so nothing under it fires or abstains.
    let store = MapStore::default()
        .set("Patient.age", Value::Integer(40))
        .set("Patient.fallRisk", Value::Boolean(true));
    let out = apply(&fall_plan(), &store).unwrap();
    assert!(out.proposed.is_empty());
    assert!(
        out.abstained.is_empty(),
        "false (not unknown) → no abstention"
    );
}

#[test]
fn unknown_condition_abstains_rather_than_fires() {
    // Age qualifies, but fallRisk is not recorded → the leaf condition is Null (unknown).
    let store = MapStore::default().set("Patient.age", Value::Integer(72));
    let out = apply(&fall_plan(), &store).unwrap();

    assert!(out.proposed.is_empty(), "must not fire on unknown evidence");
    assert_eq!(out.abstained.len(), 1);
    assert_eq!(out.abstained[0].action.id, "assess-fall-risk");
    assert!(out.abstained[0].reason.contains("unknown"));
}

#[test]
fn unknown_group_abstains_every_action_in_its_subtree() {
    // The group condition itself is unknown (no age recorded) → the whole subtree abstains.
    let store = MapStore::default().set("Patient.fallRisk", Value::Boolean(true));
    let out = apply(&fall_plan(), &store).unwrap();
    assert!(out.proposed.is_empty());
    assert_eq!(out.abstained.len(), 1);
    assert_eq!(out.abstained[0].action.id, "assess-fall-risk");
}

#[test]
fn unconditional_action_always_fires_with_empty_evidence() {
    let plan = PlanDefinition::new("p", "Always")
        .with_action(PlanAction::always().recommend(Action::new("educate", "Provide education")));
    let out = apply(&plan, &MapStore::default()).unwrap();
    assert_eq!(out.proposed.len(), 1);
    assert!(out.proposed[0].evidence.is_empty());
}

#[test]
fn a_condition_that_fails_to_parse_is_an_error() {
    let plan = PlanDefinition::new("p", "Bad")
        .with_action(PlanAction::when("Patient.age >= ").recommend(Action::new("x", "X")));
    assert!(apply(&plan, &MapStore::default()).is_err());
}
