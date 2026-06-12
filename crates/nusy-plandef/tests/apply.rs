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

// ── EX-4692: negative knowledge — suppressed-by-contraindication ────────────────

/// A plan whose action is a prohibition (do_not_perform), gated on a risk condition:
/// "do NOT prescribe the combination when the patient is already on an ACE inhibitor."
fn contraindication_plan() -> PlanDefinition {
    PlanDefinition::new("acei-arb-guard", "ACEI+ARB combination guard").with_action(
        PlanAction::when("Patient.onAceInhibitor = true")
            .recommend(Action::new("add-arb", "Add an ARB").do_not_perform()),
    )
}

#[test]
fn applicable_prohibition_is_suppressed_not_proposed() {
    // EX-4692: the contraindication's condition holds → it is an explicit negative that fired.
    let store = MapStore::default().set("Patient.onAceInhibitor", Value::Boolean(true));
    let out = apply(&contraindication_plan(), &store).unwrap();

    assert!(
        out.proposed.is_empty(),
        "a prohibition must never be proposed"
    );
    assert!(out.abstained.is_empty());
    assert_eq!(
        out.suppressed.len(),
        1,
        "applicable prohibition surfaces as suppressed"
    );
    let s = &out.suppressed[0];
    assert_eq!(s.action.id, "add-arb");
    assert!(s.action.do_not_perform);
    // Carries the provenance of WHY the contraindication applies — never silence.
    assert_eq!(s.evidence, vec!["Patient.onAceInhibitor = true"]);
}

#[test]
fn prohibition_with_false_condition_does_not_fire() {
    // Not on an ACE inhibitor → the prohibition does not apply; nothing fires.
    let store = MapStore::default().set("Patient.onAceInhibitor", Value::Boolean(false));
    let out = apply(&contraindication_plan(), &store).unwrap();
    assert!(out.proposed.is_empty());
    assert!(
        out.suppressed.is_empty(),
        "a non-applicable prohibition is not a finding"
    );
    assert!(out.abstained.is_empty());
}

#[test]
fn prohibition_under_unknown_condition_abstains_not_suppresses() {
    // EX-4692: applicability unknown (missing data) → abstain, distinct from suppression.
    // A contraindication we cannot establish must not masquerade as one that fired.
    let store = MapStore::default(); // Patient.onAceInhibitor unknown
    let out = apply(&contraindication_plan(), &store).unwrap();
    assert!(
        out.suppressed.is_empty(),
        "unknown applicability is not a fired contraindication"
    );
    assert_eq!(
        out.abstained.len(),
        1,
        "unknown-applicability prohibition abstains"
    );
    assert_eq!(out.abstained[0].action.id, "add-arb");
    assert!(out.proposed.is_empty());
}
