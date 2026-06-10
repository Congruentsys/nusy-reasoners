//! # nusy-plandef — a PlanDefinition-analog decision graph (EX-4597, VOY-V18-2)
//!
//! The **VOY-V18-2** computable guideline structure: a [`PlanDefinition`] is a tree of
//! [`PlanAction`]s, each guarded by an optional **CQL-analog condition** ([`nusy_cql`]) and
//! optionally carrying a recommended [`Action`] (the ActivityDefinition analog). Ported from
//! Paper-128's `$apply` (§4.2–4.6): [`apply`] instantiates the graph against a patient's
//! facts and returns **grounded [`ActionProposal`]s, each with an evidence chain** — the
//! conditions that had to hold for it to fire.
//!
//! ## Three-valued, abstain-on-unknown
//!
//! Conditions evaluate in CQL's Kleene logic, so a node's applicability is `true` / `false`
//! / **unknown**. An action under an *unknown* condition is **abstained**, not proposed —
//! never asserted on missing evidence. This is the provable-gate contract at the
//! decision-graph level: a recommendation fires only when its conditions are *provably* true.
//!
//! ## Example
//!
//! ```
//! use nusy_plandef::{Action, PlanAction, PlanDefinition, apply};
//! use nusy_cql::{Code, FactStore, Value};
//!
//! struct P; // patient: age 72, no recorded smoking status
//! impl FactStore for P {
//!     fn get_property(&self, e: &str, path: &[String]) -> Vec<Value> {
//!         match (e, path.first().map(String::as_str)) {
//!             ("Patient", Some("age")) => vec![Value::Integer(72)],
//!             _ => vec![],
//!         }
//!     }
//!     fn in_value_set(&self, _: &Code, _: &str) -> Option<bool> { None }
//!     fn subsumes(&self, _: &Code, _: &Code) -> Option<bool> { None }
//! }
//!
//! let plan = PlanDefinition::new("fall-prevention", "Fall prevention")
//!     .with_action(
//!         PlanAction::when("Patient.age >= 65")
//!             .recommend(Action::new("assess-fall-risk", "Assess fall risk")),
//!     );
//! let out = apply(&plan, &P).unwrap();
//! assert_eq!(out.proposed.len(), 1);
//! assert_eq!(out.proposed[0].action.id, "assess-fall-risk");
//! assert_eq!(out.proposed[0].evidence, vec!["Patient.age >= 65"]); // why it fired
//! ```

use nusy_cql::{CqlError, FactStore, Value, evaluate};

pub use nusy_cql::Code;

/// A recommended action — the ActivityDefinition analog (what to do if it fires).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Action {
    /// Stable action identifier.
    pub id: String,
    /// Human-readable description.
    pub title: String,
    /// Optional terminology code for the action (procedure / medication / …).
    pub code: Option<Code>,
}

impl Action {
    /// A coded-later action with an id and title.
    pub fn new(id: impl Into<String>, title: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            code: None,
        }
    }

    /// Anchor the action to a terminology code.
    pub fn with_code(mut self, code: Code) -> Self {
        self.code = Some(code);
        self
    }
}

/// One node of the decision graph: an optional applicability condition, an optional
/// recommended action, and optional grouped sub-actions (which inherit applicability).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanAction {
    /// CQL-analog applicability condition source; `None` = always applicable.
    pub condition: Option<String>,
    /// The recommended action at this node, if any (a group node may have none).
    pub action: Option<Action>,
    /// Sub-actions, gated on this node's condition holding.
    pub sub_actions: Vec<PlanAction>,
}

impl PlanAction {
    /// An always-applicable node (no condition).
    pub fn always() -> Self {
        Self {
            condition: None,
            action: None,
            sub_actions: Vec::new(),
        }
    }

    /// A node guarded by `condition` (a CQL-analog expression source).
    pub fn when(condition: impl Into<String>) -> Self {
        Self {
            condition: Some(condition.into()),
            action: None,
            sub_actions: Vec::new(),
        }
    }

    /// Attach a recommended action to this node.
    pub fn recommend(mut self, action: Action) -> Self {
        self.action = Some(action);
        self
    }

    /// Add a gated sub-action.
    pub fn with_sub(mut self, sub: PlanAction) -> Self {
        self.sub_actions.push(sub);
        self
    }
}

/// A computable guideline: a named tree of [`PlanAction`]s.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanDefinition {
    /// Stable guideline identifier.
    pub id: String,
    /// Human-readable title.
    pub title: String,
    /// Top-level actions.
    pub actions: Vec<PlanAction>,
}

impl PlanDefinition {
    /// A new, empty plan.
    pub fn new(id: impl Into<String>, title: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            actions: Vec::new(),
        }
    }

    /// Add a top-level action.
    pub fn with_action(mut self, action: PlanAction) -> Self {
        self.actions.push(action);
        self
    }
}

/// A grounded recommendation: the action plus the **evidence chain** — the conditions, from
/// the root of the graph down to this action, that all had to hold for it to fire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionProposal {
    /// The recommended action.
    pub action: Action,
    /// The conditions (root → action) that held; the action's mini-proof of applicability.
    pub evidence: Vec<String>,
}

/// An action that was **not** proposed because its applicability is unknown (a guarding
/// condition evaluated to `Null` / a non-boolean). The gate must surface, not silently drop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Abstention {
    /// The action whose applicability could not be established.
    pub action: Action,
    /// Why it was abstained.
    pub reason: String,
}

/// The result of `$apply`: actions provably applicable, and actions whose applicability is
/// unknown (abstained — never silently dropped).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ApplyOutcome {
    /// Provably-applicable recommendations, each with its evidence chain.
    pub proposed: Vec<ActionProposal>,
    /// Actions left undecided because a guarding condition was unknown.
    pub abstained: Vec<Abstention>,
}

/// Three-valued applicability of a node, from its condition.
enum Applicability {
    Yes,
    No,
    Unknown,
}

/// `$apply` the plan against `store`: walk the decision graph, evaluating each node's
/// condition in CQL's three-valued logic, and collect grounded proposals (provably
/// applicable) and abstentions (applicability unknown).
///
/// Returns a [`CqlError`] only if a condition fails to *parse or evaluate*; a condition that
/// evaluates to `Null` or a non-boolean is treated as **unknown** (abstain), not an error.
pub fn apply(plan: &PlanDefinition, store: &dyn FactStore) -> Result<ApplyOutcome, CqlError> {
    let mut out = ApplyOutcome::default();
    let mut evidence: Vec<String> = Vec::new();
    for node in &plan.actions {
        walk(node, store, &mut evidence, &mut out)?;
    }
    Ok(out)
}

fn applicability(node: &PlanAction, store: &dyn FactStore) -> Result<Applicability, CqlError> {
    match &node.condition {
        None => Ok(Applicability::Yes),
        Some(src) => match evaluate(src, store)? {
            Value::Boolean(true) => Ok(Applicability::Yes),
            Value::Boolean(false) => Ok(Applicability::No),
            // Null or a non-boolean result → applicability cannot be established.
            _ => Ok(Applicability::Unknown),
        },
    }
}

fn walk(
    node: &PlanAction,
    store: &dyn FactStore,
    evidence: &mut Vec<String>,
    out: &mut ApplyOutcome,
) -> Result<(), CqlError> {
    match applicability(node, store)? {
        Applicability::No => Ok(()), // pruned: condition is provably false
        Applicability::Unknown => {
            // The whole subtree is gated on an unknown condition → abstain every action in it.
            let reason = match &node.condition {
                Some(src) => {
                    format!("applicability unknown: condition `{src}` did not hold provably")
                }
                None => "applicability unknown".to_string(),
            };
            for action in collect_actions(node) {
                out.abstained.push(Abstention {
                    action,
                    reason: reason.clone(),
                });
            }
            Ok(())
        }
        Applicability::Yes => {
            let guarded = node.condition.is_some();
            if let Some(src) = &node.condition {
                evidence.push(src.clone());
            }
            if let Some(action) = &node.action {
                out.proposed.push(ActionProposal {
                    action: action.clone(),
                    evidence: evidence.clone(),
                });
            }
            for sub in &node.sub_actions {
                walk(sub, store, evidence, out)?;
            }
            if guarded {
                evidence.pop();
            }
            Ok(())
        }
    }
}

/// Every recommended action in `node`'s subtree (the node's own action + its descendants').
fn collect_actions(node: &PlanAction) -> Vec<Action> {
    let mut out = Vec::new();
    if let Some(a) = &node.action {
        out.push(a.clone());
    }
    for sub in &node.sub_actions {
        out.extend(collect_actions(sub));
    }
    out
}
