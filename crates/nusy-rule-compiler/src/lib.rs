//! # nusy-rule-compiler — cortex-extracted rule clauses → computable Y2 (EX-4606, VOY-V18-4)
//!
//! The bridge that closes the V12 / V16 gap named in the port-map (§6, `V18-PORTMAP-engine-crates.md`):
//! **cortex extracts facts well but has no path to *compile* its extracted rule clauses into
//! executable Y2** — forward-chain rules a [`nusy_forward_chain::forward_chain`] can fire, or
//! PlanDefinition decision graphs [`nusy_plandef::apply`] can walk. This crate is that compilation
//! pass.
//!
//! ## Inputs (cortex → here)
//!
//! Two structured clause shapes match what the cortex's enricher already emits (see
//! `crates/nusy-y-layers/src/enricher.rs`'s `{"if": "...", "then": "...", "label": "..."}` clauses):
//!
//! - [`RuleClause`] — an if/then with **triple-pattern bodies and heads** (e.g.
//!   `?p has_condition ?c, ?c increases_fall_risk true ⊢ ?p at_risk fall`). Compiles to a
//!   [`nusy_forward_chain::IdRule`] / [`nusy_cog_computable::NamedRule`] that the engine fires.
//! - [`PlanClause`] — a CQL-analog **condition + recommended action** (e.g.
//!   `Patient.age >= 65 ⊢ assess-fall-risk`). Compiles to a [`nusy_plandef::PlanDefinition`].
//!
//! ## Outputs
//!
//! - [`compile_rule`] / [`compile_plan`] for single clauses.
//! - [`compile_bundle`] for a clause batch → a [`nusy_cog_computable::ComputableY2`] bundle, ready
//!   for [`nusy_cog_computable::reify`] (COG transfer) or direct execution.
//!
//! ## What "replaces regex pattern matching" means
//!
//! The V12 / V16 cortex represented rules as *string patterns matched by regex*. The clinical
//! fixture harness (EX-4615), the engine (EX-4588), the gate (EX-4610), and the provenance surface
//! (EX-4612) all operate on structured, **executable** Y2 — `Rule` / `PlanDefinition` / value-sets
//! — not regex hits. This crate is the deterministic transform between the two: a cortex-side
//! string-shaped clause becomes an engine-side fireable rule, with a stable id that flows through
//! `proof.rule_ids()` and `RulePath` (EX-4611, `nusy-router`) downstream.
//!
//! ## Range-restriction (safety)
//!
//! A forward-chain rule head may not introduce variables the body has not bound — the engine
//! silently drops non-range-restricted heads (`nusy-unify::Rule::unsafe_head_vars`), which would
//! show up as a *silent* extraction loss. The compiler refuses such clauses with
//! [`CompileError::UnsafeHead`], surfacing the problem at compile time where it is fixable.
//!
//! ## Example — round-trip
//!
//! ```
//! use nusy_forward_chain::forward_chain;
//! use nusy_rule_compiler::{compile_rule, NamedRuleExt, RuleClause};
//! use nusy_unify::Triple;
//!
//! let clause = RuleClause::new(
//!     "at-risk-fall",
//!     ["?p has_condition ?c", "?c increases_fall_risk true"],
//!     ["?p at_risk fall"],
//! );
//! let named = compile_rule(&clause).unwrap();
//! let sat = forward_chain(
//!     &[named.to_id_rule()],
//!     vec![
//!         Triple::new("p1", "has_condition", "osteoporosis"),
//!         Triple::new("osteoporosis", "increases_fall_risk", "true"),
//!     ],
//! );
//! assert!(sat.contains(&Triple::new("p1", "at_risk", "fall")));
//! ```

use nusy_cog_computable::{ComputableY2, NamedRule, ValueSet};
use nusy_forward_chain::IdRule;
use nusy_plandef::{Action, PlanAction, PlanDefinition};
use nusy_unify::{Rule, TriplePattern};

/// Errors raised while compiling cortex-extracted clauses.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum CompileError {
    /// A clause id was empty — every compiled rule must carry a stable identifier so the
    /// engine's `Derivation::rule_id` (and the downstream `proof.rule_ids()` / `RulePath`)
    /// can name it.
    #[error("rule clause has an empty id (every compiled rule needs a stable id)")]
    EmptyId,

    /// A clause had no body and no head — there is nothing to compile.
    #[error("rule clause `{id}` has no body and no head")]
    Empty {
        /// Clause id.
        id: String,
    },

    /// A triple-pattern string did not parse into a `subject predicate object` triple
    /// (3 whitespace-delimited terms).
    #[error(
        "rule clause `{id}` has a malformed pattern: `{pattern}` (expected `subject predicate object` with 3 whitespace-delimited terms)"
    )]
    MalformedPattern {
        /// Clause id.
        id: String,
        /// The offending pattern source.
        pattern: String,
    },

    /// A rule's head introduces a variable the body never binds — the forward-chain engine
    /// cannot ground it (the head atom would be silently dropped). Caught at compile time
    /// rather than letting the engine silently lose the head.
    #[error(
        "rule clause `{id}` head introduces unbound variables {unbound:?} (head variables must appear in the body — range-restriction)"
    )]
    UnsafeHead {
        /// Clause id.
        id: String,
        /// Variable names appearing in the head but not the body.
        unbound: Vec<String>,
    },

    /// A PlanClause carried a CQL condition source that did not parse.
    #[error("plan clause `{id}` has an unparseable CQL condition: {message}")]
    InvalidCondition {
        /// Clause id.
        id: String,
        /// Parser diagnostic.
        message: String,
    },
}

/// One forward-chain rule clause as the cortex extracted it: triple-pattern bodies and heads,
/// plus a stable id.
///
/// Strings are in the `nusy_unify::TriplePattern::parse` `?`-prefix convention: `?x` is a
/// variable, anything else is a constant. Each pattern is 3 whitespace-delimited terms
/// (`subject predicate object`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuleClause {
    /// Stable rule id — flows into `Derivation::rule_id` and `proof.rule_ids()`.
    pub id: String,
    /// Body patterns (the LHS / "if").
    pub body: Vec<String>,
    /// Head patterns (the RHS / "then").
    pub head: Vec<String>,
}

impl RuleClause {
    /// Build a clause from string slices over body and head patterns.
    pub fn new<I, J, S>(id: impl Into<String>, body: I, head: J) -> Self
    where
        I: IntoIterator<Item = S>,
        J: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            id: id.into(),
            body: body.into_iter().map(Into::into).collect(),
            head: head.into_iter().map(Into::into).collect(),
        }
    }
}

/// One PlanDefinition clause as the cortex extracted it: a CQL-analog applicability condition,
/// a recommended action (id + title), and a stable plan id.
///
/// Compiles to a flat single-action [`PlanDefinition`]; nested decision graphs can be assembled
/// downstream by combining multiple compiled plans (`PlanDefinition::with_action`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanClause {
    /// Stable plan id.
    pub id: String,
    /// Plan title (human-readable).
    pub title: String,
    /// CQL-analog applicability condition source (e.g. `"Patient.age >= 65"`). Validated at
    /// compile time via `nusy_cql::parse`.
    pub condition_cql: String,
    /// Recommended action id (becomes `Action::id`).
    pub action_id: String,
    /// Recommended action title (becomes `Action::title`).
    pub action_title: String,
    /// EX-4692: FHIR `doNotPerform` — `true` compiles to a prohibition
    /// (`Action.do_not_perform`), surfaced as suppressed-by-contraindication in `apply`.
    /// Defaults `false`; existing constructors set it via `..Default::default()` or the
    /// field directly.
    pub do_not_perform: bool,
}

impl PlanClause {
    /// Construct a plan clause.
    pub fn new(
        id: impl Into<String>,
        title: impl Into<String>,
        condition_cql: impl Into<String>,
        action_id: impl Into<String>,
        action_title: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            condition_cql: condition_cql.into(),
            action_id: action_id.into(),
            action_title: action_title.into(),
            do_not_perform: false,
        }
    }

    /// Mark this clause a prohibition (FHIR `doNotPerform`) — compiles to a
    /// suppressed-by-contraindication action. EX-4692.
    pub fn do_not_perform(mut self) -> Self {
        self.do_not_perform = true;
        self
    }
}

/// Compile a [`RuleClause`] into a [`NamedRule`] that the forward-chain engine can fire and the
/// COG can transfer. Returns [`CompileError`] on malformed patterns, an empty id, or a head that
/// introduces unbound variables.
pub fn compile_rule(clause: &RuleClause) -> Result<NamedRule, CompileError> {
    if clause.id.is_empty() {
        return Err(CompileError::EmptyId);
    }
    if clause.body.is_empty() && clause.head.is_empty() {
        return Err(CompileError::Empty {
            id: clause.id.clone(),
        });
    }
    let lhs = parse_patterns(&clause.id, &clause.body)?;
    let rhs = parse_patterns(&clause.id, &clause.head)?;
    let rule = Rule::new(lhs, rhs);
    let unbound = rule.unsafe_head_vars();
    if !unbound.is_empty() {
        return Err(CompileError::UnsafeHead {
            id: clause.id.clone(),
            unbound,
        });
    }
    Ok(NamedRule::new(clause.id.clone(), rule))
}

/// Compile a [`PlanClause`] into a single-action [`PlanDefinition`].
///
/// The CQL condition source is validated at compile time (`nusy_cql::parse`) so a downstream
/// `apply` cannot fail on unparseable CQL.
pub fn compile_plan(clause: &PlanClause) -> Result<PlanDefinition, CompileError> {
    if clause.id.is_empty() {
        return Err(CompileError::EmptyId);
    }
    nusy_cql::parse(&clause.condition_cql).map_err(|e| CompileError::InvalidCondition {
        id: clause.id.clone(),
        message: e.to_string(),
    })?;
    // EX-4692: a doNotPerform clause compiles to a prohibition Action — when its condition
    // holds, apply() surfaces it as suppressed-by-contraindication (not proposed).
    let mut action = Action::new(clause.action_id.clone(), clause.action_title.clone());
    action.do_not_perform = clause.do_not_perform;
    Ok(PlanDefinition::new(clause.id.clone(), clause.title.clone())
        .with_action(PlanAction::when(clause.condition_cql.clone()).recommend(action)))
}

/// Compile a batch of clauses and value-sets into a [`ComputableY2`] bundle, ready for
/// [`nusy_cog_computable::reify`] (COG transfer) or direct execution.
pub fn compile_bundle(
    rules: &[RuleClause],
    plans: &[PlanClause],
    value_sets: &[ValueSet],
) -> Result<ComputableY2, CompileError> {
    let rules = rules.iter().map(compile_rule).collect::<Result<_, _>>()?;
    let plans = plans.iter().map(compile_plan).collect::<Result<_, _>>()?;
    Ok(ComputableY2 {
        rules,
        plans,
        value_sets: value_sets.to_vec(),
    })
}

/// Helper trait: `NamedRule::to_id_rule()` lets the doctest hand a compiled rule directly to
/// [`nusy_forward_chain::forward_chain`].
pub trait NamedRuleExt {
    /// The forward-chain engine's view of this named rule.
    fn to_id_rule(&self) -> IdRule;
}

impl NamedRuleExt for NamedRule {
    fn to_id_rule(&self) -> IdRule {
        IdRule::new(self.id.clone(), self.rule.clone())
    }
}

// ── Internal ────────────────────────────────────────────────────────────────────────────────

fn parse_patterns(id: &str, sources: &[String]) -> Result<Vec<TriplePattern>, CompileError> {
    sources.iter().map(|s| parse_pattern(id, s)).collect()
}

fn parse_pattern(id: &str, source: &str) -> Result<TriplePattern, CompileError> {
    let parts: Vec<&str> = source.split_whitespace().collect();
    if parts.len() != 3 {
        return Err(CompileError::MalformedPattern {
            id: id.to_string(),
            pattern: source.to_string(),
        });
    }
    Ok(TriplePattern::parse(parts[0], parts[1], parts[2]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nusy_cql::Code;
    use nusy_forward_chain::forward_chain;
    use nusy_unify::Triple;

    #[test]
    fn compile_rule_round_trips_through_forward_chain() {
        let clause = RuleClause::new(
            "at-risk-fall",
            ["?p has_condition ?c", "?c increases_fall_risk true"],
            ["?p at_risk fall"],
        );
        let named = compile_rule(&clause).expect("compiles");
        assert_eq!(named.id, "at-risk-fall");

        let sat = forward_chain(
            &[named.to_id_rule()],
            vec![
                Triple::new("p1", "has_condition", "osteoporosis"),
                Triple::new("osteoporosis", "increases_fall_risk", "true"),
            ],
        );
        let derived = Triple::new("p1", "at_risk", "fall");
        assert!(sat.contains(&derived));
        // The derivation carries the clause id — closing the loop with proof.rule_ids().
        assert_eq!(
            sat.derivation_of(&derived).expect("derived").rule_id,
            "at-risk-fall"
        );
    }

    #[test]
    fn compile_rule_rejects_empty_id() {
        let clause = RuleClause::new("", ["?p has_condition ?c"], ["?p at_risk fall"]);
        assert_eq!(compile_rule(&clause), Err(CompileError::EmptyId));
    }

    #[test]
    fn compile_rule_rejects_empty_clause() {
        let clause = RuleClause::new("nothing", Vec::<String>::new(), Vec::<String>::new());
        assert_eq!(
            compile_rule(&clause),
            Err(CompileError::Empty {
                id: "nothing".into()
            })
        );
    }

    #[test]
    fn compile_rule_rejects_malformed_pattern() {
        // Two whitespace terms — not a triple.
        let clause = RuleClause::new("bad", ["?p has_condition"], ["?p at_risk fall"]);
        let err = compile_rule(&clause).unwrap_err();
        match err {
            CompileError::MalformedPattern { id, pattern } => {
                assert_eq!(id, "bad");
                assert_eq!(pattern, "?p has_condition");
            }
            other => panic!("expected MalformedPattern, got {other:?}"),
        }
    }

    #[test]
    fn compile_rule_rejects_unsafe_head() {
        // Head introduces ?x which the body never binds.
        let clause = RuleClause::new("unsafe", ["?p has_condition ?c"], ["?p at_risk ?x"]);
        match compile_rule(&clause).unwrap_err() {
            CompileError::UnsafeHead { id, unbound } => {
                assert_eq!(id, "unsafe");
                assert_eq!(unbound, vec!["x"]);
            }
            other => panic!("expected UnsafeHead, got {other:?}"),
        }
    }

    #[test]
    fn compile_rule_handles_multi_head() {
        // Two heads share the body's bindings — both must ground.
        let clause = RuleClause::new(
            "co-fires",
            ["?p has_condition diabetes"],
            ["?p flagged true", "?p needs_review true"],
        );
        let named = compile_rule(&clause).expect("compiles");
        let sat = forward_chain(
            &[named.to_id_rule()],
            vec![Triple::new("p1", "has_condition", "diabetes")],
        );
        assert!(sat.contains(&Triple::new("p1", "flagged", "true")));
        assert!(sat.contains(&Triple::new("p1", "needs_review", "true")));
    }

    #[test]
    fn compile_plan_validates_cql_and_emits_single_action() {
        let clause = PlanClause::new(
            "fall-prevention",
            "Fall prevention",
            "Patient.age >= 65",
            "assess-fall-risk",
            "Assess fall risk",
        );
        let plan = compile_plan(&clause).expect("compiles");
        assert_eq!(plan.id, "fall-prevention");
        assert_eq!(plan.title, "Fall prevention");
        assert_eq!(plan.actions.len(), 1);
        let node = &plan.actions[0];
        assert_eq!(node.condition.as_deref(), Some("Patient.age >= 65"));
        let action = node.action.as_ref().expect("has action");
        assert_eq!(action.id, "assess-fall-risk");
        assert_eq!(action.title, "Assess fall risk");
        assert!(!action.do_not_perform, "default clause is a recommendation");
    }

    #[test]
    fn do_not_perform_clause_compiles_to_a_prohibition_action() {
        // EX-4692: a doNotPerform clause compiles to Action.do_not_perform=true, so apply()
        // surfaces it as suppressed-by-contraindication (verified in nusy-plandef tests).
        let clause = PlanClause::new(
            "acei-arb-guard",
            "ACEI+ARB combination guard",
            "Patient.onAceInhibitor = true",
            "add-arb",
            "Add an ARB",
        )
        .do_not_perform();
        let plan = compile_plan(&clause).expect("compiles");
        let action = plan.actions[0].action.as_ref().expect("has action");
        assert!(
            action.do_not_perform,
            "doNotPerform clause must compile to a prohibition"
        );
        assert_eq!(action.id, "add-arb");
    }

    #[test]
    fn compile_plan_rejects_invalid_cql() {
        let clause = PlanClause::new(
            "bad",
            "Bad CQL",
            "Patient.age >>>== 65", // not a valid CQL operator
            "x",
            "X",
        );
        match compile_plan(&clause).unwrap_err() {
            CompileError::InvalidCondition { id, .. } => assert_eq!(id, "bad"),
            other => panic!("expected InvalidCondition, got {other:?}"),
        }
    }

    #[test]
    fn compile_bundle_assembles_rules_plans_and_value_sets() {
        let rules = vec![RuleClause::new(
            "at-risk-fall",
            ["?p has_condition ?c", "?c increases_fall_risk true"],
            ["?p at_risk fall"],
        )];
        let plans = vec![PlanClause::new(
            "fall-prevention",
            "Fall prevention",
            "Patient.age >= 65",
            "assess-fall-risk",
            "Assess fall risk",
        )];
        let vs = vec![ValueSet {
            name: "RiskConditions".into(),
            codes: vec![
                Code::new("SNOMED", "osteoporosis"),
                Code::new("SNOMED", "arthritis"),
            ],
        }];
        let bundle = compile_bundle(&rules, &plans, &vs).expect("compiles");
        assert_eq!(bundle.rules.len(), 1);
        assert_eq!(bundle.plans.len(), 1);
        assert_eq!(bundle.value_sets.len(), 1);
        assert_eq!(bundle.rules[0].id, "at-risk-fall");
        assert_eq!(bundle.plans[0].id, "fall-prevention");
        assert_eq!(bundle.value_sets[0].name, "RiskConditions");
    }

    #[test]
    fn compile_bundle_short_circuits_on_first_error() {
        // The second rule is malformed; compile_bundle returns the error.
        let rules = vec![
            RuleClause::new("ok", ["?p parent ?q"], ["?p ancestor ?q"]),
            RuleClause::new("bad", ["only two"], ["?p ancestor ?q"]),
        ];
        let err = compile_bundle(&rules, &[], &[]).unwrap_err();
        assert!(matches!(err, CompileError::MalformedPattern { id, .. } if id == "bad"));
    }
}
