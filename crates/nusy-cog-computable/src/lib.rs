//! # nusy-cog-computable — transfer computable Y2 artifacts in a COG
//!
//! **VOY-V18-4 / EX-4607.** A COG ([`nusy_cog`]) transfers a being's definitional Y0–Y2
//! knowledge as **triples**. But the V18 computable Y2 — forward-chaining [`Rule`]s, FHIR-CPG
//! [`PlanDefinition`] decision graphs, and terminology value-sets — are *structured*, not flat
//! triples. For a schooled being to carry its **executable** knowledge between levels and to other
//! beings (not just the facts it derived), those artifacts must transfer **faithfully**.
//!
//! This crate **reifies** the computable artifacts into Y2 triples — graph-native, so they ride
//! `nusy-cog`'s existing triple export/import unchanged — and **de-reifies** them back, with a
//! round-trip guarantee: [`dereify`]`(`[`reify`]`(x)) == x`. Reification is fully graph-native
//! (every value is a clean triple object; child order is carried by an explicit `y2:index`), so
//! there is no string-blob escaping and the artifacts are discoverable as typed Y2 entities.
//!
//! ## What transfers
//!
//! [`ComputableY2`] bundles the three artifact kinds the expedition requires — **rules**,
//! **decision-graphs**, **ontology/value-sets**. [`reify`] turns a bundle into `Vec<Triple>` to
//! merge into a COG's Y2 layer; the receiving being [`dereify`]s them back into executable form.
//!
//! ## Example
//!
//! ```
//! use nusy_cog_computable::{ComputableY2, NamedRule, ValueSet, reify, dereify};
//! use nusy_unify::{Rule, TriplePattern};
//! use nusy_cql::Code;
//!
//! let bundle = ComputableY2 {
//!     rules: vec![NamedRule::new("grandparent", Rule::new(
//!         vec![TriplePattern::parse("?x", "parent", "?y"), TriplePattern::parse("?y", "parent", "?z")],
//!         vec![TriplePattern::parse("?x", "grandparent", "?z")],
//!     ))],
//!     plans: vec![],
//!     value_sets: vec![ValueSet { name: "Hypertension".into(), codes: vec![Code::new("SNOMED", "38341003")] }],
//! };
//! let triples = reify(&bundle);                 // → Y2 triples, transfer via nusy-cog
//! let restored = dereify(&triples).unwrap();    // receiving being reconstructs
//! assert_eq!(restored, bundle);                 // faithful
//! ```

use std::collections::HashMap;

use nusy_cql::Code;
use nusy_plandef::{Action, PlanAction, PlanDefinition};
use nusy_unify::{Rule, Term, Triple, TriplePattern};

mod error;
pub use error::DereifyError;

// ── Y2 reification vocabulary ───────────────────────────────────────────────
const TYPE: &str = "rdf:type";
const T_RULE: &str = "y2:Rule";
const T_PLAN: &str = "y2:PlanDefinition";
const T_VS: &str = "y2:ValueSet";
const P_ID: &str = "y2:id";
const P_TITLE: &str = "y2:title";
const P_LHS: &str = "y2:lhsAtom";
const P_RHS: &str = "y2:rhsAtom";
const P_SUBJ: &str = "y2:subject";
const P_PRED: &str = "y2:predicate";
const P_OBJ: &str = "y2:object";
const P_INDEX: &str = "y2:index";
const P_COND: &str = "y2:condition";
const P_ACTION: &str = "y2:action";
const P_SUB: &str = "y2:subAction";
const P_REC_ID: &str = "y2:recommendId";
const P_REC_TITLE: &str = "y2:recommendTitle";
const P_CODE_SYS: &str = "y2:codeSystem";
const P_CODE_VAL: &str = "y2:codeValue";
const P_MEMBER: &str = "y2:member";

/// A [`Rule`] with the stable id it transfers under (the forward-chaining engine's rule id).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamedRule {
    /// Stable rule identifier.
    pub id: String,
    /// The Horn rule.
    pub rule: Rule,
}

impl NamedRule {
    /// Construct a named rule.
    pub fn new(id: impl Into<String>, rule: Rule) -> Self {
        Self {
            id: id.into(),
            rule,
        }
    }
}

/// A terminology value-set: a named set of codes (the ontology artifact COGs must carry).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueSet {
    /// Value-set name (as referenced by CQL `in "<name>"`).
    pub name: String,
    /// Member codes.
    pub codes: Vec<Code>,
}

/// The computable Y2 payload a COG carries: rules, decision-graphs, and value-sets.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ComputableY2 {
    /// Forward-chaining rules (with ids).
    pub rules: Vec<NamedRule>,
    /// PlanDefinition decision graphs.
    pub plans: Vec<PlanDefinition>,
    /// Terminology value-sets / ontology.
    pub value_sets: Vec<ValueSet>,
}

// ── Term ↔ string (the `?`-prefix convention round-trips via Term::parse) ────
fn term_str(t: &Term) -> String {
    match t {
        Term::Var(n) => format!("?{n}"),
        Term::Const(c) => c.clone(),
    }
}

// ── Reify ───────────────────────────────────────────────────────────────────

/// Reify a computable-Y2 bundle into transferable Y2 triples.
pub fn reify(bundle: &ComputableY2) -> Vec<Triple> {
    let mut out = Vec::new();
    for nr in &bundle.rules {
        reify_rule(nr, &mut out);
    }
    for plan in &bundle.plans {
        reify_plan(plan, &mut out);
    }
    for vs in &bundle.value_sets {
        reify_value_set(vs, &mut out);
    }
    out
}

fn tri(s: &str, p: &str, o: impl Into<String>) -> Triple {
    Triple::new(s, p, o.into())
}

fn reify_atoms(rule_node: &str, pred: &str, atoms: &[TriplePattern], out: &mut Vec<Triple>) {
    for (i, atom) in atoms.iter().enumerate() {
        let an = format!("{rule_node}:{}:{i}", pred.replace("y2:", ""));
        out.push(tri(rule_node, pred, an.clone()));
        out.push(tri(&an, P_INDEX, i.to_string()));
        out.push(tri(&an, P_SUBJ, term_str(&atom.subject)));
        out.push(tri(&an, P_PRED, term_str(&atom.predicate)));
        out.push(tri(&an, P_OBJ, term_str(&atom.object)));
    }
}

fn reify_rule(nr: &NamedRule, out: &mut Vec<Triple>) {
    let node = format!("cog:rule:{}", nr.id);
    out.push(tri(&node, TYPE, T_RULE));
    out.push(tri(&node, P_ID, nr.id.clone()));
    reify_atoms(&node, P_LHS, &nr.rule.lhs, out);
    reify_atoms(&node, P_RHS, &nr.rule.rhs, out);
}

fn reify_action_node(node: &str, pa: &PlanAction, out: &mut Vec<Triple>) {
    if let Some(cond) = &pa.condition {
        out.push(tri(node, P_COND, cond.clone()));
    }
    if let Some(action) = &pa.action {
        out.push(tri(node, P_REC_ID, action.id.clone()));
        out.push(tri(node, P_REC_TITLE, action.title.clone()));
        if let Some(code) = &action.code {
            out.push(tri(node, P_CODE_SYS, code.system.clone()));
            out.push(tri(node, P_CODE_VAL, code.code.clone()));
        }
    }
    for (i, sub) in pa.sub_actions.iter().enumerate() {
        let sn = format!("{node}:s{i}");
        out.push(tri(node, P_SUB, sn.clone()));
        out.push(tri(&sn, P_INDEX, i.to_string()));
        reify_action_node(&sn, sub, out);
    }
}

fn reify_plan(plan: &PlanDefinition, out: &mut Vec<Triple>) {
    let node = format!("cog:plan:{}", plan.id);
    out.push(tri(&node, TYPE, T_PLAN));
    out.push(tri(&node, P_ID, plan.id.clone()));
    out.push(tri(&node, P_TITLE, plan.title.clone()));
    for (i, action) in plan.actions.iter().enumerate() {
        let an = format!("{node}:a{i}");
        out.push(tri(&node, P_ACTION, an.clone()));
        out.push(tri(&an, P_INDEX, i.to_string()));
        reify_action_node(&an, action, out);
    }
}

fn reify_value_set(vs: &ValueSet, out: &mut Vec<Triple>) {
    let node = format!("cog:vs:{}", vs.name);
    out.push(tri(&node, TYPE, T_VS));
    out.push(tri(&node, P_ID, vs.name.clone()));
    for (i, code) in vs.codes.iter().enumerate() {
        let cn = format!("{node}:c{i}");
        out.push(tri(&node, P_MEMBER, cn.clone()));
        out.push(tri(&cn, P_INDEX, i.to_string()));
        out.push(tri(&cn, P_CODE_SYS, code.system.clone()));
        out.push(tri(&cn, P_CODE_VAL, code.code.clone()));
    }
}

// ── De-reify ──────────────────────────────────────────────────────────────

/// An index of triples by subject for reconstruction: subject → predicate → objects.
struct Index<'a> {
    by_subj: HashMap<&'a str, HashMap<&'a str, Vec<&'a str>>>,
}

impl<'a> Index<'a> {
    fn new(triples: &'a [Triple]) -> Self {
        let mut by_subj: HashMap<&str, HashMap<&str, Vec<&str>>> = HashMap::new();
        for t in triples {
            by_subj
                .entry(&t.subject)
                .or_default()
                .entry(&t.predicate)
                .or_default()
                .push(&t.object);
        }
        Self { by_subj }
    }

    fn one(&self, subj: &str, pred: &str) -> Option<&'a str> {
        self.by_subj.get(subj)?.get(pred)?.first().copied()
    }

    fn many(&self, subj: &str, pred: &str) -> Vec<&'a str> {
        self.by_subj
            .get(subj)
            .and_then(|m| m.get(pred))
            .cloned()
            .unwrap_or_default()
    }

    /// Subjects carrying `(subj, rdf:type, ty)`.
    fn typed(&self, ty: &str) -> Vec<&'a str> {
        let mut out: Vec<&'a str> = self
            .by_subj
            .iter()
            .filter(|(_, m)| m.get(TYPE).is_some_and(|v| v.contains(&ty)))
            .map(|(s, _)| *s)
            .collect();
        out.sort_unstable();
        out
    }

    /// Child nodes linked from `subj` via `pred`, ordered by their `y2:index`.
    fn ordered_children(&self, subj: &str, pred: &str) -> Result<Vec<&'a str>, DereifyError> {
        let mut kids: Vec<(usize, &str)> = Vec::new();
        for child in self.many(subj, pred) {
            let idx = self
                .one(child, P_INDEX)
                .ok_or_else(|| DereifyError::Malformed(format!("{child} missing {P_INDEX}")))?
                .parse::<usize>()
                .map_err(|_| DereifyError::Malformed(format!("{child} bad {P_INDEX}")))?;
            kids.push((idx, child));
        }
        kids.sort_by_key(|(i, _)| *i);
        Ok(kids.into_iter().map(|(_, c)| c).collect())
    }
}

/// De-reify Y2 triples back into a [`ComputableY2`] bundle. Inverse of [`reify`].
///
/// Triples unrelated to the reification vocabulary are ignored, so this is safe to run over a
/// whole COG's Y2 layer. Artifacts are returned in id-sorted order (deterministic).
pub fn dereify(triples: &[Triple]) -> Result<ComputableY2, DereifyError> {
    let ix = Index::new(triples);
    let mut bundle = ComputableY2::default();

    for node in ix.typed(T_RULE) {
        let id = ix
            .one(node, P_ID)
            .ok_or_else(|| DereifyError::Malformed(format!("{node} missing {P_ID}")))?;
        let lhs = read_atoms(&ix, node, P_LHS)?;
        let rhs = read_atoms(&ix, node, P_RHS)?;
        bundle.rules.push(NamedRule::new(id, Rule::new(lhs, rhs)));
    }

    for node in ix.typed(T_PLAN) {
        let id = ix
            .one(node, P_ID)
            .ok_or_else(|| DereifyError::Malformed(format!("{node} missing {P_ID}")))?;
        let title = ix.one(node, P_TITLE).unwrap_or("");
        let mut plan = PlanDefinition::new(id, title);
        for an in ix.ordered_children(node, P_ACTION)? {
            plan = plan.with_action(read_action(&ix, an)?);
        }
        bundle.plans.push(plan);
    }

    for node in ix.typed(T_VS) {
        let name = ix
            .one(node, P_ID)
            .ok_or_else(|| DereifyError::Malformed(format!("{node} missing {P_ID}")))?;
        let mut codes = Vec::new();
        for cn in ix.ordered_children(node, P_MEMBER)? {
            let sys = ix
                .one(cn, P_CODE_SYS)
                .ok_or_else(|| DereifyError::Malformed(format!("{cn} missing {P_CODE_SYS}")))?;
            let val = ix
                .one(cn, P_CODE_VAL)
                .ok_or_else(|| DereifyError::Malformed(format!("{cn} missing {P_CODE_VAL}")))?;
            codes.push(Code::new(sys, val));
        }
        bundle.value_sets.push(ValueSet {
            name: name.to_string(),
            codes,
        });
    }

    Ok(bundle)
}

fn read_atoms(ix: &Index, node: &str, pred: &str) -> Result<Vec<TriplePattern>, DereifyError> {
    let mut atoms = Vec::new();
    for an in ix.ordered_children(node, pred)? {
        let s = ix
            .one(an, P_SUBJ)
            .ok_or_else(|| DereifyError::Malformed(format!("{an} missing {P_SUBJ}")))?;
        let p = ix
            .one(an, P_PRED)
            .ok_or_else(|| DereifyError::Malformed(format!("{an} missing {P_PRED}")))?;
        let o = ix
            .one(an, P_OBJ)
            .ok_or_else(|| DereifyError::Malformed(format!("{an} missing {P_OBJ}")))?;
        atoms.push(TriplePattern::parse(s, p, o));
    }
    Ok(atoms)
}

fn read_action(ix: &Index, node: &str) -> Result<PlanAction, DereifyError> {
    let mut pa = match ix.one(node, P_COND) {
        Some(cond) => PlanAction::when(cond),
        None => PlanAction::always(),
    };
    if let Some(rec_id) = ix.one(node, P_REC_ID) {
        let title = ix.one(node, P_REC_TITLE).unwrap_or("");
        let mut action = Action::new(rec_id, title);
        if let (Some(sys), Some(val)) = (ix.one(node, P_CODE_SYS), ix.one(node, P_CODE_VAL)) {
            action = action.with_code(Code::new(sys, val));
        }
        pa = pa.recommend(action);
    }
    for sn in ix.ordered_children(node, P_SUB)? {
        pa = pa.with_sub(read_action(ix, sn)?);
    }
    Ok(pa)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nusy_plandef::{Action, PlanAction, PlanDefinition};

    /// Normalize top-level collection order: rule/plan/value-set collections are **sets** (a
    /// being's artifacts are unordered) and `dereify` returns them id-sorted. Internal order
    /// (atoms, sub-actions, codes) is preserved by reification and is NOT normalized here.
    fn norm(mut b: ComputableY2) -> ComputableY2 {
        b.rules.sort_by(|a, c| a.id.cmp(&c.id));
        b.plans.sort_by(|a, c| a.id.cmp(&c.id));
        b.value_sets.sort_by(|a, c| a.name.cmp(&c.name));
        b
    }

    /// Round-trip a bundle, comparing as sets (top-level order normalized).
    fn rt(bundle: &ComputableY2) -> ComputableY2 {
        norm(dereify(&reify(bundle)).expect("round-trip"))
    }

    #[test]
    fn rule_round_trips_with_variables() {
        let b = ComputableY2 {
            rules: vec![NamedRule::new(
                "grandparent",
                Rule::new(
                    vec![
                        TriplePattern::parse("?x", "parent", "?y"),
                        TriplePattern::parse("?y", "parent", "?z"),
                    ],
                    vec![TriplePattern::parse("?x", "grandparent", "?z")],
                ),
            )],
            ..Default::default()
        };
        assert_eq!(rt(&b), b);
    }

    #[test]
    fn multiple_rules_preserve_atom_order() {
        // Atom order in lhs is load-bearing (the join sequence); must survive transfer.
        let b = ComputableY2 {
            rules: vec![
                NamedRule::new(
                    "anc-rec",
                    Rule::new(
                        vec![
                            TriplePattern::parse("?x", "parent", "?y"),
                            TriplePattern::parse("?y", "ancestor", "?z"),
                        ],
                        vec![TriplePattern::parse("?x", "ancestor", "?z")],
                    ),
                ),
                NamedRule::new(
                    "sibling-sym",
                    Rule::new(
                        vec![TriplePattern::parse("?a", "sibling", "?b")],
                        vec![TriplePattern::parse("?b", "sibling", "?a")],
                    ),
                ),
            ],
            ..Default::default()
        };
        let out = rt(&b);
        assert_eq!(out, b);
        // explicit: first rule's lhs order is parent-then-ancestor.
        let anc = out.rules.iter().find(|r| r.id == "anc-rec").unwrap();
        assert_eq!(anc.rule.lhs[0].predicate, Term::con("parent"));
        assert_eq!(anc.rule.lhs[1].predicate, Term::con("ancestor"));
    }

    #[test]
    fn plandefinition_with_nested_actions_and_codes_round_trips() {
        let plan = PlanDefinition::new("jnc8", "Hypertension").with_action(
            PlanAction::when("Condition.code in \"HypertensionVS\"")
                .with_sub(
                    PlanAction::when("Patient.age >= 60").recommend(
                        Action::new("bp-150-90", "Treat to <150/90")
                            .with_code(Code::new("SNOMED", "1234")),
                    ),
                )
                .with_sub(
                    PlanAction::when("Patient.age < 60")
                        .recommend(Action::new("bp-140-90", "Treat to <140/90")),
                ),
        );
        let b = ComputableY2 {
            plans: vec![plan],
            ..Default::default()
        };
        assert_eq!(rt(&b), b);
    }

    #[test]
    fn value_sets_round_trip() {
        let b = ComputableY2 {
            value_sets: vec![
                ValueSet {
                    name: "HypertensionVS".into(),
                    codes: vec![
                        Code::new("SNOMED", "38341003"),
                        Code::new("SNOMED", "59621000"),
                    ],
                },
                ValueSet {
                    name: "Empty".into(),
                    codes: vec![],
                },
            ],
            ..Default::default()
        };
        // top-level value-sets are a set → compare normalized (dereify returns id-sorted).
        assert_eq!(rt(&b), norm(b));
    }

    #[test]
    fn full_bundle_round_trips_and_is_order_independent() {
        // A complete schooled-being payload: rules + a decision graph + value-sets.
        let b = ComputableY2 {
            rules: vec![NamedRule::new(
                "grandparent",
                Rule::new(
                    vec![
                        TriplePattern::parse("?x", "parent", "?y"),
                        TriplePattern::parse("?y", "parent", "?z"),
                    ],
                    vec![TriplePattern::parse("?x", "grandparent", "?z")],
                ),
            )],
            plans: vec![PlanDefinition::new("p1", "Plan 1").with_action(
                PlanAction::when("Patient.age >= 65").recommend(Action::new("a1", "Act 1")),
            )],
            value_sets: vec![ValueSet {
                name: "VS".into(),
                codes: vec![Code::new("LOINC", "x")],
            }],
        };
        let mut triples = reify(&b);
        assert_eq!(dereify(&triples).unwrap(), b);
        // Triples are a set — shuffling them must not change the result (transfer reorders).
        triples.reverse();
        assert_eq!(dereify(&triples).unwrap(), b);
    }

    #[test]
    fn ignores_unrelated_cog_triples() {
        // dereify runs over a whole COG's Y2 layer — plain domain triples are ignored.
        let mut triples = reify(&ComputableY2 {
            value_sets: vec![ValueSet {
                name: "VS".into(),
                codes: vec![Code::new("S", "1")],
            }],
            ..Default::default()
        });
        triples.push(Triple::new("alice", "parent", "bob")); // ordinary Y1 fact
        triples.push(Triple::new("bob", "age", "40"));
        let out = dereify(&triples).unwrap();
        assert_eq!(out.value_sets.len(), 1);
        assert!(out.rules.is_empty() && out.plans.is_empty());
    }

    #[test]
    fn empty_bundle_reifies_to_nothing() {
        assert!(reify(&ComputableY2::default()).is_empty());
        assert_eq!(dereify(&[]).unwrap(), ComputableY2::default());
    }

    #[test]
    fn malformed_missing_index_errors() {
        // A rule node whose atom lacks y2:index cannot be ordered → explicit error, no panic.
        let triples = vec![
            Triple::new("cog:rule:r", TYPE, T_RULE),
            Triple::new("cog:rule:r", P_ID, "r"),
            Triple::new("cog:rule:r", P_LHS, "atomX"),
            Triple::new("atomX", P_SUBJ, "?x"),
            Triple::new("atomX", P_PRED, "p"),
            Triple::new("atomX", P_OBJ, "?y"),
            // no (atomX, y2:index, _)
        ];
        assert!(matches!(dereify(&triples), Err(DereifyError::Malformed(_))));
    }
}
