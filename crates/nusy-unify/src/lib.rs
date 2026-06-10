//! # nusy-unify — unification + variable binding for rule-LHS matching
//!
//! `nusy-unify` is the **VOY-V18-1** rule-matching primitive: it unifies triple
//! patterns containing logic variables against ground Y-graph triples, producing
//! variable [`Substitution`]s, and solves a conjunctive rule body (LHS) by joining
//! patterns on their shared variables. This is the layer the forward-chaining
//! engine (EX-4588) sits on: *"for each rule, find every LHS binding, instantiate
//! the RHS, assert the derived triples."* [`fire_rule`] is exactly that, one step.
//!
//! ## Decoupled by design
//!
//! Unification is pure term algebra, so this crate has **no workspace dependencies**.
//! Facts are passed in as a slice of ground [`Triple`]s — in the engine they come
//! from the Y-graph (Arrow/rdf-fusion), in tests from a literal vector. The
//! forward-chaining/closure engine consumes `nusy-unify`, never the reverse.
//!
//! ## Model
//!
//! - [`Term`] — a [`Term::Var`] (`?x`) or a [`Term::Const`] (a concrete value). Flat:
//!   no compound function terms, so [`unify`] needs no occurs-check.
//! - [`TriplePattern`] — one rule atom, e.g. `(?x, parent, ?y)`.
//! - [`Triple`] — a ground fact.
//! - [`Rule`] — a conjunctive body (LHS) entailing a conjunctive head (RHS).
//! - [`Substitution`] — variable bindings; [`Substitution::ground`] instantiates a
//!   head atom to a concrete [`Triple`].
//!
//! ## Example — derive grandparent from parent
//!
//! ```
//! use nusy_unify::{fire_rule, Rule, Triple, TriplePattern};
//!
//! // grandparent(?x,?z) :- parent(?x,?y), parent(?y,?z)
//! let rule = Rule::new(
//!     vec![
//!         TriplePattern::parse("?x", "parent", "?y"),
//!         TriplePattern::parse("?y", "parent", "?z"),
//!     ],
//!     vec![TriplePattern::parse("?x", "grandparent", "?z")],
//! );
//! let facts = vec![
//!     Triple::new("alice", "parent", "bob"),
//!     Triple::new("bob", "parent", "carol"),
//! ];
//! let derived = fire_rule(&rule, &facts);
//! assert_eq!(derived, vec![Triple::new("alice", "grandparent", "carol")]);
//! ```

mod subst;
mod term;
mod unify;

pub use subst::Substitution;
pub use term::{Rule, Term, Triple, TriplePattern};
pub use unify::{fire_rule, match_conjunction, match_pattern, match_triple, unify};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn term_parse_convention() {
        assert_eq!(Term::parse("?x"), Term::Var("x".into()));
        assert_eq!(Term::parse("parent"), Term::Const("parent".into()));
        assert!(Term::parse("?y").is_var());
        assert_eq!(Term::parse("foo").as_const(), Some("foo"));
    }

    #[test]
    fn unify_constants() {
        let s = Substitution::new();
        assert!(unify(&Term::con("a"), &Term::con("a"), &s).is_some());
        assert!(unify(&Term::con("a"), &Term::con("b"), &s).is_none());
    }

    #[test]
    fn unify_binds_variable_either_side() {
        let s = Substitution::new();
        let r = unify(&Term::var("x"), &Term::con("a"), &s).unwrap();
        assert_eq!(r.get_const("x"), Some("a".to_string()));

        let r2 = unify(&Term::con("a"), &Term::var("y"), &s).unwrap();
        assert_eq!(r2.get_const("y"), Some("a".to_string()));
    }

    #[test]
    fn unify_variable_consistency_conflict() {
        // ?x already bound to "a"; unifying ?x with "b" must fail.
        let s = unify(&Term::var("x"), &Term::con("a"), &Substitution::new()).unwrap();
        assert!(unify(&Term::var("x"), &Term::con("b"), &s).is_none());
        // …but re-unifying with the same constant is fine.
        assert!(unify(&Term::var("x"), &Term::con("a"), &s).is_some());
    }

    #[test]
    fn unify_var_to_var_then_ground() {
        let s = Substitution::new();
        let s = unify(&Term::var("x"), &Term::var("y"), &s).unwrap();
        let s = unify(&Term::var("y"), &Term::con("a"), &s).unwrap();
        // ?x → ?y → "a" must resolve transitively.
        assert_eq!(s.get_const("x"), Some("a".to_string()));
    }

    #[test]
    fn match_triple_binds_pattern_vars() {
        let pat = TriplePattern::parse("?x", "parent", "?y");
        let fact = Triple::new("alice", "parent", "bob");
        let s = match_triple(&pat, &fact, &Substitution::new()).unwrap();
        assert_eq!(s.get_const("x"), Some("alice".to_string()));
        assert_eq!(s.get_const("y"), Some("bob".to_string()));

        // predicate mismatch → no match
        let other = Triple::new("alice", "sibling", "bob");
        assert!(match_triple(&pat, &other, &Substitution::new()).is_none());
    }

    #[test]
    fn match_pattern_returns_all_facts() {
        let pat = TriplePattern::parse("?x", "parent", "?y");
        let facts = vec![
            Triple::new("alice", "parent", "bob"),
            Triple::new("bob", "parent", "carol"),
            Triple::new("alice", "sibling", "dave"), // filtered out
        ];
        let sols = match_pattern(&pat, &facts, &Substitution::new());
        assert_eq!(sols.len(), 2);
    }

    #[test]
    fn conjunction_joins_on_shared_variable() {
        // parent(?x,?y) ∧ parent(?y,?z) — ?y is the join key.
        let patterns = vec![
            TriplePattern::parse("?x", "parent", "?y"),
            TriplePattern::parse("?y", "parent", "?z"),
        ];
        let facts = vec![
            Triple::new("alice", "parent", "bob"),
            Triple::new("bob", "parent", "carol"),
            Triple::new("bob", "parent", "dave"),
        ];
        let sols = match_conjunction(&patterns, &facts);
        // alice→bob→carol and alice→bob→dave
        assert_eq!(sols.len(), 2);
        for s in &sols {
            assert_eq!(s.get_const("x"), Some("alice".to_string()));
            assert_eq!(s.get_const("y"), Some("bob".to_string()));
            assert!(matches!(s.get_const("z").as_deref(), Some("carol") | Some("dave")));
        }
    }

    #[test]
    fn conjunction_empty_is_one_empty_solution() {
        let sols = match_conjunction(&[], &[Triple::new("a", "b", "c")]);
        assert_eq!(sols.len(), 1);
        assert!(sols[0].is_empty());
    }

    #[test]
    fn conjunction_unsatisfiable_is_empty() {
        let patterns = vec![
            TriplePattern::parse("?x", "parent", "?y"),
            TriplePattern::parse("?y", "parent", "?z"),
        ];
        // No chain: bob is never a subject of parent.
        let facts = vec![Triple::new("alice", "parent", "bob")];
        assert!(match_conjunction(&patterns, &facts).is_empty());
    }

    #[test]
    fn fire_rule_derives_grandparent() {
        let rule = Rule::new(
            vec![
                TriplePattern::parse("?x", "parent", "?y"),
                TriplePattern::parse("?y", "parent", "?z"),
            ],
            vec![TriplePattern::parse("?x", "grandparent", "?z")],
        );
        let facts = vec![
            Triple::new("alice", "parent", "bob"),
            Triple::new("bob", "parent", "carol"),
        ];
        let derived = fire_rule(&rule, &facts);
        assert_eq!(derived, vec![Triple::new("alice", "grandparent", "carol")]);
    }

    #[test]
    fn fire_rule_excludes_existing_and_dedups() {
        // symmetric sibling rule; the derived fact already exists → no duplicate.
        let rule = Rule::new(
            vec![TriplePattern::parse("?x", "sibling", "?y")],
            vec![TriplePattern::parse("?y", "sibling", "?x")],
        );
        let facts = vec![
            Triple::new("a", "sibling", "b"),
            Triple::new("b", "sibling", "a"), // reverse already present
        ];
        // Both directions already exist → nothing new derived.
        assert!(fire_rule(&rule, &facts).is_empty());
    }

    #[test]
    fn unsafe_head_var_is_skipped_not_panicked() {
        // RHS introduces ?w, never bound by the LHS → not range-restricted.
        let rule = Rule::new(
            vec![TriplePattern::parse("?x", "a", "?y")],
            vec![TriplePattern::parse("?x", "b", "?w")],
        );
        assert_eq!(rule.unsafe_head_vars(), vec!["w".to_string()]);
        let facts = vec![Triple::new("s", "a", "o")];
        // ?w cannot ground → that head atom is skipped, no panic, nothing derived.
        assert!(fire_rule(&rule, &facts).is_empty());
    }
}
