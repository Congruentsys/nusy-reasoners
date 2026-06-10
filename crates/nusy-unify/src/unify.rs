//! The unification algorithm and rule-LHS matching built on top of it.

use crate::subst::Substitution;
use crate::term::{Rule, Term, Triple, TriplePattern};

/// Unify two terms under an existing substitution, returning the extended
/// substitution on success or `None` on conflict.
///
/// Robinson unification specialised to flat terms (variables + constants):
/// - two equal constants unify (no binding);
/// - distinct constants conflict;
/// - a variable unifies with anything, binding it (a var-to-itself is a no-op).
///
/// No occurs-check is needed because terms are not compound.
pub fn unify(a: &Term, b: &Term, subst: &Substitution) -> Option<Substitution> {
    let a = subst.walk(a);
    let b = subst.walk(b);
    match (&a, &b) {
        (Term::Const(x), Term::Const(y)) => {
            if x == y {
                Some(subst.clone())
            } else {
                None
            }
        }
        (Term::Var(x), Term::Var(y)) if x == y => Some(subst.clone()),
        (Term::Var(x), other) | (other, Term::Var(x)) => {
            let mut next = subst.clone();
            next.bind(x.clone(), other.clone());
            Some(next)
        }
    }
}

/// Match a single pattern against one ground triple under `base`, returning the
/// extended substitution if subject, predicate, and object all unify.
pub fn match_triple(
    pat: &TriplePattern,
    fact: &Triple,
    base: &Substitution,
) -> Option<Substitution> {
    let s = unify(&pat.subject, &Term::Const(fact.subject.clone()), base)?;
    let s = unify(&pat.predicate, &Term::Const(fact.predicate.clone()), &s)?;
    unify(&pat.object, &Term::Const(fact.object.clone()), &s)
}

/// All substitutions that make `pat` match some fact in `facts`, each extending `base`.
pub fn match_pattern(
    pat: &TriplePattern,
    facts: &[Triple],
    base: &Substitution,
) -> Vec<Substitution> {
    facts
        .iter()
        .filter_map(|f| match_triple(pat, f, base))
        .collect()
}

/// Solve a conjunction of patterns (a rule LHS) against `facts`: return every
/// substitution that satisfies **all** patterns simultaneously, with variables
/// shared across patterns bound consistently (the relational join).
///
/// An empty pattern list yields a single empty solution (the conjunction is
/// vacuously true), which makes [`fire_rule`] of a fact-less rule well-defined.
pub fn match_conjunction(patterns: &[TriplePattern], facts: &[Triple]) -> Vec<Substitution> {
    let mut solutions = vec![Substitution::new()];
    for pat in patterns {
        let mut next = Vec::new();
        for sol in &solutions {
            next.extend(match_pattern(pat, facts, sol));
        }
        solutions = next;
        if solutions.is_empty() {
            break; // no extension possible → the whole conjunction fails
        }
    }
    solutions
}

/// Fire a rule once over `facts`: solve the LHS, then instantiate the RHS for each
/// solution. Returns the **newly derivable** ground triples, de-duplicated and
/// excluding any already present in `facts`.
///
/// Non-range-restricted RHS atoms (a head variable unbound by the body — see
/// [`Rule::unsafe_head_vars`](crate::Rule::unsafe_head_vars)) cannot be grounded
/// and are skipped rather than panicking. This is one derivation step; the
/// forward-chaining engine (EX-4588) iterates it to a fixpoint.
pub fn fire_rule(rule: &Rule, facts: &[Triple]) -> Vec<Triple> {
    let mut derived: Vec<Triple> = Vec::new();
    for sol in match_conjunction(&rule.lhs, facts) {
        for head in &rule.rhs {
            if let Some(t) = sol.ground(head)
                && !facts.contains(&t)
                && !derived.contains(&t)
            {
                derived.push(t);
            }
        }
    }
    derived
}
