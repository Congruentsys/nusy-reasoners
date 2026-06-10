//! Variable bindings (substitutions) and their application to terms/patterns.

use crate::term::{Term, Triple, TriplePattern};
use std::collections::HashMap;

/// A substitution: a set of variable → term bindings produced by unification.
///
/// Bindings may chain (a variable bound to another variable that is itself bound);
/// [`walk`](Substitution::walk) follows the chain to the representative term.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Substitution {
    map: HashMap<String, Term>,
}

impl Substitution {
    /// An empty substitution.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of direct bindings.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Is the substitution empty (no bindings)?
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Bind variable `name` to `term`. Caller ensures consistency (unify does).
    pub(crate) fn bind(&mut self, name: String, term: Term) {
        self.map.insert(name, term);
    }

    /// Follow binding chains: resolve `term` to its current representative.
    /// A variable bound to another (possibly bound) variable is followed transitively;
    /// an unbound variable resolves to itself.
    pub fn walk(&self, term: &Term) -> Term {
        let mut cur = term.clone();
        // Flat terms + acyclic binding (unify never binds a var to itself) → terminates.
        while let Term::Var(name) = &cur {
            match self.map.get(name) {
                Some(next) if next != &cur => cur = next.clone(),
                _ => break,
            }
        }
        cur
    }

    /// Resolve `var_name` to a ground constant, if it is (transitively) bound to one.
    pub fn get_const(&self, var_name: &str) -> Option<String> {
        match self.walk(&Term::Var(var_name.to_string())) {
            Term::Const(c) => Some(c),
            Term::Var(_) => None,
        }
    }

    /// Apply this substitution to a single term (one `walk` step to its representative).
    pub fn apply_term(&self, term: &Term) -> Term {
        self.walk(term)
    }

    /// Apply to a pattern, resolving each position.
    pub fn apply_pattern(&self, pat: &TriplePattern) -> TriplePattern {
        TriplePattern::new(
            self.apply_term(&pat.subject),
            self.apply_term(&pat.predicate),
            self.apply_term(&pat.object),
        )
    }

    /// Instantiate a pattern to a **ground** triple, returning `None` if any
    /// position is still a variable after substitution (the pattern is not fully bound).
    pub fn ground(&self, pat: &TriplePattern) -> Option<Triple> {
        let s = self.apply_term(&pat.subject);
        let p = self.apply_term(&pat.predicate);
        let o = self.apply_term(&pat.object);
        match (s, p, o) {
            (Term::Const(s), Term::Const(p), Term::Const(o)) => Some(Triple::new(s, p, o)),
            _ => None,
        }
    }
}
