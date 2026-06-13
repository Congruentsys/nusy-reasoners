//! Terms, triple patterns, ground triples, and rules.
//!
//! The data model is deliberately flat — there are no compound function terms,
//! only **variables** (`?x`) and **constants** (IRIs / literals as strings). That
//! is all rule-LHS matching over an RDF/Y-graph triple store needs, and it means
//! unification needs no occurs-check.

/// A term in a triple pattern: either a variable or a ground constant.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Term {
    /// A logic variable, named without the leading `?` (e.g. `Var("x")` is `?x`).
    Var(String),
    /// A ground constant (an IRI, literal, or any concrete value).
    Const(String),
}

impl Term {
    /// Parse a term from a string using the `?`-prefix convention: a leading `?`
    /// marks a variable (`"?x"` → `Var("x")`), anything else is a constant.
    pub fn parse(s: &str) -> Self {
        if let Some(name) = s.strip_prefix('?') {
            Term::Var(name.to_string())
        } else {
            Term::Const(s.to_string())
        }
    }

    /// Construct a variable from its (un-prefixed) name.
    pub fn var(name: impl Into<String>) -> Self {
        Term::Var(name.into())
    }

    /// Construct a constant.
    pub fn con(value: impl Into<String>) -> Self {
        Term::Const(value.into())
    }

    /// Is this a variable?
    pub fn is_var(&self) -> bool {
        matches!(self, Term::Var(_))
    }

    /// The constant value, if this is a `Const`.
    pub fn as_const(&self) -> Option<&str> {
        match self {
            Term::Const(c) => Some(c),
            Term::Var(_) => None,
        }
    }
}

/// A triple pattern — one atom of a rule's LHS or RHS. Any position may be a
/// variable or a constant (e.g. `(?x, "parent", ?y)`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TriplePattern {
    /// Subject position.
    pub subject: Term,
    /// Predicate position.
    pub predicate: Term,
    /// Object position.
    pub object: Term,
}

impl TriplePattern {
    /// Construct a pattern from three terms.
    pub fn new(subject: Term, predicate: Term, object: Term) -> Self {
        Self {
            subject,
            predicate,
            object,
        }
    }

    /// Construct a pattern from three strings via the [`Term::parse`] `?`-convention.
    /// e.g. `TriplePattern::parse("?x", "parent", "?y")`.
    pub fn parse(subject: &str, predicate: &str, object: &str) -> Self {
        Self::new(
            Term::parse(subject),
            Term::parse(predicate),
            Term::parse(object),
        )
    }

    /// The variable names appearing in this pattern, in subject/predicate/object order.
    pub fn variables(&self) -> Vec<&str> {
        [&self.subject, &self.predicate, &self.object]
            .into_iter()
            .filter_map(|t| match t {
                Term::Var(n) => Some(n.as_str()),
                Term::Const(_) => None,
            })
            .collect()
    }
}

/// A ground (variable-free) triple — a concrete fact in the graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Triple {
    /// Subject.
    pub subject: String,
    /// Predicate.
    pub predicate: String,
    /// Object.
    pub object: String,
}

impl Triple {
    /// Construct a ground triple.
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
        }
    }
}

/// A Horn-style rule: a conjunctive LHS (body) entailing a conjunctive RHS (head).
///
/// e.g. grandparent: `[(?x parent ?y), (?y parent ?z)] ⊢ [(?x grandparent ?z)]`.
/// Every variable in the RHS must appear in the LHS (range-restriction); otherwise
/// firing the rule cannot ground it and that RHS atom is skipped (see
/// [`crate::fire_rule`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rule {
    /// The body: a conjunction of patterns that must all match.
    pub lhs: Vec<TriplePattern>,
    /// The head: patterns instantiated and asserted for each LHS solution.
    pub rhs: Vec<TriplePattern>,
}

impl Rule {
    /// Construct a rule from its LHS (body) and RHS (head).
    pub fn new(lhs: Vec<TriplePattern>, rhs: Vec<TriplePattern>) -> Self {
        Self { lhs, rhs }
    }

    /// RHS variables not bound anywhere in the LHS (an unsafe / non-range-restricted
    /// rule). An empty result means the rule is range-restricted and always grounds.
    pub fn unsafe_head_vars(&self) -> Vec<String> {
        let mut lhs_vars: Vec<&str> = self.lhs.iter().flat_map(|p| p.variables()).collect();
        lhs_vars.sort_unstable();
        lhs_vars.dedup();
        let mut out: Vec<String> = self
            .rhs
            .iter()
            .flat_map(|p| p.variables())
            .filter(|v| !lhs_vars.contains(v))
            .map(str::to_string)
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }
}
