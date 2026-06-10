//! Abstract syntax tree and value model for the CQL-analog expression language.
//!
//! The value model mirrors the subset of HL7 CQL relevant to computable Y2 rules:
//! booleans, numbers, strings, terminology [`Code`]s, points/intervals in time, and
//! quantities. [`Value::Null`] is CQL's "unknown" and drives three-valued logic in
//! the evaluator (see [`crate::eval`]).

/// A terminology code: a `(system, code)` pair (e.g. `("SNOMED", "38341003")`).
///
/// Equality is exact on both fields; *subsumption* (is-a) is resolved by the
/// [`crate::FactStore`], not by the code itself, because it depends on the ontology.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Code {
    /// The code system / namespace (e.g. `"SNOMED"`, `"LOINC"`, `"RxNorm"`).
    pub system: String,
    /// The code within that system.
    pub code: String,
}

impl Code {
    /// Construct a code from a system and a code string.
    pub fn new(system: impl Into<String>, code: impl Into<String>) -> Self {
        Self { system: system.into(), code: code.into() }
    }
}

/// A runtime value. `Null` represents an unknown / absent fact (CQL three-valued logic).
#[derive(Debug, Clone)]
pub enum Value {
    /// Boolean truth value.
    Boolean(bool),
    /// 64-bit signed integer.
    Integer(i64),
    /// 64-bit floating-point ("decimal" in CQL terms).
    Decimal(f64),
    /// A string literal.
    Str(String),
    /// A terminology code.
    Code(Code),
    /// A point in time, as a unit-agnostic epoch offset (decoupled from any clock).
    DateTime(i64),
    /// A closed interval `[low, high]` over `DateTime`/`Integer` points.
    Interval(Box<Value>, Box<Value>),
    /// A physical quantity: a magnitude plus a unit string (units are not converted).
    Quantity(f64, String),
    /// An ordered list of values (used by `in` list-membership).
    List(Vec<Value>),
    /// Unknown / absent. Propagates through comparisons and Kleene logic.
    Null,
}

impl Value {
    /// Treat this value as a temporal interval, promoting a point to a degenerate
    /// `[p, p]`. Returns `None` if the value is not point-like or interval-like.
    pub(crate) fn as_interval(&self) -> Option<(i64, i64)> {
        match self {
            Value::DateTime(p) | Value::Integer(p) => Some((*p, *p)),
            Value::Interval(lo, hi) => match (lo.as_point(), hi.as_point()) {
                (Some(l), Some(h)) => Some((l, h)),
                _ => None,
            },
            _ => None,
        }
    }

    /// Treat this value as a temporal point. Returns `None` if not point-like.
    pub(crate) fn as_point(&self) -> Option<i64> {
        match self {
            Value::DateTime(p) | Value::Integer(p) => Some(*p),
            _ => None,
        }
    }
}

/// Comparison operators (`=`, `!=`, `<`, `<=`, `>`, `>=`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompOp {
    /// `=`
    Eq,
    /// `!=`
    Ne,
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `>`
    Gt,
    /// `>=`
    Ge,
}

/// Allen-style temporal relations over points/intervals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalOp {
    /// `lhs` ends strictly before `rhs` begins.
    Before,
    /// `lhs` begins strictly after `rhs` ends.
    After,
    /// `lhs` is contained within `rhs` (`lhs.low >= rhs.low && lhs.high <= rhs.high`).
    During,
    /// `lhs` and `rhs` intervals share at least one point.
    Overlaps,
}

/// An expression node.
#[derive(Debug, Clone)]
pub enum Expr {
    /// A literal value.
    Literal(Value),
    /// A property path rooted at an entity, e.g. `Patient.age` → `("Patient", ["age"])`.
    /// Resolved against the [`crate::FactStore`]; an absent path yields `Null`.
    Property {
        /// The root entity name.
        entity: String,
        /// The dotted path segments after the entity.
        path: Vec<String>,
    },
    /// `exists(expr)` — true iff `expr` resolves to a present (non-null, non-empty) value.
    Exists(Box<Expr>),
    /// `not expr` (Kleene negation).
    Not(Box<Expr>),
    /// `lhs and rhs` (Kleene conjunction).
    And(Box<Expr>, Box<Expr>),
    /// `lhs or rhs` (Kleene disjunction).
    Or(Box<Expr>, Box<Expr>),
    /// A comparison `lhs <op> rhs`.
    Compare {
        /// The comparison operator.
        op: CompOp,
        /// Left operand.
        lhs: Box<Expr>,
        /// Right operand.
        rhs: Box<Expr>,
    },
    /// `value in "ValueSetName"` — terminology value-set membership (resolved by the store).
    InValueSet {
        /// The expression yielding a [`Code`].
        value: Box<Expr>,
        /// The value-set name.
        valueset: String,
    },
    /// `value in (a, b, c)` — list membership.
    InList {
        /// The expression to test.
        value: Box<Expr>,
        /// The list expression (must evaluate to a [`Value::List`]).
        list: Box<Expr>,
    },
    /// `ancestor subsumes descendant` — ontology is-a (resolved by the store).
    Subsumes {
        /// The (proposed) ancestor code expression.
        ancestor: Box<Expr>,
        /// The (proposed) descendant code expression.
        descendant: Box<Expr>,
    },
    /// A temporal relation `lhs <op> rhs`.
    Temporal {
        /// The temporal operator.
        op: TemporalOp,
        /// Left operand (point or interval).
        lhs: Box<Expr>,
        /// Right operand (point or interval).
        rhs: Box<Expr>,
    },
    /// A parenthesised list literal `(a, b, c)`.
    ListLit(Vec<Expr>),
}
