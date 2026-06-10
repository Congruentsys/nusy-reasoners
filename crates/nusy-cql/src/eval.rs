//! Evaluator for the CQL-analog expression language.
//!
//! Evaluation follows CQL **three-valued (Kleene) logic**: [`Value::Null`] models an
//! unknown/absent fact and propagates through comparisons and boolean operators
//! (`null and false == false`, `null and true == null`, etc.). All access to the
//! knowledge graph goes through the [`FactStore`] trait, so the language is fully
//! decoupled from any particular Y-layer/ontology backend.

use crate::ast::{Code, CompOp, Expr, TemporalOp, Value};
use crate::error::EvalError;

/// The seam between the expression language and the knowledge graph.
///
/// A VOY-1/VOY-2 backend implements this over the Y1 entity store, the value-set
/// registry, and the ontology; tests implement it over an in-memory map. None of
/// the language's parsing or logic depends on which backend is behind the trait.
pub trait FactStore {
    /// Resolve a property path rooted at `entity` (e.g. `entity = "Patient"`,
    /// `path = ["age"]`). Returns every matching value; an empty vector means the
    /// fact is absent (evaluates to [`Value::Null`]).
    fn get_property(&self, entity: &str, path: &[String]) -> Vec<Value>;

    /// Is `code` a member of the named value set? `None` means "unknown"
    /// (e.g. the value set is not loaded) and yields [`Value::Null`].
    fn in_value_set(&self, code: &Code, valueset: &str) -> Option<bool>;

    /// Does `ancestor` subsume `descendant` in the ontology (is-a / transitive)?
    /// `None` means the relationship cannot be determined and yields [`Value::Null`].
    fn subsumes(&self, ancestor: &Code, descendant: &Code) -> Option<bool>;
}

/// Evaluate `expr` against `store`, returning a [`Value`] (possibly [`Value::Null`]).
pub fn eval(expr: &Expr, store: &dyn FactStore) -> Result<Value, EvalError> {
    match expr {
        Expr::Literal(v) => Ok(v.clone()),

        Expr::Property { entity, path } => {
            let mut vals = store.get_property(entity, path);
            match vals.len() {
                0 => Ok(Value::Null),
                1 => Ok(vals.pop().expect("len checked == 1")),
                _ => Ok(Value::List(vals)),
            }
        }

        Expr::ListLit(items) => {
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                out.push(eval(it, store)?);
            }
            Ok(Value::List(out))
        }

        Expr::Exists(inner) => {
            let v = eval(inner, store)?;
            Ok(Value::Boolean(is_present(&v)))
        }

        Expr::Not(e) => {
            let t = truth(&eval(e, store)?, "not")?;
            Ok(from_truth(t.map(|b| !b)))
        }

        Expr::And(a, b) => {
            let ta = truth(&eval(a, store)?, "and")?;
            let tb = truth(&eval(b, store)?, "and")?;
            Ok(from_truth(kleene_and(ta, tb)))
        }

        Expr::Or(a, b) => {
            let ta = truth(&eval(a, store)?, "or")?;
            let tb = truth(&eval(b, store)?, "or")?;
            Ok(from_truth(kleene_or(ta, tb)))
        }

        Expr::Compare { op, lhs, rhs } => {
            let l = eval(lhs, store)?;
            let r = eval(rhs, store)?;
            compare(*op, &l, &r)
        }

        Expr::InValueSet { value, valueset } => {
            let v = eval(value, store)?;
            match v {
                Value::Null => Ok(Value::Null),
                Value::Code(c) => Ok(from_truth(store.in_value_set(&c, valueset))),
                other => Err(EvalError::TypeError {
                    op: "in (value set)".to_string(),
                    detail: format!("expected a Code on the left, got {}", type_name(&other)),
                }),
            }
        }

        Expr::InList { value, list } => {
            let v = eval(value, store)?;
            let l = eval(list, store)?;
            in_list(&v, &l)
        }

        Expr::Subsumes { ancestor, descendant } => {
            let a = eval(ancestor, store)?;
            let d = eval(descendant, store)?;
            match (a, d) {
                (Value::Null, _) | (_, Value::Null) => Ok(Value::Null),
                (Value::Code(anc), Value::Code(desc)) => Ok(from_truth(store.subsumes(&anc, &desc))),
                (a, d) => Err(EvalError::TypeError {
                    op: "subsumes".to_string(),
                    detail: format!("expected Code subsumes Code, got {} subsumes {}", type_name(&a), type_name(&d)),
                }),
            }
        }

        Expr::Temporal { op, lhs, rhs } => {
            let l = eval(lhs, store)?;
            let r = eval(rhs, store)?;
            temporal(*op, &l, &r)
        }
    }
}

/// True iff a value is "present" for `exists`: not null, and (for lists) has a non-null member.
fn is_present(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::List(xs) => xs.iter().any(|x| !matches!(x, Value::Null)),
        _ => true,
    }
}

/// Coerce a value into Kleene truth for boolean context.
fn truth(v: &Value, op: &str) -> Result<Option<bool>, EvalError> {
    match v {
        Value::Boolean(b) => Ok(Some(*b)),
        Value::Null => Ok(None),
        other => Err(EvalError::TypeError {
            op: op.to_string(),
            detail: format!("expected a boolean, got {}", type_name(other)),
        }),
    }
}

fn from_truth(t: Option<bool>) -> Value {
    match t {
        Some(b) => Value::Boolean(b),
        None => Value::Null,
    }
}

fn kleene_and(a: Option<bool>, b: Option<bool>) -> Option<bool> {
    match (a, b) {
        (Some(false), _) | (_, Some(false)) => Some(false),
        (Some(true), Some(true)) => Some(true),
        _ => None,
    }
}

fn kleene_or(a: Option<bool>, b: Option<bool>) -> Option<bool> {
    match (a, b) {
        (Some(true), _) | (_, Some(true)) => Some(true),
        (Some(false), Some(false)) => Some(false),
        _ => None,
    }
}

/// Numeric view of a value, unifying Integer/Decimal/DateTime for ordering.
fn as_number(v: &Value) -> Option<f64> {
    match v {
        Value::Integer(n) | Value::DateTime(n) => Some(*n as f64),
        Value::Decimal(d) => Some(*d),
        _ => None,
    }
}

/// Structural equality returning Kleene truth. `None` if either side is null.
fn value_eq(a: &Value, b: &Value) -> Option<bool> {
    if matches!(a, Value::Null) || matches!(b, Value::Null) {
        return None;
    }
    // Numbers compare across Integer/Decimal/DateTime by magnitude.
    if let (Some(x), Some(y)) = (as_number(a), as_number(b)) {
        return Some((x - y).abs() < f64::EPSILON);
    }
    let eq = match (a, b) {
        (Value::Boolean(x), Value::Boolean(y)) => x == y,
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::Code(x), Value::Code(y)) => x == y,
        (Value::Quantity(xv, xu), Value::Quantity(yv, yu)) => xu == yu && (xv - yv).abs() < f64::EPSILON,
        (Value::List(xs), Value::List(ys)) => {
            xs.len() == ys.len() && xs.iter().zip(ys).all(|(x, y)| value_eq(x, y) == Some(true))
        }
        // Different, non-numeric types are never equal.
        _ => false,
    };
    Some(eq)
}

fn compare(op: CompOp, l: &Value, r: &Value) -> Result<Value, EvalError> {
    if matches!(l, Value::Null) || matches!(r, Value::Null) {
        return Ok(Value::Null);
    }
    if matches!(op, CompOp::Eq | CompOp::Ne) {
        let eq = value_eq(l, r).expect("nulls handled above");
        let res = if matches!(op, CompOp::Eq) { eq } else { !eq };
        return Ok(Value::Boolean(res));
    }
    // Ordering operators: numeric or string.
    let ordering = if let (Some(x), Some(y)) = (as_number(l), as_number(r)) {
        x.partial_cmp(&y)
    } else if let (Value::Str(x), Value::Str(y)) = (l, r) {
        Some(x.cmp(y))
    } else {
        return Err(EvalError::TypeError {
            op: format!("{op:?}"),
            detail: format!("cannot order {} and {}", type_name(l), type_name(r)),
        });
    };
    let Some(ord) = ordering else {
        // e.g. NaN
        return Ok(Value::Null);
    };
    use std::cmp::Ordering;
    let res = match op {
        CompOp::Lt => ord == Ordering::Less,
        CompOp::Le => ord != Ordering::Greater,
        CompOp::Gt => ord == Ordering::Greater,
        CompOp::Ge => ord != Ordering::Less,
        CompOp::Eq | CompOp::Ne => unreachable!("handled above"),
    };
    Ok(Value::Boolean(res))
}

fn in_list(v: &Value, list: &Value) -> Result<Value, EvalError> {
    let Value::List(items) = list else {
        return Err(EvalError::TypeError {
            op: "in (list)".to_string(),
            detail: format!("right-hand side must be a list, got {}", type_name(list)),
        });
    };
    if matches!(v, Value::Null) {
        return Ok(Value::Null);
    }
    let mut saw_unknown = false;
    for item in items {
        match value_eq(v, item) {
            Some(true) => return Ok(Value::Boolean(true)),
            None => saw_unknown = true,
            Some(false) => {}
        }
    }
    // Not found: unknown if any element was null (membership undecidable), else false.
    Ok(if saw_unknown { Value::Null } else { Value::Boolean(false) })
}

fn temporal(op: TemporalOp, l: &Value, r: &Value) -> Result<Value, EvalError> {
    if matches!(l, Value::Null) || matches!(r, Value::Null) {
        return Ok(Value::Null);
    }
    let (Some((llo, lhi)), Some((rlo, rhi))) = (l.as_interval(), r.as_interval()) else {
        return Err(EvalError::TypeError {
            op: format!("{op:?}"),
            detail: format!("temporal operators require points/intervals, got {} and {}", type_name(l), type_name(r)),
        });
    };
    let res = match op {
        TemporalOp::Before => lhi < rlo,
        TemporalOp::After => llo > rhi,
        TemporalOp::During => llo >= rlo && lhi <= rhi,
        TemporalOp::Overlaps => llo <= rhi && rlo <= lhi,
    };
    Ok(Value::Boolean(res))
}

fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Boolean(_) => "Boolean",
        Value::Integer(_) => "Integer",
        Value::Decimal(_) => "Decimal",
        Value::Str(_) => "String",
        Value::Code(_) => "Code",
        Value::DateTime(_) => "DateTime",
        Value::Interval(_, _) => "Interval",
        Value::Quantity(_, _) => "Quantity",
        Value::List(_) => "List",
        Value::Null => "Null",
    }
}
