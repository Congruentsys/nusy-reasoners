//! Arrow-native conjunction matching (EX-4669, VY-4667 phase 2).
//!
//! [`match_conjunction_arrow`] computes the same relational join as
//! [`nusy_unify::match_conjunction`] — every substitution satisfying **all** patterns
//! with shared variables bound consistently — but over a [`TripleBatch`](crate::batch::TripleBatch)
//! instead of `&[Triple]`, working column-wise on the dictionary-encoded term columns:
//!
//! 1. **Per-pattern selection.** Within each round's [`RecordBatch`](arrow::record_batch::RecordBatch),
//!    a constant position is resolved to its dictionary key **once per round**; rows are
//!    then filtered by comparing `u32` keys over the keys array (no string compares on
//!    the scan path). A constant absent from a round's dictionary skips that round wholesale.
//! 2. **Hash-join across patterns.** Accumulated solutions join with each pattern's
//!    matches on their shared variables via a hash map keyed by the shared-variable
//!    value tuple — the classic build/probe join, one pattern at a time.
//! 3. **Adapter to the existing solution shape.** Each joined row is converted to a
//!    [`Substitution`] through the public [`nusy_unify::unify`] API, so callers iterate
//!    and `ground()` exactly as with the Vec path. (True zero-copy grounding lands with
//!    the Arrow fixpoint, EX-4670.)
//!
//! Semantics match the Vec implementation as a **multiset**: duplicate facts produce
//! duplicate solutions on both paths. Enumeration order is not part of the contract
//! (the differential tests compare canonicalized multisets).
//!
//! ## GPU mapping (documentation only — no GPU code here)
//!
//! The join layout is deliberately cuDF/RAPIDS-shaped for a future `nusy-gpu-accel`
//! path: step 1 is a per-column predicate over a dictionary-encoded column (cuDF
//! `binary_operation` on key arrays after a host-side dictionary lookup), and step 2 is
//! a multi-column inner join on the shared-variable columns (cuDF `inner_join` over the
//! bindings table). Keeping the bindings table columnar (one column per variable) means
//! the DGX path can swap the host hash-join for a libcudf join without reshaping data.

use std::collections::HashMap;

use arrow::array::{Array, StringArray, UInt32Array};
use arrow::record_batch::RecordBatch;
use nusy_unify::{Substitution, Term, TriplePattern, unify};

use crate::batch::{TripleBatch, TripleCol, round_term_ids};

/// One position of a pattern, resolved per round: either a required dictionary key
/// (constant) or the variable's index in this pattern's variable list.
enum PosMatch {
    /// Constant position: rows must have exactly this key (resolved per round).
    Key(u32),
    /// Constant position whose value is absent from this round's dictionary.
    Absent,
    /// Variable position: row value binds the pattern-variable at this index.
    Var(usize),
}

/// The values column of a round's term column, plus its keys, for value extraction.
fn round_column(batch: &RecordBatch, col: TripleCol) -> (&UInt32Array, &StringArray) {
    let keys = round_term_ids(batch, col);
    let idx = match col {
        TripleCol::Subject => 0,
        TripleCol::Predicate => 1,
        TripleCol::Object => 2,
    };
    let values = batch
        .column(idx)
        .as_any()
        .downcast_ref::<arrow::array::DictionaryArray<arrow::datatypes::UInt32Type>>()
        .expect("term column is Dictionary<UInt32, Utf8> by schema")
        .values()
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("dictionary values are Utf8 by schema");
    (keys, values)
}

/// Resolve a constant to its dictionary key in `values`, if present.
fn key_of(values: &StringArray, constant: &str) -> Option<u32> {
    (0..values.len())
        .find(|&i| values.value(i) == constant)
        .map(|i| i as u32)
}

/// The distinct variables of `patterns` in first-occurrence order (S, P, O within each).
fn pattern_vars(patterns: &[TriplePattern]) -> Vec<String> {
    let mut vars = Vec::new();
    for pat in patterns {
        for term in [&pat.subject, &pat.predicate, &pat.object] {
            if let Term::Var(name) = term
                && !vars.iter().any(|v| v == name)
            {
                vars.push(name.clone());
            }
        }
    }
    vars
}

/// Match one pattern against every row of `facts`, column-wise per round.
/// Returns the per-solution values of this pattern's variables (in `vars` order,
/// `vars` being the pattern's own distinct variables).
fn scan_pattern(pat: &TriplePattern, facts: &TripleBatch) -> (Vec<String>, Vec<Vec<String>>) {
    let vars = pattern_vars(std::slice::from_ref(pat));
    let var_index = |name: &str| vars.iter().position(|v| v == name).expect("var collected");

    let mut rows: Vec<Vec<String>> = Vec::new();
    for round in facts.rounds() {
        // Resolve each position against THIS round's dictionaries.
        let cols = [
            (TripleCol::Subject, &pat.subject),
            (TripleCol::Predicate, &pat.predicate),
            (TripleCol::Object, &pat.object),
        ];
        let mut resolved: Vec<(PosMatch, &UInt32Array, &StringArray)> = Vec::with_capacity(3);
        let mut round_dead = false;
        for (col, term) in cols {
            let (keys, values) = round_column(round, col);
            let pos = match term {
                Term::Const(c) => match key_of(values, c) {
                    Some(k) => PosMatch::Key(k),
                    None => PosMatch::Absent,
                },
                Term::Var(name) => PosMatch::Var(var_index(name)),
            };
            if matches!(pos, PosMatch::Absent) {
                round_dead = true; // constant not in this round's dictionary → no row can match
                break;
            }
            resolved.push((pos, keys, values));
        }
        if round_dead {
            continue;
        }

        'row: for row in 0..round.num_rows() {
            let mut binding: Vec<Option<&str>> = vec![None; vars.len()];
            for (pos, keys, values) in &resolved {
                let key = keys.value(row);
                match pos {
                    PosMatch::Key(required) => {
                        if key != *required {
                            continue 'row;
                        }
                    }
                    PosMatch::Var(vi) => {
                        let value = values.value(key as usize);
                        match binding[*vi] {
                            // Repeated variable within the pattern must bind equal values.
                            Some(prev) if prev != value => continue 'row,
                            _ => binding[*vi] = Some(value),
                        }
                    }
                    PosMatch::Absent => unreachable!("dead rounds are skipped above"),
                }
            }
            rows.push(
                binding
                    .into_iter()
                    .map(|v| {
                        v.expect("every pattern var binds exactly once per row")
                            .to_string()
                    })
                    .collect(),
            );
        }
    }
    (vars, rows)
}

/// Solve a conjunction of patterns against an Arrow [`TripleBatch`]: every
/// [`Substitution`] that satisfies **all** patterns simultaneously, with variables
/// shared across patterns bound consistently. The Arrow-native equivalent of
/// [`nusy_unify::match_conjunction`] (same solution multiset; order unspecified).
pub fn match_conjunction_arrow(
    patterns: &[TriplePattern],
    facts: &TripleBatch,
) -> Vec<Substitution> {
    // Accumulated bindings table: one column per variable, one row per partial solution.
    // An empty conjunction is vacuously true: a single empty solution (Vec parity).
    let mut acc_vars: Vec<String> = Vec::new();
    let mut acc_rows: Vec<Vec<String>> = vec![Vec::new()];

    for pat in patterns {
        let (pat_vars, pat_rows) = scan_pattern(pat, facts);

        // Shared variables = join key; new variables extend the table.
        let shared: Vec<(usize, usize)> = pat_vars
            .iter()
            .enumerate()
            .filter_map(|(pi, name)| acc_vars.iter().position(|v| v == name).map(|ai| (ai, pi)))
            .collect();
        let new_vars: Vec<(usize, String)> = pat_vars
            .iter()
            .enumerate()
            .filter(|(_, name)| !acc_vars.iter().any(|v| &v == name))
            .map(|(pi, name)| (pi, name.clone()))
            .collect();

        // Build side: hash the pattern matches by their shared-variable value tuple.
        let mut built: HashMap<Vec<&str>, Vec<&Vec<String>>> = HashMap::new();
        for row in &pat_rows {
            let key: Vec<&str> = shared.iter().map(|(_, pi)| row[*pi].as_str()).collect();
            built.entry(key).or_default().push(row);
        }

        // Probe side: each accumulated row joins with every matching pattern row.
        let mut next_rows: Vec<Vec<String>> =
            Vec::with_capacity(acc_rows.len().min(pat_rows.len().max(1)));
        for acc in &acc_rows {
            let key: Vec<&str> = shared.iter().map(|(ai, _)| acc[*ai].as_str()).collect();
            if let Some(matches) = built.get(&key) {
                for pat_row in matches {
                    let mut joined = acc.clone();
                    joined.extend(new_vars.iter().map(|(pi, _)| pat_row[*pi].clone()));
                    next_rows.push(joined);
                }
            }
        }

        acc_vars.extend(new_vars.into_iter().map(|(_, name)| name));
        acc_rows = next_rows;
        if acc_rows.is_empty() {
            return Vec::new(); // a dead pattern kills the whole conjunction
        }
    }

    // Adapt rows to Substitutions through the public unify API (no private access).
    acc_rows
        .into_iter()
        .map(|row| {
            let mut subst = Substitution::new();
            for (var, value) in acc_vars.iter().zip(row) {
                subst = unify(&Term::Var(var.clone()), &Term::Const(value), &subst)
                    .expect("binding a fresh variable to a constant cannot conflict");
            }
            subst
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nusy_unify::{Triple, match_conjunction};

    fn t(s: &str, p: &str, o: &str) -> Triple {
        Triple::new(s, p, o)
    }

    fn pat(s: &str, p: &str, o: &str) -> TriplePattern {
        TriplePattern::parse(s, p, o)
    }

    /// Canonicalize solutions for multiset comparison: per substitution, the sorted
    /// (var, value) pairs over `vars`; the whole set sorted.
    fn canon(subs: &[Substitution], vars: &[&str]) -> Vec<Vec<(String, String)>> {
        let mut out: Vec<Vec<(String, String)>> = subs
            .iter()
            .map(|s| {
                let mut bound: Vec<(String, String)> = vars
                    .iter()
                    .filter_map(|v| s.get_const(v).map(|c| (v.to_string(), c)))
                    .collect();
                bound.sort();
                bound
            })
            .collect();
        out.sort();
        out
    }

    fn assert_paths_agree(patterns: &[TriplePattern], facts: &[Triple], vars: &[&str]) {
        let vec_path = match_conjunction(patterns, facts);
        let arrow_path = match_conjunction_arrow(patterns, &TripleBatch::from_triples(facts));
        assert_eq!(
            canon(&arrow_path, vars),
            canon(&vec_path, vars),
            "Arrow and Vec paths disagree for {patterns:?} over {facts:?}"
        );
    }

    #[test]
    fn single_pattern_matches_agree() {
        let facts = vec![
            t("a", "parent", "b"),
            t("b", "parent", "c"),
            t("a", "likes", "c"),
        ];
        assert_paths_agree(&[pat("?x", "parent", "?y")], &facts, &["x", "y"]);
    }

    #[test]
    fn conjunction_join_agrees() {
        let facts = vec![
            t("a", "parent", "b"),
            t("b", "parent", "c"),
            t("c", "parent", "d"),
        ];
        assert_paths_agree(
            &[pat("?x", "parent", "?y"), pat("?y", "parent", "?z")],
            &facts,
            &["x", "y", "z"],
        );
    }

    #[test]
    fn constant_positions_filter() {
        let facts = vec![
            t("a", "parent", "b"),
            t("a", "likes", "b"),
            t("b", "parent", "a"),
        ];
        assert_paths_agree(&[pat("a", "parent", "?y")], &facts, &["y"]);
        assert_paths_agree(&[pat("?x", "parent", "b")], &facts, &["x"]);
        assert_paths_agree(&[pat("a", "parent", "b")], &facts, &[]);
    }

    #[test]
    fn absent_constant_yields_no_solutions() {
        let facts = vec![t("a", "parent", "b")];
        assert_paths_agree(&[pat("?x", "missing_pred", "?y")], &facts, &["x", "y"]);
        assert!(
            match_conjunction_arrow(
                &[pat("?x", "missing_pred", "?y")],
                &TripleBatch::from_triples(&facts)
            )
            .is_empty()
        );
    }

    #[test]
    fn empty_conjunction_is_vacuously_true() {
        let facts = vec![t("a", "parent", "b")];
        let arrow = match_conjunction_arrow(&[], &TripleBatch::from_triples(&facts));
        let vec_path = match_conjunction(&[], &facts);
        assert_eq!(arrow.len(), vec_path.len());
        assert_eq!(arrow.len(), 1);
    }

    #[test]
    fn repeated_variable_within_pattern_requires_equality() {
        let facts = vec![t("a", "knows", "a"), t("a", "knows", "b")];
        assert_paths_agree(&[pat("?x", "knows", "?x")], &facts, &["x"]);
    }

    #[test]
    fn duplicate_facts_produce_duplicate_solutions_on_both_paths() {
        let facts = vec![t("a", "parent", "b"), t("a", "parent", "b")];
        let vec_path = match_conjunction(&[pat("?x", "parent", "?y")], &facts);
        let arrow = match_conjunction_arrow(
            &[pat("?x", "parent", "?y")],
            &TripleBatch::from_triples(&facts),
        );
        assert_eq!(vec_path.len(), 2);
        assert_eq!(arrow.len(), 2);
    }

    #[test]
    fn matches_span_rounds() {
        let mut facts = TripleBatch::from_triples(&[t("a", "parent", "b")]);
        facts.append_triples(&[t("b", "parent", "c")]);
        let solutions = match_conjunction_arrow(
            &[pat("?x", "parent", "?y"), pat("?y", "parent", "?z")],
            &facts,
        );
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].get_const("x").as_deref(), Some("a"));
        assert_eq!(solutions[0].get_const("z").as_deref(), Some("c"));
    }

    #[test]
    fn clinical_style_multi_rule_body_agrees() {
        let facts = vec![
            t("patient1", "has_condition", "osteoporosis"),
            t("osteoporosis", "increases_fall_risk", "true"),
            t("patient1", "age_band", "over_65"),
            t("patient2", "has_condition", "osteoporosis"),
        ];
        assert_paths_agree(
            &[
                pat("?p", "has_condition", "?c"),
                pat("?c", "increases_fall_risk", "true"),
            ],
            &facts,
            &["p", "c"],
        );
    }
}
