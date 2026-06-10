//! Integration test — the `from_y_graph` → do-calculus seam (CH-4666, VOY-1).
//!
//! EX-4638 added [`CausalDag::from_y_graph`] (the Y-graph derivation front-end);
//! the Pearl do-calculus core (identifiability, confounders, intervention) was
//! already shipped. The `from_y_graph` *unit* tests assert DAG **structure** only
//! (nodes/edges/paths). This is the integration test the EX-4653 engine-crate
//! inventory named as the highest-value missing coverage: it builds a fixture
//! Y-graph, runs `from_y_graph`, then runs a **real do-calculus query** over the
//! result — proving the front-end feeds the core correctly end-to-end, not just
//! that it constructs a DAG. It also confirms non-causal triples are filtered so
//! they cannot corrupt the causal query.

use std::sync::Arc;

use arrow::array::{ArrayRef, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use nusy_reasoning_causal::CausalDag;
use nusy_reasoning_causal::identifiability::{IdentificationCriterion, verify_identifiability};
use nusy_reasoning_causal::intervention::confounders;

/// Build a Y-graph triple-schema batch (col0 = id [unused], col1 = subject,
/// col2 = predicate, col3 = object) — the column layout `from_y_graph` reads.
fn y_graph(rows: &[(&str, &str, &str)]) -> RecordBatch {
    let subjects: Vec<&str> = rows.iter().map(|(s, _, _)| *s).collect();
    let predicates: Vec<&str> = rows.iter().map(|(_, p, _)| *p).collect();
    let objects: Vec<&str> = rows.iter().map(|(_, _, o)| *o).collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("subject", DataType::Utf8, false),
        Field::new("predicate", DataType::Utf8, false),
        Field::new("object", DataType::Utf8, false),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(vec!["row"; rows.len()])),
        Arc::new(StringArray::from(subjects)),
        Arc::new(StringArray::from(predicates)),
        Arc::new(StringArray::from(objects)),
    ];
    RecordBatch::try_new(schema, cols).expect("valid y-graph batch")
}

#[test]
fn confounded_effect_is_identifiable_via_backdoor_from_y_graph() {
    // Z confounds X→Y: Z→X, Z→Y, X→Y. Plus non-causal triples the front-end must
    // filter out — otherwise they'd add spurious nodes and corrupt the query.
    let batch = y_graph(&[
        ("Z", "causes", "X"),
        ("Z", "causes", "Y"),
        ("X", "causes", "Y"),
        ("X", "rdf:type", "Treatment"), // non-causal — filtered
        ("Y", "hasLabel", "outcome"),   // non-causal — filtered
    ]);

    // Seam: Y-graph triples → CausalDag (the EX-4638 front-end).
    let dag = CausalDag::from_y_graph(&[batch]).expect("build DAG from Y-graph");
    assert_eq!(
        dag.node_count(),
        3,
        "only causal endpoints Z, X, Y — not 'Treatment'/'outcome'"
    );

    // Core: a real do-calculus identifiability query over the built DAG.
    let v = verify_identifiability(&dag, "X", "Y").expect("verify identifiability");
    assert!(v.identifiable, "X→Y is identifiable by adjusting for Z");
    assert_eq!(v.criterion, Some(IdentificationCriterion::Backdoor));
    assert!(
        v.confounders.contains("Z"),
        "Z (common cause of X and Y) is the backdoor confounder"
    );

    // The confounder query agrees, and the filtered triples added no spurious ones.
    let conf = confounders(&dag, "X", "Y").expect("confounders");
    assert_eq!(conf.len(), 1, "Z is the only confounder");
    assert!(conf.contains("Z"));
}

#[test]
fn direct_effect_from_y_graph_has_no_confounders() {
    // X→Y with no common cause — trivially identifiable, empty adjustment set.
    let dag = CausalDag::from_y_graph(&[y_graph(&[("X", "causes", "Y")])])
        .expect("build DAG from Y-graph");

    let v = verify_identifiability(&dag, "X", "Y").expect("verify identifiability");
    assert!(v.identifiable);
    assert_eq!(v.criterion, Some(IdentificationCriterion::DirectEffect));
    assert!(
        confounders(&dag, "X", "Y").expect("confounders").is_empty(),
        "a direct effect has no confounders to adjust for"
    );
}

#[test]
fn non_causal_only_y_graph_yields_no_queryable_effect() {
    // A Y-graph with NO causal predicates → empty DAG → the do-calculus query
    // refuses (treatment/outcome nodes don't exist). Proves the front-end's
    // filtering and the core's node-existence guard compose safely: a graph with
    // no causal structure cannot fabricate an identifiable effect.
    let dag = CausalDag::from_y_graph(&[y_graph(&[
        ("X", "rdf:type", "Treatment"),
        ("Y", "hasLabel", "outcome"),
    ])])
    .expect("build DAG from Y-graph");

    assert_eq!(dag.node_count(), 0, "no causal predicates → empty DAG");
    assert!(
        verify_identifiability(&dag, "X", "Y").is_err(),
        "no causal nodes → identifiability query must error, not fabricate an effect"
    );
}
