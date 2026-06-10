//! Forward-chaining engine behaviour: fixpoint derivation with provenance, the
//! recursive (transitive-closure) case, multi-hop proof chains, and the provable-only
//! invariant that nothing is asserted without a derivation.

use nusy_forward_chain::{IdRule, forward_chain};
use nusy_unify::{Rule, Triple, TriplePattern};

fn t(s: &str, p: &str, o: &str) -> Triple {
    Triple::new(s, p, o)
}

/// `head(?x,?z) :- body...` convenience.
fn rule(id: &str, body: Vec<TriplePattern>, head: TriplePattern) -> IdRule {
    IdRule::new(id, Rule::new(body, vec![head]))
}

#[test]
fn grandparent_derivation_carries_provenance() {
    let gp = rule(
        "grandparent",
        vec![
            TriplePattern::parse("?x", "parent", "?y"),
            TriplePattern::parse("?y", "parent", "?z"),
        ],
        TriplePattern::parse("?x", "grandparent", "?z"),
    );
    let seed = vec![t("a", "parent", "b"), t("b", "parent", "c")];
    let sat = forward_chain(&[gp], seed);

    let concl = t("a", "grandparent", "c");
    assert!(sat.contains(&concl));
    assert_eq!(sat.derived_count(), 1, "exactly one derived fact");

    let d = sat.derivation_of(&concl).expect("derivation recorded");
    assert_eq!(d.rule_id, "grandparent");
    // Premises are the two ground parent facts that fired the rule.
    assert!(d.premises.contains(&t("a", "parent", "b")));
    assert!(d.premises.contains(&t("b", "parent", "c")));
    assert_eq!(d.premises.len(), 2);

    // A seed fact has no derivation.
    assert!(sat.derivation_of(&t("a", "parent", "b")).is_none());
}

#[test]
fn transitive_closure_reaches_fixpoint_with_provenance() {
    // ancestor(?x,?z) :- parent(?x,?z)
    // ancestor(?x,?z) :- parent(?x,?y), ancestor(?y,?z)
    let base = rule(
        "anc-base",
        vec![TriplePattern::parse("?x", "parent", "?z")],
        TriplePattern::parse("?x", "ancestor", "?z"),
    );
    let recursive = rule(
        "anc-rec",
        vec![
            TriplePattern::parse("?x", "parent", "?y"),
            TriplePattern::parse("?y", "ancestor", "?z"),
        ],
        TriplePattern::parse("?x", "ancestor", "?z"),
    );
    let seed = vec![
        t("a", "parent", "b"),
        t("b", "parent", "c"),
        t("c", "parent", "d"),
    ];
    let sat = forward_chain(&[base, recursive], seed);

    // Every reachable ancestor pair is derived...
    for (x, z) in [
        ("a", "b"),
        ("a", "c"),
        ("a", "d"),
        ("b", "c"),
        ("b", "d"),
        ("c", "d"),
    ] {
        let fact = t(x, "ancestor", z);
        assert!(sat.contains(&fact), "missing ancestor({x},{z})");
        assert!(
            sat.derivation_of(&fact).is_some(),
            "ancestor({x},{z}) lacks provenance"
        );
    }
    // ...and nothing spurious.
    assert!(!sat.contains(&t("d", "ancestor", "a")));
    assert!(!sat.contains(&t("b", "ancestor", "a")));
}

#[test]
fn proof_chains_through_derived_premises() {
    // A derived fact can be a premise of a deeper derivation: ancestor(a,c) is
    // derived from parent(a,b) + ancestor(b,c), and ancestor(b,c) is itself derived.
    let base = rule(
        "anc-base",
        vec![TriplePattern::parse("?x", "parent", "?z")],
        TriplePattern::parse("?x", "ancestor", "?z"),
    );
    let recursive = rule(
        "anc-rec",
        vec![
            TriplePattern::parse("?x", "parent", "?y"),
            TriplePattern::parse("?y", "ancestor", "?z"),
        ],
        TriplePattern::parse("?x", "ancestor", "?z"),
    );
    let seed = vec![t("a", "parent", "b"), t("b", "parent", "c")];
    let sat = forward_chain(&[base, recursive], seed);

    let ac = t("a", "ancestor", "c");
    let d_ac = sat.derivation_of(&ac).expect("ancestor(a,c) derived");
    assert_eq!(d_ac.rule_id, "anc-rec");
    // One of its premises is ancestor(b,c), which is ITSELF a derived fact —
    // EX-4592 recurses on exactly this to build the full proof tree.
    let bc = t("b", "ancestor", "c");
    assert!(d_ac.premises.contains(&bc));
    assert!(
        sat.derivation_of(&bc).is_some(),
        "the premise is itself provable"
    );
}

#[test]
fn provable_only_no_premise_no_derivation() {
    // recommend(?p,"fall_assessment") :- at_risk(?p,"fall"), age_over(?p,"65")
    let recommend = rule(
        "recommend-fall",
        vec![
            TriplePattern::parse("?p", "at_risk", "fall"),
            TriplePattern::parse("?p", "age_over", "65"),
        ],
        TriplePattern::parse("?p", "recommend", "fall_assessment"),
    );
    // patient1 has both facts; patient2 lacks age_over.
    let seed = vec![
        t("patient1", "at_risk", "fall"),
        t("patient1", "age_over", "65"),
        t("patient2", "at_risk", "fall"),
    ];
    let sat = forward_chain(&[recommend], seed);

    assert!(sat.contains(&t("patient1", "recommend", "fall_assessment")));
    // patient2 has no age_over fact → no recommendation is invented.
    assert!(!sat.contains(&t("patient2", "recommend", "fall_assessment")));
    assert_eq!(sat.derived_count(), 1);
}

#[test]
fn empty_rules_is_identity() {
    let seed = vec![t("a", "b", "c")];
    let sat = forward_chain(&[], seed);
    assert_eq!(sat.facts.len(), 1);
    assert_eq!(sat.derived_count(), 0);
    assert!(sat.contains(&t("a", "b", "c")));
}

#[test]
fn non_range_restricted_head_is_skipped_not_panicked() {
    // RHS has variable ?w that never appears in the body → cannot ground.
    let bad = IdRule::new(
        "unsafe",
        Rule::new(
            vec![TriplePattern::parse("?x", "parent", "?y")],
            vec![TriplePattern::parse("?x", "related_to", "?w")],
        ),
    );
    let seed = vec![t("a", "parent", "b")];
    let sat = forward_chain(&[bad], seed);
    // Nothing derivable (the head can't be grounded), but no panic.
    assert_eq!(sat.derived_count(), 0);
    assert_eq!(sat.facts.len(), 1);
}
