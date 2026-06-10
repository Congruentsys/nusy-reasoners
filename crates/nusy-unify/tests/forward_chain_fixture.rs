//! Integration test: drive `nusy-unify` the way the forward-chaining engine
//! (EX-4588) and transitive-closure derivation (EX-4589) will — iterate
//! [`fire_rule`] over a rule set to a fixpoint. This shows the unification +
//! LHS-matching primitive is sufficient for the VOY-1 engine's matching layer.

use nusy_unify::{Rule, Triple, TriplePattern, fire_rule};
use std::collections::HashSet;

/// Apply `rules` to `facts` repeatedly until no new triples appear (naive fixpoint).
/// This is the shape EX-4588 will wrap with provenance/proof-trace tracking.
fn saturate(rules: &[Rule], seed: Vec<Triple>) -> HashSet<Triple> {
    let mut facts: Vec<Triple> = seed;
    loop {
        let mut new = Vec::new();
        for rule in rules {
            for t in fire_rule(rule, &facts) {
                if !facts.contains(&t) && !new.contains(&t) {
                    new.push(t);
                }
            }
        }
        if new.is_empty() {
            break;
        }
        facts.extend(new);
    }
    facts.into_iter().collect()
}

fn p(s: &str, pred: &str, o: &str) -> Triple {
    Triple::new(s, pred, o)
}

#[test]
fn transitive_ancestor_closure_reaches_fixpoint() {
    // ancestor(?x,?z) :- parent(?x,?z)
    // ancestor(?x,?z) :- parent(?x,?y), ancestor(?y,?z)
    let base = Rule::new(
        vec![TriplePattern::parse("?x", "parent", "?z")],
        vec![TriplePattern::parse("?x", "ancestor", "?z")],
    );
    let recursive = Rule::new(
        vec![
            TriplePattern::parse("?x", "parent", "?y"),
            TriplePattern::parse("?y", "ancestor", "?z"),
        ],
        vec![TriplePattern::parse("?x", "ancestor", "?z")],
    );
    // a → b → c → d chain
    let seed = vec![
        p("a", "parent", "b"),
        p("b", "parent", "c"),
        p("c", "parent", "d"),
    ];

    let closure = saturate(&[base, recursive], seed);

    // Every reachable ancestor pair must be derived.
    for (x, z) in [
        ("a", "b"),
        ("a", "c"),
        ("a", "d"),
        ("b", "c"),
        ("b", "d"),
        ("c", "d"),
    ] {
        assert!(
            closure.contains(&p(x, "ancestor", z)),
            "missing ancestor({x},{z})"
        );
    }
    // No spurious reverse edges.
    assert!(!closure.contains(&p("d", "ancestor", "a")));
    assert!(!closure.contains(&p("b", "ancestor", "a")));
}

#[test]
fn multi_rule_clinical_style_derivation() {
    // A small CPG-flavoured rule set: combine relations to a recommendation.
    // at_risk(?p, "fall") :- has_condition(?p, ?c), increases_risk(?c, "fall")
    // recommend(?p, "fall_assessment") :- at_risk(?p, "fall"), age_over(?p, "65")
    let at_risk = Rule::new(
        vec![
            TriplePattern::parse("?p", "has_condition", "?c"),
            TriplePattern::parse("?c", "increases_risk", "fall"),
        ],
        vec![TriplePattern::parse("?p", "at_risk", "fall")],
    );
    let recommend = Rule::new(
        vec![
            TriplePattern::parse("?p", "at_risk", "fall"),
            TriplePattern::parse("?p", "age_over", "65"),
        ],
        vec![TriplePattern::parse("?p", "recommend", "fall_assessment")],
    );
    let seed = vec![
        p("patient1", "has_condition", "osteoporosis"),
        p("osteoporosis", "increases_risk", "fall"),
        p("patient1", "age_over", "65"),
        p("patient2", "has_condition", "osteoporosis"), // younger → no recommendation
    ];

    let closure = saturate(&[at_risk, recommend], seed);

    assert!(closure.contains(&p("patient1", "at_risk", "fall")));
    assert!(closure.contains(&p("patient1", "recommend", "fall_assessment")));
    assert!(closure.contains(&p("patient2", "at_risk", "fall")));
    // patient2 has no age_over fact → no recommendation (provable-only, not assumed).
    assert!(!closure.contains(&p("patient2", "recommend", "fall_assessment")));
}

#[test]
fn no_rules_is_identity() {
    let seed = vec![p("a", "b", "c")];
    let closure = saturate(&[], seed);
    assert_eq!(closure.len(), 1);
    assert!(closure.contains(&p("a", "b", "c")));
}
