//! Differential integration tests for EX-4670: the Arrow-backed engine
//! (`forward_chain`, rewired) must agree with the retained Vec reference oracle
//! (`forward_chain_vec`) on the saturated fact set, derived set, and proofs —
//! on hand-built fixtures and randomized rule/fact sets.

use std::collections::HashSet;

use nusy_forward_chain::{IdRule, forward_chain, forward_chain_arrow, forward_chain_vec};
use nusy_unify::{Rule, Triple, TriplePattern};

fn t(s: &str, p: &str, o: &str) -> Triple {
    Triple::new(s, p, o)
}

fn rule(id: &str, body: &[(&str, &str, &str)], head: (&str, &str, &str)) -> IdRule {
    let lhs = body
        .iter()
        .map(|(s, p, o)| TriplePattern::parse(s, p, o))
        .collect();
    let rhs = vec![TriplePattern::parse(head.0, head.1, head.2)];
    IdRule::new(id, Rule::new(lhs, rhs))
}

fn fact_set(facts: &[Triple]) -> HashSet<Triple> {
    facts.iter().cloned().collect()
}

/// Seeded xorshift PRNG.
struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn pick<'a>(&mut self, items: &'a [&'a str]) -> &'a str {
        items[(self.next() % items.len() as u64) as usize]
    }
}

#[test]
fn ancestor_closure_agrees_with_vec_oracle() {
    let rules = vec![
        rule(
            "anc-base",
            &[("?x", "parent", "?y")],
            ("?x", "ancestor", "?y"),
        ),
        rule(
            "anc-rec",
            &[("?x", "parent", "?y"), ("?y", "ancestor", "?z")],
            ("?x", "ancestor", "?z"),
        ),
    ];
    let seed = vec![
        t("a", "parent", "b"),
        t("b", "parent", "c"),
        t("c", "parent", "d"),
    ];

    let arrow = forward_chain(&rules, seed.clone());
    let vec_oracle = forward_chain_vec(&rules, seed);

    assert_eq!(fact_set(&arrow.facts), fact_set(&vec_oracle.facts));
    assert_eq!(arrow.derived_count(), vec_oracle.derived_count());
    // Every derived fact must carry an equivalent proof on both paths (same
    // provability; proof SHAPE may differ where multiple derivations exist).
    for d in &vec_oracle.derivations {
        assert!(
            arrow.proof_of(&d.conclusion).is_some(),
            "{:?} unproven on Arrow path",
            d.conclusion
        );
    }
}

#[test]
fn arrow_saturation_views_are_consistent_with_materialization() {
    let rules = vec![
        rule(
            "anc-base",
            &[("?x", "parent", "?y")],
            ("?x", "ancestor", "?y"),
        ),
        rule(
            "anc-rec",
            &[("?x", "parent", "?y"), ("?y", "ancestor", "?z")],
            ("?x", "ancestor", "?z"),
        ),
    ];
    let seed = vec![t("a", "parent", "b"), t("b", "parent", "c")];
    let arrow_sat = forward_chain_arrow(&rules, seed.clone());
    let materialized = arrow_sat.to_saturation();

    // Membership and counts agree between the Arrow views and the Vec materialization.
    assert_eq!(arrow_sat.derived_count(), materialized.derived_count());
    for fact in &materialized.facts {
        assert!(arrow_sat.contains(fact));
    }
    // Discovery order: seed first, then deltas — seed prefix preserved exactly.
    assert_eq!(&materialized.facts[..seed.len()], &seed[..]);
    // Per-fact derivations agree, and Arrow proofs match the materialized proofs.
    for d in &materialized.derivations {
        let arrow_d = arrow_sat
            .derivation_of(&d.conclusion)
            .expect("derived on both");
        assert_eq!(&arrow_d, d);
        assert_eq!(
            arrow_sat.proof_of(&d.conclusion),
            materialized.proof_of(&d.conclusion)
        );
    }
    // Seed facts have no derivation; unknown triples have no proof.
    assert!(arrow_sat.derivation_of(&seed[0]).is_none());
    assert!(arrow_sat.proof_of(&t("nope", "nope", "nope")).is_none());
    // The delta seam: seed round + one round per fixpoint iteration (2 here).
    assert_eq!(arrow_sat.facts().rounds().len(), 3);
}

#[test]
fn randomized_rule_sets_agree_with_vec_oracle() {
    let entities = ["a", "b", "c", "d"];
    let predicates = ["p", "q", "r"];
    let body_terms = ["?x", "?y", "a", "b"];
    let head_terms = ["?x", "?y", "c"];

    let mut rng = XorShift(0x5eed_4670);
    for case in 0..100 {
        let n_facts = 1 + (rng.next() % 15) as usize;
        let seed: Vec<Triple> = (0..n_facts)
            .map(|_| {
                t(
                    rng.pick(&entities),
                    rng.pick(&predicates),
                    rng.pick(&entities),
                )
            })
            .collect();

        let n_rules = 1 + (rng.next() % 3) as usize;
        let rules: Vec<IdRule> = (0..n_rules)
            .map(|i| {
                let n_body = 1 + (rng.next() % 2) as usize;
                let body: Vec<(&str, &str, &str)> = (0..n_body)
                    .map(|_| {
                        (
                            rng.pick(&body_terms),
                            rng.pick(&predicates),
                            rng.pick(&body_terms),
                        )
                    })
                    .collect();
                let head = (
                    rng.pick(&head_terms),
                    rng.pick(&predicates),
                    rng.pick(&head_terms),
                );
                rule(&format!("r{i}"), &body, head)
            })
            .collect();

        let arrow = forward_chain(&rules, seed.clone());
        let vec_oracle = forward_chain_vec(&rules, seed.clone());
        assert_eq!(
            fact_set(&arrow.facts),
            fact_set(&vec_oracle.facts),
            "case {case}: fact sets disagree\n  rules: {rules:?}\n  seed: {seed:?}"
        );
        // Derived sets (conclusions) agree as sets; rule attribution may differ where
        // a fact has multiple derivations (first-found is order-dependent).
        let arrow_derived: HashSet<Triple> = arrow
            .derivations
            .iter()
            .map(|d| d.conclusion.clone())
            .collect();
        let vec_derived: HashSet<Triple> = vec_oracle
            .derivations
            .iter()
            .map(|d| d.conclusion.clone())
            .collect();
        assert_eq!(
            arrow_derived, vec_derived,
            "case {case}: derived sets disagree"
        );
    }
}
