//! Differential + timing tests for the semi-naive incremental engine (EX-4593, VY-4667 §5).
//!
//! **Correctness contract:** `forward_chain_arrow_incremental(prior, rules, new)` must yield
//! the *same* result — fact set and provability of every claim — as the full re-fire
//! `forward_chain_arrow(rules, prior_seed ++ new)`. These tests assert that on fixed
//! clinical/kinship cases and on randomized parent graphs.
//!
//! **Performance:** on a long transitive-closure chain, adding one edge incrementally is far
//! cheaper than re-firing the whole closure — the EX-4593 goal (beat the M-4623 full-refire
//! baseline). The timing check uses a deliberately generous margin so it proves the *trend*
//! without flaking on CI noise; the precise numbers live in the `bench_incremental_vs_refire`
//! example.

use std::collections::HashSet;
use std::time::Instant;

use nusy_forward_chain::{
    ArrowSaturation, IdRule, forward_chain_arrow, forward_chain_arrow_incremental,
};
use nusy_unify::{Rule, Triple, TriplePattern};

/// `forward_chain_arrow_incremental(prior, rules, new)` ≡ `forward_chain_arrow(rules, A ++ new)`.
fn assert_equivalent(rules: &[IdRule], seed_a: &[Triple], new: &[Triple]) {
    let prior = forward_chain_arrow(rules, seed_a.to_vec());
    let incremental = forward_chain_arrow_incremental(&prior, rules, new.to_vec());

    let mut full_seed = seed_a.to_vec();
    full_seed.extend_from_slice(new);
    let refire = forward_chain_arrow(rules, full_seed);

    let inc_facts: HashSet<Triple> = incremental.facts().to_triples().into_iter().collect();
    let refire_facts: HashSet<Triple> = refire.facts().to_triples().into_iter().collect();
    assert_eq!(
        inc_facts, refire_facts,
        "incremental and full-refire fact sets diverged"
    );

    // Every fact must be provable in both (seeds → Axiom, derived → Derived).
    for f in &refire_facts {
        assert!(
            incremental.proof_of(f).is_some(),
            "incremental could not prove {f:?}"
        );
        assert!(refire.proof_of(f).is_some(), "refire could not prove {f:?}");
    }
}

fn ancestor_rules() -> Vec<IdRule> {
    vec![
        IdRule::new(
            "anc-base",
            Rule::new(
                vec![TriplePattern::parse("?x", "parent", "?y")],
                vec![TriplePattern::parse("?x", "ancestor", "?y")],
            ),
        ),
        IdRule::new(
            "anc-trans",
            Rule::new(
                vec![
                    TriplePattern::parse("?x", "ancestor", "?y"),
                    TriplePattern::parse("?y", "ancestor", "?z"),
                ],
                vec![TriplePattern::parse("?x", "ancestor", "?z")],
            ),
        ),
    ]
}

fn parent(a: &str, b: &str) -> Triple {
    Triple::new(a, "parent", b)
}

#[test]
fn incremental_matches_refire_on_kinship_chain() {
    let rules = ancestor_rules();
    // a0 → a1 → a2 → a3 already closed; then add the edge a3 → a4.
    let seed_a = vec![parent("a0", "a1"), parent("a1", "a2"), parent("a2", "a3")];
    let new = vec![parent("a3", "a4")];
    assert_equivalent(&rules, &seed_a, &new);
}

#[test]
fn incremental_matches_refire_when_new_edge_bridges_two_components() {
    let rules = ancestor_rules();
    // Two separate chains; the new edge joins them, unlocking cross-component ancestors.
    let seed_a = vec![
        parent("a", "b"),
        parent("b", "c"),
        parent("x", "y"),
        parent("y", "z"),
    ];
    let new = vec![parent("c", "x")]; // now a..c are ancestors of x..z
    assert_equivalent(&rules, &seed_a, &new);
}

#[test]
fn incremental_matches_refire_multi_premise_clinical() {
    // at_risk(?p,"fall") :- has_condition(?p,?c), increases_fall_risk(?c,"true")
    let rules = vec![IdRule::new(
        "at-risk-fall",
        Rule::new(
            vec![
                TriplePattern::parse("?p", "has_condition", "?c"),
                TriplePattern::parse("?c", "increases_fall_risk", "true"),
            ],
            vec![TriplePattern::parse("?p", "at_risk", "fall")],
        ),
    )];
    let seed_a = vec![
        Triple::new("osteoporosis", "increases_fall_risk", "true"),
        Triple::new("p1", "has_condition", "osteoporosis"),
    ];
    // New patient + a new risk-bearing condition arrive incrementally.
    let new = vec![
        Triple::new("p2", "has_condition", "osteoporosis"),
        Triple::new("neuropathy", "increases_fall_risk", "true"),
        Triple::new("p1", "has_condition", "neuropathy"),
    ];
    assert_equivalent(&rules, &seed_a, &new);
}

#[test]
fn incremental_matches_refire_on_randomized_graphs() {
    let rules = ancestor_rules();
    // Dependency-free deterministic PRNG (no rand dep; Math.random is irrelevant in Rust).
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for trial in 0..40 {
        let n = 6 + (next() % 6) as usize; // 6..11 nodes
        let mut edges: Vec<Triple> = Vec::new();
        for _ in 0..(n + (next() % (n as u64)) as usize) {
            let u = next() % n as u64;
            let v = next() % n as u64;
            if u != v {
                edges.push(parent(&format!("n{u}"), &format!("n{v}")));
            }
        }
        if edges.len() < 2 {
            continue;
        }
        let split = 1 + (next() as usize % (edges.len() - 1));
        let (a, b) = edges.split_at(split);
        // Run under each split — incremental(prior=A, +B) must equal refire(A++B).
        assert_equivalent(&rules, a, b);
        let _ = trial;
    }
}

#[test]
fn incremental_beats_full_refire_on_long_chain() {
    let rules = ancestor_rules();
    // A chain → a quadratic transitive closure. Adding one edge. (Kept modest so the test
    // stays fast; the margin is large enough to prove the trend regardless.)
    let n = 45usize;
    let chain: Vec<Triple> = (0..n)
        .map(|i| parent(&format!("c{i}"), &format!("c{}", i + 1)))
        .collect();
    let new = vec![parent(&format!("c{n}"), "tail")];

    let reps = 5;
    let prior = forward_chain_arrow(&rules, chain.clone());

    let t_inc = Instant::now();
    let mut inc_sat: Option<ArrowSaturation> = None;
    for _ in 0..reps {
        inc_sat = Some(forward_chain_arrow_incremental(&prior, &rules, new.clone()));
    }
    let inc = t_inc.elapsed();

    let mut full_seed = chain.clone();
    full_seed.extend_from_slice(&new);
    let t_ref = Instant::now();
    for _ in 0..reps {
        let _ = forward_chain_arrow(&rules, full_seed.clone());
    }
    let refire = t_ref.elapsed();

    // Sanity: the incremental result is the full closure (correctness, not just speed).
    let inc_sat = inc_sat.unwrap();
    assert!(inc_sat.contains(&Triple::new("c0", "ancestor", "tail")));

    // Generous margin (2×) so this proves the trend without flaking on noise.
    assert!(
        inc.as_nanos() * 2 < refire.as_nanos(),
        "incremental ({inc:?}) not clearly faster than full re-fire ({refire:?})"
    );
}
