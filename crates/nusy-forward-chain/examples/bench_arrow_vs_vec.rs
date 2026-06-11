//! EX-4670 bench: full-refire saturation latency, Arrow engine vs the Vec reference
//! oracle, at clinical-fixture scale and at ~10k-triple synthetic scale.
//!
//! ```bash
//! cargo run --release -p nusy-forward-chain --example bench_arrow_vs_vec
//! ```

use std::time::Instant;

use nusy_forward_chain::{IdRule, forward_chain, forward_chain_vec};
use nusy_unify::{Rule, Triple, TriplePattern};

fn rule(id: &str, body: &[(&str, &str, &str)], head: (&str, &str, &str)) -> IdRule {
    let lhs = body
        .iter()
        .map(|(s, p, o)| TriplePattern::parse(s, p, o))
        .collect();
    let rhs = vec![TriplePattern::parse(head.0, head.1, head.2)];
    IdRule::new(id, Rule::new(lhs, rhs))
}

fn kinship_rules() -> Vec<IdRule> {
    vec![
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
        rule(
            "grandparent",
            &[("?x", "parent", "?y"), ("?y", "parent", "?z")],
            ("?x", "grandparent", "?z"),
        ),
    ]
}

/// A parent chain of `n` nodes: closure size n(n-1)/2 ancestors + (n-2) grandparents.
fn chain_seed(n: usize) -> Vec<Triple> {
    (0..n - 1)
        .map(|i| Triple::new(format!("p{i}"), "parent", format!("p{}", i + 1)))
        .collect()
}

fn clinical_seed() -> (Vec<IdRule>, Vec<Triple>) {
    let rules = vec![
        rule(
            "at-risk-fall",
            &[
                ("?p", "has_condition", "?c"),
                ("?c", "increases_fall_risk", "true"),
            ],
            ("?p", "at_risk", "fall"),
        ),
        rule(
            "recommend-fall-assessment",
            &[("?p", "at_risk", "fall"), ("?p", "age_band", "over_65")],
            ("?p", "recommend", "fall_assessment"),
        ),
    ];
    let seed = vec![
        Triple::new("patient1", "has_condition", "osteoporosis"),
        Triple::new("osteoporosis", "increases_fall_risk", "true"),
        Triple::new("patient1", "age_band", "over_65"),
    ];
    (rules, seed)
}

fn median_ms(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn bench(name: &str, rules: &[IdRule], seed: &[Triple], trials: usize) {
    let expected = forward_chain_vec(rules, seed.to_vec());

    let mut arrow_ms = Vec::with_capacity(trials);
    for _ in 0..trials {
        let start = Instant::now();
        let sat = forward_chain(rules, seed.to_vec());
        arrow_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        assert_eq!(sat.facts.len(), expected.facts.len(), "engines disagree");
    }
    let mut vec_ms = Vec::with_capacity(trials);
    for _ in 0..trials {
        let start = Instant::now();
        let sat = forward_chain_vec(rules, seed.to_vec());
        vec_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        assert_eq!(
            sat.facts.len(),
            expected.facts.len(),
            "oracle non-deterministic"
        );
    }

    let arrow = median_ms(arrow_ms);
    let vec = median_ms(vec_ms);
    println!(
        "{name}: facts={} derived={} | arrow median {arrow:.3} ms | vec median {vec:.3} ms | speedup {:.1}x",
        expected.facts.len(),
        expected.derived_count(),
        vec / arrow,
    );
}

fn main() {
    let (clinical_rules, clinical) = clinical_seed();
    bench("clinical-fixture (3 seed)", &clinical_rules, &clinical, 50);

    let rules = kinship_rules();
    bench(
        "kinship chain n=20 (~210 facts)",
        &rules,
        &chain_seed(20),
        20,
    );
    bench(
        "kinship chain n=60 (~1.9k facts)",
        &rules,
        &chain_seed(60),
        5,
    );
    bench(
        "kinship chain n=142 (~10k facts)",
        &rules,
        &chain_seed(142),
        3,
    );
}
