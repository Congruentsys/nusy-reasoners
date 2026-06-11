//! Benchmark: semi-naive incremental re-derivation vs full re-fire (EX-4593, VY-4667 §5).
//!
//! Builds a transitive-closure chain, then measures maintaining the closure after adding a
//! single edge two ways: (a) `forward_chain_arrow_incremental` (re-derive only the delta's
//! consequences) vs (b) `forward_chain_arrow` over `prior ++ new` (full re-fire). Reports the
//! median of each — the M-4623 "full-refire baseline" vs the incremental path.
//!
//! Run: `cargo run -p nusy-forward-chain --release --example bench_incremental_vs_refire`

use std::time::Instant;

use nusy_forward_chain::{IdRule, forward_chain_arrow, forward_chain_arrow_incremental};
use nusy_unify::{Rule, Triple, TriplePattern};

fn median(mut v: Vec<u128>) -> u128 {
    v.sort_unstable();
    v[v.len() / 2]
}

fn main() {
    let rules = vec![
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
    ];

    for n in [20usize, 45, 80] {
        let chain: Vec<Triple> = (0..n)
            .map(|i| Triple::new(format!("c{i}"), "parent", format!("c{}", i + 1)))
            .collect();
        let new = vec![Triple::new(format!("c{n}"), "parent", "tail")];
        let mut full_seed = chain.clone();
        full_seed.extend_from_slice(&new);

        let prior = forward_chain_arrow(&rules, chain.clone());
        let reps = 25;

        let inc: Vec<u128> = (0..reps)
            .map(|_| {
                let t = Instant::now();
                let s = forward_chain_arrow_incremental(&prior, &rules, new.clone());
                std::hint::black_box(&s);
                t.elapsed().as_micros()
            })
            .collect();
        let refire: Vec<u128> = (0..reps)
            .map(|_| {
                let t = Instant::now();
                let s = forward_chain_arrow(&rules, full_seed.clone());
                std::hint::black_box(&s);
                t.elapsed().as_micros()
            })
            .collect();

        let (mi, mr) = (median(inc), median(refire));
        let closure = forward_chain_arrow(&rules, full_seed.clone()).facts().to_triples().len();
        println!(
            "n={n:<3} closure={closure:<5}  incremental median={mi:>6} µs   full-refire median={mr:>6} µs   speedup={:.1}×",
            mr as f64 / mi.max(1) as f64
        );
    }
}
