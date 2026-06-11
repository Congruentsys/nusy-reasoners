//! Differential integration test for EX-4669: the Arrow conjunction matcher must return
//! the SAME solution multiset as `nusy_unify::match_conjunction` on randomized fact sets
//! and pattern conjunctions (seeded, deterministic — no proptest dependency).

use nusy_forward_chain::arrow_match::match_conjunction_arrow;
use nusy_forward_chain::batch::TripleBatch;
use nusy_unify::{Substitution, Term, Triple, TriplePattern, match_conjunction};

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

/// Canonical multiset of solutions: per substitution, sorted (var, value) pairs over
/// the conjunction's variables; the whole list sorted.
fn canon(subs: &[Substitution], patterns: &[TriplePattern]) -> Vec<Vec<(String, String)>> {
    let mut vars: Vec<String> = Vec::new();
    for p in patterns {
        for term in [&p.subject, &p.predicate, &p.object] {
            if let Term::Var(name) = term
                && !vars.contains(name)
            {
                vars.push(name.clone());
            }
        }
    }
    let mut out: Vec<Vec<(String, String)>> = subs
        .iter()
        .map(|s| {
            let mut bound: Vec<(String, String)> = vars
                .iter()
                .filter_map(|v| s.get_const(v).map(|c| (v.clone(), c)))
                .collect();
            bound.sort();
            bound
        })
        .collect();
    out.sort();
    out
}

#[test]
fn randomized_conjunctions_agree_with_vec_path() {
    let entities = ["a", "b", "c", "d", "e"];
    let predicates = ["parent", "likes", "knows"];
    let term_pool = ["?x", "?y", "?z", "a", "b", "c"];
    let pred_pool = ["parent", "likes", "knows", "?p"];

    let mut rng = XorShift(0x5eed_4669);
    for case in 0..200 {
        // Random fact set (with possible duplicates — multiset semantics must hold).
        let n_facts = 1 + (rng.next() % 25) as usize;
        let facts: Vec<Triple> = (0..n_facts)
            .map(|_| {
                Triple::new(
                    rng.pick(&entities),
                    rng.pick(&predicates),
                    rng.pick(&entities),
                )
            })
            .collect();

        // Random conjunction of 1-3 patterns drawing vars and constants from the pools.
        let n_pats = 1 + (rng.next() % 3) as usize;
        let patterns: Vec<TriplePattern> = (0..n_pats)
            .map(|_| {
                TriplePattern::parse(
                    rng.pick(&term_pool),
                    rng.pick(&pred_pool),
                    rng.pick(&term_pool),
                )
            })
            .collect();

        // Split facts into 1-3 rounds to exercise cross-round matching.
        let cut1 = (rng.next() % (n_facts as u64 + 1)) as usize;
        let cut2 = cut1 + (rng.next() % ((n_facts - cut1) as u64 + 1)) as usize;
        let mut batch = TripleBatch::new();
        batch.append_triples(&facts[..cut1]);
        batch.append_triples(&facts[cut1..cut2]);
        batch.append_triples(&facts[cut2..]);

        let vec_path = match_conjunction(&patterns, &facts);
        let arrow_path = match_conjunction_arrow(&patterns, &batch);
        assert_eq!(
            canon(&arrow_path, &patterns),
            canon(&vec_path, &patterns),
            "case {case}: paths disagree\n  patterns: {patterns:?}\n  facts: {facts:?}"
        );
    }
}
