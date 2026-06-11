//! Integration tests for the Arrow substrate (EX-4668): the batches must losslessly
//! carry a REAL saturation — facts and derivations from an actual `forward_chain` run —
//! not just hand-built rows, plus a seeded randomized round-trip sweep.

use nusy_forward_chain::batch::{DerivationBatch, TripleBatch, TripleCol};
use nusy_forward_chain::{IdRule, forward_chain};
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

/// Saturate the recursive-ancestor fixture and round-trip the WHOLE saturation
/// (facts + derivations) through TripleBatch/DerivationBatch.
#[test]
fn real_saturation_round_trips_through_batches() {
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
    let sat = forward_chain(&rules, seed.clone());
    assert!(sat.derived_count() > 0, "fixture must actually derive");

    // Facts: seed as round 1, then each derived fact appended (engine-shaped usage:
    // seed batch + derived rows).
    let mut facts = TripleBatch::from_triples(&seed);
    let derived: Vec<Triple> = sat
        .derivations
        .iter()
        .map(|d| d.conclusion.clone())
        .collect();
    facts.append_triples(&derived);
    assert_eq!(
        facts.to_triples(),
        sat.facts,
        "fact order must be preserved"
    );

    // Derivations: encode against the fact batch, decode back, compare exactly.
    let db = DerivationBatch::from_derivations(&sat.derivations, &facts).unwrap();
    assert_eq!(db.len(), sat.derivations.len());
    let decoded = db.to_derivations(&facts).unwrap();
    assert_eq!(decoded, sat.derivations);
}

/// The per-round delta seam: seed round + one round per fixpoint iteration, with
/// rounds() exposing each delta's RecordBatch.
#[test]
fn rounds_model_fixpoint_deltas() {
    let seed = vec![t("a", "parent", "b"), t("b", "parent", "c")];
    let round1_delta = vec![t("a", "ancestor", "b"), t("b", "ancestor", "c")];
    let round2_delta = vec![t("a", "ancestor", "c")];

    let mut facts = TripleBatch::from_triples(&seed);
    facts.append_triples(&round1_delta);
    facts.append_triples(&round2_delta);

    assert_eq!(facts.rounds().len(), 3);
    assert_eq!(facts.rounds()[0].num_rows(), 2);
    assert_eq!(facts.rounds()[1].num_rows(), 2);
    assert_eq!(facts.rounds()[2].num_rows(), 1);
    assert_eq!(facts.len(), 5);
    // Global rows address across rounds.
    assert_eq!(facts.triple_at(4), t("a", "ancestor", "c"));
    assert_eq!(facts.term_at(4, TripleCol::Predicate), "ancestor");
}

/// Seeded xorshift PRNG — deterministic randomized cases without a proptest dependency.
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
}

#[test]
fn randomized_round_trips_are_lossless() {
    let mut rng = XorShift(0x5eed_4668);
    let subjects = ["p0", "p1", "patientë", "変数", "x", ""];
    let predicates = ["parent", "ancestor", "hâs_condition", "p", "rel-1"];
    let objects = ["b", "c", "östeoporosis", "値", "true", ""];

    for case in 0..50 {
        let n = (rng.next() % 40) as usize;
        let triples: Vec<Triple> = (0..n)
            .map(|_| {
                t(
                    subjects[(rng.next() % subjects.len() as u64) as usize],
                    predicates[(rng.next() % predicates.len() as u64) as usize],
                    objects[(rng.next() % objects.len() as u64) as usize],
                )
            })
            .collect();

        // Split into 1-3 rounds at random boundaries.
        let mut tb = TripleBatch::new();
        let cut1 = if n == 0 {
            0
        } else {
            (rng.next() % (n as u64 + 1)) as usize
        };
        let cut2 = if n == 0 {
            0
        } else {
            cut1 + (rng.next() % ((n - cut1) as u64 + 1)) as usize
        };
        tb.append_triples(&triples[..cut1]);
        tb.append_triples(&triples[cut1..cut2]);
        tb.append_triples(&triples[cut2..]);

        assert_eq!(tb.to_triples(), triples, "case {case}: round-trip mismatch");
        assert_eq!(tb.len(), n, "case {case}: length mismatch");
        for (row, expected) in triples.iter().enumerate() {
            assert_eq!(&tb.triple_at(row), expected, "case {case} row {row}");
        }
    }
}
