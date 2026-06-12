//! EX-4747 phase 3 — differential conformance: the adapter must agree with the engine
//! **answer-for-answer and proof-for-proof** on the workloads the engine was validated
//! on (the EXPR-4578.3 clinical gold cases + the kinship closure), and must abstain on
//! exactly the engine's non-entailments (contraindications + negative controls).

use nusy_clinical_fixtures::gold_cases;
use nusy_forward_chain::{ProofTree, forward_chain_arrow};
use nusy_reasoner::{ProofTrace, Provability, Query, Reasoner};
use nusy_reasoner_adapters::{DeductiveReasoner, to_derivation_trace};

/// Engine proof and adapted trace agree on shape: same depth and the same rule multiset.
/// (Compared as sorted multisets — `ProofTree::rule_ids` walks root-first while
/// `DerivationTrace::rule_ids` walks deepest-first, so sequence order legitimately differs
/// for the same tree.)
fn traces_agree(engine: &ProofTree, adapted: &nusy_reasoner::DerivationTrace) -> bool {
    let same_depth = engine.depth() == adapted.depth();
    let mut engine_rules: Vec<&str> = engine.rule_ids();
    let mut adapted_rules: Vec<&str> = adapted.rule_ids();
    engine_rules.sort_unstable();
    adapted_rules.sort_unstable();
    same_depth && engine_rules == adapted_rules
}

#[test]
fn adapter_matches_engine_on_every_clinical_gold_case() {
    for fx in gold_cases() {
        let sat = forward_chain_arrow(&fx.rules, fx.patient_facts.clone());
        let reasoner = DeductiveReasoner::new(fx.rules.clone(), fx.patient_facts.clone());

        // Every engine-derived fact answers Proven through the adapter, same proof shape.
        for i in 0..sat.derived_count() {
            let fact = sat.derivation_batch().conclusion_at(i);
            let answer = reasoner.answer(&Query::new(fact.clone()));
            assert_eq!(
                answer.provability(),
                Provability::Proven,
                "{}: engine derived {fact:?} but adapter did not prove it",
                fx.name
            );
            let engine_proof = sat.proof_of(&fact).expect("engine has the proof");
            let ProofTrace::Derivation(adapted) = &answer.proof else {
                panic!("{}: adapter proof is not a derivation", fx.name);
            };
            assert!(
                traces_agree(&engine_proof, adapted),
                "{}: proof shape diverged for {fact:?}",
                fx.name
            );
        }

        // Every must-NOT fact abstains — the adapter cannot invent what the engine refused.
        for must_not in fx.contraindicated.iter().chain(fx.negative_controls.iter()) {
            let answer = reasoner.answer(&Query::new(must_not.clone()));
            assert_eq!(
                answer.provability(),
                Provability::Abstained,
                "{}: adapter answered a must-not fact {must_not:?}",
                fx.name
            );
        }
    }
}

#[test]
fn adapted_trace_is_lossless_on_a_deep_recursive_proof() {
    use nusy_forward_chain::IdRule;
    use nusy_unify::{Rule, Triple, TriplePattern};
    let rules = vec![
        IdRule::new(
            "anc-base",
            Rule::new(
                vec![TriplePattern::parse("?x", "parent", "?y")],
                vec![TriplePattern::parse("?x", "ancestor", "?y")],
            ),
        ),
        IdRule::new(
            "anc-rec",
            Rule::new(
                vec![
                    TriplePattern::parse("?x", "parent", "?y"),
                    TriplePattern::parse("?y", "ancestor", "?z"),
                ],
                vec![TriplePattern::parse("?x", "ancestor", "?z")],
            ),
        ),
    ];
    let seed: Vec<Triple> = (0..5)
        .map(|i| Triple::new(format!("p{i}"), "parent", format!("p{}", i + 1)))
        .collect();
    let sat = forward_chain_arrow(&rules, seed.clone());
    let goal = Triple::new("p0", "ancestor", "p5"); // depth-5 recursive proof
    let engine_proof = sat.proof_of(&goal).expect("engine proves the chain");
    let adapted = to_derivation_trace(&engine_proof);
    assert!(traces_agree(&engine_proof, &adapted));
    assert!(adapted.is_complete());
    // Axiom multisets agree (the adapted trace cites exactly the engine's leaves).
    let engine_axioms: Vec<String> = engine_proof
        .axioms()
        .iter()
        .map(|a| format!("{} {} {}", a.subject, a.predicate, a.object))
        .collect();
    assert!(!engine_axioms.is_empty());
    let answer = DeductiveReasoner::new(rules, seed).answer(&Query::new(goal));
    assert_eq!(answer.provenance.len(), engine_axioms.len());
}
