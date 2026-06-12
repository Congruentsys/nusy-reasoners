//! EX-4749 phase 3 acceptance — COMPOSE over real engines.
//!
//! Genuine context threading on the actual forward-chain adapters: stage 2's
//! engine has **no facts of its own** and cannot prove the goal alone (asserted);
//! it proves only because stage 1's answer flows in through the pipeline context.
//! The composite trace then **grafts** stage 1's full derivation in place of the
//! context fact stage 2 leaned on — so the end-to-end proof carries both stages'
//! rules and computes `Proven` (the minimum law's Proven side, no mocks).

use nusy_forward_chain::IdRule;
use nusy_reasoner::{Pipeline, ProofTrace, Provability, Query, Reasoner};
use nusy_reasoner_adapters::DeductiveReasoner;
use nusy_unify::{Rule, Triple, TriplePattern};

#[test]
fn context_threading_and_grafting_on_real_engines() {
    // Stage 1: the full clinical chain — proves the goal with a 2-rule derivation.
    let chain_rules = vec![
        IdRule::new(
            "frail-from-condition",
            Rule::new(
                vec![
                    TriplePattern::parse("?p", "has_condition", "?c"),
                    TriplePattern::parse("?c", "increases_fall_risk", "true"),
                ],
                vec![TriplePattern::parse("?p", "frail", "true")],
            ),
        ),
        IdRule::new(
            "at-risk-from-frail",
            Rule::new(
                vec![TriplePattern::parse("?p", "frail", "true")],
                vec![TriplePattern::parse("?p", "at_risk", "fall")],
            ),
        ),
    ];
    let seed_facts = vec![
        Triple::new("p1", "has_condition", "osteoporosis"),
        Triple::new("osteoporosis", "increases_fall_risk", "true"),
    ];

    // Stage 2: NO rules that reach the goal from seeds, NO facts at all — it can
    // only answer through what the pipeline context carries.
    let bare_stage = DeductiveReasoner::new(vec![], vec![]);
    let goal = Query::new(Triple::new("p1", "at_risk", "fall"));
    assert_eq!(
        bare_stage.answer(&goal).provability(),
        Provability::Abstained,
        "stage 2 alone must be unable to prove the goal"
    );

    let pipe = Pipeline::new("deductive→deductive")
        .then(Box::new(DeductiveReasoner::new(chain_rules, seed_facts)))
        .then(Box::new(DeductiveReasoner::new(vec![], vec![])));

    let a = pipe.answer(&goal);
    assert_eq!(
        a.provability(),
        Provability::Proven,
        "stage 1's contribution flows through context; composite stays Proven"
    );

    // The graft threads end-to-end: stage 2 proved the goal as a context fact
    // (axiom), and composition replaced that axiom with stage 1's real 2-rule
    // derivation — both engine rules appear in the composite proof.
    let ProofTrace::Derivation(d) = &a.proof else {
        panic!("expected a composite derivation, got {:?}", a.proof);
    };
    let rules = d.rule_ids();
    assert!(rules.contains(&"frail-from-condition"), "{rules:?}");
    assert!(rules.contains(&"at-risk-from-frail"), "{rules:?}");
    assert!(
        d.is_complete(),
        "grafted composite bottoms out at seed axioms"
    );
    assert!(d.depth() >= 2, "threaded proof keeps the multi-hop depth");
}
