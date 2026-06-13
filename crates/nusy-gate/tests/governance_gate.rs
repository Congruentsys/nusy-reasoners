//! Non-clinical genericity smoke test, gate tier (EX-4700 / EXPR-4699, V19-H3): the
//! provable-claim gate answers *governance/CI policy* claims exactly as it answers
//! clinical ones — Proven with a proof when derivable, Unproven (loud, never asserted)
//! otherwise. Same engine, same gate, different content.

use nusy_forward_chain::{IdRule, forward_chain, forward_chain_arrow};
use nusy_gate::ProvableClaimGate;
use nusy_unify::{Rule, Triple, TriplePattern};

fn t(s: &str, p: &str, o: &str) -> Triple {
    Triple::new(s, p, o)
}

fn rules() -> Vec<IdRule> {
    vec![
        IdRule::new(
            "cross-agent-approval",
            Rule::new(
                vec![
                    TriplePattern::parse("?p", "authored_by", "?a"),
                    TriplePattern::parse("?p", "has_approval", "?r"),
                    TriplePattern::parse("?a", "distinct_from", "?r"),
                ],
                vec![TriplePattern::parse("?p", "cross_agent_approved", "true")],
            ),
        ),
        IdRule::new(
            "no-self-approve",
            Rule::new(
                vec![
                    TriplePattern::parse("?p", "authored_by", "?a"),
                    TriplePattern::parse("?p", "has_approval", "?a"),
                ],
                vec![TriplePattern::parse(
                    "?p",
                    "violates_policy",
                    "self-approval",
                )],
            ),
        ),
        IdRule::new(
            "mergeable",
            Rule::new(
                vec![
                    TriplePattern::parse("?p", "cross_agent_approved", "true"),
                    TriplePattern::parse("?p", "comments_resolved", "true"),
                ],
                vec![TriplePattern::parse("?p", "mergeable", "true")],
            ),
        ),
    ]
}

fn facts() -> Vec<Triple> {
    vec![
        t("m5", "distinct_from", "mini"),
        t("prop1", "authored_by", "m5"),
        t("prop1", "has_approval", "mini"),
        t("prop1", "comments_resolved", "true"),
        t("prop2", "authored_by", "dgx"),
        t("prop2", "has_approval", "dgx"),
    ]
}

#[test]
fn gate_proves_derivable_policy_claims_with_their_derivation() {
    let gate = ProvableClaimGate::new(forward_chain(&rules(), facts()));

    let verdict = gate.gate(&t("prop1", "mergeable", "true"));
    assert!(verdict.is_proven());
    let rendered = verdict.render();
    // The rendered answer carries the rule chain that justifies the merge.
    assert!(rendered.contains("PROVEN"));
    assert!(rendered.contains("mergeable"));

    let violation = gate.gate(&t("prop2", "violates_policy", "self-approval"));
    assert!(violation.is_proven());
    assert!(
        violation
            .proof()
            .expect("proof present")
            .rule_ids()
            .contains(&"no-self-approve")
    );
}

#[test]
fn gate_abstains_loudly_on_underivable_policy_claims() {
    let gate = ProvableClaimGate::new(forward_chain(&rules(), facts()));

    // prop2 is self-approved: mergeable is NOT derivable and must never be asserted.
    let verdict = gate.gate(&t("prop2", "mergeable", "true"));
    assert!(!verdict.is_proven());
    assert!(verdict.render().contains("UNPROVEN"));

    // A proposal the graph has never seen: flagged, not guessed.
    assert!(!gate.gate(&t("prop99", "mergeable", "true")).is_proven());

    // Batch summary: exactly the derivable claims are proven, everything else flagged.
    let summary = gate.summarize(&[
        t("prop1", "mergeable", "true"),
        t("prop2", "violates_policy", "self-approval"),
        t("prop2", "mergeable", "true"),
        t("prop99", "mergeable", "true"),
    ]);
    assert_eq!(summary.proven, 2);
    assert_eq!(summary.unproven, 2);
}

#[test]
fn arrow_backed_gate_gives_identical_policy_verdicts() {
    let vec_gate = ProvableClaimGate::new(forward_chain(&rules(), facts()));
    let arrow_gate = ProvableClaimGate::from_arrow(forward_chain_arrow(&rules(), facts()));

    for claim in [
        t("prop1", "mergeable", "true"),
        t("prop2", "violates_policy", "self-approval"),
        t("prop2", "mergeable", "true"),
        t("prop99", "mergeable", "true"),
    ] {
        assert_eq!(
            vec_gate.gate(&claim).is_proven(),
            arrow_gate.gate(&claim).is_proven(),
            "backend verdict mismatch on {claim:?}"
        );
    }
}
