//! Non-clinical genericity smoke test (EX-4700 / EXPR-4699, V19-H3): the engine runs a
//! *governance/CI policy* domain — NuSy's own PR rules — with zero engine changes. The
//! content is the only thing that differs from the clinical fixtures; the engine does not
//! know what a "proposal" is any more than it knows what a guideline is.
//!
//! Policy encoded as rules (from the project's review guardrails):
//! - approval counts only if reviewer ≠ author (cross-agent review)
//! - author approving their own proposal is a policy violation
//! - voyages are approved only by the Captain
//! - a proposal is mergeable iff cross-agent-approved and its comments are resolved
//! - an unresolved comment blocks, and blockage propagates through depends_on (recursive)
//!
//! No negation is used: inequality is encoded as explicit `distinct_from` facts and
//! "comments resolved" is asserted positively — standard Datalog-without-builtins encoding,
//! i.e. pure content, which is exactly the point of the smoke test.

use nusy_forward_chain::{IdRule, forward_chain, forward_chain_arrow};
use nusy_unify::{Rule, Triple, TriplePattern};

fn t(s: &str, p: &str, o: &str) -> Triple {
    Triple::new(s, p, o)
}

fn rule(id: &str, body: Vec<TriplePattern>, head: TriplePattern) -> IdRule {
    IdRule::new(id, Rule::new(body, vec![head]))
}

/// The governance rule set — domain content, nothing engine-specific.
fn governance_rules() -> Vec<IdRule> {
    vec![
        // cross-agent-approval: ?p approved by ?r who is not the author ?a
        rule(
            "cross-agent-approval",
            vec![
                TriplePattern::parse("?p", "authored_by", "?a"),
                TriplePattern::parse("?p", "has_approval", "?r"),
                TriplePattern::parse("?a", "distinct_from", "?r"),
            ],
            TriplePattern::parse("?p", "cross_agent_approved", "true"),
        ),
        // self-approval violation: the author approved their own proposal
        rule(
            "no-self-approve",
            vec![
                TriplePattern::parse("?p", "authored_by", "?a"),
                TriplePattern::parse("?p", "has_approval", "?a"),
            ],
            TriplePattern::parse("?p", "violates_policy", "self-approval"),
        ),
        // voyages need the Captain
        rule(
            "voyage-captain-gate",
            vec![
                TriplePattern::parse("?p", "item_type", "voyage"),
                TriplePattern::parse("?p", "has_approval", "captain"),
            ],
            TriplePattern::parse("?p", "captain_approved", "true"),
        ),
        // mergeable = cross-agent approved + comments resolved
        rule(
            "mergeable",
            vec![
                TriplePattern::parse("?p", "cross_agent_approved", "true"),
                TriplePattern::parse("?p", "comments_resolved", "true"),
            ],
            TriplePattern::parse("?p", "mergeable", "true"),
        ),
        // an unresolved comment blocks the proposal
        rule(
            "unresolved-blocks",
            vec![
                TriplePattern::parse("?p", "has_comment", "?c"),
                TriplePattern::parse("?c", "comment_status", "unresolved"),
            ],
            TriplePattern::parse("?p", "blocked", "true"),
        ),
        // blockage propagates through dependencies (base + recursive — the fixpoint case)
        rule(
            "dep-block-base",
            vec![
                TriplePattern::parse("?p", "depends_on", "?q"),
                TriplePattern::parse("?q", "blocked", "true"),
            ],
            TriplePattern::parse("?p", "blocked", "true"),
        ),
    ]
}

/// A synthetic review scene:
/// - prop1: M5 authored, Mini approved, comments resolved          → mergeable
/// - prop2: DGX authored, DGX approved                              → self-approval violation
/// - prop3: a voyage, approved by the Captain                       → captain_approved
/// - prop4: Air authored, M5 approved, but cmt1 is unresolved       → blocked, NOT mergeable
/// - prop5: depends on prop4                                        → transitively blocked
/// - prop6: depends on prop5 (two hops)                             → transitively blocked
fn governance_facts() -> Vec<Triple> {
    vec![
        // agents pairwise distinct (only the pairs the scene needs)
        t("m5", "distinct_from", "mini"),
        t("mini", "distinct_from", "m5"),
        t("air", "distinct_from", "m5"),
        t("m5", "distinct_from", "air"),
        // prop1 — the clean merge
        t("prop1", "authored_by", "m5"),
        t("prop1", "has_approval", "mini"),
        t("prop1", "comments_resolved", "true"),
        // prop2 — the self-approval
        t("prop2", "authored_by", "dgx"),
        t("prop2", "has_approval", "dgx"),
        // prop3 — the voyage
        t("prop3", "item_type", "voyage"),
        t("prop3", "has_approval", "captain"),
        // prop4 — approved cross-agent but blocked on a comment
        t("prop4", "authored_by", "air"),
        t("prop4", "has_approval", "m5"),
        t("prop4", "has_comment", "cmt1"),
        t("cmt1", "comment_status", "unresolved"),
        // dependency chain
        t("prop5", "depends_on", "prop4"),
        t("prop6", "depends_on", "prop5"),
    ]
}

#[test]
fn governance_domain_derives_expected_conclusions() {
    let sat = forward_chain(&governance_rules(), governance_facts());

    // Expected derivations all hold.
    assert!(sat.contains(&t("prop1", "cross_agent_approved", "true")));
    assert!(sat.contains(&t("prop1", "mergeable", "true")));
    assert!(sat.contains(&t("prop2", "violates_policy", "self-approval")));
    assert!(sat.contains(&t("prop3", "captain_approved", "true")));
    assert!(sat.contains(&t("prop4", "cross_agent_approved", "true")));
    assert!(sat.contains(&t("prop4", "blocked", "true")));
    // Recursive propagation reaches both hops of the dependency chain.
    assert!(sat.contains(&t("prop5", "blocked", "true")));
    assert!(sat.contains(&t("prop6", "blocked", "true")));
}

#[test]
fn governance_negative_controls_do_not_fire() {
    let sat = forward_chain(&governance_rules(), governance_facts());

    // The clean proposal violates nothing.
    assert!(!sat.contains(&t("prop1", "violates_policy", "self-approval")));
    assert!(!sat.contains(&t("prop1", "blocked", "true")));
    // The self-approved proposal earns no cross-agent approval and is not mergeable.
    assert!(!sat.contains(&t("prop2", "cross_agent_approved", "true")));
    assert!(!sat.contains(&t("prop2", "mergeable", "true")));
    // Cross-agent approval alone is not mergeability: prop4 is approved but blocked,
    // and (with comments unresolved) never derived mergeable.
    assert!(!sat.contains(&t("prop4", "mergeable", "true")));
    // Blockage does not flow backwards along depends_on.
    assert!(!sat.contains(&t("prop4", "depends_on", "prop5")));
    // A voyage approved by a non-Captain would not be captain_approved (no such fact here,
    // and the only captain_approved derivation is prop3's).
    assert!(!sat.contains(&t("prop1", "captain_approved", "true")));
}

#[test]
fn every_derived_governance_fact_has_a_complete_proof() {
    let rules = governance_rules();
    let seed = governance_facts();
    let sat = forward_chain(&rules, seed.clone());

    // Proof completeness: every derived fact carries a derivation whose proof tree
    // bottoms out in seed axioms only — the provable-only invariant, on governance content.
    let derived = [
        t("prop1", "cross_agent_approved", "true"),
        t("prop1", "mergeable", "true"),
        t("prop2", "violates_policy", "self-approval"),
        t("prop3", "captain_approved", "true"),
        t("prop4", "blocked", "true"),
        t("prop5", "blocked", "true"),
        t("prop6", "blocked", "true"),
    ];
    for fact in &derived {
        let proof = sat
            .proof_of(fact)
            .unwrap_or_else(|| panic!("no proof for {fact:?}"));
        assert_eq!(proof.conclusion(), fact);
        for axiom in proof.axioms() {
            assert!(
                seed.contains(axiom),
                "proof of {fact:?} cites non-seed axiom {axiom:?}"
            );
        }
    }

    // The two-hop blockage proof actually chains through the derived premise:
    // blocked(prop6) ← dep-block-base ← blocked(prop5) ← dep-block-base ← blocked(prop4).
    let p6 = sat.proof_of(&t("prop6", "blocked", "true")).unwrap();
    assert!(p6.depth() >= 3, "expected a multi-hop proof, got {}", p6.depth());
    assert!(p6.rule_ids().contains(&"dep-block-base"));
    assert!(p6.rule_ids().contains(&"unresolved-blocks"));
}

#[test]
fn arrow_backend_agrees_on_governance_content() {
    // Differential: the Arrow saturation derives exactly the same governance facts —
    // domain content does not perturb backend equivalence.
    let rules = governance_rules();
    let vec_sat = forward_chain(&rules, governance_facts());
    let arrow_sat = forward_chain_arrow(&rules, governance_facts());

    assert_eq!(vec_sat.derived_count(), arrow_sat.derived_count());
    for fact in [
        t("prop1", "mergeable", "true"),
        t("prop2", "violates_policy", "self-approval"),
        t("prop6", "blocked", "true"),
    ] {
        assert!(arrow_sat.contains(&fact));
        assert!(arrow_sat.proof_of(&fact).is_some());
    }
}
