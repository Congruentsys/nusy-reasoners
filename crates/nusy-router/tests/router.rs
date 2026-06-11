//! The router/gate contract: the pre-filter routes in-domain claims to the symbolic gate,
//! out-of-domain claims straight to neural (skipping the engine entirely), and the
//! rule-path surfacer extracts a stable identifier from the gate's proof — the end-to-end
//! "route → gate → classify" pipeline the V18 caller assembles.

use nusy_forward_chain::{IdRule, forward_chain};
use nusy_gate::{GateResponse, ProvableClaimGate};
use nusy_router::{RouteClassifier, RouteDecision, RulePath};
use nusy_unify::{Rule, Triple, TriplePattern};

fn t(s: &str, p: &str, o: &str) -> Triple {
    Triple::new(s, p, o)
}

/// Two-step fall-risk pipeline (matches the gate crate's fixture): at_risk ← condition+risk,
/// recommend ← at_risk+age.
fn fall_engine() -> (Vec<IdRule>, ProvableClaimGate, RouteClassifier) {
    let at_risk = IdRule::new(
        "at-risk-fall",
        Rule::new(
            vec![
                TriplePattern::parse("?p", "has_condition", "?c"),
                TriplePattern::parse("?c", "increases_fall_risk", "true"),
            ],
            vec![TriplePattern::parse("?p", "at_risk", "fall")],
        ),
    );
    let recommend = IdRule::new(
        "recommend-fall-assessment",
        Rule::new(
            vec![
                TriplePattern::parse("?p", "at_risk", "fall"),
                TriplePattern::parse("?p", "age_band", "over_65"),
            ],
            vec![TriplePattern::parse("?p", "recommend", "fall_assessment")],
        ),
    );
    let rules = vec![at_risk, recommend];
    let facts = vec![
        t("p1", "has_condition", "osteoporosis"),
        t("osteoporosis", "increases_fall_risk", "true"),
        t("p1", "age_band", "over_65"),
    ];
    let sat = forward_chain(&rules, facts);
    let gate = ProvableClaimGate::new(sat.clone());
    let router = RouteClassifier::from_engine(&sat, &rules);
    (rules, gate, router)
}

#[test]
fn in_domain_claim_routes_to_gate_and_is_proven_with_path() {
    let (_, gate, router) = fall_engine();
    let claim = t("p1", "recommend", "fall_assessment");

    // The router says: invoke the gate.
    assert!(matches!(router.classify(&claim), RouteDecision::Symbolic));

    // The gate proves it.
    let resp = gate.gate(&claim);
    assert!(resp.is_proven());

    // The router surfaces the rule path used by the proof: outer rule first, then
    // the rule for the recursive premise.
    let path = RulePath::from_response(&resp).expect("proven → has a path");
    assert_eq!(path.depth(), 2);
    assert_eq!(
        path.rule_ids,
        vec!["recommend-fall-assessment", "at-risk-fall"]
    );
    assert_eq!(path.id(), "recommend-fall-assessment>at-risk-fall");
}

#[test]
fn out_of_domain_claim_skips_gate() {
    let (_, gate, router) = fall_engine();
    let foreign = t("p1", "weather_today", "sunny");

    // The pre-filter rules this out — the engine has no facts or rules about weather.
    match router.classify(&foreign) {
        RouteDecision::Neural { reason } => {
            assert!(reason.contains("weather_today"));
            assert!(reason.contains("symbolic domain"));
        }
        RouteDecision::Symbolic => panic!("out-of-domain predicate must route to neural"),
    }

    // Sanity: if you DID invoke the gate, it would have returned Unproven — the router
    // saves the engine call, the verdict would be the same flagging.
    assert!(!gate.gate(&foreign).is_proven());
}

#[test]
fn in_domain_unprovable_claim_still_routes_to_gate_and_is_flagged() {
    // The pre-filter is about the predicate domain, not whether the specific claim is
    // provable — it lets the gate be the authority on provability.
    let (_, gate, router) = fall_engine();
    let in_domain_but_false = t("p2", "recommend", "fall_assessment"); // unknown patient

    // Router: in domain → symbolic. The gate is the right authority for this claim.
    assert!(matches!(
        router.classify(&in_domain_but_false),
        RouteDecision::Symbolic
    ));

    // Gate: not derivable → flagged.
    let resp = gate.gate(&in_domain_but_false);
    assert!(!resp.is_proven());
    assert!(RulePath::from_response(&resp).is_none());
    match resp {
        GateResponse::Unproven { .. } => {}
        GateResponse::Proven { .. } => panic!("hallucination: provable on unknown patient"),
    }
}

#[test]
fn batch_routes_partition_into_symbolic_and_neural() {
    let (_, _gate, router) = fall_engine();
    let claims = vec![
        t("p1", "recommend", "fall_assessment"), // in domain
        t("p1", "weather_today", "sunny"),       // out of domain
        t("p1", "at_risk", "fall"),              // in domain
        t("p1", "stock_price", "100"),           // out of domain
    ];
    let decisions = router.classify_all(&claims);
    assert_eq!(decisions.len(), 4);
    assert!(decisions[0].is_symbolic());
    assert!(decisions[1].is_neural());
    assert!(decisions[2].is_symbolic());
    assert!(decisions[3].is_neural());

    let summary = router.summarize(&claims);
    assert_eq!(summary.symbolic, 2);
    assert_eq!(summary.neural, 2);
}

#[test]
fn rule_path_tallies_across_a_batch_dedup_by_path() {
    // Two patients hit the same rule chain — their proofs share a path id.
    let at_risk = IdRule::new(
        "at-risk-fall",
        Rule::new(
            vec![
                TriplePattern::parse("?p", "has_condition", "?c"),
                TriplePattern::parse("?c", "increases_fall_risk", "true"),
            ],
            vec![TriplePattern::parse("?p", "at_risk", "fall")],
        ),
    );
    let facts = vec![
        t("p1", "has_condition", "osteoporosis"),
        t("p2", "has_condition", "osteoporosis"),
        t("osteoporosis", "increases_fall_risk", "true"),
    ];
    let sat = forward_chain(&[at_risk], facts);
    let gate = ProvableClaimGate::new(sat);

    let claims = vec![t("p1", "at_risk", "fall"), t("p2", "at_risk", "fall")];
    let mut tally: std::collections::HashMap<RulePath, usize> = std::collections::HashMap::new();
    for c in &claims {
        let resp = gate.gate(c);
        if let Some(path) = RulePath::from_response(&resp) {
            *tally.entry(path).or_insert(0) += 1;
        }
    }
    // Both proofs used the same rule path — one bucket, count 2.
    assert_eq!(tally.len(), 1);
    let (path, count) = tally.into_iter().next().unwrap();
    assert_eq!(path.id(), "at-risk-fall");
    assert_eq!(count, 2);
}
