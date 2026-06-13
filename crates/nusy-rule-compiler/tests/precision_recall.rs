//! The named missing test from `V18-PORTMAP-engine-crates.md` §6 (nusy-cortex,
//! cross-crate-finding #3): **computable-rule-extraction precision/recall** — cortex-extracted
//! rule clauses → CQL-analog / forward-chain rules → forward-chain execution **reproduces the
//! extracted facts**. This is the seam that proves the extraction → compilation → execution
//! contract end-to-end.
//!
//! The fixtures here mirror the clinical gold cases (`nusy-clinical-fixtures::cases`) but build
//! the rules through the compiler rather than constructing `IdRule` directly. A perfect score
//! means the compiler is the identity transform across the contract: anything the engine could
//! derive from a hand-written `IdRule` it also derives from a compiled `RuleClause`.

use nusy_forward_chain::{Saturation, forward_chain};
use nusy_rule_compiler::{NamedRuleExt, RuleClause, compile_bundle, compile_rule};
use nusy_unify::Triple;

fn t(s: &str, p: &str, o: &str) -> Triple {
    Triple::new(s, p, o)
}

/// (clauses, seeds, expected_derived) for one end-to-end case.
struct EndToEndCase {
    name: &'static str,
    clauses: Vec<RuleClause>,
    seeds: Vec<Triple>,
    expected_derived: Vec<Triple>,
}

/// Two-step fall-risk: at-risk-fall, then recommend-fall-assessment (mirrors the clinical-fixtures
/// `fall_risk_assessment_fires` case, but built via the compiler).
fn fall_risk_case() -> EndToEndCase {
    EndToEndCase {
        name: "fall_risk_assessment_fires",
        clauses: vec![
            RuleClause::new(
                "at-risk-fall",
                ["?p has_condition ?c", "?c increases_fall_risk true"],
                ["?p at_risk fall"],
            ),
            RuleClause::new(
                "recommend-fall-assessment",
                ["?p at_risk fall", "?p age_band over_65"],
                ["?p recommend fall_assessment"],
            ),
        ],
        seeds: vec![
            t("p1", "has_condition", "osteoporosis"),
            t("osteoporosis", "increases_fall_risk", "true"),
            t("p1", "age_band", "over_65"),
        ],
        expected_derived: vec![
            t("p1", "at_risk", "fall"),
            t("p1", "recommend", "fall_assessment"),
        ],
    }
}

/// Glycemic intensification gated on renal status (mirrors
/// `glycemic_intensify_renal_contraindication`). patient3's candidacy is correctly derivable;
/// only patient4's recommendation fires (renal_status=adequate).
fn glycemic_case() -> EndToEndCase {
    EndToEndCase {
        name: "glycemic_intensify_renal_contraindication",
        clauses: vec![
            RuleClause::new(
                "intensify-candidate",
                [
                    "?p has_condition type2_diabetes",
                    "?p has_observation hba1c_elevated",
                ],
                ["?p intensify_candidate glycemic"],
            ),
            RuleClause::new(
                "recommend-intensify",
                [
                    "?p intensify_candidate glycemic",
                    "?p renal_status adequate",
                ],
                ["?p recommend intensify_glycemic"],
            ),
        ],
        seeds: vec![
            t("patient3", "has_condition", "type2_diabetes"),
            t("patient3", "has_observation", "hba1c_elevated"),
            t("patient3", "renal_status", "impaired"),
            t("patient4", "has_condition", "type2_diabetes"),
            t("patient4", "has_observation", "hba1c_elevated"),
            t("patient4", "renal_status", "adequate"),
        ],
        expected_derived: vec![
            t("patient3", "intensify_candidate", "glycemic"),
            t("patient4", "intensify_candidate", "glycemic"),
            t("patient4", "recommend", "intensify_glycemic"),
        ],
    }
}

/// Recursive kinship: a 4-node chain p0→p1→p2→p3 with base + recursive ancestor and a
/// grandparent rule. Tests transitive closure compiled through the rule compiler.
fn kinship_case() -> EndToEndCase {
    let clauses = vec![
        RuleClause::new("ancestor-base", ["?x parent ?y"], ["?x ancestor ?y"]),
        RuleClause::new(
            "ancestor-rec",
            ["?x parent ?y", "?y ancestor ?z"],
            ["?x ancestor ?z"],
        ),
        RuleClause::new(
            "grandparent",
            ["?x parent ?y", "?y parent ?z"],
            ["?x grandparent ?z"],
        ),
    ];
    let seeds = vec![
        t("p0", "parent", "p1"),
        t("p1", "parent", "p2"),
        t("p2", "parent", "p3"),
    ];
    let mut expected_derived = Vec::new();
    // C(4,2) = 6 ancestors over a 4-node chain.
    for i in 0..4 {
        for j in (i + 1)..4 {
            expected_derived.push(Triple::new(format!("p{i}"), "ancestor", format!("p{j}")));
        }
    }
    // 2 grandparents: p0→p2, p1→p3.
    expected_derived.push(t("p0", "grandparent", "p2"));
    expected_derived.push(t("p1", "grandparent", "p3"));

    EndToEndCase {
        name: "kinship_closure_4_node_chain",
        clauses,
        seeds,
        expected_derived,
    }
}

fn run_case(case: &EndToEndCase) -> (usize, usize, usize, Saturation) {
    let compiled: Vec<_> = case
        .clauses
        .iter()
        .map(|c| compile_rule(c).expect("compiles").to_id_rule())
        .collect();
    let sat = forward_chain(&compiled, case.seeds.clone());
    let derived: Vec<&Triple> = sat.derivations.iter().map(|d| &d.conclusion).collect();
    let true_positive = derived
        .iter()
        .filter(|f| case.expected_derived.contains(f))
        .count();
    let false_positive = derived.len() - true_positive;
    let false_negative = case
        .expected_derived
        .iter()
        .filter(|f| !sat.contains(f))
        .count();
    (true_positive, false_positive, false_negative, sat)
}

#[test]
fn fall_risk_round_trips_perfectly() {
    let case = fall_risk_case();
    let (tp, fp, fn_, _) = run_case(&case);
    assert_eq!(
        tp,
        case.expected_derived.len(),
        "case {}: missed",
        case.name
    );
    assert_eq!(fp, 0, "case {}: spurious derivation", case.name);
    assert_eq!(fn_, 0, "case {}: missed derivation", case.name);
}

#[test]
fn glycemic_round_trips_perfectly_and_renal_impaired_is_not_recommended() {
    let case = glycemic_case();
    let (tp, fp, fn_, sat) = run_case(&case);
    assert_eq!(tp, case.expected_derived.len());
    assert_eq!(fp, 0);
    assert_eq!(fn_, 0);
    // Contraindication: patient3 must NOT be recommended intensification.
    assert!(!sat.contains(&t("patient3", "recommend", "intensify_glycemic")));
}

#[test]
fn kinship_round_trips_full_transitive_closure() {
    let case = kinship_case();
    let (tp, fp, fn_, _) = run_case(&case);
    assert_eq!(
        tp,
        case.expected_derived.len(),
        "case {}: missed",
        case.name
    );
    assert_eq!(fp, 0, "case {}: spurious derivation", case.name);
    assert_eq!(fn_, 0, "case {}: missed derivation", case.name);
}

#[test]
fn aggregate_precision_and_recall_are_perfect_across_all_cases() {
    let cases = [fall_risk_case(), glycemic_case(), kinship_case()];
    let mut total_tp = 0usize;
    let mut total_fp = 0usize;
    let mut total_fn = 0usize;
    let mut total_expected = 0usize;
    let mut total_derived = 0usize;
    for case in &cases {
        let (tp, fp, fn_, sat) = run_case(case);
        total_tp += tp;
        total_fp += fp;
        total_fn += fn_;
        total_expected += case.expected_derived.len();
        total_derived += sat.derivations.len();
    }
    // Total ground-truth count: 2 (fall) + 3 (glycemic) + 8 (kinship: 6 ancestors + 2 grandparents) = 13.
    assert_eq!(total_expected, 13);
    assert_eq!(total_derived, 13);
    let precision = total_tp as f64 / total_derived as f64;
    let recall = total_tp as f64 / total_expected as f64;
    assert!(
        (precision - 1.0).abs() < 1e-9,
        "precision = {precision}, expected 1.0"
    );
    assert!(
        (recall - 1.0).abs() < 1e-9,
        "recall = {recall}, expected 1.0"
    );
    assert_eq!(total_fp, 0);
    assert_eq!(total_fn, 0);
}

#[test]
fn rule_ids_flow_through_to_derivation_provenance() {
    // The compiled rule's id must reach Derivation::rule_id so downstream consumers
    // (proof.rule_ids(), nusy_router::RulePath) can identify the rule that fired.
    let case = fall_risk_case();
    let compiled: Vec<_> = case
        .clauses
        .iter()
        .map(|c| compile_rule(c).expect("compiles").to_id_rule())
        .collect();
    let sat = forward_chain(&compiled, case.seeds);
    let at_risk = t("p1", "at_risk", "fall");
    let recommend = t("p1", "recommend", "fall_assessment");
    assert_eq!(
        sat.derivation_of(&at_risk).expect("derived").rule_id,
        "at-risk-fall"
    );
    assert_eq!(
        sat.derivation_of(&recommend).expect("derived").rule_id,
        "recommend-fall-assessment"
    );
}

#[test]
fn bundle_compilation_round_trips_through_forward_chain() {
    // Compile a multi-rule bundle (the cog-transfer path) and run it end-to-end.
    let case = fall_risk_case();
    let bundle = compile_bundle(&case.clauses, &[], &[]).expect("bundle compiles");
    assert_eq!(bundle.rules.len(), 2);
    let compiled: Vec<_> = bundle.rules.iter().map(|r| r.to_id_rule()).collect();
    let sat = forward_chain(&compiled, case.seeds);
    for expected in &case.expected_derived {
        assert!(sat.contains(expected), "missing {expected:?}");
    }
}
