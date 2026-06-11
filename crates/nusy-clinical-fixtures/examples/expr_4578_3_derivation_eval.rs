//! EXPR-4578.3 — derivation correctness vs ground-truth fixture (VOY-V18-1 eval).
//!
//! Runs the engine (`nusy-forward-chain`) over fixed Y-graph fixtures with **hand-derived
//! complete closures** and scores the output against that ground truth:
//!
//! - **M-4620 derivation precision** — fraction of engine-derived facts in the ground truth.
//! - **M-4621 derivation recall** — fraction of ground-truth derivations the engine finds.
//! - **M-4622 proof completeness** — fraction of derived facts whose proof tree resolves to
//!   seed-fact axioms AND whose surfaced [`nusy_provenance::Provenance`] is grounded.
//! - **M-4623 re-derive latency** — median ms to re-derive after a single triple change.
//!   **Caveat:** the engine is full-refire (no incremental mode until EX-4593), so this is
//!   the full-refire baseline EX-4593 must beat, not an incremental measurement.
//!
//! Cases: the three clinical gold cases from [`nusy_clinical_fixtures::gold_cases`] (closures
//! hand-derived below, keyed by fixture name so a fixture change breaks this eval loudly)
//! plus one synthetic kinship-closure case (6-node parent chain; ancestor + grandparent
//! rules) whose 19-fact closure is exhaustively hand-computed for statistical volume.
//!
//! Emits the eval JSON on stdout (redirect to `research/shared/eval-data/v18-expr-4578.3/`)
//! and a human-readable summary on stderr.
//!
//! ```bash
//! cargo run --release -p nusy-clinical-fixtures --example expr_4578_3_derivation_eval \
//!     > research/shared/eval-data/v18-expr-4578.3/DERIVATION-CORRECTNESS.json
//! ```

use std::time::Instant;

use nusy_clinical_fixtures::{IdRule, Rule, TriplePattern, gold_cases};
use nusy_forward_chain::{Saturation, forward_chain};
use nusy_provenance::surface;
use nusy_unify::Triple;

const TRIALS_PER_CASE: usize = 200;
const WARMUP_RUNS: usize = 10;

fn t(s: &str, p: &str, o: &str) -> Triple {
    Triple::new(s, p, o)
}

/// One eval case: a fixture plus its hand-derived complete expected closure.
struct EvalCase {
    name: String,
    seeds: Vec<Triple>,
    rules: Vec<IdRule>,
    /// COMPLETE set of facts the engine must derive (hand-computed closure minus seeds).
    expected_derived: Vec<Triple>,
    /// Facts that must NOT be derivable (contraindications + negative controls).
    must_not_derive: Vec<Triple>,
    /// A single-triple change that re-triggers derivation, for the M-4623 latency probe.
    /// `k` varies per trial so subjects differ across trials where it matters.
    delta: fn(k: usize) -> Triple,
}

/// The three clinical gold cases with their hand-derived complete closures.
///
/// Closures derived by hand from `cases.rs` (2026-06-11): each rule body is matched
/// against the seed facts; conclusions are added and matching repeats to fixpoint.
fn clinical_cases() -> Vec<EvalCase> {
    let mut out = Vec::new();
    for fx in gold_cases() {
        let (expected_derived, delta): (Vec<Triple>, fn(usize) -> Triple) = match fx.name.as_str() {
            // at-risk-fall fires for patient1 (osteoporosis ↑ fall risk), then
            // recommend-fall-assessment fires (at_risk + over_65). Fixpoint after 2 facts.
            "fall_risk_assessment_fires" => (
                vec![
                    t("patient1", "at_risk", "fall"),
                    t("patient1", "recommend", "fall_assessment"),
                ],
                |k| Triple::new(format!("delta_patient{k}"), "has_condition", "osteoporosis"),
            ),
            // at-risk-fall fires for patient2; recommend does NOT (no age_band seed).
            "fall_risk_no_age_does_not_fire" => (vec![t("patient2", "at_risk", "fall")], |k| {
                Triple::new(format!("delta_patient{k}"), "has_condition", "osteoporosis")
            }),
            // intensify-candidate fires for patient3 AND patient4 (both diabetic with
            // elevated hba1c); recommend-intensify fires only for patient4 (renal adequate).
            // patient3's candidacy is correctly derivable — only the recommendation is gated.
            "glycemic_intensify_renal_contraindication" => (
                vec![
                    t("patient3", "intensify_candidate", "glycemic"),
                    t("patient4", "intensify_candidate", "glycemic"),
                    t("patient4", "recommend", "intensify_glycemic"),
                ],
                // Adding renal adequacy for patient3 re-triggers recommend-intensify.
                |_| Triple::new("patient3", "renal_status", "adequate"),
            ),
            other => panic!("gold case '{other}' has no hand-derived closure in this eval"),
        };
        let must_not = fx
            .contraindicated
            .iter()
            .chain(fx.negative_controls.iter())
            .cloned()
            .collect();
        out.push(EvalCase {
            name: fx.name,
            seeds: fx.patient_facts,
            rules: fx.rules,
            expected_derived,
            must_not_derive: must_not,
            delta,
        });
    }
    out
}

/// Synthetic kinship closure: p0→p1→…→p5 parent chain; ancestor (base + recursive) and
/// grandparent rules. Hand-computed closure: ancestor(pi,pj) for all 0≤i<j≤5 (C(6,2)=15)
/// plus grandparent(pi,pi+2) for i=0..=3 (4) — 19 derived facts.
fn kinship_case() -> EvalCase {
    let rules = vec![
        IdRule::new(
            "ancestor-base",
            Rule::new(
                vec![TriplePattern::parse("?x", "parent", "?y")],
                vec![TriplePattern::parse("?x", "ancestor", "?y")],
            ),
        ),
        IdRule::new(
            "ancestor-rec",
            Rule::new(
                vec![
                    TriplePattern::parse("?x", "parent", "?y"),
                    TriplePattern::parse("?y", "ancestor", "?z"),
                ],
                vec![TriplePattern::parse("?x", "ancestor", "?z")],
            ),
        ),
        IdRule::new(
            "grandparent",
            Rule::new(
                vec![
                    TriplePattern::parse("?x", "parent", "?y"),
                    TriplePattern::parse("?y", "parent", "?z"),
                ],
                vec![TriplePattern::parse("?x", "grandparent", "?z")],
            ),
        ),
    ];
    let seeds: Vec<Triple> = (0..5)
        .map(|i| Triple::new(format!("p{i}"), "parent", format!("p{}", i + 1)))
        .collect();
    let mut expected = Vec::new();
    for i in 0..6 {
        for j in (i + 1)..6 {
            expected.push(Triple::new(format!("p{i}"), "ancestor", format!("p{j}")));
        }
    }
    for i in 0..4 {
        expected.push(Triple::new(
            format!("p{i}"),
            "grandparent",
            format!("p{}", i + 2),
        ));
    }
    EvalCase {
        name: "kinship_closure_6_node_chain".to_string(),
        seeds,
        rules,
        expected_derived: expected,
        must_not_derive: vec![
            t("p5", "ancestor", "p0"),    // closure must not invert
            t("p0", "grandparent", "p1"), // depth-1 is not a grandparent
            t("p0", "ancestor", "p0"),    // no reflexive ancestry
        ],
        // A new leaf edge re-triggers the recursive closure (6 ancestors + 1 grandparent).
        delta: |k| Triple::new("p5", "parent", format!("q{k}")),
    }
}

struct CaseResult {
    name: String,
    seed_count: usize,
    derived_count: usize,
    expected_count: usize,
    true_positive: usize,
    false_positive: usize,
    false_negative: usize,
    proof_complete: usize,
    false_provable: usize,
    must_not_count: usize,
    initial_saturation_ms: f64,
    rederive_median_ms: f64,
    rederive_p95_ms: f64,
}

/// Proof completeness for one derived fact: a proof tree exists, every axiom leaf is a seed
/// fact, and the surfaced provenance is grounded (every premise justified before use).
fn proof_is_complete(sat: &Saturation, seeds: &[Triple], fact: &Triple) -> bool {
    let Some(tree) = sat.proof_of(fact) else {
        return false;
    };
    if !tree.axioms().iter().all(|ax| seeds.contains(ax)) {
        return false;
    }
    surface(sat, fact).is_some_and(|prov| prov.is_grounded())
}

fn percentile(sorted_ms: &[f64], p: f64) -> f64 {
    if sorted_ms.is_empty() {
        return 0.0;
    }
    let idx = ((sorted_ms.len() as f64 - 1.0) * p).round() as usize;
    sorted_ms[idx]
}

fn run_case(case: &EvalCase) -> CaseResult {
    // Initial saturation (timed once after warmup).
    for _ in 0..WARMUP_RUNS {
        let _ = forward_chain(&case.rules, case.seeds.clone());
    }
    let start = Instant::now();
    let sat = forward_chain(&case.rules, case.seeds.clone());
    let initial_saturation_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Correctness: engine-derived set vs hand-derived ground truth.
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

    // M-4622: every derived fact must carry a full proof back to seed axioms.
    let proof_complete = derived
        .iter()
        .filter(|f| proof_is_complete(&sat, &case.seeds, f))
        .count();

    // False-provable: contraindications / negative controls that the engine derived.
    let false_provable = case
        .must_not_derive
        .iter()
        .filter(|f| sat.contains(f))
        .count();

    // M-4623: re-derive after a single triple change (full-refire baseline; see header).
    let mut trial_ms: Vec<f64> = Vec::with_capacity(TRIALS_PER_CASE);
    for k in 0..TRIALS_PER_CASE {
        let mut perturbed = case.seeds.clone();
        perturbed.push((case.delta)(k));
        let start = Instant::now();
        let resat = forward_chain(&case.rules, perturbed);
        trial_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        assert!(
            resat.facts.len() >= sat.facts.len(),
            "perturbed saturation lost facts in case {}",
            case.name
        );
    }
    trial_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());

    CaseResult {
        name: case.name.clone(),
        seed_count: case.seeds.len(),
        derived_count: derived.len(),
        expected_count: case.expected_derived.len(),
        true_positive,
        false_positive,
        false_negative,
        proof_complete,
        false_provable,
        must_not_count: case.must_not_derive.len(),
        initial_saturation_ms,
        rederive_median_ms: percentile(&trial_ms, 0.5),
        rederive_p95_ms: percentile(&trial_ms, 0.95),
    }
}

fn ratio(num: usize, den: usize) -> f64 {
    if den == 0 {
        1.0
    } else {
        num as f64 / den as f64
    }
}

fn main() {
    let mut cases = clinical_cases();
    cases.push(kinship_case());
    let results: Vec<CaseResult> = cases.iter().map(run_case).collect();

    let tp: usize = results.iter().map(|r| r.true_positive).sum();
    let derived: usize = results.iter().map(|r| r.derived_count).sum();
    let expected: usize = results.iter().map(|r| r.expected_count).sum();
    let proof_complete: usize = results.iter().map(|r| r.proof_complete).sum();
    let false_provable: usize = results.iter().map(|r| r.false_provable).sum();
    let must_not: usize = results.iter().map(|r| r.must_not_count).sum();
    let mut medians: Vec<f64> = results.iter().map(|r| r.rederive_median_ms).collect();
    medians.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let worst_median = *medians.last().unwrap();

    let precision = ratio(tp, derived);
    let recall = ratio(tp, expected);
    let completeness = ratio(proof_complete, derived);
    let false_provable_rate = false_provable as f64 / must_not.max(1) as f64;

    // Human summary → stderr; JSON → stdout.
    eprintln!(
        "EXPR-4578.3 derivation correctness — {} cases",
        results.len()
    );
    for r in &results {
        eprintln!(
            "  {}: derived {}/{} expected (tp={}, fp={}, fn={}), proofs {}/{}, false-provable {}/{}, rederive median {:.4} ms",
            r.name,
            r.derived_count,
            r.expected_count,
            r.true_positive,
            r.false_positive,
            r.false_negative,
            r.proof_complete,
            r.derived_count,
            r.false_provable,
            r.must_not_count,
            r.rederive_median_ms,
        );
    }
    eprintln!(
        "M-4620 precision={precision:.4}  M-4621 recall={recall:.4}  M-4622 proof-completeness={completeness:.4}  M-4623 worst-case median rederive={worst_median:.4} ms  false-provable-rate={false_provable_rate:.4}"
    );

    let case_json: Vec<String> = results
        .iter()
        .map(|r| {
            format!(
                r#"    {{
      "name": "{}",
      "seed_count": {},
      "derived_count": {},
      "expected_count": {},
      "true_positive": {},
      "false_positive": {},
      "false_negative": {},
      "precision": {:.6},
      "recall": {:.6},
      "proof_complete": {},
      "proof_completeness": {:.6},
      "false_provable": {},
      "must_not_count": {},
      "initial_saturation_ms": {:.6},
      "rederive_median_ms": {:.6},
      "rederive_p95_ms": {:.6}
    }}"#,
                r.name,
                r.seed_count,
                r.derived_count,
                r.expected_count,
                r.true_positive,
                r.false_positive,
                r.false_negative,
                ratio(r.true_positive, r.derived_count),
                ratio(r.true_positive, r.expected_count),
                r.proof_complete,
                ratio(r.proof_complete, r.derived_count),
                r.false_provable,
                r.must_not_count,
                r.initial_saturation_ms,
                r.rederive_median_ms,
                r.rederive_p95_ms,
            )
        })
        .collect();

    println!(
        r#"{{
  "experiment": "EXPR-4578.3",
  "title": "Derivation correctness vs ground-truth fixture",
  "engine": "nusy-forward-chain (Vec fixpoint, full-refire)",
  "machine": "M5",
  "trials_per_case": {TRIALS_PER_CASE},
  "cases": [
{}
  ],
  "aggregate": {{
    "derivation_precision_M4620": {precision:.6},
    "derivation_recall_M4621": {recall:.6},
    "proof_completeness_M4622": {completeness:.6},
    "rederive_latency_worst_case_median_ms_M4623": {worst_median:.6},
    "false_provable_rate": {false_provable_rate:.6},
    "true_positive": {tp},
    "derived_total": {derived},
    "expected_total": {expected}
  }},
  "caveats": [
    "M-4623 is the FULL-REFIRE baseline: the engine has no incremental mode (EX-4593 pending); latency = full re-saturation after adding one triple.",
    "Ground truth = hand-derived complete closures for the 3 clinical gold cases (cases.rs, keyed by fixture name) + 1 synthetic 19-fact kinship closure.",
    "Fixtures are small (engine-correctness scale); scale/perf evaluation is EX-4600/EX-4593 territory."
  ]
}}"#,
        case_json.join(",\n")
    );
}
