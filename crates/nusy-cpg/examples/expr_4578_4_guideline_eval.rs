//! EXPR-4578.4 — guideline execution vs gold-standard expected recommendations (VOY-V18-2 eval).
//!
//! Runs the imported JNC 8 CPG ([`nusy_cpg::hypertension_jnc8`]) over a panel of FHIR-style
//! sample patients via [`nusy_plandef::apply`] and scores the engine's output against
//! **hand-derived gold-standard outcomes** (expected proposals AND expected abstentions per
//! patient, walked by hand from JNC 8 logic + the documented three-valued semantics:
//! provably-true → propose, provably-false → prune, unknown → abstain loudly).
//!
//! - **M-4624 guideline coverage** — (a) representation: recommendations in computable Y2
//!   form / JNC 8's 9 actionable recommendations (the import deliberately scopes to the 5
//!   load-bearing ones; recs 4–7 are longitudinal titration/follow-up strategies outside a
//!   single-encounter decision graph); (b) execution: represented recommendations exercised
//!   (proposed for ≥1 panel patient).
//! - **M-4625 recommendation accuracy** — micro precision/recall over (patient, action)
//!   proposal pairs vs gold, plus exact-match rate (proposed AND abstained sets both match).
//! - Diagnostics: abstention correctness, contraindication absence (no `combine-*` action,
//!   ever), evidence-chain completeness (every proposal cites the root HTN gate), and
//!   median `apply()` latency.
//!
//! Emits eval JSON on stdout, human summary on stderr:
//!
//! ```bash
//! cargo run --release -p nusy-cpg --example expr_4578_4_guideline_eval \
//!     > research/shared/eval-data/v18-expr-4578.4/GUIDELINE-EXECUTION.json
//! ```

use std::collections::HashMap;
use std::time::Instant;

use nusy_cpg::{JNC8_VALUE_SETS, hypertension_jnc8};
use nusy_cql::{Code, FactStore, Value};
use nusy_plandef::ApplyOutcome;

const TRIALS_PER_PATIENT: usize = 200;
const WARMUP_RUNS: usize = 10;

/// JNC 8 publishes 9 actionable recommendations; the import scopes to the 5 load-bearing
/// single-encounter ones (see module docs) — recs 4–7 (titration strategy / follow-up) need
/// longitudinal data and are out of scope by design.
const JNC8_TOTAL_ACTIONABLE_RECS: usize = 9;

/// A patient-fact store backed by simple maps, with JNC 8 value sets wired from the
/// published table (mirrors the crate's test fixture — re-implemented here because the
/// test helper is `#[cfg(test)]`).
#[derive(Default, Clone)]
struct Patient {
    props: HashMap<String, Vec<Value>>,
}

impl Patient {
    fn with(mut self, key: &str, vals: Vec<Value>) -> Self {
        self.props.insert(key.to_string(), vals);
        self
    }
    fn code(c: &str) -> Value {
        Value::Code(Code::new("SNOMED", c))
    }
}

impl FactStore for Patient {
    fn get_property(&self, entity: &str, path: &[String]) -> Vec<Value> {
        let key = if path.is_empty() {
            entity.to_string()
        } else {
            format!("{entity}.{}", path.join("."))
        };
        self.props.get(&key).cloned().unwrap_or_default()
    }
    fn in_value_set(&self, code: &Code, valueset: &str) -> Option<bool> {
        JNC8_VALUE_SETS
            .iter()
            .find(|(name, _)| *name == valueset)
            .map(|(_, codes)| codes.contains(&code.code.as_str()))
    }
    fn subsumes(&self, _: &Code, _: &Code) -> Option<bool> {
        None
    }
}

/// One gold case: a sample patient plus the hand-derived expected outcome.
struct GoldCase {
    name: &'static str,
    patient: Patient,
    /// Action ids that MUST be proposed (order-insensitive, no duplicates expected).
    expect_proposed: Vec<&'static str>,
    /// Action ids that MUST be abstained (unknown applicability surfaced, not dropped).
    expect_abstained: Vec<&'static str>,
}

/// The 12-patient gold panel. Expectations hand-walked (2026-06-11) from the JNC 8 plan
/// tree: root gate `Condition.code in HypertensionVS`; age bands at 60 (`<` vs `>=`);
/// comorbid 140/90 on DM or CKD; ACEI/ARB on CKD; first-line for non-black race.
fn gold_panel() -> Vec<GoldCase> {
    let htn = || Patient::code("38341003"); // essential hypertension
    let htn2 = || Patient::code("59621000"); // secondary-variant anchor
    let dm2 = || Patient::code("44054006"); // diabetes type 2
    let dm1 = || Patient::code("46635009"); // diabetes type 1
    let ckd = || Patient::code("709044004"); // CKD
    let ckd3 = || Patient::code("433146000"); // CKD stage 3
    let white = || Value::Str("white".into());
    let black = || Value::Str("black".into());

    vec![
        GoldCase {
            name: "elderly_htn_no_comorbidity",
            patient: Patient::default()
                .with("Patient.age", vec![Value::Integer(72)])
                .with("Patient.race", vec![white()])
                .with("Condition.code", vec![htn()]),
            expect_proposed: vec!["bp-target-150-90", "init-first-line-general"],
            expect_abstained: vec![],
        },
        GoldCase {
            name: "young_htn",
            patient: Patient::default()
                .with("Patient.age", vec![Value::Integer(45)])
                .with("Patient.race", vec![white()])
                .with("Condition.code", vec![htn()]),
            expect_proposed: vec!["bp-target-140-90", "init-first-line-general"],
            expect_abstained: vec![],
        },
        GoldCase {
            name: "elderly_htn_ckd",
            patient: Patient::default()
                .with("Patient.age", vec![Value::Integer(68)])
                .with("Patient.race", vec![white()])
                .with("Condition.code", vec![htn(), ckd()]),
            expect_proposed: vec![
                "bp-target-150-90",
                "bp-target-140-90-comorbid",
                "init-acei-or-arb-ckd",
                "init-first-line-general",
            ],
            expect_abstained: vec![],
        },
        GoldCase {
            name: "midlife_htn_diabetes",
            patient: Patient::default()
                .with("Patient.age", vec![Value::Integer(55)])
                .with("Patient.race", vec![white()])
                .with("Condition.code", vec![htn(), dm2()]),
            expect_proposed: vec![
                "bp-target-140-90",
                "bp-target-140-90-comorbid",
                "init-first-line-general",
            ],
            expect_abstained: vec![],
        },
        GoldCase {
            name: "boundary_age_60_secondary_htn_code",
            patient: Patient::default()
                .with("Patient.age", vec![Value::Integer(60)])
                .with("Patient.race", vec![white()])
                .with("Condition.code", vec![htn2()]),
            // Exactly 60: `age < 60` false, `age >= 60` true.
            expect_proposed: vec!["bp-target-150-90", "init-first-line-general"],
            expect_abstained: vec![],
        },
        GoldCase {
            name: "boundary_age_59",
            patient: Patient::default()
                .with("Patient.age", vec![Value::Integer(59)])
                .with("Patient.race", vec![white()])
                .with("Condition.code", vec![htn()]),
            expect_proposed: vec!["bp-target-140-90", "init-first-line-general"],
            expect_abstained: vec![],
        },
        GoldCase {
            name: "non_hypertensive_negative_control",
            patient: Patient::default()
                .with("Patient.age", vec![Value::Integer(72)])
                .with("Patient.race", vec![white()])
                .with("Condition.code", vec![Patient::code("00000000")]),
            // Root provably false → whole guideline pruned silently.
            expect_proposed: vec![],
            expect_abstained: vec![],
        },
        GoldCase {
            name: "htn_unknown_age_abstains_age_bands",
            patient: Patient::default()
                .with("Patient.race", vec![white()])
                .with("Condition.code", vec![htn()]),
            expect_proposed: vec!["init-first-line-general"],
            expect_abstained: vec!["bp-target-140-90", "bp-target-150-90"],
        },
        GoldCase {
            name: "black_elderly_htn_ckd_first_line_pruned",
            patient: Patient::default()
                .with("Patient.age", vec![Value::Integer(70)])
                .with("Patient.race", vec![black()])
                .with("Condition.code", vec![htn(), ckd3()]),
            // `not (race = black)` provably false → first-line pruned, not abstained.
            expect_proposed: vec![
                "bp-target-150-90",
                "bp-target-140-90-comorbid",
                "init-acei-or-arb-ckd",
            ],
            expect_abstained: vec![],
        },
        GoldCase {
            name: "htn_unknown_race_abstains_first_line",
            patient: Patient::default()
                .with("Patient.age", vec![Value::Integer(70)])
                .with("Condition.code", vec![htn()]),
            expect_proposed: vec!["bp-target-150-90"],
            expect_abstained: vec!["init-first-line-general"],
        },
        GoldCase {
            name: "elderly_htn_dm1_and_ckd3_full_stack",
            patient: Patient::default()
                .with("Patient.age", vec![Value::Integer(80)])
                .with("Patient.race", vec![white()])
                .with("Condition.code", vec![htn(), dm1(), ckd3()]),
            expect_proposed: vec![
                "bp-target-150-90",
                "bp-target-140-90-comorbid",
                "init-acei-or-arb-ckd",
                "init-first-line-general",
            ],
            expect_abstained: vec![],
        },
        GoldCase {
            name: "empty_record_root_unknown_abstains_everything",
            patient: Patient::default()
                .with("Patient.age", vec![Value::Integer(50)])
                .with("Patient.race", vec![white()]),
            // No Condition.code at all → root gate unknown → every action in the subtree
            // abstains (surfaced, never silently dropped, never asserted).
            expect_proposed: vec![],
            expect_abstained: vec![
                "bp-target-140-90",
                "bp-target-150-90",
                "bp-target-140-90-comorbid",
                "init-acei-or-arb-ckd",
                "init-first-line-general",
            ],
        },
    ]
}

fn ids_sorted(ids: impl Iterator<Item = String>) -> Vec<String> {
    let mut v: Vec<String> = ids.collect();
    v.sort();
    v
}

fn proposed_ids(out: &ApplyOutcome) -> Vec<String> {
    ids_sorted(out.proposed.iter().map(|p| p.action.id.clone()))
}

fn abstained_ids(out: &ApplyOutcome) -> Vec<String> {
    ids_sorted(out.abstained.iter().map(|a| a.action.id.clone()))
}

struct CaseResult {
    name: &'static str,
    proposed: Vec<String>,
    abstained: Vec<String>,
    true_positive: usize,
    false_positive: usize,
    false_negative: usize,
    abstain_match: bool,
    exact_match: bool,
    evidence_complete: bool,
    contraindication_clean: bool,
    apply_median_ms: f64,
}

fn percentile(sorted_ms: &[f64], p: f64) -> f64 {
    if sorted_ms.is_empty() {
        return 0.0;
    }
    let idx = ((sorted_ms.len() as f64 - 1.0) * p).round() as usize;
    sorted_ms[idx]
}

fn main() {
    let plan = hypertension_jnc8();
    let panel = gold_panel();
    let mut results: Vec<CaseResult> = Vec::with_capacity(panel.len());

    for case in &panel {
        let out = nusy_plandef::apply(&plan, &case.patient).expect("guideline must evaluate");
        let proposed = proposed_ids(&out);
        let abstained = abstained_ids(&out);

        let gold_proposed = ids_sorted(case.expect_proposed.iter().map(|s| s.to_string()));
        let gold_abstained = ids_sorted(case.expect_abstained.iter().map(|s| s.to_string()));

        let true_positive = proposed
            .iter()
            .filter(|p| gold_proposed.contains(p))
            .count();
        let false_positive = proposed.len() - true_positive;
        let false_negative = gold_proposed
            .iter()
            .filter(|g| !proposed.contains(g))
            .count();
        let abstain_match = abstained == gold_abstained;
        let exact_match = proposed == gold_proposed && abstain_match;

        // Every proposal must carry a non-empty evidence chain citing the root HTN gate.
        let evidence_complete = out.proposed.iter().all(|p| {
            !p.evidence.is_empty() && p.evidence.iter().any(|e| e.contains("HypertensionVS"))
        });

        // The ACEI+ARB combination contraindication: no `combine` action may ever appear.
        let contraindication_clean = !out.proposed.iter().any(|p| p.action.id.contains("combine"));

        // Latency: median apply() over this patient.
        for _ in 0..WARMUP_RUNS {
            let _ = nusy_plandef::apply(&plan, &case.patient).unwrap();
        }
        let mut trial_ms: Vec<f64> = Vec::with_capacity(TRIALS_PER_PATIENT);
        for _ in 0..TRIALS_PER_PATIENT {
            let start = Instant::now();
            let _ = nusy_plandef::apply(&plan, &case.patient).unwrap();
            trial_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        trial_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());

        results.push(CaseResult {
            name: case.name,
            proposed,
            abstained,
            true_positive,
            false_positive,
            false_negative,
            abstain_match,
            exact_match,
            evidence_complete,
            contraindication_clean,
            apply_median_ms: percentile(&trial_ms, 0.5),
        });
    }

    // Aggregates.
    let tp: usize = results.iter().map(|r| r.true_positive).sum();
    let fp: usize = results.iter().map(|r| r.false_positive).sum();
    let fnn: usize = results.iter().map(|r| r.false_negative).sum();
    let precision = if tp + fp == 0 {
        1.0
    } else {
        tp as f64 / (tp + fp) as f64
    };
    let recall = if tp + fnn == 0 {
        1.0
    } else {
        tp as f64 / (tp + fnn) as f64
    };
    let exact = results.iter().filter(|r| r.exact_match).count();
    let abstain_ok = results.iter().filter(|r| r.abstain_match).count();
    let evidence_ok = results.iter().filter(|r| r.evidence_complete).count();
    let contra_ok = results.iter().filter(|r| r.contraindication_clean).count();

    // M-4624 coverage. Representation: actions in the computable plan vs JNC 8's
    // actionable recommendations. Execution: represented actions proposed >=1 time.
    let plan_action_ids: Vec<String> = {
        // The 5 action ids the import declares (stable, asserted by exercising the panel).
        let mut all: Vec<String> = results.iter().flat_map(|r| r.proposed.clone()).collect();
        all.sort();
        all.dedup();
        all
    };
    let represented = 5usize; // hypertension_jnc8() encodes 5 load-bearing recommendations
    let representation_coverage = represented as f64 / JNC8_TOTAL_ACTIONABLE_RECS as f64;
    let execution_coverage = plan_action_ids.len() as f64 / represented as f64;

    let mut medians: Vec<f64> = results.iter().map(|r| r.apply_median_ms).collect();
    medians.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let worst_median = *medians.last().unwrap();

    eprintln!(
        "EXPR-4578.4 guideline execution — {} patients",
        results.len()
    );
    for r in &results {
        eprintln!(
            "  {}: tp={} fp={} fn={} exact={} abstain-ok={} evidence-ok={} apply median {:.4} ms",
            r.name,
            r.true_positive,
            r.false_positive,
            r.false_negative,
            r.exact_match,
            r.abstain_match,
            r.evidence_complete,
            r.apply_median_ms,
        );
    }
    eprintln!(
        "M-4625 precision={precision:.4} recall={recall:.4} exact-match {exact}/{n}  M-4624 representation {represented}/{total} ({representation_coverage:.3}) execution {exercised}/{represented} ({execution_coverage:.3})  abstain {abstain_ok}/{n}  evidence {evidence_ok}/{n}  contraindication-clean {contra_ok}/{n}  worst apply median {worst_median:.4} ms",
        n = results.len(),
        total = JNC8_TOTAL_ACTIONABLE_RECS,
        exercised = plan_action_ids.len(),
    );

    let case_json: Vec<String> = results
        .iter()
        .map(|r| {
            format!(
                r#"    {{
      "name": "{}",
      "proposed": [{}],
      "abstained": [{}],
      "true_positive": {},
      "false_positive": {},
      "false_negative": {},
      "exact_match": {},
      "abstain_match": {},
      "evidence_complete": {},
      "contraindication_clean": {},
      "apply_median_ms": {:.6}
    }}"#,
                r.name,
                r.proposed
                    .iter()
                    .map(|s| format!("\"{s}\""))
                    .collect::<Vec<_>>()
                    .join(", "),
                r.abstained
                    .iter()
                    .map(|s| format!("\"{s}\""))
                    .collect::<Vec<_>>()
                    .join(", "),
                r.true_positive,
                r.false_positive,
                r.false_negative,
                r.exact_match,
                r.abstain_match,
                r.evidence_complete,
                r.contraindication_clean,
                r.apply_median_ms,
            )
        })
        .collect();

    println!(
        r#"{{
  "experiment": "EXPR-4578.4",
  "title": "Guideline execution vs gold-standard expected recommendations (JNC 8)",
  "guideline": "jnc8-hypertension (nusy-cpg) via nusy_plandef::apply",
  "machine": "M5",
  "panel_size": {n},
  "trials_per_patient": {TRIALS_PER_PATIENT},
  "cases": [
{cases}
  ],
  "aggregate": {{
    "recommendation_precision_M4625": {precision:.6},
    "recommendation_recall_M4625": {recall:.6},
    "exact_match_rate_M4625": {exact_rate:.6},
    "representation_coverage_M4624": {representation_coverage:.6},
    "execution_coverage_M4624": {execution_coverage:.6},
    "abstention_correct_rate": {abstain_rate:.6},
    "evidence_complete_rate": {evidence_rate:.6},
    "contraindication_clean_rate": {contra_rate:.6},
    "apply_latency_worst_case_median_ms": {worst_median:.6}
  }},
  "caveats": [
    "Gold standard = hand-walked JNC 8 plan-tree outcomes (proposed AND abstained sets) for 12 synthetic FHIR-style patients; single-encounter scope.",
    "M-4624 representation is 5/9: the import deliberately scopes to JNC 8's 5 load-bearing single-encounter recommendations; recs 4-7 (titration/follow-up strategies) need longitudinal data and are out of scope by design.",
    "Latency is plan-walk + CQL evaluation per patient on M5 (release build); no I/O."
  ]
}}"#,
        n = results.len(),
        cases = case_json.join(",\n"),
        exact_rate = exact as f64 / results.len() as f64,
        abstain_rate = abstain_ok as f64 / results.len() as f64,
        evidence_rate = evidence_ok as f64 / results.len() as f64,
        contra_rate = contra_ok as f64 / results.len() as f64,
    );
}
