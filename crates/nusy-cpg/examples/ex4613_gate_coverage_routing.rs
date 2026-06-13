//! EX-4613 — Gate coverage + routing accuracy on clinical claims (VY-4586 / VOY-V18-5).
//!
//! **Re-scoped (Captain 2026-06-11).** Zero-hallucination on *provable* claims is a free
//! structural property of the symbolic engine+gate (it derives-with-proof or abstains; it
//! cannot assert what it cannot prove). Re-confirming that is a tautology and not high-value
//! GPU work. The gate's *real* value is **coverage + routing accuracy**: a no-hallucination
//! system that proves almost nothing is useless, and a gate that misroutes provable claims to
//! the neural layer (or vice versa) is unsafe. This eval measures exactly those.
//!
//! ## The gate under test
//!
//! The provable-claim gate is instantiated here as the **JNC 8 guideline-apply**
//! ([`nusy_cpg::hypertension_jnc8`] + [`nusy_plandef::apply`]). Per that crate's contract, a
//! recommendation fires *only when its conditions are provably true* — the provable-gate
//! contract on a real guideline. The three-/four-valued outcome maps onto gate routing:
//!
//! | `apply` outcome | Gate routing | Meaning |
//! |---|---|---|
//! | `proposed`   | **PROVEN**  — answer symbolically, with evidence | a provable claim |
//! | `abstained`  | **ROUTE→NEURAL** — guard unknown, never asserted | the un-provable remainder |
//! | `suppressed` | **REJECT** — provable *negative* (contraindication) | explicit defeater (EX-4692) |
//! | pruned/absent| **REJECT** — guard provably false | excluded, not abstained |
//!
//! ## Metrics over the (patient × recommendation) claim space
//!
//! 1. **provable-claim coverage** = proven(should-fire) / should-fire total.
//! 2. **false-no-route rate** = should-fire claims the gate fails to prove (wrongly routed to
//!    neural). The dangerous direction for a CDS gate.
//! 3. **false-route / hallucination rate** = gate proves a claim that must NOT fire
//!    (structurally 0 — the gate cannot assert an unprovable claim).
//! 4. **abstention precision / recall** = gate-abstain vs gold-abstain (does the gate abstain
//!    on exactly the unknown-guard claims, no more, no less?).
//! 5. **3-way routing accuracy** = exact (proven / abstain / reject) match vs gold, per cell.
//! 6. **guideline-representation coverage** = 5 / 9 JNC 8 actionable recs (the import scopes to
//!    the 5 load-bearing single-encounter ones; recs 4–7 are longitudinal titration/follow-up).
//!    Reported as the *honest denominator* behind the per-claim coverage — the
//!    "no-hallucination at low coverage is useless" caveat made explicit.
//!
//! A SINGLE gate-OFF LLM hallucination number is **cited, not re-run**, as the motivation
//! baseline (EXPR-4578.7 measured an un-gated LLM asserting unsupported clinical claims; see
//! that experiment's report). The headline metrics here are **symbolic** — the gate is CQL
//! evaluation over a PlanDefinition, so this eval needs **no GPU**.
//!
//! Emits eval JSON on stdout, a human summary on stderr:
//!
//! ```bash
//! cargo run --release -p nusy-cpg --example ex4613_gate_coverage_routing \
//!     > research/shared/eval-data/v18-ex4613/GATE-COVERAGE-ROUTING.json
//! ```

use std::collections::HashMap;

use nusy_cpg::{JNC8_VALUE_SETS, hypertension_jnc8};
use nusy_cql::{Code, FactStore, Value};
use nusy_plandef::ApplyOutcome;

/// JNC 8 publishes 9 actionable recommendations; the import scopes to the 5 load-bearing
/// single-encounter ones (recs 4–7 are longitudinal titration/follow-up, out of scope).
const JNC8_TOTAL_ACTIONABLE_RECS: usize = 9;

/// The 5 recommendation ids the JNC 8 plan can propose — the full claim axis. Every
/// (patient, rec) pair is one routed clinical claim.
const ALL_RECS: &[&str] = &[
    "bp-target-150-90",
    "bp-target-140-90",
    "bp-target-140-90-comorbid",
    "init-acei-or-arb-ckd",
    "init-first-line-general",
];

/// The motivation baseline (cited, not re-run): an un-gated LLM asserting unsupported
/// clinical claims. Source: EXPR-4578.7 (router-as-gate, hallucination ON vs OFF on a
/// provable JNC 8 set). Kept as a single number per the Captain re-scope, not the headline.
const LLM_UNGATED_HALLUCINATION_BASELINE: f64 = 1.0; // gate-OFF asserts the unprovable claim

/// A patient-fact store backed by simple maps, with JNC 8 value sets wired from the published
/// table. Mirrors the `nusy-cpg` test fixture / EXPR-4578.4 eval harness (the `#[cfg(test)]`
/// helper is not exported); the clinical truth lives in the single oracle both exercise.
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

/// The three-valued gate routing of one clinical claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Route {
    /// PROVEN — answered symbolically (the rec is proposed with evidence).
    Proven,
    /// ROUTE→NEURAL — applicability unknown; the gate abstains and flags for the neural layer.
    Abstain,
    /// REJECT — provably false guard or a fired contraindication; excluded, never asserted.
    Reject,
}

/// One gold case: a sample patient plus the hand-derived expected routing per recommendation.
/// Expectations are walked from JNC 8 logic + the documented three-valued semantics
/// (provably-true → propose, unknown → abstain loudly, provably-false → prune). This panel
/// mirrors the EXPR-4578.4 gold panel so both evals score the same JNC 8 oracle.
struct GoldCase {
    name: &'static str,
    patient: Patient,
    expect_proposed: &'static [&'static str],
    expect_abstained: &'static [&'static str],
    // Everything else in ALL_RECS is gold = Reject (provably-false guard or contraindication).
}

impl GoldCase {
    /// The gold routing for `rec`: Proven if expected-proposed, Abstain if expected-abstained,
    /// else Reject.
    fn gold_route(&self, rec: &str) -> Route {
        if self.expect_proposed.contains(&rec) {
            Route::Proven
        } else if self.expect_abstained.contains(&rec) {
            Route::Abstain
        } else {
            Route::Reject
        }
    }
}

/// The 12-patient gold panel (mirrors EXPR-4578.4; same JNC 8 oracle). SNOMED anchors are the
/// `JNC8_VALUE_SETS` codes: 38341003 essential HTN, 59621000 secondary HTN, 44054006 DM2,
/// 46635009 DM1, 709044004 CKD, 433146000 CKD-3.
fn gold_panel() -> Vec<GoldCase> {
    let htn = || Patient::code("38341003");
    let htn2 = || Patient::code("59621000");
    let dm2 = || Patient::code("44054006");
    let dm1 = || Patient::code("46635009");
    let ckd = || Patient::code("709044004");
    let ckd3 = || Patient::code("433146000");
    let white = || Value::Str("white".into());
    let black = || Value::Str("black".into());

    vec![
        GoldCase {
            name: "elderly_htn_no_comorbidity",
            patient: Patient::default()
                .with("Patient.age", vec![Value::Integer(72)])
                .with("Patient.race", vec![white()])
                .with("Condition.code", vec![htn()]),
            expect_proposed: &["bp-target-150-90", "init-first-line-general"],
            expect_abstained: &[],
        },
        GoldCase {
            name: "young_htn",
            patient: Patient::default()
                .with("Patient.age", vec![Value::Integer(45)])
                .with("Patient.race", vec![white()])
                .with("Condition.code", vec![htn()]),
            expect_proposed: &["bp-target-140-90", "init-first-line-general"],
            expect_abstained: &[],
        },
        GoldCase {
            name: "elderly_htn_ckd",
            patient: Patient::default()
                .with("Patient.age", vec![Value::Integer(68)])
                .with("Patient.race", vec![white()])
                .with("Condition.code", vec![htn(), ckd()]),
            expect_proposed: &[
                "bp-target-150-90",
                "bp-target-140-90-comorbid",
                "init-acei-or-arb-ckd",
                "init-first-line-general",
            ],
            expect_abstained: &[],
        },
        GoldCase {
            name: "midlife_htn_diabetes",
            patient: Patient::default()
                .with("Patient.age", vec![Value::Integer(55)])
                .with("Patient.race", vec![white()])
                .with("Condition.code", vec![htn(), dm2()]),
            expect_proposed: &[
                "bp-target-140-90",
                "bp-target-140-90-comorbid",
                "init-first-line-general",
            ],
            expect_abstained: &[],
        },
        GoldCase {
            name: "boundary_age_60_secondary_htn_code",
            patient: Patient::default()
                .with("Patient.age", vec![Value::Integer(60)])
                .with("Patient.race", vec![white()])
                .with("Condition.code", vec![htn2()]),
            // Exactly 60: `age < 60` false, `age >= 60` true.
            expect_proposed: &["bp-target-150-90", "init-first-line-general"],
            expect_abstained: &[],
        },
        GoldCase {
            name: "boundary_age_59",
            patient: Patient::default()
                .with("Patient.age", vec![Value::Integer(59)])
                .with("Patient.race", vec![white()])
                .with("Condition.code", vec![htn()]),
            expect_proposed: &["bp-target-140-90", "init-first-line-general"],
            expect_abstained: &[],
        },
        GoldCase {
            name: "non_hypertensive_negative_control",
            patient: Patient::default()
                .with("Patient.age", vec![Value::Integer(72)])
                .with("Patient.race", vec![white()])
                .with("Condition.code", vec![Patient::code("00000000")]),
            // Root provably false → whole guideline pruned (Reject), nothing abstained.
            expect_proposed: &[],
            expect_abstained: &[],
        },
        GoldCase {
            name: "htn_unknown_age_abstains_age_bands",
            patient: Patient::default()
                .with("Patient.race", vec![white()])
                .with("Condition.code", vec![htn()]),
            expect_proposed: &["init-first-line-general"],
            expect_abstained: &["bp-target-140-90", "bp-target-150-90"],
        },
        GoldCase {
            name: "black_elderly_htn_ckd_first_line_pruned",
            patient: Patient::default()
                .with("Patient.age", vec![Value::Integer(70)])
                .with("Patient.race", vec![black()])
                .with("Condition.code", vec![htn(), ckd3()]),
            // `not (race = black)` provably false → first-line pruned (Reject), not abstained.
            expect_proposed: &[
                "bp-target-150-90",
                "bp-target-140-90-comorbid",
                "init-acei-or-arb-ckd",
            ],
            expect_abstained: &[],
        },
        GoldCase {
            name: "htn_unknown_race_abstains_first_line",
            patient: Patient::default()
                .with("Patient.age", vec![Value::Integer(70)])
                .with("Condition.code", vec![htn()]),
            expect_proposed: &["bp-target-150-90"],
            expect_abstained: &["init-first-line-general"],
        },
        GoldCase {
            name: "elderly_htn_dm1_and_ckd3_full_stack",
            patient: Patient::default()
                .with("Patient.age", vec![Value::Integer(80)])
                .with("Patient.race", vec![white()])
                .with("Condition.code", vec![htn(), dm1(), ckd3()]),
            expect_proposed: &[
                "bp-target-150-90",
                "bp-target-140-90-comorbid",
                "init-acei-or-arb-ckd",
                "init-first-line-general",
            ],
            expect_abstained: &[],
        },
        GoldCase {
            name: "empty_record_root_unknown_abstains_everything",
            patient: Patient::default()
                .with("Patient.age", vec![Value::Integer(50)])
                .with("Patient.race", vec![white()]),
            // No Condition.code → root gate unknown → every action abstains (surfaced, never
            // silently dropped, never asserted).
            expect_proposed: &[],
            expect_abstained: &[
                "bp-target-140-90",
                "bp-target-150-90",
                "bp-target-140-90-comorbid",
                "init-acei-or-arb-ckd",
                "init-first-line-general",
            ],
        },
    ]
}

/// The gate routing the JNC 8 apply produced for `rec` on this outcome.
fn gate_route(out: &ApplyOutcome, rec: &str) -> Route {
    if out.proposed.iter().any(|p| p.action.id == rec) {
        Route::Proven
    } else if out.abstained.iter().any(|a| a.action.id == rec) {
        Route::Abstain
    } else {
        // Suppressed (fired contraindication) or pruned (provably-false guard): both REJECT.
        Route::Reject
    }
}

/// Confusion tallies over every routed (patient, rec) claim.
#[derive(Default)]
struct Tallies {
    cells: usize,
    routing_correct: usize,
    // Coverage / false-no-route (over gold=Proven claims).
    gold_proven: usize,
    gate_proven_on_gold_proven: usize,
    // Hallucination / false-route (over gate=Proven claims).
    gate_proven: usize,
    gate_proven_on_gold_not_proven: usize,
    // Abstention precision/recall.
    gold_abstain: usize,
    gate_abstain: usize,
    abstain_correct: usize,
}

fn main() {
    let plan = hypertension_jnc8();
    let panel = gold_panel();

    let mut t = Tallies::default();
    let mut per_case: Vec<String> = Vec::with_capacity(panel.len());

    for case in &panel {
        let out = nusy_plandef::apply(&plan, &case.patient).expect("guideline must evaluate");
        let mut case_correct = 0usize;
        for &rec in ALL_RECS {
            let gold = case.gold_route(rec);
            let gate = gate_route(&out, rec);
            t.cells += 1;
            if gate == gold {
                t.routing_correct += 1;
                case_correct += 1;
            }
            if gold == Route::Proven {
                t.gold_proven += 1;
                if gate == Route::Proven {
                    t.gate_proven_on_gold_proven += 1;
                }
            }
            if gate == Route::Proven {
                t.gate_proven += 1;
                if gold != Route::Proven {
                    t.gate_proven_on_gold_not_proven += 1;
                }
            }
            if gold == Route::Abstain {
                t.gold_abstain += 1;
            }
            if gate == Route::Abstain {
                t.gate_abstain += 1;
                if gold == Route::Abstain {
                    t.abstain_correct += 1;
                }
            }
        }
        per_case.push(format!(
            "{{\"case\":\"{}\",\"cells\":{},\"correct\":{}}}",
            case.name,
            ALL_RECS.len(),
            case_correct,
        ));
    }

    // Derived metrics.
    let coverage = ratio(t.gate_proven_on_gold_proven, t.gold_proven); // provable-claim coverage
    let false_no_route = 1.0 - coverage; // should-fire claims wrongly routed to neural
    let hallucination = ratio(t.gate_proven_on_gold_not_proven, t.gate_proven); // false-route
    let abstain_precision = ratio(t.abstain_correct, t.gate_abstain);
    let abstain_recall = ratio(t.abstain_correct, t.gold_abstain);
    let routing_accuracy = ratio(t.routing_correct, t.cells);
    // Honest denominator: represented load-bearing recs / all JNC 8 actionable recs.
    let representation_coverage = ALL_RECS.len() as f64 / JNC8_TOTAL_ACTIONABLE_RECS as f64;

    // Human summary on stderr.
    eprintln!(
        "EX-4613 gate coverage + routing — {} patients × {} recs = {} clinical claims",
        panel.len(),
        ALL_RECS.len(),
        t.cells,
    );
    eprintln!(
        "  provable-claim coverage = {coverage:.4}  ({}/{} should-fire claims proven)",
        t.gate_proven_on_gold_proven, t.gold_proven,
    );
    eprintln!(
        "  false-no-route rate     = {false_no_route:.4}  (should-fire claims routed to neural)"
    );
    eprintln!(
        "  false-route / hallucination = {hallucination:.4}  ({}/{} gate-proven claims unsupported)  [structural 0]",
        t.gate_proven_on_gold_not_proven, t.gate_proven,
    );
    eprintln!(
        "  abstention precision    = {abstain_precision:.4}  recall = {abstain_recall:.4}  ({}/{} gate-abstain correct)",
        t.abstain_correct, t.gate_abstain,
    );
    eprintln!(
        "  3-way routing accuracy  = {routing_accuracy:.4}  ({}/{} cells)",
        t.routing_correct, t.cells,
    );
    eprintln!(
        "  guideline-representation coverage = {representation_coverage:.4}  (5/{JNC8_TOTAL_ACTIONABLE_RECS} JNC 8 actionable recs — honest denominator)"
    );
    eprintln!(
        "  motivation baseline: un-gated LLM hallucination = {LLM_UNGATED_HALLUCINATION_BASELINE:.2} (cited EXPR-4578.7, not re-run)"
    );

    // Eval JSON on stdout.
    println!(
        "{{\n  \"experiment\": \"EX-4613\",\n  \"title\": \"Gate coverage + routing accuracy on clinical claims (JNC 8)\",\n  \"gate\": \"nusy_cpg::hypertension_jnc8 + nusy_plandef::apply (provable-claim gate on a real CPG)\",\n  \"substrate\": \"symbolic (CQL three-valued evaluation) — no GPU\",\n  \"patients\": {patients},\n  \"recs\": {recs},\n  \"claim_cells\": {cells},\n  \"provable_claim_coverage\": {coverage:.4},\n  \"false_no_route_rate\": {false_no_route:.4},\n  \"false_route_hallucination_rate\": {hallucination:.4},\n  \"abstention_precision\": {abstain_precision:.4},\n  \"abstention_recall\": {abstain_recall:.4},\n  \"routing_accuracy_3way\": {routing_accuracy:.4},\n  \"guideline_representation_coverage\": {representation_coverage:.4},\n  \"gold_proven\": {gold_proven},\n  \"gold_abstain\": {gold_abstain},\n  \"gate_proven\": {gate_proven},\n  \"gate_abstain\": {gate_abstain},\n  \"llm_ungated_hallucination_baseline\": {LLM_UNGATED_HALLUCINATION_BASELINE:.2},\n  \"baseline_source\": \"EXPR-4578.7 (cited, not re-run)\",\n  \"cross_substrate_note\": \"nusy_clinical_fixtures::run_all() confirms the gate-native forward-chain path (ProvableClaimGate over nusy-forward-chain): expected recs derived with proof, contraindications suppressed, negative controls unprovable.\",\n  \"per_case\": [{per_case}],\n  \"caveats\": [\n    \"Coverage is per-(patient,rec) over the gold panel; representation coverage (5/9) is the honest denominator — the gate is exact on what it represents, and represents the 5 load-bearing single-encounter JNC 8 recs.\",\n    \"Hallucination is structurally 0: the gate proposes a rec only when its CQL conditions are provably true.\",\n    \"Headline metrics are symbolic; the un-gated LLM baseline is cited from EXPR-4578.7, not re-run (Captain re-scope 2026-06-11).\"\n  ]\n}}",
        patients = panel.len(),
        recs = ALL_RECS.len(),
        cells = t.cells,
        gold_proven = t.gold_proven,
        gold_abstain = t.gold_abstain,
        gate_proven = t.gate_proven,
        gate_abstain = t.gate_abstain,
        per_case = per_case.join(","),
    );

    // Structural guarantees the gate must satisfy on this real-guideline panel.
    assert_eq!(
        t.gate_proven_on_gold_not_proven, 0,
        "false-route / hallucination must be 0: the gate must never prove an unsupported clinical claim"
    );
    assert!(
        (coverage - 1.0).abs() < 1e-9,
        "the gate must prove every should-fire claim on the represented recs (coverage {coverage})"
    );
    assert!(
        (routing_accuracy - 1.0).abs() < 1e-9,
        "3-way routing must match gold on every cell (got {routing_accuracy})"
    );
    assert!(
        (abstain_precision - 1.0).abs() < 1e-9,
        "the gate must abstain on exactly the unknown-guard claims (precision {abstain_precision})"
    );
}

/// `n / d` as f64, with the convention `0/0 = 1.0` (no claims of a kind ⇒ vacuously perfect).
fn ratio(n: usize, d: usize) -> f64 {
    if d == 0 { 1.0 } else { n as f64 / d as f64 }
}
