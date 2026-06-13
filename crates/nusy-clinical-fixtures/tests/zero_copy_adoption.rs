//! Zero-copy-adoption integration test (EX-4671, VY-4667 phase 4).
//!
//! Drives every clinical gold case end-to-end through the **Arrow** path —
//! `forward_chain_arrow` → `ProvableClaimGate::from_arrow` / `surface_arrow` — and proves
//! two things:
//!
//! 1. **Correctness is preserved.** Expected recommendations gate as `Proven` and surface
//!    provenance; negative controls gate as `Unproven` and surface nothing (the gate
//!    abstains). I.e. the zero-copy consumers agree with the contract.
//! 2. **No `Vec<Triple>` materialization on the hot path.** Using the engine's
//!    `materialization-counter` feature, the Arrow path runs the whole gold-case battery
//!    with the boundary counter at **0** — the gate and provenance read the engine's Arrow
//!    batches directly, never crossing `ArrowSaturation::to_saturation`. A contrast test
//!    confirms the counter *does* fire on the legacy Vec path, so the `0` is meaningful.

use std::sync::Mutex;

use nusy_clinical_fixtures::gold_cases;
use nusy_forward_chain::{forward_chain, forward_chain_arrow, matcount};
use nusy_gate::ProvableClaimGate;
use nusy_provenance::{surface, surface_arrow};

/// `matcount` is a process-global counter; cargo runs tests in this binary in parallel.
/// Every test that touches the counter (resets it, or calls `forward_chain` which bumps
/// it) holds this guard for its body, so measured windows never overlap. `into_inner`
/// ignores poisoning (a poisoned lock just means another test already failed).
static COUNTER_GUARD: Mutex<()> = Mutex::new(());

#[test]
fn arrow_path_gates_gold_cases_with_zero_materialization() {
    let _guard = COUNTER_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    let cases = gold_cases();
    assert!(!cases.is_empty(), "no gold cases to exercise");

    matcount::reset();
    for fx in &cases {
        // Engine on the Arrow substrate — no Vec saturation materialized.
        let sat = forward_chain_arrow(&fx.rules, fx.patient_facts.clone());
        // Gate reads the Arrow batches directly (zero-copy back-end).
        let gate = ProvableClaimGate::from_arrow(sat.clone());

        for rec in &fx.expected_recommendations {
            assert!(
                gate.gate(rec).is_proven(),
                "{}: expected recommendation not proven on the Arrow path",
                fx.name
            );
            assert!(
                surface_arrow(&sat, rec).is_some(),
                "{}: no provenance surfaced for an expected recommendation",
                fx.name
            );
        }
        for neg in &fx.negative_controls {
            assert!(
                !gate.gate(neg).is_proven(),
                "{}: a negative control was proven on the Arrow path",
                fx.name
            );
            assert!(
                surface_arrow(&sat, neg).is_none(),
                "{}: provenance surfaced for a negative control (should abstain)",
                fx.name
            );
        }
    }

    // The load-bearing assertion: the entire Arrow battery materialized 0 Vec saturations.
    assert_eq!(
        matcount::count(),
        0,
        "Arrow hot path materialized {} Vec saturation(s) — zero-copy adoption regressed",
        matcount::count()
    );
}

#[test]
fn arrow_and_vec_paths_agree_on_every_gold_claim() {
    let _guard = COUNTER_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    for fx in &gold_cases() {
        let arrow =
            ProvableClaimGate::from_arrow(forward_chain_arrow(&fx.rules, fx.patient_facts.clone()));
        let vec = ProvableClaimGate::new(forward_chain(&fx.rules, fx.patient_facts.clone()));
        for claim in fx
            .expected_recommendations
            .iter()
            .chain(&fx.negative_controls)
            .chain(&fx.contraindicated)
        {
            assert_eq!(
                arrow.gate(claim).is_proven(),
                vec.gate(claim).is_proven(),
                "{}: Arrow and Vec gate verdicts diverged on a claim",
                fx.name
            );
        }
    }
}

/// Contrast: the legacy Vec path *does* cross the materialization boundary, so the `0`
/// in the Arrow test is a real measurement rather than a dead counter.
#[test]
fn vec_path_materializes_as_baseline_contrast() {
    let _guard = COUNTER_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    let cases = gold_cases();
    let fx = &cases[0];

    matcount::reset();
    let sat = forward_chain(&fx.rules, fx.patient_facts.clone()); // = forward_chain_arrow(..).to_saturation()
    let _gate = ProvableClaimGate::new(sat.clone());
    if let Some(rec) = fx.expected_recommendations.first() {
        let _ = surface(&sat, rec);
    }
    assert!(
        matcount::count() >= 1,
        "the Vec path should materialize at least one saturation (counter never fired)"
    );
}
