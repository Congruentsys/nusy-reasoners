//! EX-4748 phase 3 — PAR (Provable Answer Rate) standing regression guard.
//!
//! The V18 zero-hallucination-by-construction invariant, generalized to the
//! reasoner-router: on the clinical gold panel (the EXPR-4578.3 fixtures),
//!
//! - **PAR == 1.0** on the entailed set — every expected recommendation routes to a
//!   Proven answer through the envelope-dispatching router;
//! - **false_proofs == 0** — no contraindicated or negative-control claim is ever
//!   answered Proven;
//! - abstention stays **loud** — every must-not-prove claim yields a reasoned
//!   outcome, never a silent drop.
//!
//! This runs on every `cargo test`, so a regression in the router, the adapters, or
//! the engine that lowers PAR (or mints a false proof) fails the workspace, not a
//! quarterly eval. EX-4613/EX-4614's batteries drive the same `ReasonerRouter::par`
//! surface at full scale.

use nusy_clinical_fixtures::gold_cases;
use nusy_reasoner::Query;
use nusy_reasoner_adapters::DeductiveReasoner;
use nusy_router::ReasonerRouter;

#[test]
fn par_is_one_and_false_proofs_zero_on_the_clinical_gold_panel() {
    let mut total_expected = 0usize;
    for fx in gold_cases() {
        let mut router = ReasonerRouter::new();
        router.push(Box::new(DeductiveReasoner::new(
            fx.rules.clone(),
            fx.patient_facts.clone(),
        )));

        let mut panel: Vec<(Query, bool)> = Vec::new();
        for rec in &fx.expected_recommendations {
            panel.push((Query::new(rec.clone()), true));
        }
        for bad in fx.contraindicated.iter().chain(&fx.negative_controls) {
            panel.push((Query::new(bad.clone()), false));
        }
        total_expected += fx.expected_recommendations.len();

        let report = router.par(&panel);
        assert_eq!(
            report.false_proofs, 0,
            "{}: a must-not-prove claim was answered Proven",
            fx.name
        );
        assert_eq!(report.silent_drops, 0, "{}: silent drop", fx.name);
        assert_eq!(
            report.missed,
            0,
            "{}: an expected recommendation failed to prove through the router (PAR {})",
            fx.name,
            report.par()
        );
        assert!(
            (report.par() - 1.0).abs() < 1e-9,
            "{}: PAR regressed",
            fx.name
        );
    }
    assert!(
        total_expected > 0,
        "gold panel must contain entailed claims — empty panel would pass vacuously"
    );
}
