//! Run the clinical gold cases end-to-end and assert the six §12.4 elements hold:
//! expected recs derived with proof, contraindications suppressed, negative controls
//! unprovable, and proof paths with the right rule chain + depth.

use nusy_clinical_fixtures::{ProofNode, gold_cases, proof_path, run_all, run_fixture};
use nusy_forward_chain::forward_chain;
use nusy_unify::Triple;

#[test]
fn all_gold_cases_pass() {
    let reports = run_all();
    assert!(!reports.is_empty(), "there should be gold cases");
    for r in &reports {
        assert!(r.passed(), "fixture '{}' failed: {:?}", r.name, r.failures);
    }
}

#[test]
fn recommendation_proof_path_has_rule_chain_and_depth() {
    // The fall-risk fixture: recommend(fall_assessment) is a 2-step derivation.
    let fx = gold_cases()
        .into_iter()
        .find(|f| f.name == "fall_risk_assessment_fires")
        .expect("fixture present");
    let sat = forward_chain(&fx.rules, fx.patient_facts.clone());

    let rec = Triple::new("patient1", "recommend", "fall_assessment");
    let proof = proof_path(&sat, &rec);

    // Depth 2: recommend ← at_risk ← (seed facts).
    assert_eq!(proof.depth(), 2, "two-step derivation");
    // Both rules appear in the proof, deepest-first.
    let ids = proof.rule_ids();
    assert!(
        ids.contains(&"at-risk-fall"),
        "proof cites the at-risk rule"
    );
    assert!(
        ids.contains(&"recommend-fall-assessment"),
        "proof cites the recommend rule"
    );
    // The recommend rule is applied last (root of the tree).
    assert_eq!(ids.last(), Some(&"recommend-fall-assessment"));

    // The leaves of the proof are seed patient facts (asserted, not derived).
    if let ProofNode::Derived { premises, .. } = &proof {
        let at_risk = premises
            .iter()
            .find(|n| matches!(n, ProofNode::Derived { .. }))
            .expect("at_risk premise is itself derived");
        if let ProofNode::Derived {
            premises: leaves, ..
        } = at_risk
        {
            assert!(
                leaves.iter().all(|n| matches!(n, ProofNode::Fact(_))),
                "at_risk's premises are seed facts"
            );
        }
    }
}

#[test]
fn contraindication_suppresses_recommendation() {
    let fx = gold_cases()
        .into_iter()
        .find(|f| f.name == "glycemic_intensify_renal_contraindication")
        .expect("fixture present");
    let sat = forward_chain(&fx.rules, fx.patient_facts.clone());

    // patient4 (renal adequate) → recommended.
    assert!(sat.contains(&Triple::new("patient4", "recommend", "intensify_glycemic")));
    // patient3 (renal impaired) → suppressed, even though indicated.
    assert!(sat.contains(&Triple::new("patient3", "intensify_candidate", "glycemic")));
    assert!(!sat.contains(&Triple::new("patient3", "recommend", "intensify_glycemic")));

    let report = run_fixture(&fx);
    assert!(report.passed(), "{:?}", report.failures);
}

#[test]
fn negative_control_is_unprovable() {
    let fx = gold_cases()
        .into_iter()
        .find(|f| f.name == "fall_risk_assessment_fires")
        .expect("fixture present");
    let sat = forward_chain(&fx.rules, fx.patient_facts.clone());

    // An arbitrary claim with no derivation must be absent → the gate abstains/flags it.
    let bogus = Triple::new("patient1", "recommend", "kidney_transplant");
    assert!(!sat.contains(&bogus));
    assert!(sat.derivation_of(&bogus).is_none());
}
