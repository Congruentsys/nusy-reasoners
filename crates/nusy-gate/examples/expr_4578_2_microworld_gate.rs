//! EXPR-4578.2 gate-ON arm — the REAL provable-claim gate over the micro-world (EX-4662).
//!
//! Reads the validated closure (`gate-facts.tsv`) and the 256 provable claims
//! (`gate-claims.tsv`) written by `run_microworld_gate.py export`, runs each claim through
//! [`ProvableClaimGate`], and writes `gate-answers.tsv` (`S\tP\tO\ttruth\tproven`). The gate
//! is sound + complete over the closed world, so it answers every provable query exactly —
//! the zero-hallucination-on-provable contract (EX-4610), contrasted against the LLM arm.
//!
//! Run: `cargo run -p nusy-gate --example expr_4578_2_microworld_gate -- <experiment-dir>`
//! (dir defaults to `research/expr-v18/EXPR-4578.2-gate-hallucination` relative to the repo root).

use std::fs;
use std::path::PathBuf;

use nusy_forward_chain::{IdRule, forward_chain};
use nusy_gate::ProvableClaimGate;
use nusy_unify::Triple;

fn main() {
    let dir: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../research/expr-v18/EXPR-4578.2-gate-hallucination")
        });

    // 1. Load the world's validated derivation closure as the gate's saturation.
    let facts_raw = fs::read_to_string(dir.join("gate-facts.tsv")).expect("read gate-facts.tsv");
    let facts: Vec<Triple> = facts_raw
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let mut it = l.split('\t');
            let (s, p, o) = (it.next().unwrap(), it.next().unwrap(), it.next().unwrap());
            Triple::new(s, p, o)
        })
        .collect();
    let n_facts = facts.len();

    // No rules: the seed already IS the closure (derivation validated separately, EXPR-4578.3);
    // here the gate is tested as the admit/reject hallucination filter over that closure.
    let no_rules: Vec<IdRule> = Vec::new();
    let gate = ProvableClaimGate::new(forward_chain(&no_rules, facts));

    // 2. Gate each of the 256 claims; record proven (1) / unproven (0) vs ground truth.
    let claims_raw = fs::read_to_string(dir.join("gate-claims.tsv")).expect("read gate-claims.tsv");
    let mut out = String::new();
    let (mut n, mut correct, mut false_yes, mut truth_no) = (0u32, 0u32, 0u32, 0u32);
    for line in claims_raw.lines().filter(|l| !l.trim().is_empty()) {
        let mut it = line.split('\t');
        let (s, p, o, truth) = (
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
        );
        let proven = gate.gate(&Triple::new(s, p, o)).is_proven();
        let gate_yn = if proven { "yes" } else { "no" };
        n += 1;
        if gate_yn == truth {
            correct += 1;
        }
        if truth == "no" {
            truth_no += 1;
            if proven {
                false_yes += 1;
            }
        }
        out.push_str(&format!(
            "{s}\t{p}\t{o}\t{truth}\t{}\n",
            if proven { 1 } else { 0 }
        ));
    }
    fs::write(dir.join("gate-answers.tsv"), out).expect("write gate-answers.tsv");

    let acc = correct as f64 / n as f64;
    let halluc = if truth_no > 0 {
        false_yes as f64 / truth_no as f64
    } else {
        0.0
    };
    println!(
        "[gate-ON] {n_facts} closure facts; {n} claims  accuracy={acc:.4}  hallucination_rate={halluc:.4}  → gate-answers.tsv"
    );
    // The gate must be the exact oracle on the provable set.
    assert_eq!(
        false_yes, 0,
        "the provable gate must NEVER assert an unprovable claim"
    );
    assert!(
        (acc - 1.0).abs() < 1e-9,
        "gate-ON must be exact on the closed provable set, got {acc}"
    );
}
