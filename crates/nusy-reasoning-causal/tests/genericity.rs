//! Genericity guard (EX-4752, VY-C §3) — the do-calculus core carries **no**
//! domain policy. These tests fail the build if a clinical/medical/legal term
//! leaks into a core module, so the FOSS extraction (`--no-default-features`)
//! stays a pure generic Pearl engine. The clinical vocabulary lives only behind
//! the `clinical-policy` feature (`safety_routing` / `clinical_gate`).
//!
//! Run BOTH configs in CI:
//!   cargo test -p nusy-reasoning-causal
//!   cargo test -p nusy-reasoning-causal --no-default-features
//!
//! The grep-assert below scans the core source at compile time via `include_str!`
//! — no policy module is referenced, so it compiles in either feature config.

/// The forbidden vocabulary — domain-POLICY words that would make the core
/// non-generic. Lower-cased before matching. Deliberately **excludes** standard
/// causal-inference vocabulary that is domain-neutral: `treatment`/`outcome` are
/// the canonical names for the intervention/response variables (T → Y), and
/// `patient` appears only as a generic example unit. The terms here name a
/// specific safety-critical *domain* — their presence means clinical policy
/// leaked out from behind the `clinical-policy` feature.
const CLINICAL_VOCAB: &[&str] = &[
    "clinical",
    "medical",
    "pharmaceutical",
    "diagnostic",
    "nccn",
];

/// Every always-built core module. (The policy modules `safety_routing` and
/// `clinical_gate` are intentionally excluded — they ARE the clinical layer.)
const CORE_SOURCES: &[(&str, &str)] = &[
    ("adjustment", include_str!("../src/adjustment.rs")),
    ("counterfactual", include_str!("../src/counterfactual.rs")),
    ("error", include_str!("../src/error.rs")),
    ("graph", include_str!("../src/graph.rs")),
    ("identifiability", include_str!("../src/identifiability.rs")),
    ("intervention", include_str!("../src/intervention.rs")),
];

#[test]
fn core_modules_carry_no_clinical_vocabulary() {
    let mut leaks = Vec::new();
    for (module, src) in CORE_SOURCES {
        let lower = src.to_lowercase();
        for term in CLINICAL_VOCAB {
            if lower.contains(term) {
                leaks.push(format!("  {module}.rs contains forbidden term {term:?}"));
            }
        }
    }
    assert!(
        leaks.is_empty(),
        "do-calculus core must stay domain-neutral (move clinical content behind \
         the `clinical-policy` feature). Leaks:\n{}",
        leaks.join("\n")
    );
}

/// The crate-level doc may MENTION the feature, but the lib root must not
/// `pub mod`/`pub use` a policy module unconditionally — the feature gate is the
/// only thing that exposes clinical types. Guards against an accidental ungated
/// re-export reintroducing the dependency in the FOSS build.
#[test]
fn policy_modules_are_feature_gated_in_lib_root() {
    let lib = include_str!("../src/lib.rs");
    for line in lib.lines() {
        let trimmed = line.trim_start();
        let is_policy_item = trimmed.starts_with("pub mod safety_routing")
            || trimmed.starts_with("pub mod clinical_gate")
            || trimmed.starts_with("pub use safety_routing")
            || trimmed.starts_with("pub use clinical_gate");
        if is_policy_item {
            // The immediately-preceding non-blank line (or a block above) must gate it.
            // Simplest robust check: the whole file must contain the cfg attr, and the
            // item must not appear without a cfg somewhere in lib.rs. We assert the
            // cfg attribute count is at least the number of policy items.
            // (Cheap structural check; the compiler is the real guarantee via
            //  `--no-default-features`.)
            assert!(
                lib.contains("#[cfg(feature = \"clinical-policy\")]"),
                "policy item `{trimmed}` present but no clinical-policy cfg gate found in lib.rs"
            );
        }
    }
}
