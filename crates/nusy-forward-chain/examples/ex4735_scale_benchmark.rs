//! EX-4735 — scale benchmark for the forward-chain engine (H-4725 / EXPR-4727).
//!
//! Answers the V19 FOSS-launch's weakest-leg question (V19-STRATEGY #3): **does the
//! Arrow forward-chain engine reach closure at 100k–1M derived facts, and how fast?**
//! Prior data stopped at ~10k-fact fixtures (`bench_arrow_vs_vec`); this pushes to 1M.
//!
//! ## Two rulesets, chosen for *known* closure complexity (the load-bearing decision)
//!
//! A scale benchmark is meaningless unless the ruleset's closure size is understood —
//! otherwise "1M seed facts" might derive nothing and prove nothing. So:
//!
//! * **LINEAR (`subsumption`)** — the realistic clinical/ontology shape. `K`-level class
//!   chain `c0 ⊑ c1 ⊑ … ⊑ c_{K-1}`; `N` entities each `type(e, c0)`. Rule
//!   `type(?x,?super) :- type(?x,?sub), subClassOf(?sub,?super)` walks each entity up the
//!   chain → **N·(K-1)** derived `type` facts, **O(N)** for fixed `K`. This is how a real
//!   typed knowledge graph saturates.
//! * **QUADRATIC (`transitive`)** — the fixpoint stress test. A parent chain of `n` nodes
//!   under `ancestor(?x,?z) :- parent(?x,?y), ancestor(?y,?z)` → **n(n-1)/2** ancestors,
//!   **O(n²)**. Exercises deep recursive derivation, not just a shallow sweep.
//!
//! Scale points are chosen by *derived-fact target* (~10k / ~100k / ~1M), not seed size.
//!
//! ## Measures (H-4725 / M-4726)
//! seed facts, derived facts, total closure size, wall-clock to closure (median of trials),
//! derivations/sec throughput, and peak resident memory (Linux `VmHWM`/`VmRSS`).
//!
//! FOSS baselines (oxigraph / ascent / crepe) are **deliberately out of scope for v0.1** here
//! and are logged as NOT-RUN (no silent omission) — they add external deps and belong in a
//! follow-up once the NuSy-side scaling curve is established.
//!
//! ```bash
//! cargo run --release -p nusy-forward-chain --example ex4735_scale_benchmark \
//!     > research/shared/eval-data/v19-h5-scale/SCALE-BENCH.json
//! ```
//! (human summary on stderr; eval JSON on stdout.)

use std::time::Instant;

use nusy_forward_chain::{IdRule, forward_chain};
use nusy_unify::{Rule, Triple, TriplePattern};

fn rule(id: &str, body: &[(&str, &str, &str)], head: (&str, &str, &str)) -> IdRule {
    let lhs = body
        .iter()
        .map(|(s, p, o)| TriplePattern::parse(s, p, o))
        .collect();
    let rhs = vec![TriplePattern::parse(head.0, head.1, head.2)];
    IdRule::new(id, Rule::new(lhs, rhs))
}

// ---- LINEAR ruleset: subsumption over a K-level class chain --------------------------------

const K_LEVELS: usize = 6; // c0..c5 → 5 derived type-levels per entity

fn subsumption_rules() -> Vec<IdRule> {
    vec![rule(
        "type-subsumption",
        &[("?x", "type", "?sub"), ("?sub", "subClassOf", "?super")],
        ("?x", "type", "?super"),
    )]
}

/// `n_entities` each typed at `c0`, over the fixed K-level chain. Closure = n·(K-1) types.
fn subsumption_seed(n_entities: usize) -> Vec<Triple> {
    let mut seed = Vec::with_capacity(n_entities + K_LEVELS);
    for i in 0..K_LEVELS - 1 {
        seed.push(Triple::new(
            format!("c{i}"),
            "subClassOf",
            format!("c{}", i + 1),
        ));
    }
    for e in 0..n_entities {
        seed.push(Triple::new(format!("e{e}"), "type", "c0"));
    }
    seed
}

// ---- QUADRATIC ruleset: transitive ancestor over a parent chain ----------------------------

fn transitive_rules() -> Vec<IdRule> {
    vec![
        rule(
            "anc-base",
            &[("?x", "parent", "?y")],
            ("?x", "ancestor", "?y"),
        ),
        rule(
            "anc-rec",
            &[("?x", "parent", "?y"), ("?y", "ancestor", "?z")],
            ("?x", "ancestor", "?z"),
        ),
    ]
}

/// A parent chain of `n` nodes: closure n(n-1)/2 ancestors.
fn chain_seed(n: usize) -> Vec<Triple> {
    (0..n.saturating_sub(1))
        .map(|i| Triple::new(format!("p{i}"), "parent", format!("p{}", i + 1)))
        .collect()
}

// ---- memory probe (Linux) ------------------------------------------------------------------

/// Current resident set (`VmRSS`) in MB, read from /proc/self/status. 0.0 if unavailable.
fn rss_mb() -> f64 {
    read_status_kb("VmRSS:") / 1024.0
}
/// Peak resident set (`VmHWM`) in MB — the high-water mark over the whole process.
fn peak_rss_mb() -> f64 {
    read_status_kb("VmHWM:") / 1024.0
}
fn read_status_kb(key: &str) -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with(key))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|kb| kb.parse::<f64>().ok())
        })
        .unwrap_or(0.0)
}

// ---- one scale point -----------------------------------------------------------------------

struct ScaleResult {
    ruleset: &'static str,
    param: usize,
    seed: usize,
    derived: usize,
    total: usize,
    median_ms: f64,
    derivations_per_sec: f64,
    rss_mb: f64,
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Run `trials` closures of `(rules, seed)`, dropping each saturation before the next so the
/// RSS sampled after the final trial reflects ONE saturation, not the sum.
fn run_scale(
    ruleset: &'static str,
    param: usize,
    rules: &[IdRule],
    seed: Vec<Triple>,
    trials: usize,
) -> ScaleResult {
    let seed_n = seed.len();
    let mut times = Vec::with_capacity(trials);
    let mut derived = 0usize;
    let mut total = 0usize;
    let mut rss = 0.0;
    for t in 0..trials {
        let s = seed.clone();
        let start = Instant::now();
        let sat = forward_chain(rules, s);
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        times.push(ms);
        derived = sat.derived_count();
        total = sat.facts.len();
        if t == trials - 1 {
            rss = rss_mb(); // sample while the final saturation is still alive
        }
        drop(sat);
    }
    let median_ms = median(times);
    let r = ScaleResult {
        ruleset,
        param,
        seed: seed_n,
        derived,
        total,
        median_ms,
        derivations_per_sec: if median_ms > 0.0 {
            derived as f64 / (median_ms / 1000.0)
        } else {
            0.0
        },
        rss_mb: rss,
    };
    // Print incrementally so a later scale timing out never loses earlier data.
    eprintln!(
        "  → {:<22} param={:>7} seed={:>8} derived={:>9} median={:>9.2}ms {:>12.0} derivs/s rss={:.1}MB",
        r.ruleset, r.param, r.seed, r.derived, r.median_ms, r.derivations_per_sec, r.rss_mb
    );
    r
}

fn main() {
    // Scale points chosen so derived-fact counts land near 10k / 100k / 1M.
    // LINEAR: derived = n·(K-1) = n·5  → n = 2k / 20k / 200k.
    // QUADRATIC: derived ≈ n²/2       → n = 142 / 448 / 1414.
    let mut results = Vec::new();

    // LINEAR is the realistic shape and the 1M-derived headline (200k entities × 5 = 1M).
    let sub_rules = subsumption_rules();
    for (n, trials) in [(2_000usize, 5), (20_000, 3), (200_000, 2)] {
        eprintln!("[ex4735] subsumption n_entities={n} …");
        results.push(run_scale(
            "subsumption-linear",
            n,
            &sub_rules,
            subsumption_seed(n),
            trials,
        ));
    }

    // QUADRATIC is the recursive-fixpoint stress test; it saturates far more slowly per
    // derived fact, so it's capped where it completes in budget and the curve is reported
    // honestly (the n²/iteration re-fire cost is the finding, not a bug to hide).
    // n≥800 does NOT finish in a multi-minute budget (the naive-refire n²/iteration wall —
    // recorded as a finding in the report, not run here so JSON always emits).
    let tr_rules = transitive_rules();
    for (n, trials) in [(142usize, 3), (448, 1)] {
        eprintln!("[ex4735] transitive chain n={n} …");
        results.push(run_scale(
            "transitive-quadratic",
            n,
            &tr_rules,
            chain_seed(n),
            trials,
        ));
    }

    let peak = peak_rss_mb();

    // Human summary on stderr.
    eprintln!("\nEX-4735 forward-chain scale benchmark (Arrow engine, release)");
    eprintln!(
        "{:<22} {:>8} {:>10} {:>10} {:>11} {:>14} {:>9}",
        "ruleset", "param", "seed", "derived", "median_ms", "derivs/sec", "rss_mb"
    );
    for r in &results {
        eprintln!(
            "{:<22} {:>8} {:>10} {:>10} {:>11.2} {:>14.0} {:>9.1}",
            r.ruleset, r.param, r.seed, r.derived, r.median_ms, r.derivations_per_sec, r.rss_mb
        );
    }
    eprintln!("peak RSS over run: {peak:.1} MB");
    let hit_1m = results.iter().any(|r| r.derived >= 1_000_000);
    eprintln!(
        "reached ≥1M derived facts: {} (H-4725 primary refutation = OOM/super-linear before 1M)",
        if hit_1m { "YES" } else { "NO" }
    );

    // Eval JSON on stdout.
    let rows: Vec<String> = results
        .iter()
        .map(|r| {
            format!(
                "{{\"ruleset\":\"{}\",\"param\":{},\"seed\":{},\"derived\":{},\"total\":{},\"median_ms\":{:.2},\"derivations_per_sec\":{:.0},\"rss_mb\":{:.1}}}",
                r.ruleset, r.param, r.seed, r.derived, r.total, r.median_ms, r.derivations_per_sec, r.rss_mb
            )
        })
        .collect();
    println!(
        "{{\n  \"experiment\": \"EX-4735 / EXPR-4727\",\n  \"hypothesis\": \"H-4725\",\n  \"engine\": \"nusy_forward_chain::forward_chain (Arrow, release)\",\n  \"rulesets\": {{\"linear\": \"subsumption type(x,super):-type(x,sub),subClassOf(sub,super), K={K_LEVELS}\", \"quadratic\": \"transitive ancestor(x,z):-parent(x,y),ancestor(y,z)\"}},\n  \"reached_1m_derived\": {hit_1m},\n  \"peak_rss_mb\": {peak:.1},\n  \"scales\": [{}],\n  \"baselines_not_run\": [\"oxigraph\", \"ascent\", \"crepe\"],\n  \"baselines_note\": \"FOSS-comparison baselines deferred to a follow-up (external deps); v0.1 establishes the NuSy-side scaling curve only — logged, not silently omitted.\",\n  \"caveats\": [\"Single-process wall-clock medians; RSS is VmRSS sampled after the final trial's closure (one saturation live), peak is process VmHWM.\", \"Closure complexity is by-design: linear = N*(K-1), quadratic = n(n-1)/2 — so derived counts are meaningful, not accidental.\"]\n}}",
        rows.join(",")
    );

    // The load-bearing assertion: the engine must reach 1M derived facts without OOM.
    assert!(
        hit_1m,
        "H-4725 refuted at this config: engine did not reach 1M derived facts"
    );
}
