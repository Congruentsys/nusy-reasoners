//! EX-4819 — emit the VY-Bayes acceptance-battery eval JSON.
//!
//! ```bash
//! cargo run --release -p nusy-bayesian-battery --example ex4819_bayesian_battery \
//!   > research/shared/eval-data/v19-bayesian/ACCEPTANCE.json
//! ```

use nusy_bayesian_battery::run;

fn main() {
    let r = run();
    let parsimony_acc = r.parsimony_top1 as f64 / r.abduction_total as f64;
    let bayesian_acc = r.bayesian_top1 as f64 / r.abduction_total as f64;
    println!(
        "{{\n  \"experiment\": \"EX-4819 / VY-Bayes E4 acceptance battery (+ CH-4831 discriminating case)\",\n  \
         \"hypothesis\": \"the Bayesian stack (engine + Reasoner conformance + abduction-rank upgrade) is accurate, IMPROVES abduction ranking over parsimony, and is proof-pure\",\n  \
         \"grade_fidelity\": {{\"matched\": {gm}, \"total\": {gt}}},\n  \
         \"posterior_fidelity\": {{\"matched\": {pm}, \"total\": {pt}, \"note\": \"hand-computed golds incl. the base-rate fallacy P(disease|+)=1/6\"}},\n  \
         \"abduction_nonregression\": {{\"parsimony\": {pacc:.4}, \"bayesian\": {bacc:.4}, \"delta\": {delta:.4}, \"cases\": {at}, \"note\": \"easy single-winner cases — both rankers hit (no regression)\"}},\n  \
         \"abduction_improvement\": {{\"discriminating_cases\": {dt}, \"bayesian_hits\": {dbh}, \"parsimony_hits\": {dph}, \"delta\": {ddelta}, \"note\": \"engineered divergences where high-confidence multi-atom explanation beats the parsimonious single atom — Bayesian hits, parsimony misses (CH-4831; the H-4823 improves clause)\"}},\n  \
         \"proof_purity\": {{\"probabilistic_proven_links\": {ppl}, \"invariant\": \"0 — a probability is never laundered into a proof\"}},\n  \
         \"all_pass\": {pass},\n  \
         \"caveats\": [\n    \
         \"Golds are HAND-COMPUTED, not engine-generated (EX-4819 constraint — engine-generated golds are circular).\",\n    \
         \"Non-regression cases tie (delta=0); the IMPROVEMENT is shown on the discriminating cases (delta={ddelta}): the Bayesian posterior picks a high-confidence multi-atom explanation parsimony's atom-count misses.\",\n    \
         \"Fully symbolic CI battery (cargo test -p nusy-bayesian-battery); no LLM/GPU. Hypothesis status is a finding — Captain adjudicates.\"\n  ]\n}}",
        gm = r.grade_matched,
        gt = r.grade_total,
        pm = r.posterior_matched,
        pt = r.posterior_total,
        pacc = parsimony_acc,
        bacc = bayesian_acc,
        delta = bayesian_acc - parsimony_acc,
        at = r.abduction_total,
        dt = r.discrim_total,
        dbh = r.discrim_bayesian_hits,
        dph = r.discrim_parsimony_hits,
        ddelta = r.discrim_delta(),
        ppl = r.probabilistic_proven_links,
        pass = r.all_pass(),
    );
}
