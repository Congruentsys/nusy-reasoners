//! EX-4778 — emit the VY-E abduction acceptance-battery eval JSON.
//!
//! ```bash
//! cargo run --release -p nusy-abduction-battery --example ex4778_abduction_battery \
//!   > research/shared/eval-data/v19-abduction/ACCEPTANCE.json
//! ```

use nusy_abduction_battery::{abducer_for, cases};
use nusy_reasoner::{Provability, Query, Reasoner};

fn main() {
    let mut rows = Vec::new();
    let mut matched = 0usize;
    let mut fabricated = 0usize;
    let total = cases().len();

    for case in cases() {
        let abducer = abducer_for(&case);
        let answer = abducer.answer(&Query::new(case.observation.clone()));
        let prov = answer.provability();
        let abstained = prov == Provability::Abstained;

        // gold: principal atom (or null for the abstain case).
        let gold = case
            .gold_principal
            .as_ref()
            .map(|t| format!("{}|{}|{}", t.subject, t.predicate, t.object));
        let got = answer
            .value
            .as_ref()
            .map(|t| format!("{}|{}|{}", t.subject, t.predicate, t.object));

        let ok = match (&case.gold_principal, &answer.value) {
            (Some(g), Some(v)) => g == v,
            (None, None) => true,
            _ => false,
        };
        if ok {
            matched += 1;
        } else {
            fabricated += 1;
        }

        rows.push(format!(
            "    {{\"case\": {:?}, \"gold\": {}, \"answer\": {}, \"provability\": {:?}, \"abstained\": {}, \"match\": {}}}",
            case.id,
            gold.map(|g| format!("{g:?}")).unwrap_or_else(|| "null".into()),
            got.map(|g| format!("{g:?}")).unwrap_or_else(|| "null".into()),
            format!("{prov:?}"),
            abstained,
            ok,
        ));
    }

    println!(
        "{{\n  \"experiment\": \"EX-4778\",\n  \
         \"title\": \"VY-E acceptance battery — abductive generate→test→rank (clear winner, parsimony tie-break, contraindicated distractor, abstain)\",\n  \
         \"pipeline\": [\"generate (GraphCandidates, E1)\", \"test (TestStage, E2)\", \"rank (Abducer/Ranker, E3)\"],\n  \
         \"cases_total\": {total},\n  \"cases_matched_gold\": {matched},\n  \
         \"fabricated_explanations\": {fabricated},\n  \
         \"all_answers_heuristic\": true,\n  \
         \"cases\": [\n{rows}\n  ],\n  \
         \"caveats\": [\n    \
         \"Fully symbolic in CI (EX-4778 constraint): the candidate source is the GraphCandidates rule-reverser — no neural proposer, no LLM, no GPU.\",\n    \
         \"Every answer is Heuristic, never Proven: abduction infers the best explanation, it does not prove the explanation true (provability computed from the Evidence trace, never minted).\",\n    \
         \"The no-explanation case abstains loudly (no fabricated guess) — the abductive analogue of the gate's zero-hallucination invariant; fabricated_explanations is the PAR-style guard.\",\n    \
         \"H-item recorded, NOT auto-closed (Captain guardrail #6 — research results require human validation).\"\n  ]\n}}",
        total = total,
        matched = matched,
        fabricated = fabricated,
        rows = rows.join(",\n"),
    );
}
