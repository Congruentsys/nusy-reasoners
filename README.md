# nusy-reasoners

**Proof-carrying reasoning engines over [Apache Arrow](https://arrow.apache.org/): derivations
you can audit, abstention you can trust.**

Most systems that "reason" generate token streams and ask you to believe them. This suite takes
the other branch: **reasoning is executed, not generated**. A query is answered by an engine
that derives the answer over a knowledge graph and hands back the complete derivation — every
rule that fired, every premise it consumed, down to the seed facts — or it **abstains, loudly**.
A claim without a proof is never presented as one.

The whole family speaks one contract:

- **Answer + proof trace + provability tag + provenance** — the proof is the universal currency.
- **Provability is computed from the proof, never asserted** — a heuristic stage is structurally
  unable to mint `Proven`, and composing reasoners propagates the *minimum* guarantee
  (a best-effort link is never laundered into a proof).
- **Competence as data** — each reasoner declares the query shapes it soundly covers; a router
  dispatches by envelope and certifies the result, or abstains.

The engines are generic over three shapes — **rules** (Horn-style forward chaining to fixpoint,
semi-naive), **decision graphs / DAGs** (causal identification via do-calculus, counterfactual
estimates that honestly report as estimates), and **ontology inference** (subsumption,
terminology) — on an Arrow-native substrate measured to a million derived facts in seconds on
a laptop.

## Quickstart

> **Pre-release.** The suite ships as a git dependency (Wave-2, `v0.2.0-rc`); crates are not
> yet on crates.io. Add just the pieces you need from the workspace:

```toml
[dependencies]
nusy-reasoner = { git = "https://github.com/Congruentsys/nusy-reasoners" }
nusy-unify    = { git = "https://github.com/Congruentsys/nusy-reasoners" }
```

Every reasoner returns an `Answer`, and its **provability is computed from the proof, not
asserted**. A complete derivation yields `Proven`; neural evidence is always `Heuristic` —
approximate output is *structurally* unable to mint a proof:

```rust
use nusy_reasoner::*;
use nusy_unify::Triple;

// A symbolic reasoner answers WITH a complete derivation → Proven.
let proven = Answer {
    value: Some(Triple::new("p1", "at_risk", "fall")),
    proof: ProofTrace::Derivation(DerivationTrace::Derived {
        conclusion: Triple::new("p1", "at_risk", "fall"),
        rule_id: "at-risk-fall".into(),
        premises: vec![DerivationTrace::Axiom(
            Triple::new("p1", "has_condition", "osteoporosis"),
        )],
    }),
    provenance: vec!["chunk-7".into()],
};
assert_eq!(proven.provability(), Provability::Proven);

// A neural reasoner answers WITH evidence → Heuristic, never Proven.
let neural = Answer {
    value: Some(Triple::new("p1", "at_risk", "stroke")),
    proof: ProofTrace::Evidence { confidence: 0.82, why: vec!["age + bp pattern".into()] },
    provenance: vec![],
};
assert_eq!(neural.provability(), Provability::Heuristic);
```

Build and run the suite from a clone:

```bash
git clone https://github.com/Congruentsys/nusy-reasoners
cd nusy-reasoners
cargo test --workspace                                        # conformance + reasoning batteries
cargo run -p nusy-cpg --example ex4613_gate_coverage_routing  # a worked, gated derivation
```

## Status

**Pre-0.1 scaffolding.** The Wave-1 crates move here from the NuSy monorepo (a relocation with
history, not a mirror): `nusy-unify`, `nusy-forward-chain`, `nusy-provenance`, `nusy-gate`,
`nusy-router`, `nusy-reasoning-causal`, `nusy-cql`, `nusy-cpg`, `nusy-cog-computable`,
`nusy-reasoner`, `nusy-reasoner-adapters`. Until then this repository proves the licensing,
CI, and governance the code will land into. Releases are git tags + GitHub releases.

## Ecosystem

Part of the open-source stack behind [Congruent Systems](https://congruentsys.com):

- **[nusy-kanban](https://github.com/hankh95/nusy-kanban)** — Arrow-native, distributed kanban for multi-agent teams, with a built-in Hypothesis-Driven-Development research workflow.
- **[noesis-ship](https://github.com/hankh95/noesis-ship)** — pluggable multi-agent communication platform on NATS (EventBus, KV, object store).
- **[acf-framework](https://github.com/hankh95/acf-framework)** — a graph-based framework for measuring AI capability against human professional standards.

## License

[MIT](LICENSE) — © 2026 Hank Head / Congruent Systems LLC.
