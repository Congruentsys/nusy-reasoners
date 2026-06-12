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

## Status

**Pre-0.1 scaffolding.** The Wave-1 crates move here from the NuSy monorepo (a relocation with
history, not a mirror): `nusy-unify`, `nusy-forward-chain`, `nusy-provenance`, `nusy-gate`,
`nusy-router`, `nusy-reasoning-causal`, `nusy-cql`, `nusy-cpg`, `nusy-cog-computable`,
`nusy-reasoner`, `nusy-reasoner-adapters`. Until then this repository proves the licensing,
CI, and governance the code will land into. Releases are git tags + GitHub releases.

## License

[MIT](LICENSE) — © 2026 Hank Head / Congruent Systems LLC.
