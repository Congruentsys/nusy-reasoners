# Contributing

Thanks for your interest. Ground rules — short, because the proofs do the arguing:

- **License:** by contributing you agree your contribution is MIT-licensed (see
  [LICENSE](LICENSE)). No CLA.
- **Maintainership:** each crate names a steward in its README; the steward merges. Until the
  0.1 release the steward for the workspace is the Congruent Systems team.
- **The bar every PR meets:**
  - `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, and
    `cargo fmt --all --check` are green.
  - New reasoning behavior ships with tests that would fail if the behavior were wrong —
    including the negative direction (what must *not* be derivable).
  - **The guarantee invariant is non-negotiable:** nothing may mint `Proven` without a complete
    derivation, and no change may let a heuristic answer masquerade as one. PRs that weaken
    this are closed regardless of how useful they are.
- **Conduct:** be direct, be kind, argue from evidence.
