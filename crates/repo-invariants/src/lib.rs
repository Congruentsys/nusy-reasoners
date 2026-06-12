//! Repository invariants, CI-enforced from day zero so they cannot silently drift:
//! the license and governance files the Wave-1 code will land into. This crate is
//! deleted when the first real crate moves in — its tests migrate to that crate's CI.

#[cfg(test)]
mod tests {
    /// Repository root, resolved from this crate's manifest.
    fn repo_root() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .expect("repo root resolves")
    }

    fn read(rel: &str) -> String {
        std::fs::read_to_string(repo_root().join(rel))
            .unwrap_or_else(|e| panic!("{rel} must exist at the repo root: {e}"))
    }

    #[test]
    fn license_is_mit_with_the_ruled_copyright_line() {
        let license = read("LICENSE");
        assert!(
            license.starts_with("MIT License"),
            "license must be MIT (ruling D5)"
        );
        assert!(
            license.contains("Hank Head / Congruent Systems LLC"),
            "copyright line must match the SG-4716 ruling"
        );
    }

    #[test]
    fn governance_files_exist_and_carry_the_guarantee_invariant() {
        assert!(read("README.md").contains("reasoning is executed, not generated"));
        assert!(read("CONTRIBUTING.md").contains("guarantee invariant"));
    }
}
