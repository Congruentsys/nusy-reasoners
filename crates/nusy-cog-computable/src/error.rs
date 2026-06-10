//! Error type for de-reification.

use thiserror::Error;

/// A failure reconstructing computable Y2 artifacts from triples.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum DereifyError {
    /// The reified triples are structurally incomplete or inconsistent (a required
    /// predicate is missing, or an ordering `y2:index` is absent/unparseable).
    #[error("malformed reification: {0}")]
    Malformed(String),
}
