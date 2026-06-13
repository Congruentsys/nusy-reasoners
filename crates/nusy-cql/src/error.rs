//! Error types for the CQL-analog expression language.

use thiserror::Error;

/// A parse-time error (lexing or parsing).
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// An unexpected character was encountered while tokenizing.
    #[error("unexpected character {ch:?} at byte offset {pos}")]
    UnexpectedChar { ch: char, pos: usize },

    /// A string or code literal was opened but never closed.
    #[error("unterminated string literal starting at byte offset {pos}")]
    UnterminatedString { pos: usize },

    /// The parser reached the end of input while still expecting more tokens.
    #[error("unexpected end of input: expected {expected}")]
    UnexpectedEof { expected: String },

    /// A token was found where a different one was expected.
    #[error("unexpected token {found:?}: expected {expected}")]
    UnexpectedToken { found: String, expected: String },

    /// A numeric literal could not be parsed.
    #[error("invalid number literal {literal:?}: {reason}")]
    InvalidNumber { literal: String, reason: String },
}

/// An evaluation-time error.
///
/// Note: a *missing* fact or a comparison involving an unknown operand is **not**
/// an error — it evaluates to [`crate::Value::Null`] per CQL three-valued logic.
/// `EvalError` is reserved for genuine type misuse the parser cannot catch.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum EvalError {
    /// An operator was applied to operand types it does not support
    /// (e.g. `during` on two plain integers with no interval).
    #[error("type error in {op}: {detail}")]
    TypeError { op: String, detail: String },
}
