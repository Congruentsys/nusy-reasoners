//! Tokenizer for the CQL-analog expression language.

use crate::error::ParseError;

/// A lexical token.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Literals
    /// An integer literal.
    Int(i64),
    /// A decimal literal.
    Decimal(f64),
    /// A single- or double-quoted string literal.
    Str(String),
    /// An identifier (entity / property segment / keyword candidate).
    Ident(String),

    // Keywords
    /// `and`
    And,
    /// `or`
    Or,
    /// `not`
    Not,
    /// `in`
    In,
    /// `exists`
    Exists,
    /// `subsumes`
    Subsumes,
    /// `before`
    Before,
    /// `after`
    After,
    /// `during`
    During,
    /// `overlaps`
    Overlaps,
    /// `true`
    True,
    /// `false`
    False,
    /// `null`
    Null,

    // Punctuation / operators
    /// `(`
    LParen,
    /// `)`
    RParen,
    /// `,`
    Comma,
    /// `.`
    Dot,
    /// `=`
    Eq,
    /// `!=`
    Ne,
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `>`
    Gt,
    /// `>=`
    Ge,
}

/// Tokenize `input` into a flat token stream. Whitespace is skipped.
pub fn lex(input: &str) -> Result<Vec<Token>, ParseError> {
    let bytes = input.as_bytes();
    let chars: Vec<char> = input.chars().collect();
    // Map char index → byte offset for error reporting.
    let mut byte_offsets = Vec::with_capacity(chars.len() + 1);
    let mut acc = 0usize;
    for c in &chars {
        byte_offsets.push(acc);
        acc += c.len_utf8();
    }
    byte_offsets.push(bytes.len());

    let mut tokens = Vec::new();
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        match c {
            '(' => {
                tokens.push(Token::LParen);
                i += 1;
            }
            ')' => {
                tokens.push(Token::RParen);
                i += 1;
            }
            ',' => {
                tokens.push(Token::Comma);
                i += 1;
            }
            '.' => {
                tokens.push(Token::Dot);
                i += 1;
            }
            '=' => {
                tokens.push(Token::Eq);
                i += 1;
            }
            '!' => {
                if chars.get(i + 1) == Some(&'=') {
                    tokens.push(Token::Ne);
                    i += 2;
                } else {
                    return Err(ParseError::UnexpectedChar {
                        ch: '!',
                        pos: byte_offsets[i],
                    });
                }
            }
            '<' => {
                if chars.get(i + 1) == Some(&'=') {
                    tokens.push(Token::Le);
                    i += 2;
                } else {
                    tokens.push(Token::Lt);
                    i += 1;
                }
            }
            '>' => {
                if chars.get(i + 1) == Some(&'=') {
                    tokens.push(Token::Ge);
                    i += 2;
                } else {
                    tokens.push(Token::Gt);
                    i += 1;
                }
            }
            '\'' | '"' => {
                let quote = c;
                let start = byte_offsets[i];
                i += 1;
                let mut s = String::new();
                let mut closed = false;
                while i < chars.len() {
                    let ch = chars[i];
                    if ch == '\\' && i + 1 < chars.len() {
                        // Minimal escapes: \', \", \\, \n, \t.
                        let next = chars[i + 1];
                        let mapped = match next {
                            'n' => '\n',
                            't' => '\t',
                            other => other,
                        };
                        s.push(mapped);
                        i += 2;
                        continue;
                    }
                    if ch == quote {
                        closed = true;
                        i += 1;
                        break;
                    }
                    s.push(ch);
                    i += 1;
                }
                if !closed {
                    return Err(ParseError::UnterminatedString { pos: start });
                }
                tokens.push(Token::Str(s));
            }
            c if c.is_ascii_digit() => {
                let start = i;
                let mut seen_dot = false;
                while i < chars.len() {
                    let ch = chars[i];
                    if ch.is_ascii_digit() {
                        i += 1;
                    } else if ch == '.'
                        && !seen_dot
                        && chars.get(i + 1).is_some_and(|d| d.is_ascii_digit())
                    {
                        seen_dot = true;
                        i += 1;
                    } else {
                        break;
                    }
                }
                let literal: String = chars[start..i].iter().collect();
                if seen_dot {
                    let v: f64 = literal.parse().map_err(|e: std::num::ParseFloatError| {
                        ParseError::InvalidNumber {
                            literal: literal.clone(),
                            reason: e.to_string(),
                        }
                    })?;
                    tokens.push(Token::Decimal(v));
                } else {
                    let v: i64 = literal.parse().map_err(|e: std::num::ParseIntError| {
                        ParseError::InvalidNumber {
                            literal: literal.clone(),
                            reason: e.to_string(),
                        }
                    })?;
                    tokens.push(Token::Int(v));
                }
            }
            c if c.is_alphabetic() || c == '_' => {
                let start = i;
                while i < chars.len() {
                    let ch = chars[i];
                    if ch.is_alphanumeric() || ch == '_' {
                        i += 1;
                    } else {
                        break;
                    }
                }
                let word: String = chars[start..i].iter().collect();
                let token = match word.to_ascii_lowercase().as_str() {
                    "and" => Token::And,
                    "or" => Token::Or,
                    "not" => Token::Not,
                    "in" => Token::In,
                    "exists" => Token::Exists,
                    "subsumes" => Token::Subsumes,
                    "before" => Token::Before,
                    "after" => Token::After,
                    "during" => Token::During,
                    "overlaps" => Token::Overlaps,
                    "true" => Token::True,
                    "false" => Token::False,
                    "null" => Token::Null,
                    _ => Token::Ident(word),
                };
                tokens.push(token);
            }
            other => {
                return Err(ParseError::UnexpectedChar {
                    ch: other,
                    pos: byte_offsets[i],
                });
            }
        }
    }
    Ok(tokens)
}
