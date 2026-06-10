//! Recursive-descent parser for the CQL-analog expression language.
//!
//! Grammar (lowest to highest precedence):
//! ```text
//! expr        := or_expr
//! or_expr     := and_expr ("or" and_expr)*
//! and_expr    := not_expr ("and" not_expr)*
//! not_expr    := "not" not_expr | rel_expr
//! rel_expr    := primary ( rel_op primary )?            // non-associative, at most one
//! rel_op      := "=" | "!=" | "<" | "<=" | ">" | ">="
//!              | "subsumes" | "before" | "after" | "during" | "overlaps"
//!              | "in" (string | list)
//! primary     := literal | "exists" "(" expr ")" | ctor | property
//!              | "(" expr ("," expr)* ")"               // parenthesised expr or list literal
//! ctor        := ("Code"|"DateTime"|"Quantity"|"Interval") "(" args ")"
//! property    := ident ("." ident)*
//! ```

use crate::ast::{Code, CompOp, Expr, TemporalOp, Value};
use crate::error::ParseError;
use crate::lexer::{lex, Token};

/// Parse a source string into an [`Expr`].
pub fn parse(input: &str) -> Result<Expr, ParseError> {
    let tokens = lex(input)?;
    let mut p = Parser { tokens, pos: 0 };
    let expr = p.parse_or()?;
    if p.pos != p.tokens.len() {
        return Err(ParseError::UnexpectedToken {
            found: format!("{:?}", p.tokens[p.pos]),
            expected: "end of input".to_string(),
        });
    }
    Ok(expr)
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<Token> {
        let t = self.tokens.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn expect(&mut self, want: &Token, ctx: &str) -> Result<(), ParseError> {
        match self.peek() {
            Some(t) if t == want => {
                self.pos += 1;
                Ok(())
            }
            Some(t) => Err(ParseError::UnexpectedToken {
                found: format!("{t:?}"),
                expected: format!("{want:?} ({ctx})"),
            }),
            None => Err(ParseError::UnexpectedEof { expected: format!("{want:?} ({ctx})") }),
        }
    }

    fn parse_or(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_and()?;
        while matches!(self.peek(), Some(Token::Or)) {
            self.pos += 1;
            let rhs = self.parse_and()?;
            lhs = Expr::Or(Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_and(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_not()?;
        while matches!(self.peek(), Some(Token::And)) {
            self.pos += 1;
            let rhs = self.parse_not()?;
            lhs = Expr::And(Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_not(&mut self) -> Result<Expr, ParseError> {
        if matches!(self.peek(), Some(Token::Not)) {
            self.pos += 1;
            let inner = self.parse_not()?;
            return Ok(Expr::Not(Box::new(inner)));
        }
        self.parse_rel()
    }

    fn parse_rel(&mut self) -> Result<Expr, ParseError> {
        let lhs = self.parse_primary()?;
        let op = match self.peek() {
            Some(Token::Eq) => Some(CompOp::Eq),
            Some(Token::Ne) => Some(CompOp::Ne),
            Some(Token::Lt) => Some(CompOp::Lt),
            Some(Token::Le) => Some(CompOp::Le),
            Some(Token::Gt) => Some(CompOp::Gt),
            Some(Token::Ge) => Some(CompOp::Ge),
            _ => None,
        };
        if let Some(op) = op {
            self.pos += 1;
            let rhs = self.parse_primary()?;
            return Ok(Expr::Compare { op, lhs: Box::new(lhs), rhs: Box::new(rhs) });
        }
        match self.peek() {
            Some(Token::Subsumes) => {
                self.pos += 1;
                let rhs = self.parse_primary()?;
                Ok(Expr::Subsumes { ancestor: Box::new(lhs), descendant: Box::new(rhs) })
            }
            Some(Token::Before) => self.temporal(lhs, TemporalOp::Before),
            Some(Token::After) => self.temporal(lhs, TemporalOp::After),
            Some(Token::During) => self.temporal(lhs, TemporalOp::During),
            Some(Token::Overlaps) => self.temporal(lhs, TemporalOp::Overlaps),
            Some(Token::In) => {
                self.pos += 1;
                match self.peek() {
                    Some(Token::Str(name)) => {
                        let name = name.clone();
                        self.pos += 1;
                        Ok(Expr::InValueSet { value: Box::new(lhs), valueset: name })
                    }
                    Some(Token::LParen) => {
                        let list = self.parse_primary()?;
                        Ok(Expr::InList { value: Box::new(lhs), list: Box::new(list) })
                    }
                    Some(t) => Err(ParseError::UnexpectedToken {
                        found: format!("{t:?}"),
                        expected: "a value-set name (string) or a list literal after `in`".to_string(),
                    }),
                    None => Err(ParseError::UnexpectedEof {
                        expected: "a value-set name or list after `in`".to_string(),
                    }),
                }
            }
            _ => Ok(lhs),
        }
    }

    fn temporal(&mut self, lhs: Expr, op: TemporalOp) -> Result<Expr, ParseError> {
        self.pos += 1;
        let rhs = self.parse_primary()?;
        Ok(Expr::Temporal { op, lhs: Box::new(lhs), rhs: Box::new(rhs) })
    }

    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        match self.peek() {
            Some(Token::True) => {
                self.pos += 1;
                Ok(Expr::Literal(Value::Boolean(true)))
            }
            Some(Token::False) => {
                self.pos += 1;
                Ok(Expr::Literal(Value::Boolean(false)))
            }
            Some(Token::Null) => {
                self.pos += 1;
                Ok(Expr::Literal(Value::Null))
            }
            Some(Token::Int(n)) => {
                let n = *n;
                self.pos += 1;
                Ok(Expr::Literal(Value::Integer(n)))
            }
            Some(Token::Decimal(d)) => {
                let d = *d;
                self.pos += 1;
                Ok(Expr::Literal(Value::Decimal(d)))
            }
            Some(Token::Str(s)) => {
                let s = s.clone();
                self.pos += 1;
                Ok(Expr::Literal(Value::Str(s)))
            }
            Some(Token::Exists) => {
                self.pos += 1;
                self.expect(&Token::LParen, "after `exists`")?;
                let inner = self.parse_or()?;
                self.expect(&Token::RParen, "closing `exists(`")?;
                Ok(Expr::Exists(Box::new(inner)))
            }
            Some(Token::LParen) => {
                self.pos += 1;
                let first = self.parse_or()?;
                if matches!(self.peek(), Some(Token::Comma)) {
                    let mut items = vec![first];
                    while matches!(self.peek(), Some(Token::Comma)) {
                        self.pos += 1;
                        items.push(self.parse_or()?);
                    }
                    self.expect(&Token::RParen, "closing list literal")?;
                    Ok(Expr::ListLit(items))
                } else {
                    self.expect(&Token::RParen, "closing `(`")?;
                    Ok(first)
                }
            }
            Some(Token::Ident(name)) => {
                let name = name.clone();
                self.pos += 1;
                if matches!(self.peek(), Some(Token::LParen)) {
                    self.parse_ctor(&name)
                } else {
                    // Property path: entity ("." segment)*
                    let mut path = Vec::new();
                    while matches!(self.peek(), Some(Token::Dot)) {
                        self.pos += 1;
                        match self.advance() {
                            Some(Token::Ident(seg)) => path.push(seg),
                            other => {
                                return Err(ParseError::UnexpectedToken {
                                    found: format!("{other:?}"),
                                    expected: "a property segment after `.`".to_string(),
                                });
                            }
                        }
                    }
                    Ok(Expr::Property { entity: name, path })
                }
            }
            Some(t) => Err(ParseError::UnexpectedToken {
                found: format!("{t:?}"),
                expected: "a literal, property, constructor, `exists`, or `(`".to_string(),
            }),
            None => Err(ParseError::UnexpectedEof { expected: "an expression".to_string() }),
        }
    }

    /// Parse a comma-separated argument list (the opening `(` is the next token).
    fn parse_ctor(&mut self, name: &str) -> Result<Expr, ParseError> {
        self.expect(&Token::LParen, "constructor arguments")?;
        let mut args = Vec::new();
        if !matches!(self.peek(), Some(Token::RParen)) {
            args.push(self.parse_or()?);
            while matches!(self.peek(), Some(Token::Comma)) {
                self.pos += 1;
                args.push(self.parse_or()?);
            }
        }
        self.expect(&Token::RParen, "closing constructor")?;

        match name {
            "Code" => {
                let [sys, code] = take_string_args::<2>(name, &args)?;
                Ok(Expr::Literal(Value::Code(Code::new(sys, code))))
            }
            "DateTime" => {
                let n = take_int_arg(name, &args)?;
                Ok(Expr::Literal(Value::DateTime(n)))
            }
            "Quantity" => {
                if args.len() != 2 {
                    return Err(arity_err(name, 2, args.len()));
                }
                let mag = take_number(name, &args[0])?;
                let unit = take_string(name, &args[1])?;
                Ok(Expr::Literal(Value::Quantity(mag, unit)))
            }
            "Interval" => {
                if args.len() != 2 {
                    return Err(arity_err(name, 2, args.len()));
                }
                let lo = lit_value(name, &args[0])?;
                let hi = lit_value(name, &args[1])?;
                Ok(Expr::Literal(Value::Interval(Box::new(lo), Box::new(hi))))
            }
            other => Err(ParseError::UnexpectedToken {
                found: format!("{other}(...)"),
                expected: "a known constructor: Code, DateTime, Quantity, or Interval".to_string(),
            }),
        }
    }
}

fn arity_err(name: &str, want: usize, got: usize) -> ParseError {
    ParseError::UnexpectedToken {
        found: format!("{got} argument(s)"),
        expected: format!("{want} argument(s) for {name}(...)"),
    }
}

fn lit_value(ctor: &str, e: &Expr) -> Result<Value, ParseError> {
    match e {
        Expr::Literal(v) => Ok(v.clone()),
        _ => Err(ParseError::UnexpectedToken {
            found: "non-literal expression".to_string(),
            expected: format!("a literal argument to {ctor}(...)"),
        }),
    }
}

fn take_string(ctor: &str, e: &Expr) -> Result<String, ParseError> {
    match lit_value(ctor, e)? {
        Value::Str(s) => Ok(s),
        _ => Err(ParseError::UnexpectedToken {
            found: "non-string literal".to_string(),
            expected: format!("a string argument to {ctor}(...)"),
        }),
    }
}

fn take_number(ctor: &str, e: &Expr) -> Result<f64, ParseError> {
    match lit_value(ctor, e)? {
        Value::Integer(n) => Ok(n as f64),
        Value::Decimal(d) => Ok(d),
        _ => Err(ParseError::UnexpectedToken {
            found: "non-numeric literal".to_string(),
            expected: format!("a number argument to {ctor}(...)"),
        }),
    }
}

fn take_int_arg(ctor: &str, args: &[Expr]) -> Result<i64, ParseError> {
    if args.len() != 1 {
        return Err(arity_err(ctor, 1, args.len()));
    }
    match lit_value(ctor, &args[0])? {
        Value::Integer(n) => Ok(n),
        _ => Err(ParseError::UnexpectedToken {
            found: "non-integer literal".to_string(),
            expected: format!("an integer argument to {ctor}(...)"),
        }),
    }
}

fn take_string_args<const N: usize>(ctor: &str, args: &[Expr]) -> Result<[String; N], ParseError> {
    if args.len() != N {
        return Err(arity_err(ctor, N, args.len()));
    }
    let mut out: [String; N] = std::array::from_fn(|_| String::new());
    for (slot, arg) in out.iter_mut().zip(args.iter()) {
        *slot = take_string(ctor, arg)?;
    }
    Ok(out)
}
