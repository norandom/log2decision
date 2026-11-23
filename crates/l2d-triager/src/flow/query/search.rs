//! The `search` command. Bare terms and quoted phrases follow Splunk's
//! segmentation, `TERM()` and `CASE()` adjust it, fields compare with `=`,
//! `!=`, `<`, `<=`, `>`, `>=` and `IN`, `earliest`/`latest` bound occurrence
//! time, and the operators bind like SPL: NOT, then OR, then AND.
use std::{cmp::Ordering, collections::VecDeque, iter::Peekable, str::CharIndices};

use super::field::{self, FieldRef, Fields, Piece, Row, Value};

const MAX_TOKENS: usize = 256;
pub(super) const MAX_DEPTH: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Compare {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl Compare {
    pub fn test(self, order: Ordering) -> bool {
        match self {
            Self::Eq => order.is_eq(),
            Self::Ne => order.is_ne(),
            Self::Lt => order.is_lt(),
            Self::Le => order.is_le(),
            Self::Gt => order.is_gt(),
            Self::Ge => order.is_ge(),
        }
    }
}

/// A field value in a search. Equality is a case-insensitive whole-value
/// wildcard match, or numeric when both sides are numbers.
pub(super) struct Literal {
    pieces: Vec<Piece>,
    number: Option<f64>,
    text: String,
}

impl Literal {
    fn new(raw: &str) -> Self {
        let pieces = field::wildcard(raw);
        let text = field::literal_text(&pieces);
        let number = if pieces.contains(&Piece::Star) {
            None
        } else {
            field::parse_number(&text)
        };
        Self {
            pieces,
            number,
            text,
        }
    }

    fn equals(&self, value: &Value<'_>) -> bool {
        if let (Some(a), Some(b)) = (value.number(), self.number) {
            return a == b;
        }
        value
            .as_text()
            .is_some_and(|t| field::glob_full(&self.pieces, &t, true))
    }

    fn order(&self, value: &Value<'_>) -> Option<Ordering> {
        if let (Some(a), Some(b)) = (value.number(), self.number) {
            return a.partial_cmp(&b);
        }
        Some(value.as_text()?.as_ref().cmp(self.text.as_str()))
    }
}

pub(super) enum Expr {
    Term {
        pieces: Vec<Piece>,
        insensitive: bool,
        major_only: bool,
    },
    Compare {
        field: FieldRef,
        op: Compare,
        value: Literal,
    },
    In {
        field: FieldRef,
        values: Vec<Literal>,
    },
    Time {
        earliest: bool,
        at: f64,
    },
    Not(Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
}

impl Expr {
    /// Two-valued, as in Splunk: a missing field fails `f=v` and `f!=v`, so
    /// `NOT f=v` keeps events without the field.
    pub fn matches(&self, row: &Row<'_, '_>) -> bool {
        match self {
            Self::Term {
                pieces,
                insensitive,
                major_only,
            } => field::term_in(pieces, &row.record.text, *insensitive, *major_only),
            Self::Compare { field, op, value } => {
                let v = field.get(row);
                if v.is_null() {
                    return false;
                }
                match op {
                    Compare::Eq => value.equals(&v),
                    Compare::Ne => !value.equals(&v),
                    _ => value.order(&v).is_some_and(|o| op.test(o)),
                }
            }
            Self::In { field, values } => {
                let v = field.get(row);
                !v.is_null() && values.iter().any(|l| l.equals(&v))
            }
            Self::Time { earliest, at } => match FieldRef::Time.get(row).number() {
                Some(t) if *earliest => t >= *at,
                Some(t) => t < *at,
                None => false,
            },
            Self::Not(e) => !e.matches(row),
            Self::And(a, b) => a.matches(row) && b.matches(row),
            Self::Or(a, b) => a.matches(row) || b.matches(row),
        }
    }

    pub fn uses_occurrence(&self) -> bool {
        match self {
            Self::Term { .. } => false,
            Self::Compare { field, .. } | Self::In { field, .. } => field.occurrence(),
            Self::Time { .. } => true,
            Self::Not(e) => e.uses_occurrence(),
            Self::And(a, b) | Self::Or(a, b) => a.uses_occurrence() || b.uses_occurrence(),
        }
    }
}

pub(super) fn parse(source: &str, fields: &Fields) -> Result<Expr, String> {
    let mut parser = Parser {
        tokens: lex(source)?,
        fields,
    };
    let expr = parser.and(0)?;
    if !parser.tokens.is_empty() {
        return Err("Unexpected closing parenthesis in search.".into());
    }
    check_time(&expr, true)?;
    Ok(expr)
}

fn check_time(expr: &Expr, top: bool) -> Result<(), String> {
    match expr {
        Expr::Time { .. } if !top => {
            Err("earliest and latest bound the whole search. Use them outside OR and NOT.".into())
        }
        Expr::And(a, b) => check_time(a, top).and(check_time(b, top)),
        Expr::Or(a, b) => check_time(a, false).and(check_time(b, false)),
        Expr::Not(e) => check_time(e, false),
        _ => Ok(()),
    }
}

#[derive(Debug)]
enum Token {
    Word(String),
    Quoted(String),
    /// `TERM(...)` (false) or `CASE(...)` (true).
    Directive(bool, String),
    Compare(Compare),
    Open,
    Close,
    Comma,
    And,
    Or,
    Not,
    In,
}

fn stop(c: char) -> bool {
    c.is_whitespace() || matches!(c, '(' | ')' | '"' | '=' | '!' | '<' | '>' | ',' | '|')
}

fn quoted(chars: &mut Peekable<CharIndices<'_>>, start: usize) -> Result<String, String> {
    let mut word = String::new();
    loop {
        let Some((_, c)) = chars.next() else {
            return Err(format!("Unclosed quote at byte {}.", start + 1));
        };
        match c {
            '"' => return Ok(word),
            '\\' if chars.next_if(|(_, n)| *n == '"').is_some() => word.push('"'),
            // Keep an escaped backslash escaped for wildcard translation.
            '\\' if chars.next_if(|(_, n)| *n == '\\').is_some() => word.push_str(r"\\"),
            c => word.push(c),
        }
    }
}

fn directive(chars: &mut Peekable<CharIndices<'_>>, name: &str) -> Result<String, String> {
    while chars.next_if(|(_, c)| c.is_whitespace()).is_some() {}
    let inner = if let Some((start, _)) = chars.next_if(|(_, c)| *c == '"') {
        let text = quoted(chars, start)?;
        while chars.next_if(|(_, c)| c.is_whitespace()).is_some() {}
        if chars.next_if(|(_, c)| *c == ')').is_none() {
            return Err(format!("{name}() takes one term."));
        }
        text
    } else {
        let mut text = String::new();
        loop {
            match chars.next() {
                Some((_, ')')) => break,
                Some((_, c)) => text.push(c),
                None => return Err(format!("{name}( is missing its closing parenthesis.")),
            }
        }
        text.trim().to_owned()
    };
    if inner.is_empty() {
        return Err(format!("{name}() needs a term."));
    }
    Ok(inner)
}

fn lex(source: &str) -> Result<VecDeque<Token>, String> {
    let mut tokens = VecDeque::new();
    let mut chars = source.char_indices().peekable();
    while let Some((start, ch)) = chars.next() {
        if ch.is_whitespace() {
            continue;
        }
        if tokens.len() == MAX_TOKENS {
            return Err(format!("A search stage is limited to {MAX_TOKENS} tokens."));
        }
        let token = match ch {
            '(' => Token::Open,
            ')' => Token::Close,
            ',' => Token::Comma,
            '"' => Token::Quoted(quoted(&mut chars, start)?),
            '=' => {
                if chars.next_if(|(_, n)| *n == '=').is_some() {
                    return Err("search compares fields with =. == belongs to where.".into());
                }
                Token::Compare(Compare::Eq)
            }
            '!' => {
                if chars.next_if(|(_, n)| *n == '=').is_none() {
                    return Err("Use NOT to negate an expression or != to compare a field.".into());
                }
                Token::Compare(Compare::Ne)
            }
            '<' | '>' => {
                let equal = chars.next_if(|(_, n)| *n == '=').is_some();
                Token::Compare(match (ch, equal) {
                    ('<', true) => Compare::Le,
                    ('<', false) => Compare::Lt,
                    (_, true) => Compare::Ge,
                    (_, false) => Compare::Gt,
                })
            }
            '|' => return Err("Unexpected | inside a search stage.".into()),
            '[' => return Err("Subsearches ([ ... ]) are not supported.".into()),
            '`' => return Err("Search macros are not supported.".into()),
            _ => {
                let mut word = String::new();
                let mut c = ch;
                loop {
                    if c == '\\' {
                        match chars.peek().map(|&(_, n)| n) {
                            Some(n @ ('*' | '\\')) => {
                                word.push('\\');
                                word.push(n);
                                chars.next();
                            }
                            Some(n) if stop(n) => {
                                word.push(n);
                                chars.next();
                            }
                            _ => word.push('\\'),
                        }
                    } else {
                        word.push(c);
                    }
                    match chars.next_if(|&(_, n)| !stop(n)) {
                        Some((_, n)) => c = n,
                        None => break,
                    }
                }
                match word.as_str() {
                    "AND" => Token::And,
                    "OR" => Token::Or,
                    "NOT" => Token::Not,
                    "IN" => Token::In,
                    "XOR" => {
                        return Err(
                            "XOR is not a search operator. Use where, or quote it to search the text."
                                .into(),
                        );
                    }
                    "TERM" | "CASE" if chars.next_if(|(_, n)| *n == '(').is_some() => {
                        Token::Directive(word == "CASE", directive(&mut chars, &word)?)
                    }
                    _ => Token::Word(word),
                }
            }
        };
        tokens.push_back(token);
    }
    Ok(tokens)
}

struct Parser<'f> {
    tokens: VecDeque<Token>,
    fields: &'f Fields,
}

impl Parser<'_> {
    // Explicit or implicit AND binds loosest.
    fn and(&mut self, depth: usize) -> Result<Expr, String> {
        let mut expr = self.or(depth)?;
        loop {
            match self.tokens.front() {
                None | Some(Token::Close) => return Ok(expr),
                Some(Token::Comma) => {
                    self.tokens.pop_front();
                    continue;
                }
                Some(Token::And) => {
                    self.tokens.pop_front();
                }
                _ => {}
            }
            expr = Expr::And(Box::new(expr), Box::new(self.or(depth)?));
        }
    }

    fn or(&mut self, depth: usize) -> Result<Expr, String> {
        let mut expr = self.unary(depth)?;
        while matches!(self.tokens.front(), Some(Token::Or)) {
            self.tokens.pop_front();
            expr = Expr::Or(Box::new(expr), Box::new(self.unary(depth)?));
        }
        Ok(expr)
    }

    fn unary(&mut self, depth: usize) -> Result<Expr, String> {
        if depth > MAX_DEPTH {
            return Err(format!("Search nesting is limited to {MAX_DEPTH} levels."));
        }
        match self.tokens.pop_front() {
            Some(Token::Not) => Ok(Expr::Not(Box::new(self.unary(depth + 1)?))),
            Some(Token::Open) => {
                let expr = self.and(depth + 1)?;
                if !matches!(self.tokens.pop_front(), Some(Token::Close)) {
                    return Err("Missing closing parenthesis in search.".into());
                }
                Ok(expr)
            }
            Some(Token::Word(key)) if matches!(self.tokens.front(), Some(Token::Compare(_))) => {
                let Some(Token::Compare(op)) = self.tokens.pop_front() else {
                    unreachable!()
                };
                let Some(Token::Word(value) | Token::Quoted(value)) = self.tokens.pop_front()
                else {
                    return Err(format!("Expected a value after the comparison for {key}."));
                };
                self.comparison(&key, op, &value)
            }
            Some(Token::Word(key)) if matches!(self.tokens.front(), Some(Token::In)) => {
                self.tokens.pop_front();
                self.list(&key)
            }
            Some(Token::Word(value) | Token::Quoted(value)) => {
                if value.is_empty() {
                    return Err("A search term cannot be empty.".into());
                }
                Ok(Expr::Term {
                    pieces: field::wildcard(&value),
                    insensitive: true,
                    major_only: false,
                })
            }
            Some(Token::Directive(case, value)) => Ok(Expr::Term {
                pieces: field::wildcard(&value),
                insensitive: !case,
                major_only: !case,
            }),
            _ => Err(
                "Expected a search term, quoted phrase, field comparison or parenthesized expression."
                    .into(),
            ),
        }
    }

    fn comparison(&mut self, key: &str, op: Compare, value: &str) -> Result<Expr, String> {
        if matches!(key, "earliest" | "latest") {
            if op != Compare::Eq {
                return Err(format!("{key} takes a time with =."));
            }
            return Ok(Expr::Time {
                earliest: key == "earliest",
                at: field::parse_time_modifier(value)?,
            });
        }
        Ok(Expr::Compare {
            field: self.fields.resolve(key),
            op,
            value: Literal::new(value),
        })
    }

    fn list(&mut self, key: &str) -> Result<Expr, String> {
        if !matches!(self.tokens.pop_front(), Some(Token::Open)) {
            return Err(format!("{key} IN needs a parenthesized list of values."));
        }
        let mut values = Vec::new();
        loop {
            match self.tokens.pop_front() {
                Some(Token::Word(v) | Token::Quoted(v)) => values.push(Literal::new(&v)),
                Some(Token::Comma) => {}
                Some(Token::Close) if !values.is_empty() => break,
                _ => return Err(format!("{key} IN takes values such as (a, \"b c\", d*).")),
            }
        }
        Ok(Expr::In {
            field: self.fields.resolve(key),
            values,
        })
    }
}
