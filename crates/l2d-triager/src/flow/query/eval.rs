//! The eval expression language used by `where` and `head (...)`. Bare words
//! are field names, double quotes delimit strings and single quotes delimit
//! field names. String comparisons are case-sensitive. Operators bind as in
//! SPL: `* / %`, `+ - .`, `< <= > >=`, `= == != LIKE`, NOT, AND, then OR/XOR.
use std::{borrow::Cow, collections::VecDeque, net::IpAddr};

use regex::Regex;

use super::{
    Compiler,
    field::{self, FieldRef, Piece, Row, Value, compare_values},
    search::{self, MAX_DEPTH},
};

const MAX_TOKENS: usize = 256;
/// Rust prints an f64 without an exponent, so 1e300 takes about 310 bytes.
pub(super) const NUMBER_TEXT: u64 = 400;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Concat,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    Xor,
    Not,
    Like,
}

#[derive(Debug, PartialEq)]
enum Tok {
    Num(f64),
    Str(String),
    Ident(String),
    Field(String),
    Op(Op),
    Open,
    Close,
    Comma,
}

pub(super) enum Func {
    If,
    Case,
    Coalesce,
    IsNull,
    IsNotNull,
    IsNum,
    IsStr,
    IsInt,
    Null,
    True,
    False,
    Len,
    Lower,
    Upper,
    Substr,
    Trim,
    Ltrim,
    Rtrim,
    Replace(Box<Regex>, String),
    Match(Box<Regex>),
    ToNumber,
    ToString,
    Abs,
    Round,
    Floor,
    Ceiling,
    Sqrt,
    Exp,
    Ln,
    Log,
    Pow,
    Min,
    Max,
    In,
    CidrMatch,
    SearchMatch(Box<search::Expr>),
    TypeOf,
}

pub(super) enum Node {
    Lit(Value<'static>),
    Field(FieldRef),
    Neg(Box<Node>),
    Not(Box<Node>),
    Bin(Op, Box<Node>, Box<Node>),
    Like(Box<Node>, Vec<Piece>),
    Call(Func, Vec<Node>),
}

pub(super) fn parse(source: &str, compiler: &mut Compiler<'_>) -> Result<Node, String> {
    let mut parser = Parser {
        tokens: lex(source)?,
        compiler,
    };
    if parser.tokens.is_empty() {
        return Err("Expected an eval expression.".into());
    }
    let node = parser.or(0)?;
    if let Some(token) = parser.tokens.front() {
        return Err(format!("Unexpected {token:?} in eval expression."));
    }
    Ok(node)
}

fn lex(source: &str) -> Result<VecDeque<Tok>, String> {
    let mut tokens = VecDeque::new();
    let mut chars = source.char_indices().peekable();
    while let Some((start, c)) = chars.next() {
        if c.is_whitespace() {
            continue;
        }
        if tokens.len() == MAX_TOKENS {
            return Err(format!(
                "An eval expression is limited to {MAX_TOKENS} tokens."
            ));
        }
        let token = match c {
            '(' => Tok::Open,
            ')' => Tok::Close,
            ',' => Tok::Comma,
            '+' => Tok::Op(Op::Add),
            '-' => Tok::Op(Op::Sub),
            '*' => Tok::Op(Op::Mul),
            '/' => Tok::Op(Op::Div),
            '%' => Tok::Op(Op::Mod),
            '.' if !chars.peek().is_some_and(|(_, n)| n.is_ascii_digit()) => Tok::Op(Op::Concat),
            '=' => {
                chars.next_if(|(_, n)| *n == '=');
                Tok::Op(Op::Eq)
            }
            '!' => {
                if chars.next_if(|(_, n)| *n == '=').is_none() {
                    return Err("Use NOT to negate in eval; != compares.".into());
                }
                Tok::Op(Op::Ne)
            }
            '<' | '>' => {
                let equal = chars.next_if(|(_, n)| *n == '=').is_some();
                Tok::Op(match (c, equal) {
                    ('<', true) => Op::Le,
                    ('<', false) => Op::Lt,
                    (_, true) => Op::Ge,
                    (_, false) => Op::Gt,
                })
            }
            '"' => {
                let mut text = String::new();
                loop {
                    match chars.next() {
                        None => return Err(format!("Unclosed quote at byte {}.", start + 1)),
                        Some((_, '"')) => break,
                        Some((_, '\\')) => match chars.next_if(|(_, n)| matches!(n, '"' | '\\')) {
                            Some((_, n)) => text.push(n),
                            None => text.push('\\'),
                        },
                        Some((_, c)) => text.push(c),
                    }
                }
                Tok::Str(text)
            }
            '\'' => {
                let mut name = String::new();
                loop {
                    match chars.next() {
                        None => return Err(format!("Unclosed field quote at byte {}.", start + 1)),
                        Some((_, '\'')) => break,
                        Some((_, c)) => name.push(c),
                    }
                }
                Tok::Field(name)
            }
            c if c.is_ascii_digit() || c == '.' => {
                let mut end = start + 1;
                while let Some((i, n)) = chars.next_if(|(_, n)| n.is_ascii_digit() || *n == '.') {
                    end = i + n.len_utf8();
                }
                if let Some((i, e)) = chars.next_if(|(_, n)| matches!(n, 'e' | 'E')) {
                    end = i + e.len_utf8();
                    if let Some((i, s)) = chars.next_if(|(_, n)| matches!(n, '+' | '-')) {
                        end = i + s.len_utf8();
                    }
                    while let Some((i, n)) = chars.next_if(|(_, n)| n.is_ascii_digit()) {
                        end = i + n.len_utf8();
                    }
                }
                let text = &source[start..end];
                Tok::Num(
                    text.parse::<f64>()
                        .ok()
                        .filter(|n| n.is_finite())
                        .ok_or_else(|| format!("Cannot read number {text:?}."))?,
                )
            }
            c if c.is_alphabetic() || c == '_' => {
                let mut end = start + c.len_utf8();
                while let Some((i, n)) = chars.next_if(|(_, n)| n.is_alphanumeric() || *n == '_') {
                    end = i + n.len_utf8();
                }
                match &source[start..end] {
                    "AND" => Tok::Op(Op::And),
                    "OR" => Tok::Op(Op::Or),
                    "XOR" => Tok::Op(Op::Xor),
                    "NOT" => Tok::Op(Op::Not),
                    "LIKE" => Tok::Op(Op::Like),
                    word => Tok::Ident(word.to_owned()),
                }
            }
            c => return Err(format!("Unexpected {c:?} in eval expression.")),
        };
        tokens.push_back(token);
    }
    Ok(tokens)
}

struct Parser<'c, 'b> {
    tokens: VecDeque<Tok>,
    compiler: &'c mut Compiler<'b>,
}

impl Parser<'_, '_> {
    fn take(&mut self, ops: &[Op]) -> Option<Op> {
        match self.tokens.front() {
            Some(Tok::Op(op)) if ops.contains(op) => {
                let op = *op;
                self.tokens.pop_front();
                Some(op)
            }
            _ => None,
        }
    }

    fn binary(
        &mut self,
        depth: usize,
        ops: &[Op],
        next: fn(&mut Self, usize) -> Result<Node, String>,
    ) -> Result<Node, String> {
        let mut node = next(self, depth)?;
        while let Some(op) = self.take(ops) {
            node = Node::Bin(op, Box::new(node), Box::new(next(self, depth)?));
        }
        Ok(node)
    }

    fn or(&mut self, depth: usize) -> Result<Node, String> {
        self.binary(depth, &[Op::Or, Op::Xor], Self::and)
    }

    fn and(&mut self, depth: usize) -> Result<Node, String> {
        self.binary(depth, &[Op::And], Self::not)
    }

    fn not(&mut self, depth: usize) -> Result<Node, String> {
        if depth > MAX_DEPTH {
            return Err(format!("Eval nesting is limited to {MAX_DEPTH} levels."));
        }
        if self.take(&[Op::Not]).is_some() {
            return Ok(Node::Not(Box::new(self.not(depth + 1)?)));
        }
        self.equality(depth)
    }

    fn equality(&mut self, depth: usize) -> Result<Node, String> {
        let mut node = self.relational(depth)?;
        while let Some(op) = self.take(&[Op::Eq, Op::Ne, Op::Like]) {
            if op == Op::Like {
                let Some(Tok::Str(pattern)) = self.tokens.pop_front() else {
                    return Err("LIKE needs a quoted pattern such as \"%error%\".".into());
                };
                node = Node::Like(Box::new(node), field::like(&pattern));
            } else {
                node = Node::Bin(op, Box::new(node), Box::new(self.relational(depth)?));
            }
        }
        Ok(node)
    }

    fn relational(&mut self, depth: usize) -> Result<Node, String> {
        self.binary(depth, &[Op::Lt, Op::Le, Op::Gt, Op::Ge], Self::additive)
    }

    fn additive(&mut self, depth: usize) -> Result<Node, String> {
        self.binary(depth, &[Op::Add, Op::Sub, Op::Concat], Self::multiplicative)
    }

    fn multiplicative(&mut self, depth: usize) -> Result<Node, String> {
        self.binary(depth, &[Op::Mul, Op::Div, Op::Mod], Self::unary)
    }

    fn unary(&mut self, depth: usize) -> Result<Node, String> {
        if depth > MAX_DEPTH {
            return Err(format!("Eval nesting is limited to {MAX_DEPTH} levels."));
        }
        if self.take(&[Op::Sub]).is_some() {
            return Ok(Node::Neg(Box::new(self.unary(depth + 1)?)));
        }
        if self.take(&[Op::Add]).is_some() {
            return self.unary(depth + 1);
        }
        match self.tokens.pop_front() {
            Some(Tok::Num(n)) => Ok(Node::Lit(Value::Num(n))),
            Some(Tok::Str(s)) => Ok(Node::Lit(Value::Str(Cow::Owned(s)))),
            Some(Tok::Field(name)) => Ok(Node::Field(self.compiler.fields.resolve(&name))),
            Some(Tok::Ident(name)) if self.tokens.front() == Some(&Tok::Open) => {
                self.tokens.pop_front();
                self.call(&name, depth + 1)
            }
            Some(Tok::Ident(name)) => Ok(Node::Field(self.compiler.fields.resolve(&name))),
            Some(Tok::Open) => {
                let node = self.or(depth + 1)?;
                if self.tokens.pop_front() != Some(Tok::Close) {
                    return Err("Missing closing parenthesis in eval expression.".into());
                }
                Ok(node)
            }
            other => Err(format!(
                "Expected a value, field or function in eval expression, found {other:?}."
            )),
        }
    }

    fn call(&mut self, name: &str, depth: usize) -> Result<Node, String> {
        let mut args = Vec::new();
        if self.tokens.front() == Some(&Tok::Close) {
            self.tokens.pop_front();
        } else {
            loop {
                args.push(self.or(depth)?);
                match self.tokens.pop_front() {
                    Some(Tok::Comma) => {}
                    Some(Tok::Close) => break,
                    _ => {
                        return Err(format!(
                            "{name}() is missing a comma or closing parenthesis."
                        ));
                    }
                }
            }
        }
        let lower = name.to_ascii_lowercase();
        let arity = |min: usize, max: usize| {
            if (min..=max).contains(&args.len()) {
                Ok(())
            } else if min == max {
                Err(format!("{lower}() takes {min} argument(s)."))
            } else {
                Err(format!("{lower}() takes {min} to {max} arguments."))
            }
        };
        let func = match lower.as_str() {
            "if" => arity(3, 3).map(|()| Func::If)?,
            "case" => {
                if args.is_empty() || args.len() % 2 != 0 {
                    return Err("case() takes condition, value pairs.".into());
                }
                Func::Case
            }
            "coalesce" => arity(1, 64).map(|()| Func::Coalesce)?,
            "isnull" => arity(1, 1).map(|()| Func::IsNull)?,
            "isnotnull" => arity(1, 1).map(|()| Func::IsNotNull)?,
            "isnum" => arity(1, 1).map(|()| Func::IsNum)?,
            "isstr" => arity(1, 1).map(|()| Func::IsStr)?,
            "isint" => arity(1, 1).map(|()| Func::IsInt)?,
            "null" => arity(0, 0).map(|()| Func::Null)?,
            "true" => arity(0, 0).map(|()| Func::True)?,
            "false" => arity(0, 0).map(|()| Func::False)?,
            "len" => arity(1, 1).map(|()| Func::Len)?,
            "lower" => arity(1, 1).map(|()| Func::Lower)?,
            "upper" => arity(1, 1).map(|()| Func::Upper)?,
            "substr" => arity(2, 3).map(|()| Func::Substr)?,
            "trim" => arity(1, 2).map(|()| Func::Trim)?,
            "ltrim" => arity(1, 2).map(|()| Func::Ltrim)?,
            "rtrim" => arity(1, 2).map(|()| Func::Rtrim)?,
            "replace" => {
                arity(3, 3)?;
                let replacement = literal(&args[2], &lower)?;
                let regex = self.compiler.regex(&literal(&args[1], &lower)?)?;
                args.truncate(1);
                Func::Replace(Box::new(regex), backreferences(&replacement))
            }
            "match" => {
                arity(2, 2)?;
                let regex = self.compiler.regex(&literal(&args[1], &lower)?)?;
                args.truncate(1);
                Func::Match(Box::new(regex))
            }
            "like" => {
                arity(2, 2)?;
                let pattern = field::like(&literal(&args[1], &lower)?);
                return Ok(Node::Like(Box::new(args.swap_remove(0)), pattern));
            }
            "tonumber" => arity(1, 2).map(|()| Func::ToNumber)?,
            "tostring" => arity(1, 1).map(|()| Func::ToString)?,
            "abs" => arity(1, 1).map(|()| Func::Abs)?,
            "round" => arity(1, 2).map(|()| Func::Round)?,
            "floor" => arity(1, 1).map(|()| Func::Floor)?,
            "ceil" | "ceiling" => arity(1, 1).map(|()| Func::Ceiling)?,
            "sqrt" => arity(1, 1).map(|()| Func::Sqrt)?,
            "exp" => arity(1, 1).map(|()| Func::Exp)?,
            "ln" => arity(1, 1).map(|()| Func::Ln)?,
            "log" => arity(1, 2).map(|()| Func::Log)?,
            "pow" => arity(2, 2).map(|()| Func::Pow)?,
            "min" => arity(1, 64).map(|()| Func::Min)?,
            "max" => arity(1, 64).map(|()| Func::Max)?,
            "in" => arity(2, 64).map(|()| Func::In)?,
            "cidrmatch" => arity(2, 2).map(|()| Func::CidrMatch)?,
            "searchmatch" => {
                arity(1, 1)?;
                let expr = search::parse(&literal(&args[0], &lower)?, &self.compiler.fields)?;
                args.clear();
                Func::SearchMatch(Box::new(expr))
            }
            "typeof" => arity(1, 1).map(|()| Func::TypeOf)?,
            "now" | "time" | "random" | "relative_time" => {
                return Err(format!(
                    "{lower}() is not supported: a saved view must replay identically."
                ));
            }
            _ => {
                return Err(format!(
                    "Unknown eval function {lower}(). Supported: if, case, coalesce, isnull, \
                     isnotnull, isnum, isstr, isint, null, true, false, len, lower, upper, substr, \
                     trim, ltrim, rtrim, replace, match, like, tonumber, tostring, abs, round, \
                     floor, ceiling, sqrt, exp, ln, log, pow, min, max, in, cidrmatch, \
                     searchmatch, typeof."
                ));
            }
        };
        Ok(Node::Call(func, args))
    }
}

fn literal(node: &Node, name: &str) -> Result<String, String> {
    match node {
        Node::Lit(Value::Str(s)) => Ok(s.to_string()),
        _ => Err(format!("{name}() needs a quoted string literal here.")),
    }
}

/// Splunk writes back-references as `\1`; the regex crate as `${1}`.
fn backreferences(replacement: &str) -> String {
    let mut out = String::with_capacity(replacement.len() + 8);
    let mut chars = replacement.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '$' => out.push_str("$$"),
            '\\' if chars.peek().is_some_and(char::is_ascii_digit) => {
                out.push_str("${");
                while let Some(d) = chars.next_if(char::is_ascii_digit) {
                    out.push(d);
                }
                out.push('}');
            }
            '\\' if chars.next_if_eq(&'\\').is_some() => out.push('\\'),
            c => out.push(c),
        }
    }
    out
}

fn truth(value: &Value<'_>) -> Option<bool> {
    match value {
        Value::Bool(b) => Some(*b),
        _ => None,
    }
}

fn text_value<'a>(text: String) -> Value<'a> {
    Value::Str(Cow::Owned(text))
}

fn number(value: Option<f64>) -> Value<'static> {
    value
        .filter(|n| n.is_finite())
        .map_or(Value::Null, Value::Num)
}

fn cidr_match(cidr: &str, ip: &str) -> bool {
    let Some((network, bits)) = cidr.split_once('/') else {
        return false;
    };
    let (Ok(network), Ok(bits), Ok(ip)) = (
        network.trim().parse::<IpAddr>(),
        bits.trim().parse::<u32>(),
        ip.trim().parse::<IpAddr>(),
    ) else {
        return false;
    };
    match (network, ip) {
        (IpAddr::V4(n), IpAddr::V4(i)) if bits <= 32 => {
            let mask = u32::MAX.checked_shl(32 - bits).unwrap_or(0);
            u32::from(n) & mask == u32::from(i) & mask
        }
        (IpAddr::V6(n), IpAddr::V6(i)) if bits <= 128 => {
            let mask = u128::MAX.checked_shl(128 - bits).unwrap_or(0);
            u128::from(n) & mask == u128::from(i) & mask
        }
        _ => false,
    }
}

impl Node {
    pub fn uses_occurrence(&self) -> bool {
        match self {
            Self::Lit(_) => false,
            Self::Field(f) => f.occurrence(),
            Self::Neg(n) | Self::Not(n) | Self::Like(n, _) => n.uses_occurrence(),
            Self::Bin(_, a, b) => a.uses_occurrence() || b.uses_occurrence(),
            Self::Call(func, args) => {
                matches!(func, Func::SearchMatch(e) if e.uses_occurrence())
                    || args.iter().any(Self::uses_occurrence)
            }
        }
    }

    /// The most bytes of heap text this node's value can own, for an original
    /// whose text fields are at most `t` bytes. Saturates instead of overflowing.
    fn text_bound(&self, t: u64) -> u64 {
        let max = |args: &[Node]| args.iter().map(|n| n.text_bound(t)).max().unwrap_or(0);
        match self {
            Self::Lit(Value::Str(s)) => s.len() as u64,
            Self::Lit(_) | Self::Neg(_) | Self::Not(_) | Self::Like(..) => NUMBER_TEXT,
            Self::Field(FieldRef::Punct) => field::PUNCT_BYTES,
            Self::Field(_) => t.max(NUMBER_TEXT),
            Self::Bin(Op::Add | Op::Concat, a, b) => {
                a.text_bound(t).saturating_add(b.text_bound(t))
            }
            // A comparison may format both operands.
            Self::Bin(..) => 2 * NUMBER_TEXT,
            Self::Call(func, args) => match func {
                Func::If | Func::Case | Func::Coalesce | Func::Min | Func::Max => max(args),
                // Unicode case mapping grows text at most threefold.
                Func::Lower | Func::Upper => args[0].text_bound(t).saturating_mul(3),
                Func::Substr | Func::Trim | Func::Ltrim | Func::Rtrim => args[0].text_bound(t),
                Func::ToString => args[0].text_bound(t).max(NUMBER_TEXT),
                // Every position, including the end, may match and insert the replacement.
                Func::Replace(_, replacement) => {
                    let input = args[0].text_bound(t);
                    input
                        .saturating_add(1)
                        .saturating_mul(replacement.len() as u64)
                        .saturating_add(input)
                }
                _ => NUMBER_TEXT,
            },
        }
    }

    /// Transient bytes while one row is evaluated: every intermediate value
    /// may be alive at once, plus per-function working space.
    pub fn scratch(&self, t: u64) -> u64 {
        let children: u64 = match self {
            Self::Lit(_) | Self::Field(_) => 0,
            Self::Neg(n) | Self::Not(n) | Self::Like(n, _) => n.scratch(t),
            Self::Bin(_, a, b) => a.scratch(t).saturating_add(b.scratch(t)),
            Self::Call(_, args) => args
                .iter()
                .fold(0u64, |sum, n| sum.saturating_add(n.scratch(t))),
        };
        let working = match self {
            // min() and max() keep every argument's value.
            Self::Call(Func::Min | Func::Max, args) => {
                (args.len() * std::mem::size_of::<Value<'_>>()) as u64
            }
            // trim() keeps its character set.
            Self::Call(Func::Trim | Func::Ltrim | Func::Rtrim, args) if args.len() == 2 => {
                args[1].text_bound(t)
            }
            _ => 0,
        };
        // Growing strings may double their capacity before they settle.
        self.text_bound(t)
            .saturating_mul(2)
            .saturating_add(children)
            .saturating_add(working)
            .saturating_add(64)
    }

    pub fn eval<'a>(&'a self, row: &Row<'_, 'a>) -> Value<'a> {
        match self {
            Self::Lit(Value::Str(s)) => Value::Str(Cow::Borrowed(s.as_ref())),
            Self::Lit(v) => v.clone(),
            Self::Field(f) => f.get(row),
            Self::Neg(n) => number(n.eval(row).number().map(|x| -x)),
            Self::Not(n) => truth(&n.eval(row)).map_or(Value::Null, |b| Value::Bool(!b)),
            Self::Bin(op, a, b) => binary(*op, a.eval(row), b.eval(row)),
            Self::Like(n, pattern) => n.eval(row).as_text().map_or(Value::Null, |t| {
                Value::Bool(field::glob_full(pattern, &t, false))
            }),
            Self::Call(func, args) => call(func, args, row),
        }
    }
}

fn binary<'a>(op: Op, a: Value<'a>, b: Value<'a>) -> Value<'a> {
    let numbers = (a.number(), b.number());
    match op {
        Op::Add => match numbers {
            (Some(x), Some(y)) => number(Some(x + y)),
            _ => match (&a, &b) {
                (Value::Str(x) | Value::Field(x), Value::Str(y) | Value::Field(y)) => {
                    text_value(format!("{x}{y}"))
                }
                _ => Value::Null,
            },
        },
        Op::Sub | Op::Mul | Op::Div | Op::Mod => {
            let (Some(x), Some(y)) = numbers else {
                return Value::Null;
            };
            number(match op {
                Op::Sub => Some(x - y),
                Op::Mul => Some(x * y),
                _ if y == 0.0 => None,
                Op::Div => Some(x / y),
                _ => Some(x % y),
            })
        }
        Op::Concat => match (a.as_text(), b.as_text()) {
            (Some(x), Some(y)) => text_value(format!("{x}{y}")),
            _ => Value::Null,
        },
        Op::Eq | Op::Ne | Op::Lt | Op::Le | Op::Gt | Op::Ge => {
            compare_values(&a, &b).map_or(Value::Null, |o| {
                Value::Bool(match op {
                    Op::Eq => o.is_eq(),
                    Op::Ne => o.is_ne(),
                    Op::Lt => o.is_lt(),
                    Op::Le => o.is_le(),
                    Op::Gt => o.is_gt(),
                    _ => o.is_ge(),
                })
            })
        }
        Op::And | Op::Or | Op::Xor => {
            let (x, y) = (truth(&a), truth(&b));
            let result = match op {
                Op::And => match (x, y) {
                    (Some(false), _) | (_, Some(false)) => Some(false),
                    (Some(true), Some(true)) => Some(true),
                    _ => None,
                },
                Op::Or => match (x, y) {
                    (Some(true), _) | (_, Some(true)) => Some(true),
                    (Some(false), Some(false)) => Some(false),
                    _ => None,
                },
                _ => x.zip(y).map(|(x, y)| x != y),
            };
            result.map_or(Value::Null, Value::Bool)
        }
        Op::Not | Op::Like => Value::Null,
    }
}

fn call<'a>(func: &'a Func, args: &'a [Node], row: &Row<'_, 'a>) -> Value<'a> {
    let arg = |i: usize| args.get(i).map_or(Value::Null, |n| n.eval(row));
    let num = |i: usize| arg(i).number();
    let unary = |f: fn(f64) -> f64| number(num(0).map(f));
    match func {
        Func::If => {
            if truth(&arg(0)) == Some(true) {
                arg(1)
            } else {
                arg(2)
            }
        }
        Func::Case => args
            .chunks(2)
            .find(|pair| truth(&pair[0].eval(row)) == Some(true))
            .map_or(Value::Null, |pair| pair[1].eval(row)),
        Func::Coalesce => args
            .iter()
            .map(|n| n.eval(row))
            .find(|v| !v.is_null())
            .unwrap_or(Value::Null),
        Func::IsNull => Value::Bool(arg(0).is_null()),
        Func::IsNotNull => Value::Bool(!arg(0).is_null()),
        Func::IsNum => Value::Bool(match arg(0) {
            Value::Num(_) => true,
            Value::Field(s) => field::parse_number(&s).is_some(),
            _ => false,
        }),
        Func::IsStr => Value::Bool(match arg(0) {
            Value::Str(_) => true,
            Value::Field(s) => field::parse_number(&s).is_none(),
            _ => false,
        }),
        Func::IsInt => {
            let v = arg(0);
            Value::Bool(!matches!(v, Value::Str(_)) && v.number().is_some_and(|n| n.fract() == 0.0))
        }
        Func::Null => Value::Null,
        Func::True => Value::Bool(true),
        Func::False => Value::Bool(false),
        Func::Len => number(arg(0).as_text().map(|t| t.chars().count() as f64)),
        Func::Lower => arg(0)
            .as_text()
            .map_or(Value::Null, |t| text_value(t.to_lowercase())),
        Func::Upper => arg(0)
            .as_text()
            .map_or(Value::Null, |t| text_value(t.to_uppercase())),
        Func::Substr => {
            let value = arg(0);
            let (Some(text), Some(start)) = (value.as_text(), num(1)) else {
                return Value::Null;
            };
            // Character positions; `as` saturates huge or infinite numbers.
            let len = text.chars().count() as i64;
            let start = start as i64;
            let from = if start < 0 {
                len.saturating_add(start).max(0)
            } else {
                start.saturating_sub(1).max(0)
            }
            .min(len);
            let count = match args.len() {
                3 => match num(2) {
                    Some(n) if n >= 0.0 => n as i64,
                    _ => return Value::Null,
                },
                _ => len,
            };
            let to = from.saturating_add(count).min(len);
            let byte = |i: i64| {
                text.char_indices()
                    .nth(i as usize)
                    .map_or(text.len(), |(b, _)| b)
            };
            text_value(text[byte(from)..byte(to)].to_owned())
        }
        Func::Trim | Func::Ltrim | Func::Rtrim => {
            let value = arg(0);
            let Some(text) = value.as_text() else {
                return Value::Null;
            };
            let set = if args.len() == 2 {
                match arg(1).into_text() {
                    Some(s) => s.into_owned(),
                    None => return Value::Null,
                }
            } else {
                " \t".to_owned()
            };
            let strip = |c: char| set.contains(c);
            text_value(
                match func {
                    Func::Trim => text.trim_matches(strip),
                    Func::Ltrim => text.trim_start_matches(strip),
                    _ => text.trim_end_matches(strip),
                }
                .to_owned(),
            )
        }
        Func::Replace(regex, replacement) => arg(0).as_text().map_or(Value::Null, |t| {
            text_value(regex.replace_all(&t, replacement.as_str()).into_owned())
        }),
        Func::Match(regex) => arg(0)
            .as_text()
            .map_or(Value::Null, |t| Value::Bool(regex.is_match(&t))),
        Func::ToNumber => {
            let value = arg(0);
            if let Value::Num(n) = value {
                return Value::Num(n);
            }
            let Some(text) = value.as_text() else {
                return Value::Null;
            };
            match args.len() {
                2 => match num(1) {
                    Some(base) if (2.0..=36.0).contains(&base) && base.fract() == 0.0 => {
                        let digits = text.trim();
                        let digits = if base == 16.0 {
                            digits
                                .strip_prefix("0x")
                                .or_else(|| digits.strip_prefix("0X"))
                                .unwrap_or(digits)
                        } else {
                            digits
                        };
                        number(
                            i64::from_str_radix(digits, base as u32)
                                .ok()
                                .map(|n| n as f64),
                        )
                    }
                    _ => Value::Null,
                },
                _ => number(field::parse_number(&text)),
            }
        }
        Func::ToString => match arg(0) {
            Value::Bool(b) => Value::Str(Cow::Borrowed(if b { "True" } else { "False" })),
            v => v.into_text().map_or(Value::Null, Value::Str),
        },
        Func::Abs => unary(f64::abs),
        Func::Floor => unary(f64::floor),
        Func::Ceiling => unary(f64::ceil),
        Func::Sqrt => unary(f64::sqrt),
        Func::Exp => unary(f64::exp),
        Func::Ln => unary(f64::ln),
        Func::Round => {
            let places = if args.len() == 2 { num(1) } else { Some(0.0) };
            let (Some(x), Some(places)) = (num(0), places) else {
                return Value::Null;
            };
            let scale = 10f64.powi(places as i32);
            number(Some((x * scale).round() / scale))
        }
        Func::Log => {
            let base = if args.len() == 2 { num(1) } else { Some(10.0) };
            number(num(0).zip(base).map(|(x, b)| x.log(b)))
        }
        Func::Pow => number(num(0).zip(num(1)).map(|(x, y)| x.powf(y))),
        Func::Min | Func::Max => {
            let values: Vec<Value<'a>> = args
                .iter()
                .map(|n| n.eval(row))
                .filter(|v| !v.is_null())
                .collect();
            let numeric = values.iter().all(|v| v.number().is_some());
            let pick = values.into_iter().reduce(|best, v| {
                let order = if numeric {
                    v.number().partial_cmp(&best.number())
                } else {
                    v.as_text().zip(best.as_text()).map(|(a, b)| a.cmp(&b))
                };
                let better = match func {
                    Func::Min => order.is_some_and(|o| o.is_lt()),
                    _ => order.is_some_and(|o| o.is_gt()),
                };
                if better { v } else { best }
            });
            pick.unwrap_or(Value::Null)
        }
        Func::In => {
            let value = arg(0);
            Value::Bool(
                !value.is_null()
                    && args[1..]
                        .iter()
                        .any(|n| compare_values(&value, &n.eval(row)).is_some_and(|o| o.is_eq())),
            )
        }
        Func::CidrMatch => {
            let (cidr, ip) = (arg(0), arg(1));
            match (cidr.as_text(), ip.as_text()) {
                (Some(c), Some(i)) => Value::Bool(cidr_match(&c, &i)),
                _ => Value::Null,
            }
        }
        Func::SearchMatch(expr) => Value::Bool(expr.matches(row)),
        Func::TypeOf => Value::Str(Cow::Borrowed(match arg(0) {
            Value::Num(_) => "Number",
            Value::Field(s) if field::parse_number(&s).is_some() => "Number",
            Value::Str(_) | Value::Field(_) => "String",
            Value::Bool(_) => "Boolean",
            Value::Null => "Invalid",
        })),
    }
}
