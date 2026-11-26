//! Field values for the SPL subset: built-in fields of a retained original and
//! its scoped occurrence, Splunk-style automatic `key=value` extraction from
//! `_raw`, `rex` captures, Splunk's breaker classes and wildcard matching.
use std::{borrow::Cow, cmp::Ordering, collections::HashMap};

use crate::flow::{Occurrence, Record};

/// One original as a pipeline stage sees it. `group` is set when a stage
/// evaluates a single scoped occurrence (fields `host` and `_time`).
pub(super) struct Row<'s, 'a> {
    pub record: &'a Record,
    pub group: Option<&'a Occurrence>,
    pub slots: &'s [Option<&'a str>],
}

/// A value during evaluation. `Field` is text read from an event, which SPL
/// treats as a number wherever it parses as one. `Str` is a quoted literal or
/// a computed string and is never converted implicitly.
#[derive(Clone, Debug)]
pub(super) enum Value<'a> {
    Null,
    Bool(bool),
    Num(f64),
    Str(Cow<'a, str>),
    Field(Cow<'a, str>),
}

impl<'a> Value<'a> {
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    pub fn number(&self) -> Option<f64> {
        match self {
            Self::Num(n) => Some(*n),
            Self::Field(s) => parse_number(s),
            _ => None,
        }
    }

    pub fn as_text(&self) -> Option<Cow<'_, str>> {
        match self {
            Self::Str(s) | Self::Field(s) => Some(Cow::Borrowed(s.as_ref())),
            Self::Num(n) => Some(Cow::Owned(format_number(*n))),
            Self::Bool(b) => Some(Cow::Borrowed(if *b { "true" } else { "false" })),
            Self::Null => None,
        }
    }

    pub fn into_text(self) -> Option<Cow<'a, str>> {
        match self {
            Self::Str(s) | Self::Field(s) => Some(s),
            Self::Num(n) => Some(Cow::Owned(format_number(n))),
            Self::Bool(b) => Some(Cow::Borrowed(if b { "true" } else { "false" })),
            Self::Null => None,
        }
    }
}

pub(crate) fn parse_number(text: &str) -> Option<f64> {
    let text = text.trim();
    let first = text.chars().next()?;
    if !(first.is_ascii_digit() || matches!(first, '-' | '+' | '.')) {
        return None;
    }
    text.parse::<f64>().ok().filter(|n| n.is_finite())
}

pub(super) fn format_number(n: f64) -> String {
    if n == 0.0 { "0".into() } else { n.to_string() }
}

/// A field reference resolved when the query is compiled.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum FieldRef {
    Raw,
    Id,
    Host,
    Time,
    Score,
    C,
    Reason,
    Punct,
    Linecount,
    /// A `rex` capture (newest slot first) and, unless hidden by `fields`,
    /// the automatic `key=value` extraction from `_raw`.
    Extracted {
        name: Box<str>,
        slots: Vec<usize>,
        kv: bool,
    },
    /// Removed by `fields`; always null.
    Removed,
}

pub(super) const BUILT_IN: [&str; 9] = [
    "_raw",
    "id",
    "host",
    "_time",
    "score",
    "c",
    "reason",
    "punct",
    "linecount",
];

impl FieldRef {
    fn built_in(name: &str) -> Option<Self> {
        Some(match name {
            "_raw" => Self::Raw,
            "id" => Self::Id,
            "host" => Self::Host,
            "_time" => Self::Time,
            "score" => Self::Score,
            "c" => Self::C,
            "reason" => Self::Reason,
            "punct" => Self::Punct,
            "linecount" => Self::Linecount,
            _ => return None,
        })
    }

    /// Fields that differ between occurrences of one original.
    pub fn occurrence(&self) -> bool {
        matches!(self, Self::Host | Self::Time)
    }

    /// Fields whose text can be borrowed from the event (sources for `rex`).
    pub fn borrowable(&self) -> bool {
        matches!(
            self,
            Self::Raw | Self::Id | Self::C | Self::Reason | Self::Extracted { .. }
        )
    }

    pub fn get<'a>(&self, row: &Row<'_, 'a>) -> Value<'a> {
        let record = row.record;
        match self {
            Self::Score => Value::Num(record.score),
            Self::Time => row
                .group
                .and_then(|g| g.time.as_deref())
                .and_then(parse_iso)
                .map_or(Value::Null, Value::Num),
            Self::Punct => Value::Field(Cow::Owned(punct(&record.text))),
            Self::Linecount => Value::Num((record.text.matches('\n').count() + 1) as f64),
            Self::Removed => Value::Null,
            _ => self
                .text(row)
                .map_or(Value::Null, |s| Value::Field(Cow::Borrowed(s))),
        }
    }

    pub fn text<'a>(&self, row: &Row<'_, 'a>) -> Option<&'a str> {
        let record = row.record;
        match self {
            Self::Raw => Some(&record.text),
            Self::Id => Some(&record.id),
            Self::Reason => Some(&record.reason),
            Self::C => Some(record.recommendation.label()),
            Self::Host => row.group.and_then(|g| g.host.as_deref()),
            Self::Extracted { name, slots, kv } => slots
                .iter()
                .find_map(|&slot| row.slots[slot])
                .or_else(|| kv.then(|| kv_value(&record.text, name)).flatten()),
            _ => None,
        }
    }
}

/// Which fields later stages see. `fields` and `rex` change it while the
/// pipeline is compiled, so evaluation never consults stage history.
#[derive(Default)]
pub(super) struct Fields {
    slots: usize,
    extracted: HashMap<String, (Vec<usize>, bool)>,
    removed: Vec<Vec<Piece>>,
    keep: Vec<Vec<Vec<Piece>>>,
}

impl Fields {
    pub fn slots(&self) -> usize {
        self.slots
    }

    fn visible(&self, name: &str) -> bool {
        !self.removed.iter().any(|p| glob_full(p, name, false))
            && (matches!(name, "_raw" | "_time")
                || self
                    .keep
                    .iter()
                    .all(|list| list.iter().any(|p| glob_full(p, name, false))))
    }

    pub fn resolve(&self, name: &str) -> FieldRef {
        if let Some((slots, kv)) = self.extracted.get(name) {
            return FieldRef::Extracted {
                name: name.into(),
                // Newest capture wins.
                slots: slots.iter().rev().copied().collect(),
                kv: *kv,
            };
        }
        if !self.visible(name) {
            return FieldRef::Removed;
        }
        FieldRef::built_in(name).unwrap_or_else(|| FieldRef::Extracted {
            name: name.into(),
            slots: Vec::new(),
            kv: true,
        })
    }

    pub fn define(&mut self, name: &str) -> Result<usize, String> {
        if BUILT_IN.contains(&name) {
            return Err(format!("rex cannot replace the built-in field {name}."));
        }
        let slot = self.slots;
        self.slots += 1;
        let visible = self.visible(name);
        self.extracted
            .entry(name.to_owned())
            .or_insert_with(|| (Vec::new(), visible))
            .0
            .push(slot);
        Ok(slot)
    }

    pub fn remove(&mut self, patterns: Vec<Vec<Piece>>) {
        self.extracted
            .retain(|name, _| !patterns.iter().any(|p| glob_full(p, name, false)));
        self.removed.extend(patterns);
    }

    pub fn keep(&mut self, patterns: Vec<Vec<Piece>>) {
        self.extracted
            .retain(|name, _| patterns.iter().any(|p| glob_full(p, name, false)));
        self.keep.push(patterns);
    }
}

/// Splunk's default major breakers (`segmenters.conf`), as single characters.
pub(super) fn is_major(c: char) -> bool {
    c.is_whitespace()
        || matches!(
            c,
            '[' | ']'
                | '<'
                | '>'
                | '('
                | ')'
                | '{'
                | '}'
                | '|'
                | '!'
                | ';'
                | ','
                | '\''
                | '"'
                | '*'
                | '&'
                | '?'
                | '+'
        )
}

/// Splunk's default minor breakers.
pub(super) fn is_minor(c: char) -> bool {
    matches!(
        c,
        '/' | ':' | '=' | '@' | '.' | '-' | '$' | '#' | '%' | '\\' | '_'
    )
}

fn is_breaker(c: char) -> bool {
    is_major(c) || is_minor(c)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Piece {
    Char(char),
    /// `*` in SPL, `%` in LIKE.
    Star,
    /// `_` in LIKE.
    One,
}

/// SPL wildcard text: `*` is the only wildcard; `\*` and `\\` are literal.
pub(super) fn wildcard(text: &str) -> Vec<Piece> {
    let mut pieces = Vec::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '*' => {
                if pieces.last() != Some(&Piece::Star) {
                    pieces.push(Piece::Star);
                }
            }
            '\\' => pieces.push(Piece::Char(
                chars.next_if(|n| matches!(n, '*' | '\\')).unwrap_or('\\'),
            )),
            c => pieces.push(Piece::Char(c)),
        }
    }
    pieces
}

/// SQL LIKE text: `%` matches any run and `_` one character.
pub(super) fn like(text: &str) -> Vec<Piece> {
    text.chars()
        .map(|c| match c {
            '%' => Piece::Star,
            '_' => Piece::One,
            c => Piece::Char(c),
        })
        .collect()
}

pub(super) fn literal_text(pieces: &[Piece]) -> String {
    pieces
        .iter()
        .map(|p| match p {
            Piece::Char(c) => *c,
            Piece::Star => '*',
            Piece::One => '_',
        })
        .collect()
}

fn fold(c: char) -> char {
    if c.is_ascii() {
        c.to_ascii_lowercase()
    } else {
        c.to_lowercase().next().unwrap_or(c)
    }
}

fn same(a: char, b: char, insensitive: bool) -> bool {
    a == b || (insensitive && fold(a) == fold(b))
}

/// Match `pieces` from byte `t` of `text`. Stars may only consume characters
/// accepted by `star`. Succeeds at the first end position `end` accepts.
fn match_at(
    pieces: &[Piece],
    text: &str,
    mut t: usize,
    insensitive: bool,
    star: impl Fn(char) -> bool,
    end: impl Fn(usize) -> bool,
) -> bool {
    let mut p = 0;
    let mut resume: Option<(usize, usize)> = None;
    loop {
        if p == pieces.len() {
            if end(t) {
                return true;
            }
        } else {
            let next = text[t..].chars().next();
            match (pieces[p], next) {
                (Piece::Star, _) => {
                    resume = Some((p + 1, t));
                    p += 1;
                    continue;
                }
                (Piece::One, Some(c)) => {
                    p += 1;
                    t += c.len_utf8();
                    continue;
                }
                (Piece::Char(want), Some(c)) if same(want, c, insensitive) => {
                    p += 1;
                    t += c.len_utf8();
                    continue;
                }
                _ => {}
            }
        }
        // Let the most recent star consume one more character.
        let Some((after, from)) = resume else {
            return false;
        };
        let Some(c) = text[from..].chars().next() else {
            return false;
        };
        if !star(c) {
            return false;
        }
        resume = Some((after, from + c.len_utf8()));
        p = after;
        t = from + c.len_utf8();
    }
}

/// Whole-value wildcard match, as for `field=value`.
pub(super) fn glob_full(pieces: &[Piece], text: &str, insensitive: bool) -> bool {
    match_at(pieces, text, 0, insensitive, |_| true, |t| t == text.len())
}

/// Splunk term semantics over `_raw`: the term must begin and end at a breaker
/// (major or minor) unless its own edge character is a breaker or a wildcard.
/// Wildcards stay inside one major segment. `TERM()` uses major breakers only.
pub(super) fn term_in(pieces: &[Piece], text: &str, insensitive: bool, major_only: bool) -> bool {
    let boundary = |c: char| {
        if major_only {
            is_major(c)
        } else {
            is_breaker(c)
        }
    };
    let needs_left = matches!(pieces.first(), Some(Piece::Char(c)) if !boundary(*c));
    let needs_right = matches!(pieces.last(), Some(Piece::Char(c)) if !boundary(*c));
    // A leading star absorbs any start inside a major segment, so trying each
    // segment's first position is enough and keeps the scan linear in segments.
    let leading_star = pieces.first() == Some(&Piece::Star);
    let end = |t: usize| !needs_right || text[t..].chars().next().is_none_or(boundary);
    let mut previous: Option<char> = None;
    for (start, c) in text
        .char_indices()
        .chain(std::iter::once((text.len(), '\0')))
    {
        let start_ok = if leading_star {
            previous.is_none_or(is_major)
        } else {
            !needs_left || previous.is_none_or(boundary)
        };
        if start_ok && match_at(pieces, text, start, insensitive, |c| !is_major(c), end) {
            return true;
        }
        previous = Some(c);
    }
    false
}

/// Splunk-style automatic extraction (`KV_MODE = auto`): the first `key=value`
/// in `_raw` whose key is not part of a longer key. A value is a quoted string
/// or runs to whitespace, `,`, `;`, `&` or `"`.
pub(crate) fn kv_value<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    if key.is_empty() {
        return None;
    }
    let key_char = |c: char| c.is_alphanumeric() || matches!(c, '_' | '.' | '-');
    let mut from = 0;
    while let Some(found) = text[from..].find(key) {
        let start = from + found;
        let end = start + key.len();
        from = start + text[start..].chars().next().map_or(1, char::len_utf8);
        if text[..start].chars().next_back().is_some_and(key_char) {
            continue;
        }
        let Some(rest) = text[end..].strip_prefix('=') else {
            continue;
        };
        let value = match rest.strip_prefix('"') {
            Some(quoted) => quoted.find('"').map(|i| &quoted[..i]),
            None => {
                let stop = rest
                    .find(|c: char| c.is_whitespace() || matches!(c, ',' | ';' | '&' | '"'))
                    .unwrap_or(rest.len());
                Some(&rest[..stop])
            }
        };
        if let Some(value) = value.filter(|v| !v.is_empty()) {
            return Some(value);
        }
    }
    None
}

/// Heap bytes of one `punct` value: 30 characters of up to 4 bytes each.
pub(super) const PUNCT_BYTES: u64 = 120;

/// Splunk's `punct`: the first 30 punctuation characters of `_raw`, with a
/// space as `_`, a tab as `t` and a line break as `n`.
pub(super) fn punct(text: &str) -> String {
    let mut out = String::with_capacity(PUNCT_BYTES as usize);
    let mut count = 0;
    for c in text.chars() {
        if count == 30 {
            break;
        }
        let mapped = match c {
            ' ' => '_',
            '\t' => 't',
            '\n' => 'n',
            '\r' => 'r',
            c if c.is_alphanumeric() || c.is_whitespace() => continue,
            c => c,
        };
        out.push(mapped);
        count += 1;
    }
    out
}

fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (month + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn epoch(parts: [i64; 6]) -> Option<f64> {
    let [year, month, day, hour, minute, second] = parts;
    let days_in_month = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 => 29,
        2 => 28,
        _ => return None,
    };
    if !(1..=days_in_month).contains(&day) || hour > 23 || minute > 59 || second > 60 {
        return None;
    }
    Some((days_from_civil(year, month, day) * 86_400 + hour * 3600 + minute * 60 + second) as f64)
}

fn digits(text: &str, len: usize) -> Option<(i64, &str)> {
    let head = text.get(..len)?;
    if !head.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some((head.parse().ok()?, &text[len..]))
}

/// `YYYY-MM-DDTHH:MM[:SS[.fff]]` followed by `Z` or `±HH:MM`, as epoch seconds.
pub(crate) fn parse_iso(text: &str) -> Option<f64> {
    let (year, rest) = digits(text, 4)?;
    let (month, rest) = digits(rest.strip_prefix('-')?, 2)?;
    let (day, rest) = digits(rest.strip_prefix('-')?, 2)?;
    let rest = rest.strip_prefix(['T', ' '])?;
    let (hour, rest) = digits(rest, 2)?;
    let (minute, mut rest) = digits(rest.strip_prefix(':')?, 2)?;
    let mut second = 0;
    let mut fraction = 0.0;
    if let Some(after) = rest.strip_prefix(':') {
        let (s, after) = digits(after, 2)?;
        second = s;
        rest = after;
        if let Some(after) = rest.strip_prefix('.') {
            let len = after.bytes().take_while(u8::is_ascii_digit).count();
            if len == 0 {
                return None;
            }
            fraction = format!("0.{}", &after[..len]).parse().ok()?;
            rest = &after[len..];
        }
    }
    let offset = if rest == "Z" {
        0
    } else {
        let sign = match rest.as_bytes().first()? {
            b'+' => 1,
            b'-' => -1,
            _ => return None,
        };
        let (h, after) = digits(&rest[1..], 2)?;
        let (m, after) = digits(after.strip_prefix(':')?, 2)?;
        if !after.is_empty() || h > 23 || m > 59 {
            return None;
        }
        sign * (h * 3600 + m * 60)
    };
    Some(epoch([year, month, day, hour, minute, second])? - offset as f64 + fraction)
}

/// An absolute `earliest`/`latest` value: epoch seconds, Splunk's
/// `%m/%d/%Y:%H:%M:%S` (read as UTC) or ISO 8601.
pub(super) fn parse_time_modifier(text: &str) -> Result<f64, String> {
    if text.starts_with(['-', '+']) || text.contains('@') || text.eq_ignore_ascii_case("now") {
        return Err(format!(
            "Relative time {text:?} is not supported: a saved view must replay identically. \
             Give an absolute time as epoch seconds, %m/%d/%Y:%H:%M:%S (UTC) or ISO 8601."
        ));
    }
    if let Some(n) = parse_number(text) {
        return Ok(n);
    }
    let splunk = || {
        let (month, rest) = digits(text, 2)?;
        let (day, rest) = digits(rest.strip_prefix('/')?, 2)?;
        let (year, rest) = digits(rest.strip_prefix('/')?, 4)?;
        let (hour, rest) = digits(rest.strip_prefix(':')?, 2)?;
        let (minute, rest) = digits(rest.strip_prefix(':')?, 2)?;
        let (second, rest) = digits(rest.strip_prefix(':')?, 2)?;
        rest.is_empty()
            .then(|| epoch([year, month, day, hour, minute, second]))
            .flatten()
    };
    splunk().or_else(|| parse_iso(text)).ok_or_else(|| {
        format!(
            "Cannot read time {text:?}. Use epoch seconds, %m/%d/%Y:%H:%M:%S (UTC) or ISO 8601."
        )
    })
}

/// Numeric when both sides are numbers, otherwise byte order of the text.
pub(super) fn compare_values(a: &Value<'_>, b: &Value<'_>) -> Option<Ordering> {
    if a.is_null() || b.is_null() {
        return None;
    }
    if let (Value::Bool(x), Value::Bool(y)) = (a, b) {
        return Some(x.cmp(y));
    }
    if let (Some(x), Some(y)) = (a.number(), b.number()) {
        return x.partial_cmp(&y);
    }
    Some(a.as_text()?.as_ref().cmp(b.as_text()?.as_ref()))
}
