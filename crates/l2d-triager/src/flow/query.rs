//! A Splunk SPL subset over retained originals. A query is a `search`
//! expression followed by `|` stages that filter (`search`, `where`, `regex`),
//! extract (`rex`, `fields`) or order and truncate (`sort`, `head`, `tail`,
//! `dedup`). Commands that turn events into tables (`stats`, `eval`, `table`,
//! `timechart`, ...) do not apply to a view of originals and are refused.
//!
//! Rows are originals. Fields `host` and `_time` belong to occurrences: a
//! filter that reads them keeps only the matching occurrences of each original,
//! and ordering commands refuse them. The query is compiled once per drill and
//! every allocation made while it runs is checked against the memory budget.
mod eval;
pub(crate) mod field;
mod search;

use std::{borrow::Cow, cmp::Ordering, collections::HashMap, mem::size_of, net::IpAddr};

use l2d_core::{
    MemoryBudget,
    marks::REGEX_HELD_BYTES,
    selections::{SearchQuery, compile},
};
use regex::Regex;

use super::{Member, Record};
use field::{FieldRef, Fields, Row, Value};

const MAX_BYTES: usize = 8192;
const MAX_STAGES: usize = 32;
const MAX_REGEXES: usize = 16;
const DEFAULT_COUNT: usize = 10;
const DEFAULT_SORT_LIMIT: usize = 10_000;
// Tokens, expression trees, wildcard pieces and option text of one query.
const PARSER_BYTES: u64 = 256 * 1024;
const SUPPORTED: &str = "search, where, regex, rex, head, tail, sort, dedup and fields";

pub(super) struct Query {
    stages: Vec<Stage>,
    slots: usize,
    uses_occurrence: bool,
    predicate: bool,
    reserved_bytes: u64,
}

/// Where occurrence-level stages read and clear the scoped groups of each
/// original: bit `offsets[record] + i` stands for `records[record].occurrences[i]`.
pub(super) struct Occurrences<'m> {
    pub mask: &'m mut [u64],
    pub offsets: &'m [usize],
}

impl Occurrences<'_> {
    fn bits(&self, record: usize) -> std::ops::Range<usize> {
        self.offsets[record]..self.offsets[record + 1]
    }
    fn get(&self, bit: usize) -> bool {
        self.mask[bit / 64] & (1 << (bit % 64)) != 0
    }
    fn clear(&mut self, bit: usize) {
        self.mask[bit / 64] &= !(1 << (bit % 64));
    }
    fn drop_record(&mut self, record: usize) {
        for bit in self.bits(record) {
            self.clear(bit);
        }
    }
}

fn drop_rows(scope: &mut Option<Occurrences<'_>>, dropped: &[(usize, Member)]) {
    if let Some(scope) = scope {
        for (_, m) in dropped {
            scope.drop_record(m.record);
        }
    }
}

enum Filter {
    Search(search::Expr),
    Where(eval::Node),
    Regex {
        field: FieldRef,
        regex: Regex,
        negate: bool,
    },
}

impl Filter {
    fn keep(&self, row: &Row<'_, '_>) -> bool {
        match self {
            Self::Search(expr) => expr.matches(row),
            Self::Where(node) => matches!(node.eval(row), Value::Bool(true)),
            Self::Regex {
                field,
                regex,
                negate,
            } => {
                let value = field.get(row);
                let matched = value.as_text().is_some_and(|t| regex.is_match(&t));
                matched != *negate
            }
        }
    }

    fn uses_occurrence(&self) -> bool {
        match self {
            Self::Search(expr) => expr.uses_occurrence(),
            Self::Where(node) => node.uses_occurrence(),
            Self::Regex { field, .. } => field.occurrence(),
        }
    }

    /// Transient bytes while one row is tested. A search or regex owns at most
    /// a formatted number and a `punct` value at a time.
    fn scratch(&self, text_bytes: u64) -> u64 {
        match self {
            Self::Where(node) => node.scratch(text_bytes),
            _ => 2 * (eval::NUMBER_TEXT + field::PUNCT_BYTES) + 256,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum SortMode {
    Auto,
    Str,
    Num,
    Ip,
}

struct SortKey {
    field: FieldRef,
    mode: SortMode,
    descending: bool,
}

enum Ordered<'v> {
    Num(f64),
    Text(Cow<'v, str>),
    Ip(u8, u128),
}

impl SortKey {
    fn key<'v>(&self, value: &'v Value<'_>) -> Option<Ordered<'v>> {
        match self.mode {
            SortMode::Num => value.number().map(Ordered::Num),
            SortMode::Str => value.as_text().map(Ordered::Text),
            SortMode::Ip => value
                .as_text()?
                .trim()
                .parse::<IpAddr>()
                .ok()
                .map(|ip| match ip {
                    IpAddr::V4(v4) => Ordered::Ip(4, u128::from(u32::from(v4))),
                    IpAddr::V6(v6) => Ordered::Ip(6, u128::from(v6)),
                }),
            // Like Splunk's auto mode: numbers, then IP addresses, then text.
            SortMode::Auto => value.number().map(Ordered::Num).or_else(|| {
                let text = value.as_text()?;
                match text.trim().parse::<IpAddr>() {
                    Ok(IpAddr::V4(v4)) => Some(Ordered::Ip(4, u128::from(u32::from(v4)))),
                    Ok(IpAddr::V6(v6)) => Some(Ordered::Ip(6, u128::from(v6))),
                    Err(_) => Some(Ordered::Text(text)),
                }
            }),
        }
    }

    /// Missing values sort last in either direction; ties keep input order.
    fn compare(&self, a: &Value<'_>, b: &Value<'_>) -> Ordering {
        let order = match (self.key(a), self.key(b)) {
            (None, None) => return Ordering::Equal,
            (None, Some(_)) => return Ordering::Greater,
            (Some(_), None) => return Ordering::Less,
            (Some(x), Some(y)) => match (x, y) {
                (Ordered::Num(x), Ordered::Num(y)) => x.total_cmp(&y),
                (Ordered::Num(_), _) => Ordering::Less,
                (_, Ordered::Num(_)) => Ordering::Greater,
                (Ordered::Ip(f, x), Ordered::Ip(g, y)) => (f, x).cmp(&(g, y)),
                (Ordered::Ip(..), _) => Ordering::Less,
                (_, Ordered::Ip(..)) => Ordering::Greater,
                (Ordered::Text(x), Ordered::Text(y)) => x.as_bytes().cmp(y.as_bytes()),
            },
        };
        if self.descending {
            order.reverse()
        } else {
            order
        }
    }
}

struct Sort {
    keys: Vec<SortKey>,
    /// Zero keeps every row.
    limit: usize,
}

struct Dedup {
    count: usize,
    fields: Vec<FieldRef>,
    keepempty: bool,
    consecutive: bool,
    sortby: Option<Sort>,
}

struct Rex {
    field: FieldRef,
    regex: Regex,
    slots: Vec<(usize, usize)>,
}

enum Head {
    Count(usize),
    While {
        node: eval::Node,
        keeplast: bool,
        null: bool,
    },
}

enum Stage {
    Filter(Filter),
    Rex(Rex),
    Head(Head),
    Tail(usize),
    Sort(Sort),
    Dedup(Dedup),
}

/// Compile-time state shared by the stage parsers.
struct Compiler<'b> {
    budget: &'b MemoryBudget,
    fields: Fields,
    reserved_bytes: u64,
    regexes: usize,
}

impl Compiler<'_> {
    fn regex(&mut self, pattern: &str) -> Result<Regex, String> {
        self.regexes += 1;
        if self.regexes > MAX_REGEXES {
            return Err(format!(
                "A query is limited to {MAX_REGEXES} regular expressions."
            ));
        }
        self.reserved_bytes += REGEX_HELD_BYTES;
        self.budget
            .require_whole(self.reserved_bytes, "compiling search regular expressions")
            .map_err(|e| e.to_string())?;
        let query = SearchQuery::Regex {
            pattern: pattern.to_owned(),
            insensitive: false,
        };
        compile(&query, self.budget).map_err(|e| e.to_string())
    }
}

/// Split at `|` outside double quotes and drop ```comments```.
fn stages(source: &str) -> Result<Vec<String>, String> {
    let mut out = vec![String::new()];
    let mut chars = source.chars().peekable();
    let mut quoted = false;
    while let Some(c) = chars.next() {
        let current = out.last_mut().expect("one stage");
        match c {
            '\\' => {
                current.push(c);
                if let Some(n) = chars.next() {
                    current.push(n);
                }
            }
            '"' => {
                quoted = !quoted;
                current.push(c);
            }
            '`' if !quoted && chars.next_if_eq(&'`').is_some() => {
                if chars.next_if_eq(&'`').is_none() {
                    return Err("Search macros are not supported.".into());
                }
                let mut ticks = 0;
                loop {
                    match chars.next() {
                        Some('`') => {
                            ticks += 1;
                            if ticks == 3 {
                                break;
                            }
                        }
                        Some(_) => ticks = 0,
                        None => return Err("Unclosed ``` comment.".into()),
                    }
                }
                current.push(' ');
            }
            '|' if !quoted => {
                if out.len() == MAX_STAGES {
                    return Err(format!("A query is limited to {MAX_STAGES} stages."));
                }
                out.push(String::new());
            }
            c => current.push(c),
        }
    }
    if quoted {
        return Err("Unclosed quote in search.".into());
    }
    Ok(out)
}

/// Command arguments: words, quoted strings, `=`, `!=`, commas and parentheses.
#[derive(Debug, PartialEq)]
enum Arg {
    Word(String),
    Quoted(String),
    Eq,
    Ne,
    Comma,
    Open,
    Close,
}

fn args(text: &str) -> Result<Vec<Arg>, String> {
    let mut out = Vec::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        out.push(match c {
            c if c.is_whitespace() => continue,
            '=' => Arg::Eq,
            ',' => Arg::Comma,
            '(' => Arg::Open,
            ')' => Arg::Close,
            '!' if chars.next_if_eq(&'=').is_some() => Arg::Ne,
            '"' => {
                let mut s = String::new();
                loop {
                    match chars.next() {
                        None => return Err("Unclosed quote in command arguments.".into()),
                        Some('"') => break,
                        // As the stage splitter does: \" is a quote and \\ stays
                        // an escaped backslash for the regex.
                        Some('\\') => match chars.next_if(|n| matches!(n, '"' | '\\')) {
                            Some('"') => s.push('"'),
                            Some(_) => s.push_str(r"\\"),
                            None => s.push('\\'),
                        },
                        Some(c) => s.push(c),
                    }
                }
                Arg::Quoted(s)
            }
            c => {
                let mut s = String::from(c);
                loop {
                    match chars.peek() {
                        None => break,
                        Some(&n)
                            if n.is_whitespace() || matches!(n, '=' | ',' | '(' | ')' | '"') =>
                        {
                            break;
                        }
                        Some('!') => {
                            let mut ahead = chars.clone();
                            ahead.next();
                            if ahead.peek() == Some(&'=') {
                                break;
                            }
                            s.push('!');
                            chars.next();
                        }
                        Some(&n) => {
                            s.push(n);
                            chars.next();
                        }
                    }
                }
                Arg::Word(s)
            }
        });
    }
    Ok(out)
}

/// Splits `key=value` options from positional arguments.
fn options(
    list: Vec<Arg>,
    known: &[&str],
    command: &str,
) -> Result<(Vec<Arg>, HashMap<String, String>), String> {
    let mut positional = Vec::new();
    let mut options = HashMap::new();
    let mut items = list.into_iter().peekable();
    while let Some(item) = items.next() {
        if let Arg::Word(word) = &item
            && items.peek() == Some(&Arg::Eq)
            && known.contains(&word.as_str())
        {
            items.next();
            let value = match items.next() {
                Some(Arg::Word(v) | Arg::Quoted(v)) => v,
                _ => return Err(format!("{command} option {word} needs a value.")),
            };
            options.insert(word.clone(), value);
            continue;
        }
        positional.push(item);
    }
    Ok((positional, options))
}

fn boolean(value: &str, option: &str) -> Result<bool, String> {
    match value.to_ascii_lowercase().as_str() {
        "true" | "t" | "1" | "yes" => Ok(true),
        "false" | "f" | "0" | "no" => Ok(false),
        _ => Err(format!("{option} takes true or false, got {value:?}.")),
    }
}

fn count(value: &str, command: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|_| format!("{command} expects a whole number, got {value:?}."))
}

fn row_field(fields: &Fields, name: &str, command: &str) -> Result<FieldRef, String> {
    let field = fields.resolve(name);
    if field.occurrence() {
        return Err(format!(
            "{command} orders originals, but {name} varies by occurrence. Filter on {name} with search or where first."
        ));
    }
    Ok(field)
}

impl Compiler<'_> {
    fn search(&self, text: &str) -> Result<Stage, String> {
        search::parse(text, &self.fields).map(|e| Stage::Filter(Filter::Search(e)))
    }

    fn regex_stage(&mut self, text: &str) -> Result<Stage, String> {
        let list = args(text)?;
        let (field, negate, pattern) = match list.as_slice() {
            [Arg::Word(p) | Arg::Quoted(p)] => (FieldRef::Raw, false, p.clone()),
            [
                Arg::Word(f),
                op @ (Arg::Eq | Arg::Ne),
                Arg::Word(p) | Arg::Quoted(p),
            ] => (self.fields.resolve(f), *op == Arg::Ne, p.clone()),
            _ => return Err("Use regex [field=|field!=]\"<regular expression>\".".into()),
        };
        Ok(Stage::Filter(Filter::Regex {
            field,
            regex: self.regex(&pattern)?,
            negate,
        }))
    }

    fn rex(&mut self, text: &str) -> Result<Stage, String> {
        let (positional, options) = options(
            args(text)?,
            &["field", "max_match", "mode", "offset_field"],
            "rex",
        )?;
        if options.contains_key("mode") {
            return Err(
                "rex mode=sed rewrites fields, which views of originals do not hold.".into(),
            );
        }
        if options.contains_key("offset_field") {
            return Err("rex offset_field is not supported.".into());
        }
        if let Some(n) = options.get("max_match")
            && n != "1"
        {
            return Err("rex supports max_match=1; multivalue fields are not supported.".into());
        }
        let [Arg::Word(pattern) | Arg::Quoted(pattern)] = positional.as_slice() else {
            return Err("Use rex [field=<field>] \"<regex with (?<name>...) groups>\".".into());
        };
        let name = options.get("field").map_or("_raw", String::as_str);
        let field = self.fields.resolve(name);
        if field.occurrence() {
            return Err(format!(
                "rex extracts from one original, but {name} varies by occurrence."
            ));
        }
        if !field.borrowable() {
            return Err(format!("rex reads text fields; {name} is computed."));
        }
        let regex = self.regex(pattern)?;
        let names: Vec<(usize, String)> = regex
            .capture_names()
            .enumerate()
            .filter_map(|(i, n)| n.map(|n| (i, n.to_owned())))
            .collect();
        if names.is_empty() {
            return Err("rex needs at least one named group such as (?<user>\\w+).".into());
        }
        let mut slots = Vec::with_capacity(names.len());
        for (group, name) in names {
            slots.push((group, self.fields.define(&name)?));
        }
        Ok(Stage::Rex(Rex {
            field,
            regex,
            slots,
        }))
    }

    fn head(&mut self, text: &str) -> Result<Stage, String> {
        let trimmed = text.trim_start();
        if let Some(inner) = trimmed.strip_prefix('(') {
            let mut depth = 1;
            let mut quoted = false;
            let mut end = None;
            for (i, c) in inner.char_indices() {
                match c {
                    '"' => quoted = !quoted,
                    '(' if !quoted => depth += 1,
                    ')' if !quoted => {
                        depth -= 1;
                        if depth == 0 {
                            end = Some(i);
                            break;
                        }
                    }
                    _ => {}
                }
            }
            let end = end.ok_or("head ( is missing its closing parenthesis.")?;
            let node = eval::parse(&inner[..end], self)?;
            if node.uses_occurrence() {
                return Err("head orders originals; host and _time vary by occurrence.".into());
            }
            let (rest, options) = options(args(&inner[end + 1..])?, &["keeplast", "null"], "head")?;
            if !rest.is_empty() {
                return Err("Use head (<eval expression>) [keeplast=bool] [null=bool].".into());
            }
            return Ok(Stage::Head(Head::While {
                node,
                keeplast: options
                    .get("keeplast")
                    .map_or(Ok(false), |v| boolean(v, "keeplast"))?,
                null: options
                    .get("null")
                    .map_or(Ok(false), |v| boolean(v, "null"))?,
            }));
        }
        let (rest, options) = options(args(text)?, &["limit"], "head")?;
        let n = match (rest.as_slice(), options.get("limit")) {
            ([], None) => DEFAULT_COUNT,
            ([Arg::Word(n)], None) | ([], Some(n)) => count(n, "head")?,
            _ => return Err("Use head [<count>] or head limit=<count>.".into()),
        };
        Ok(Stage::Head(Head::Count(n)))
    }

    fn tail(&self, text: &str) -> Result<Stage, String> {
        match args(text)?.as_slice() {
            [] => Ok(Stage::Tail(DEFAULT_COUNT)),
            [Arg::Word(n)] => Ok(Stage::Tail(count(n, "tail")?)),
            _ => Err("Use tail [<count>].".into()),
        }
    }

    fn sort_keys(&self, list: &[Arg], command: &str) -> Result<Vec<SortKey>, String> {
        let mut keys = Vec::new();
        let mut items = list.iter().peekable();
        let mut sign: Option<bool> = None;
        while let Some(item) = items.next() {
            let word = match item {
                Arg::Comma => continue,
                Arg::Word(w) => w.as_str(),
                Arg::Quoted(w) => w.as_str(),
                _ => return Err(format!("Unexpected {item:?} in {command} fields.")),
            };
            let (descending, name) = match word {
                "-" | "+" => {
                    sign = Some(word == "-");
                    continue;
                }
                w if w.len() > 1 && w.starts_with('-') => (true, &w[1..]),
                w if w.len() > 1 && w.starts_with('+') => (false, &w[1..]),
                w => (sign.unwrap_or(false), w),
            };
            sign = None;
            let mode = match name {
                "auto" => Some(SortMode::Auto),
                "str" => Some(SortMode::Str),
                "num" => Some(SortMode::Num),
                "ip" => Some(SortMode::Ip),
                _ => None,
            };
            let (mode, field_name) = match mode {
                Some(mode) if items.peek() == Some(&&Arg::Open) => {
                    items.next();
                    let Some(Arg::Word(inner) | Arg::Quoted(inner)) = items.next() else {
                        return Err(format!("{name}() needs a field name."));
                    };
                    if items.next() != Some(&Arg::Close) {
                        return Err(format!("{name}( is missing its closing parenthesis."));
                    }
                    (mode, inner.as_str())
                }
                _ => (SortMode::Auto, name),
            };
            keys.push(SortKey {
                field: row_field(&self.fields, field_name, command)?,
                mode,
                descending,
            });
        }
        Ok(keys)
    }

    fn sort(&self, text: &str) -> Result<Sort, String> {
        let (mut rest, options) = options(args(text)?, &["limit"], "sort")?;
        let mut limit = options.get("limit").map(|n| count(n, "sort")).transpose()?;
        if let Some(Arg::Word(n)) = rest.first()
            && n.bytes().all(|b| b.is_ascii_digit())
        {
            limit = Some(count(n, "sort")?);
            rest.remove(0);
        }
        let reverse = matches!(rest.last(), Some(Arg::Word(w)) if w == "desc" || w == "d");
        if reverse {
            rest.pop();
        }
        let mut keys = self.sort_keys(&rest, "sort")?;
        if keys.is_empty() {
            return Err("sort needs at least one field, such as sort - score.".into());
        }
        if reverse {
            for key in &mut keys {
                key.descending = !key.descending;
            }
        }
        Ok(Sort {
            keys,
            limit: limit.unwrap_or(DEFAULT_SORT_LIMIT),
        })
    }

    fn dedup(&self, text: &str) -> Result<Stage, String> {
        let mut list = args(text)?;
        let sortby = list
            .iter()
            .position(|a| *a == Arg::Word("sortby".into()))
            .map(|i| list.split_off(i));
        let (mut rest, options) =
            options(list, &["keepempty", "consecutive", "keepevents"], "dedup")?;
        if let Some(v) = options.get("keepevents")
            && boolean(v, "keepevents")?
        {
            return Err("dedup keepevents=true keeps events but clears fields, which views of originals do not hold.".into());
        }
        let mut count_value = 1;
        if let Some(Arg::Word(n)) = rest.first()
            && n.bytes().all(|b| b.is_ascii_digit())
        {
            count_value = count(n, "dedup")?;
            rest.remove(0);
        }
        let mut fields = Vec::new();
        for item in &rest {
            match item {
                Arg::Comma => {}
                Arg::Word(name) | Arg::Quoted(name) => {
                    fields.push(row_field(&self.fields, name, "dedup")?)
                }
                other => return Err(format!("Unexpected {other:?} in dedup fields.")),
            }
        }
        if fields.is_empty() {
            return Err("dedup needs at least one field.".into());
        }
        let sortby = match sortby {
            Some(clause) => {
                let keys = self.sort_keys(&clause[1..], "dedup sortby")?;
                if keys.is_empty() {
                    return Err("dedup sortby needs at least one field.".into());
                }
                Some(Sort { keys, limit: 0 })
            }
            None => None,
        };
        Ok(Stage::Dedup(Dedup {
            count: count_value.max(1),
            fields,
            keepempty: options
                .get("keepempty")
                .map_or(Ok(false), |v| boolean(v, "keepempty"))?,
            consecutive: options
                .get("consecutive")
                .map_or(Ok(false), |v| boolean(v, "consecutive"))?,
            sortby,
        }))
    }

    fn fields_stage(&mut self, text: &str) -> Result<(), String> {
        let mut list = args(text)?;
        let mut remove = false;
        if let Some(Arg::Word(w)) = list.first_mut()
            && (w.starts_with('-') || w.starts_with('+'))
        {
            remove = w.starts_with('-');
            // "fields -a" is "fields - a".
            w.remove(0);
            if w.is_empty() {
                list.remove(0);
            }
        }
        let mut patterns = Vec::new();
        for item in list {
            match item {
                Arg::Comma => {}
                Arg::Word(name) | Arg::Quoted(name) => patterns.push(field::wildcard(&name)),
                other => return Err(format!("Unexpected {other:?} in fields.")),
            }
        }
        if patterns.is_empty() {
            return Err("fields needs at least one field name.".into());
        }
        if remove {
            self.fields.remove(patterns);
        } else {
            self.fields.keep(patterns);
        }
        Ok(())
    }
}

impl Query {
    pub(super) fn parse(source: &str, budget: &MemoryBudget) -> Result<Self, String> {
        if source.len() > MAX_BYTES {
            return Err(format!("A query is limited to {MAX_BYTES} bytes."));
        }
        budget
            .require_whole(PARSER_BYTES, "parsing search")
            .map_err(|e| e.to_string())?;
        let texts = stages(source)?;
        let mut compiler = Compiler {
            budget,
            fields: Fields::default(),
            reserved_bytes: PARSER_BYTES,
            regexes: 0,
        };
        let mut stages = Vec::with_capacity(texts.len());
        for (index, text) in texts.iter().enumerate() {
            let text = text.trim();
            let (command, rest) = match text.split_once(char::is_whitespace) {
                Some((c, r)) => (c, r),
                None => (text, ""),
            };
            let lower = command.to_ascii_lowercase();
            if index == 0 {
                if text.is_empty() && texts.len() > 1 {
                    return Err(
                        "A query starts with search terms. A leading | starts a generating command, which is not supported."
                            .into(),
                    );
                }
                let expression = if lower == "search" { rest } else { text };
                stages.push(compiler.search(expression)?);
                continue;
            }
            if text.is_empty() {
                return Err("Empty pipeline stage between | characters.".into());
            }
            let stage = match lower.as_str() {
                "search" => compiler.search(rest)?,
                "where" => {
                    let node = eval::parse(rest, &mut compiler)?;
                    Stage::Filter(Filter::Where(node))
                }
                "regex" => compiler.regex_stage(rest)?,
                "rex" => compiler.rex(rest)?,
                "head" => compiler.head(rest)?,
                "tail" => compiler.tail(rest)?,
                "sort" => Stage::Sort(compiler.sort(rest)?),
                "dedup" => compiler.dedup(rest)?,
                "fields" => {
                    compiler.fields_stage(rest)?;
                    continue;
                }
                "stats" | "eval" | "table" | "timechart" | "chart" | "top" | "rare"
                | "eventstats" | "streamstats" | "transaction" | "rename" | "lookup" | "join"
                | "append" | "bin" | "bucket" | "spath" | "makemv" | "mvexpand" | "convert"
                | "fillnull" | "addinfo" | "outputlookup" | "inputlookup" | "tstats" => {
                    return Err(format!(
                        "{lower} builds tables or new fields, which a view of originals cannot hold. Supported commands: {SUPPORTED}."
                    ));
                }
                _ => {
                    return Err(format!(
                        "Unknown command {command:?}. Supported commands: {SUPPORTED}."
                    ));
                }
            };
            stages.push(stage);
        }
        let uses_occurrence = stages
            .iter()
            .any(|s| matches!(s, Stage::Filter(f) if f.uses_occurrence()));
        let predicate = stages
            .iter()
            .all(|s| matches!(s, Stage::Filter(_) | Stage::Rex(_)));
        Ok(Self {
            stages,
            slots: compiler.fields.slots(),
            uses_occurrence,
            predicate,
            reserved_bytes: compiler.reserved_bytes,
        })
    }

    /// Whether a stage reads `host` or `_time`, so the query narrows occurrences.
    pub(super) fn uses_occurrence(&self) -> bool {
        self.uses_occurrence
    }

    /// Whether the query only selects originals, without ordering or truncating.
    pub(super) fn is_predicate(&self) -> bool {
        self.predicate
    }

    pub(super) fn reserved_bytes(&self) -> u64 {
        self.reserved_bytes
    }

    /// Transient memory of one run over `rows` originals whose text fields
    /// (`_raw`, `id`, `reason`) are at most `text_bytes`. It covers the working
    /// rows, the dropped-row list, capture slots, the result vector, the largest
    /// sort or dedup, and the largest per-row evaluation. Saturates.
    fn run_bytes(&self, rows: u64, text_bytes: u64) -> u64 {
        let value = size_of::<Value<'_>>() as u64;
        let row = size_of::<(usize, Member)>() as u64;
        // An owned key: a formatted number or a punct value.
        let owned = eval::NUMBER_TEXT.max(field::PUNCT_BYTES);
        let base =
            2 * row + size_of::<Member>() as u64 + (self.slots * size_of::<Option<&str>>()) as u64;
        // Keys with their owned text, an index order, merge-sort scratch and
        // a reordered copy of the rows.
        let sort = |keys: usize| keys as u64 * (value + owned) + 16 + row;
        let mut extra = 0;
        let mut scratch = 0;
        for stage in &self.stages {
            let stage_extra = match stage {
                Stage::Sort(s) => sort(s.keys.len()),
                Stage::Dedup(dedup) => {
                    // Key vector and owned text, and a hash slot during a resize.
                    let keys = dedup.fields.len() as u64 * (value + owned) + 128;
                    keys.max(dedup.sortby.as_ref().map_or(0, |s| sort(s.keys.len())))
                }
                Stage::Filter(filter) => {
                    scratch = scratch.max(filter.scratch(text_bytes));
                    0
                }
                Stage::Head(Head::While { node, .. }) => {
                    scratch = scratch.max(node.scratch(text_bytes));
                    0
                }
                _ => 0,
            };
            extra = extra.max(stage_extra);
        }
        rows.saturating_mul(base + extra)
            .saturating_add(scratch)
            .saturating_add(64 * 1024)
    }

    /// Run the pipeline over `members` in their current order. Occurrence
    /// filters clear the bits of non-matching groups. Dropped rows have all
    /// their bits cleared. `beside` is what the caller has live at the same
    /// time beyond held memory (its scratch, the compiled query, a mask and
    /// the `members` vector); the run is admitted together with it.
    pub(super) fn select<'a>(
        &'a self,
        records: &'a [Record],
        members: Vec<Member>,
        mut scope: Option<Occurrences<'_>>,
        beside: u64,
        budget: &MemoryBudget,
    ) -> Result<Vec<Member>, String> {
        if self.uses_occurrence && scope.is_none() {
            return Err(
                "This search reads host or _time, which needs occurrence locations.".into(),
            );
        }
        let text_bytes = members
            .iter()
            .map(|m| {
                let r = &records[m.record];
                let host = r
                    .occurrences
                    .iter()
                    .filter_map(|g| g.host.as_deref())
                    .map(str::len)
                    .max()
                    .unwrap_or(0);
                r.text.len().max(r.id.len()).max(r.reason.len()).max(host)
            })
            .max()
            .unwrap_or(0) as u64;
        budget
            .require_whole(
                beside.saturating_add(self.run_bytes(members.len() as u64, text_bytes)),
                "running the search pipeline",
            )
            .map_err(|e| e.to_string())?;
        let width = self.slots;
        let mut slots: Vec<Option<&'a str>> = vec![None; members.len() * width];
        let mut rows: Vec<(usize, Member)> = members.into_iter().enumerate().collect();
        for stage in &self.stages {
            match stage {
                Stage::Filter(filter) if filter.uses_occurrence() => {
                    let scope = scope.as_mut().expect("checked above");
                    rows.retain(|&(id, m)| {
                        let record = &records[m.record];
                        let mut any = false;
                        for (bit, group) in scope.bits(m.record).zip(&record.occurrences) {
                            if !scope.get(bit) {
                                continue;
                            }
                            let row = Row {
                                record,
                                group: Some(group),
                                slots: &slots[id * width..(id + 1) * width],
                            };
                            if filter.keep(&row) {
                                any = true;
                            } else {
                                scope.clear(bit);
                            }
                        }
                        any
                    });
                }
                Stage::Filter(filter) => {
                    let mut dropped = Vec::new();
                    rows.retain(|&(id, m)| {
                        let row = Row {
                            record: &records[m.record],
                            group: None,
                            slots: &slots[id * width..(id + 1) * width],
                        };
                        let keep = filter.keep(&row);
                        if !keep && scope.is_some() {
                            dropped.push((id, m));
                        }
                        keep
                    });
                    drop_rows(&mut scope, &dropped);
                }
                Stage::Rex(rex) => {
                    let mut locations = rex.regex.capture_locations();
                    for &(id, m) in &rows {
                        let record = &records[m.record];
                        let base = id * width;
                        let source = rex.field.text(&Row {
                            record,
                            group: None,
                            slots: &slots[base..base + width],
                        });
                        let Some(text) = source else { continue };
                        if rex.regex.captures_read(&mut locations, text).is_none() {
                            continue;
                        }
                        for &(group, slot) in &rex.slots {
                            if let Some((start, end)) = locations.get(group) {
                                slots[base + slot] = Some(&text[start..end]);
                            }
                        }
                    }
                }
                Stage::Head(head) => {
                    let keep = match head {
                        Head::Count(n) => (*n).min(rows.len()),
                        Head::While {
                            node,
                            keeplast,
                            null,
                        } => rows
                            .iter()
                            .position(|&(id, m)| {
                                let row = Row {
                                    record: &records[m.record],
                                    group: None,
                                    slots: &slots[id * width..(id + 1) * width],
                                };
                                match node.eval(&row) {
                                    Value::Bool(b) => !b,
                                    Value::Null => !null,
                                    _ => true,
                                }
                            })
                            .map_or(rows.len(), |i| if *keeplast { i + 1 } else { i }),
                    };
                    drop_rows(&mut scope, &rows[keep..]);
                    rows.truncate(keep);
                }
                Stage::Tail(n) => {
                    let skip = rows.len().saturating_sub(*n);
                    drop_rows(&mut scope, &rows[..skip]);
                    rows.drain(..skip);
                    rows.reverse();
                }
                Stage::Sort(sort) => {
                    sort_rows(sort, &mut rows, records, &slots, width, &mut scope);
                }
                Stage::Dedup(dedup) => {
                    let mut seen: HashMap<Vec<Cow<'a, str>>, usize> = HashMap::new();
                    let mut previous: Option<Vec<Cow<'a, str>>> = None;
                    let mut run = 0;
                    let mut dropped = Vec::new();
                    rows.retain(|&(id, m)| {
                        let row = Row {
                            record: &records[m.record],
                            group: None,
                            slots: &slots[id * width..(id + 1) * width],
                        };
                        let key: Option<Vec<Cow<'a, str>>> = dedup
                            .fields
                            .iter()
                            .map(|f| f.get(&row).into_text())
                            .collect();
                        let keep = match key {
                            None => dedup.keepempty,
                            Some(key) if dedup.consecutive => {
                                if previous.as_ref() == Some(&key) {
                                    run += 1;
                                } else {
                                    run = 1;
                                    previous = Some(key);
                                }
                                run <= dedup.count
                            }
                            Some(key) => {
                                let seen = seen.entry(key).or_insert(0);
                                *seen += 1;
                                *seen <= dedup.count
                            }
                        };
                        if !keep {
                            dropped.push((id, m));
                        }
                        keep
                    });
                    drop(seen);
                    drop_rows(&mut scope, &dropped);
                    if let Some(sort) = &dedup.sortby {
                        sort_rows(sort, &mut rows, records, &slots, width, &mut scope);
                    }
                }
            }
        }
        Ok(rows.into_iter().map(|(_, m)| m).collect())
    }
}

fn sort_rows<'a>(
    sort: &Sort,
    rows: &mut Vec<(usize, Member)>,
    records: &'a [Record],
    slots: &[Option<&'a str>],
    width: usize,
    scope: &mut Option<Occurrences<'_>>,
) {
    let fields = sort.keys.len();
    let mut keys: Vec<Value<'a>> = Vec::with_capacity(rows.len() * fields);
    for &(id, m) in rows.iter() {
        let row = Row {
            record: &records[m.record],
            group: None,
            slots: &slots[id * width..(id + 1) * width],
        };
        keys.extend(sort.keys.iter().map(|k| k.field.get(&row)));
    }
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by(|&a, &b| {
        sort.keys
            .iter()
            .enumerate()
            .map(|(k, key)| key.compare(&keys[a * fields + k], &keys[b * fields + k]))
            .find(|o| o.is_ne())
            .unwrap_or(Ordering::Equal)
    });
    drop(keys);
    let mut sorted: Vec<(usize, Member)> = order.iter().map(|&i| rows[i]).collect();
    let keep = if sort.limit == 0 {
        sorted.len()
    } else {
        sort.limit.min(sorted.len())
    };
    drop_rows(scope, &sorted[keep..]);
    sorted.truncate(keep);
    *rows = sorted;
}

#[cfg(test)]
mod tests;
