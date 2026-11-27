//! The terminal workbench over the investigation service (triager task 3.2,
//! requirements 4.1, 5.1 to 5.4 and 6.3).
//!
//! The workbench makes the tool calls `l2d mcp` serves, with the same arguments, on
//! the same [`Service`]: a view it opens is the view an MCP client composes, an
//! allocation is `triager_compose` with `mode=allocate`, and a saved hunt is a
//! `triager-receipt` another client reopens. It starts from a recipe of tool calls
//! (the handles the recipe bound) or from a saved receipt.
//!
//! What is drawn comes from results already returned: a resize, a divider move or a
//! redraw makes no call, so the service's work counters (`activity`) stay where they
//! were. A call that starts a job returns at once; the workbench polls the job and
//! cancels it on Esc. Every call, frame and resize can be written to a trace (JSON
//! lines) that keeps layout and drawing time apart from the time of the calls.

mod draw;

use std::collections::VecDeque;
use std::io::{self, IsTerminal, Write};
use std::path::Path;
use std::time::{Duration, Instant};

use crossterm::event::{
    self, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton,
    MouseEvent, MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{EnterAlternateScreen, enable_raw_mode};
use ratatui::layout::{Position, Rect};
use ratatui::{Terminal, backend::CrosstermBackend};
use serde_json::{Map, Value, json};

use crate::service::Service;
use crate::ui::layout::Panes;

/// Rows a records page asks for: the service's default page.
const PAGE_ROWS: u64 = 25;
/// Values a field menu lists.
const VALUE_ROWS: u64 = 50;
/// Named sets that get a membership column and a place in the regions table.
const REGION_SETS: usize = 3;
/// The most frames (views) one session keeps.
const MAX_FRAMES: usize = 256;

/// Where the workbench opens: a snapshot, the named sets that get membership columns,
/// further views to offer as branches and the view to show first.
#[derive(Debug, Clone, PartialEq)]
pub struct Start {
    pub snapshot: String,
    pub named: Vec<(String, String)>,
    pub views: Vec<(String, String)>,
    pub active: Option<String>,
    /// What the start was: "recipe" or the receipt's handle.
    pub origin: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Snapshot,
    Selection,
    View,
    Final,
}

impl Kind {
    fn of(handle: &str) -> Option<Self> {
        Some(match handle.split_once('-')?.0 {
            "snap" => Self::Snapshot,
            "set" => Self::Selection,
            "view" => Self::View,
            "final" => Self::Final,
            _ => return None,
        })
    }
}

impl Start {
    /// The start of a recipe: the snapshot bound last; every bound selection as a
    /// named set (by its binding's name, in lower case); every bound view or final
    /// set as a branch; the handle bound last as the first view.
    pub fn from_bindings(bound: &[(String, Value)]) -> Result<Self, String> {
        let mut start = Self {
            snapshot: String::new(),
            named: Vec::new(),
            views: Vec::new(),
            active: None,
            origin: "recipe".into(),
        };
        for (name, value) in bound {
            let Some(handle) = value.as_str() else {
                continue;
            };
            match Kind::of(handle) {
                Some(Kind::Snapshot) => {
                    start.snapshot = handle.to_string();
                    start.active = Some(handle.to_string());
                }
                Some(Kind::Selection) => {
                    let name = name.to_lowercase();
                    start.named.retain(|(known, _)| known != &name);
                    start.named.push((name, handle.to_string()));
                }
                Some(Kind::View | Kind::Final) => {
                    start.views.push((name.clone(), handle.to_string()));
                    start.active = Some(handle.to_string());
                }
                None => {}
            }
        }
        if start.snapshot.is_empty() {
            return Err("the recipe binds no snapshot: bind the `snapshot` of its \
                        `triager_open`, for example \"bind\": {\"S\": \"snapshot\"}"
                .into());
        }
        Ok(start)
    }

    /// The start of a reopened receipt (`triager_open {"receipt": R}`): its snapshot,
    /// the sets it names, every other set as a branch and the saved set first.
    pub fn from_reopened(result: &Value) -> Result<Self, String> {
        let snapshot = result["snapshot"]
            .as_str()
            .ok_or("the reopen result names no snapshot")?
            .to_string();
        let mut start = Self {
            snapshot,
            named: Vec::new(),
            views: Vec::new(),
            active: result["set"].as_str().map(str::to_string),
            origin: result["receipt"].as_str().unwrap_or("receipt").to_string(),
        };
        for set in result["sets"].as_array().into_iter().flatten() {
            let Some(handle) = set["handle"].as_str() else {
                continue;
            };
            if let Some(name) = set["name"].as_str() {
                start.named.push((name.to_string(), handle.to_string()));
            } else if matches!(Kind::of(handle), Some(Kind::View | Kind::Final)) {
                let label = format!(
                    "saved {} ({})",
                    set["kind"].as_str().unwrap_or("view"),
                    set["producer"].as_str().unwrap_or("expression")
                );
                start.views.push((label, handle.to_string()));
            }
        }
        Ok(start)
    }
}

/// What a pane shows before, while and after its call.
#[derive(Debug, Clone)]
enum Load<T> {
    Idle,
    /// Cancelled by the analyst: drawn as such, and loaded again like `Idle`.
    Cancelled,
    Loading,
    Ready(T),
    Unavailable(String),
}

impl<T> Load<T> {
    fn ready(&self) -> Option<&T> {
        match self {
            Self::Ready(value) => Some(value),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
struct Row {
    id: String,
    locator: String,
    occurrences: u64,
    /// The record's state in every named set: `+` in, `-` out, `?` not evaluated.
    states: Vec<char>,
    /// The pool and slot of a member of a final work set.
    allocation: Option<String>,
}

#[derive(Debug, Clone)]
struct Page {
    rows: Vec<Row>,
    /// The page's index and the cursors that fetch each page so far.
    index: usize,
    cursors: Vec<Option<String>>,
    next: Option<String>,
    total: (u64, u64),
}

#[derive(Debug, Clone)]
struct RegionRow {
    /// The state of each named set, by index: "in", "out" or "not_evaluated".
    states: Vec<(usize, String)>,
    originals: u64,
    occurrences: u64,
}

#[derive(Debug, Clone)]
struct FieldRow {
    name: String,
    distinct: u64,
    exact: bool,
    with_value: (u64, u64),
    missing: (u64, u64),
}

#[derive(Debug, Clone)]
struct Bucket {
    start: String,
    start_seconds: i64,
    originals: u64,
    occurrences: u64,
}

#[derive(Debug, Clone)]
struct Histogram {
    bucket_seconds: i64,
    buckets: Vec<Bucket>,
    with_time: (u64, u64),
    missing: (u64, u64),
    invalid: (u64, u64),
    timezone: String,
}

/// The reduction a child view states of itself: in and parent in both units.
#[derive(Debug, Clone, Copy)]
struct Reduction {
    originals: (u64, u64),
    occurrences: (u64, u64),
    unknown: u64,
}

struct Frame {
    handle: String,
    kind: Kind,
    title: String,
    parent: Option<usize>,
    members: Option<(u64, u64)>,
    reduction: Option<Reduction>,
    /// Matching occurrences of a field or time view, apart from its originals.
    matching: Option<u64>,
    page: Load<Page>,
    selected: usize,
    regions: Load<(Vec<RegionRow>, Vec<RegionRow>)>,
    fields: Load<Vec<FieldRow>>,
    histogram: Load<Histogram>,
    draft: String,
    syntax: usize,
}

impl Frame {
    fn new(handle: String, title: String, parent: Option<usize>) -> Self {
        Self {
            kind: Kind::of(&handle).unwrap_or(Kind::View),
            handle,
            title,
            parent,
            members: None,
            reduction: None,
            matching: None,
            page: Load::Idle,
            selected: 0,
            regions: Load::Idle,
            fields: Load::Idle,
            histogram: Load::Idle,
            draft: String::new(),
            syntax: 0,
        }
    }
}

/// The syntaxes of a search: the filter producer's.
const SYNTAXES: [&str; 3] = ["glob", "literal", "regex"];

#[derive(Debug, Clone)]
enum Op {
    Overview {
        frame: usize,
    },
    Records {
        frame: usize,
        index: usize,
        cursor: Option<String>,
        select_last: bool,
    },
    Regions {
        frame: usize,
    },
    Fields {
        frame: usize,
    },
    Histogram {
        frame: usize,
    },
    Values {
        frame: usize,
        field: String,
    },
    Evidence {
        frame: usize,
        record: String,
        excerpt: bool,
    },
    Select {
        from: usize,
        title: String,
        syntax: &'static str,
        query: String,
    },
    Child {
        from: usize,
        title: String,
        tool: &'static str,
        args: Value,
        key: &'static str,
    },
    Save {
        frame: usize,
    },
}

impl Op {
    /// What the status line names while the op runs.
    fn label(&self) -> String {
        match self {
            Op::Overview { .. } => "work counters".into(),
            Op::Records { .. } => "records page".into(),
            Op::Regions { .. } => "regions".into(),
            Op::Fields { .. } => "fields".into(),
            Op::Histogram { .. } => "histogram".into(),
            Op::Values { field, .. } => format!("values of {}", draw::safe(field)),
            Op::Evidence { excerpt: true, .. } => "excerpt (a read of the source)".into(),
            Op::Evidence { .. } => "evidence".into(),
            Op::Select { query, .. } => format!("search {}", draw::excerpt(query, 40)),
            Op::Child { title, .. } => draw::excerpt(title, 48),
            Op::Save { .. } => "saving a receipt".into(),
        }
    }
}

struct Pending {
    op: Op,
    job: String,
    started: Instant,
    cancel: bool,
}

#[derive(Debug, Clone)]
enum Action {
    Op(Op),
    Go(usize),
}

#[derive(Debug, Clone)]
struct Item {
    label: String,
    action: Action,
}

enum Mode {
    Browse,
    Search,
    Menu {
        title: String,
        items: Vec<Item>,
        selected: usize,
    },
    Evidence {
        record: String,
        data: Load<Value>,
        excerpt: Load<Value>,
        scroll: u16,
    },
    Allocate(String),
    Help(u16),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Focus {
    Results,
    Regions,
    Fields,
}

#[derive(Clone, Copy)]
enum Divider {
    Fields,
    Search,
}

/// JSON lines of what the workbench did and how long it took.
struct Trace {
    out: io::BufWriter<std::fs::File>,
    started: Instant,
}

impl Trace {
    fn event(&mut self, mut event: Value) -> io::Result<()> {
        event["t_ms"] = json!(self.started.elapsed().as_secs_f64() * 1e3);
        writeln!(self.out, "{event}")?;
        self.out.flush()
    }
}

/// The work counters the service reports in an overview.
const WORK: [&str; 6] = [
    "archive_scans",
    "bytes_unpacked",
    "rows_scored",
    "rows_encoded",
    "genai_rows_submitted",
    "evidence_bytes_disclosed",
];

/// The workbench: see the module description.
pub struct Workbench {
    service: Service,
    snapshot: String,
    named: Vec<(String, String)>,
    origin: String,
    frames: Vec<Frame>,
    active: usize,
    queue: VecDeque<Op>,
    pending: Option<Pending>,
    mode: Mode,
    notice: String,
    sidebar: u16,
    search_rows: u16,
    panes: Panes,
    geometry: Option<(Rect, u16, u16)>,
    focus: Focus,
    region_index: usize,
    field_index: usize,
    drag: Option<Divider>,
    trace: Option<Trace>,
    last_call_ms: Option<f64>,
    last_layout_us: u64,
    /// The layout time of the frame being drawn, when its geometry was solved anew.
    solved_us: Option<u64>,
    last_draw_us: u64,
    activity: Option<Value>,
    /// The first error writing the trace.
    trace_error: Option<String>,
}

fn payload(json: &str) -> Value {
    serde_json::from_str(json).unwrap_or_else(|_| json!({}))
}

fn pair(value: &Value) -> (u64, u64) {
    (
        value["distinct_originals"].as_u64().unwrap_or(0),
        value["occurrences"].as_u64().unwrap_or(0),
    )
}

fn fault(payload: &Value) -> String {
    let code = payload["error"]["code"].as_str().unwrap_or("error");
    let message = payload["error"]["message"].as_str().unwrap_or("");
    draw::excerpt(&format!("{code}: {message}"), 600)
}

/// Seconds since 1970 of `YYYY-MM-DDTHH:MM:SSZ`, the form histogram buckets state.
fn utc_seconds(text: &str) -> Option<i64> {
    let b = text.as_bytes();
    if b.len() != 20 || b[4] != b'-' || b[7] != b'-' || b[10] != b'T' || b[19] != b'Z' {
        return None;
    }
    let number = |from: usize, to: usize| text.get(from..to)?.parse::<i64>().ok();
    let (year, month, day) = (number(0, 4)?, number(5, 7)?, number(8, 10)?);
    let (hour, minute, second) = (number(11, 13)?, number(14, 16)?, number(17, 19)?);
    let (y, m) = if month <= 2 {
        (year - 1, month + 9)
    } else {
        (year, month - 3)
    };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * m + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some(days * 86_400 + hour * 3_600 + minute * 60 + second)
}

/// `YYYY-MM-DDTHH:MM:SSZ` of seconds since 1970.
fn utc_text(seconds: i64) -> String {
    let days = seconds.div_euclid(86_400);
    let rest = seconds.rem_euclid(86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        rest / 3_600,
        rest % 3_600 / 60,
        rest % 60
    )
}

impl Workbench {
    /// A workbench at `start` over `service`, writing a trace to `trace` when given.
    pub fn new(service: Service, start: Start, trace: Option<&Path>) -> Result<Self, String> {
        let trace = match trace {
            Some(path) => Some(Trace {
                out: io::BufWriter::new(
                    std::fs::File::create(path)
                        .map_err(|error| format!("creating the trace failed: {error}"))?,
                ),
                started: Instant::now(),
            }),
            None => None,
        };
        let root_title = if start.origin == "recipe" {
            "Snapshot".to_string()
        } else {
            format!("Receipt {}", start.origin)
        };
        let mut frames = vec![Frame::new(start.snapshot.clone(), root_title, None)];
        let mut active = 0;
        for (label, handle) in &start.views {
            frames.push(Frame::new(handle.clone(), label.clone(), Some(0)));
            if start.active.as_deref() == Some(handle.as_str()) {
                active = frames.len() - 1;
            }
        }
        let mut named = start.named;
        named.truncate(8);
        let mut bench = Self {
            service,
            snapshot: start.snapshot,
            named,
            origin: start.origin,
            frames,
            active,
            queue: VecDeque::new(),
            pending: None,
            mode: Mode::Browse,
            notice: String::new(),
            sidebar: 34,
            search_rows: 3,
            panes: Panes::default(),
            geometry: None,
            focus: Focus::Results,
            region_index: 0,
            field_index: 0,
            drag: None,
            trace,
            last_call_ms: None,
            last_layout_us: 0,
            solved_us: None,
            last_draw_us: 0,
            activity: None,
            trace_error: None,
        };
        bench.queue.push_back(Op::Overview { frame: 0 });
        bench.load(active);
        Ok(bench)
    }

    /// Writes a trace event. A trace that cannot be written ends there: the notice
    /// says so, and the session ends with that error.
    fn trace(&mut self, event: Value) {
        if let Some(trace) = &mut self.trace
            && let Err(error) = trace.event(event)
        {
            self.trace = None;
            let error = format!("the trace could not be written: {error}");
            self.notice = error.clone();
            self.trace_error.get_or_insert(error);
        }
    }

    fn frame(&self) -> &Frame {
        &self.frames[self.active]
    }

    fn named_map(&self) -> Value {
        Value::Object(
            self.named
                .iter()
                .map(|(name, handle)| (name.clone(), json!(handle)))
                .collect::<Map<String, Value>>(),
        )
    }

    /// Queues what a view shows the first time it is opened.
    fn load(&mut self, frame: usize) {
        let f = &mut self.frames[frame];
        if matches!(f.page, Load::Idle | Load::Cancelled) {
            f.page = Load::Loading;
            self.queue.push_back(Op::Records {
                frame,
                index: 0,
                cursor: None,
                select_last: false,
            });
        }
        let f = &mut self.frames[frame];
        if f.members.is_none() && f.reduction.is_none() && frame != 0 {
            self.queue.push_back(Op::Overview { frame });
        }
        let has_sets = !self.named.is_empty();
        let f = &mut self.frames[frame];
        if has_sets && matches!(f.regions, Load::Idle | Load::Cancelled) {
            f.regions = Load::Loading;
            self.queue.push_back(Op::Regions { frame });
        }
        let f = &mut self.frames[frame];
        if matches!(f.fields, Load::Idle | Load::Cancelled) {
            f.fields = Load::Loading;
            self.queue.push_back(Op::Fields { frame });
        }
        let f = &mut self.frames[frame];
        if matches!(f.histogram, Load::Idle | Load::Cancelled) {
            f.histogram = Load::Loading;
            self.queue.push_back(Op::Histogram { frame });
        }
    }

    /// The service the workbench calls: another client of the same session may call
    /// it too, as an MCP client beside the terminal would.
    pub fn service(&self) -> &Service {
        &self.service
    }

    /// The handle of the view shown.
    pub fn active_handle(&self) -> &str {
        &self.frame().handle
    }

    /// Whether a call is running or waiting to be made.
    pub fn busy(&self) -> bool {
        self.pending.is_some() || !self.queue.is_empty()
    }

    fn call(&mut self, tool: &str, args: &Value) -> (Value, bool) {
        let started = Instant::now();
        let reply = self.service.call(tool, args);
        let ms = started.elapsed().as_secs_f64() * 1e3;
        let value = payload(&reply.json);
        if tool != "triager_job" {
            self.last_call_ms = Some(ms);
        }
        self.trace(json!({
            "event": "call", "tool": tool,
            "view": args.get("view").or_else(|| args.get("mode")).cloned().unwrap_or(Value::Null),
            "action": args.get("action").cloned().unwrap_or(Value::Null),
            "ms": ms, "bytes": reply.json.len(),
            "status": value["status"].clone(), "error": reply.is_error,
        }));
        (value, reply.is_error)
    }

    fn request(&self, op: &Op) -> (&'static str, Value) {
        let handle = |frame: &usize| self.frames[*frame].handle.clone();
        match op {
            Op::Overview { frame } => (
                "triager_inspect",
                json!({"v": 1, "handle": handle(frame), "view": "overview"}),
            ),
            Op::Records { frame, cursor, .. } => {
                let mut projection = vec!["id", "locator", "occurrences", "membership"];
                if self.frames[*frame].kind == Kind::Final {
                    projection.push("allocation");
                }
                let mut args = json!({"v": 1, "handle": handle(frame), "view": "records",
                                      "limit": PAGE_ROWS, "projection": projection});
                if !self.named.is_empty() {
                    args["sets"] = self.named_map();
                }
                if let Some(cursor) = cursor {
                    args["cursor"] = json!(cursor);
                }
                ("triager_inspect", args)
            }
            Op::Regions { frame } => {
                let sets: Map<String, Value> = self
                    .named
                    .iter()
                    .take(REGION_SETS)
                    .map(|(name, handle)| (name.clone(), json!(handle)))
                    .collect();
                (
                    "triager_inspect",
                    json!({"v": 1, "handle": handle(frame), "view": "regions", "sets": sets}),
                )
            }
            Op::Fields { frame } => (
                "triager_inspect",
                json!({"v": 1, "handle": handle(frame), "view": "fields"}),
            ),
            Op::Histogram { frame } => (
                "triager_inspect",
                json!({"v": 1, "handle": handle(frame), "view": "histogram"}),
            ),
            Op::Values { frame, field } => (
                "triager_inspect",
                json!({"v": 1, "handle": handle(frame), "view": "fields", "field": field,
                       "limit": VALUE_ROWS}),
            ),
            Op::Evidence {
                frame,
                record,
                excerpt,
            } => {
                let mut args = json!({"v": 1, "handle": handle(frame), "view": "evidence",
                                      "record": record});
                if !self.named.is_empty() {
                    args["sets"] = self.named_map();
                }
                if *excerpt {
                    args["projection"] =
                        json!(["id", "locator", "occurrences", "membership", "excerpt"]);
                }
                ("triager_inspect", args)
            }
            Op::Select {
                from,
                syntax,
                query,
                ..
            } => (
                "triager_select",
                json!({"v": 1, "scope": handle(from),
                       "producer": {"kind": "filter", "syntax": syntax, "query": query}}),
            ),
            Op::Child { tool, args, .. } => (tool, args.clone()),
            Op::Save { frame } => {
                let mut args = json!({"v": 1, "handle": handle(frame)});
                if !self.named.is_empty() {
                    args["sets"] = self.named_map();
                }
                ("triager_save", args)
            }
        }
    }

    fn start(&mut self, op: Op) {
        let begun = Instant::now();
        let (tool, args) = self.request(&op);
        let (value, is_error) = self.call(tool, &args);
        if is_error {
            self.done(&op, begun, "error", &Value::Null);
            self.fail(&op, &value);
        } else if value["status"] == "running"
            && let Some(job) = value["job"].as_str()
        {
            self.pending = Some(Pending {
                op,
                job: job.to_string(),
                started: begun,
                cancel: false,
            });
        } else {
            self.done(&op, begun, "ready", &value);
            self.apply(op, &value);
        }
    }

    /// The trace event of an operation that ended: `ms` from its call to its result as
    /// the workbench saw it, with the polls of its job; `service_ms` and `queue_ms` as the
    /// service states them in the result's cost (the time of the work itself, apart from
    /// the polling).
    fn done(&mut self, op: &Op, begun: Instant, outcome: &str, result: &Value) {
        let ms = begun.elapsed().as_secs_f64() * 1e3;
        let cost = &result["cost"];
        self.trace(
            json!({"event": "op", "op": op.label(), "outcome": outcome, "ms": ms,
                          "service_ms": cost["elapsed_ms"], "queue_ms": cost["queue_ms"]}),
        );
    }

    /// How long the event loop waits before it polls the running job again: often at
    /// first, so a short job is seen soon after it ends, then at most every 100 ms.
    fn poll_interval(&self) -> Option<Duration> {
        let pending = self.pending.as_ref()?;
        let age = pending.started.elapsed();
        Some((age / 4).clamp(Duration::from_millis(5), Duration::from_millis(100)))
    }

    /// Makes progress: polls the running job, or makes the next queued call. Returns
    /// whether anything that is drawn changed.
    pub fn tick(&mut self) -> bool {
        if let Some(pending) = &self.pending {
            let job = pending.job.clone();
            let (status, is_error) = self.call(
                "triager_job",
                &json!({"v": 1, "job": job, "action": "status"}),
            );
            if is_error {
                let pending = self.pending.take().expect("polled");
                self.done(&pending.op, pending.started, "error", &Value::Null);
                self.fail(&pending.op, &status);
                return true;
            }
            return match status["status"].as_str() {
                Some("completed") => {
                    let pending = self.pending.take().expect("polled");
                    let result = status["result"].clone();
                    self.done(&pending.op, pending.started, "ready", &result);
                    self.apply(pending.op, &result);
                    true
                }
                Some("cancelled") => {
                    let pending = self.pending.take().expect("polled");
                    self.done(&pending.op, pending.started, "cancelled", &Value::Null);
                    self.cancelled(&pending.op);
                    true
                }
                Some("failed" | "interrupted") => {
                    let pending = self.pending.take().expect("polled");
                    self.done(&pending.op, pending.started, "failed", &Value::Null);
                    self.fail(&pending.op, &json!({"error": status["error"].clone()}));
                    true
                }
                // The clock of the running job is drawn.
                _ => true,
            };
        }
        match self.queue.pop_front() {
            Some(op) => {
                self.start(op);
                true
            }
            None => false,
        }
    }

    /// Runs every queued call and job to its end: for `--json` and for tests.
    pub fn settle(&mut self) {
        while self.busy() {
            if self.pending.is_some() {
                std::thread::sleep(Duration::from_millis(5));
            }
            self.tick();
        }
    }

    /// Asks the service to cancel the running job.
    fn cancel(&mut self) -> bool {
        let Some(pending) = &mut self.pending else {
            return false;
        };
        if pending.cancel {
            return true;
        }
        pending.cancel = true;
        let job = pending.job.clone();
        let label = pending.op.label();
        let (value, is_error) = self.call(
            "triager_job",
            &json!({"v": 1, "job": job, "action": "cancel"}),
        );
        self.notice = if is_error {
            fault(&value)
        } else {
            format!("Cancelling {label}…")
        };
        true
    }

    /// A cancelled job commits nothing, and what is queued behind it still runs: a
    /// search's follow-up is queued only when the search succeeded. A cancelled pane
    /// is idle again, so revisiting the view or `r` loads it.
    fn cancelled(&mut self, op: &Op) {
        self.unload(op, "cancelled");
        self.idle(op);
        self.notice = format!(
            "Cancelled {}; nothing was changed. r loads it again.",
            op.label()
        );
    }

    /// The pane of a pane op, back to idle.
    fn idle(&mut self, op: &Op) {
        match op {
            Op::Records { frame, .. } if self.frames[*frame].page.ready().is_none() => {
                self.frames[*frame].page = Load::Cancelled
            }
            Op::Regions { frame } => self.frames[*frame].regions = Load::Cancelled,
            Op::Fields { frame } => self.frames[*frame].fields = Load::Cancelled,
            Op::Histogram { frame } => self.frames[*frame].histogram = Load::Cancelled,
            _ => {}
        }
    }

    /// Loads again every pane of the active view that was cancelled or refused.
    fn reload(&mut self) {
        let frame = &mut self.frames[self.active];
        if matches!(frame.page, Load::Unavailable(_)) {
            frame.page = Load::Idle;
        }
        if matches!(frame.regions, Load::Unavailable(_)) {
            frame.regions = Load::Idle;
        }
        if matches!(frame.fields, Load::Unavailable(_)) {
            frame.fields = Load::Idle;
        }
        if matches!(frame.histogram, Load::Unavailable(_)) {
            frame.histogram = Load::Idle;
        }
        self.notice.clear();
        self.load(self.active);
    }

    fn unload(&mut self, op: &Op, why: &str) {
        let why = why.to_string();
        match op {
            Op::Records { frame, .. } if self.frames[*frame].page.ready().is_none() => {
                self.frames[*frame].page = Load::Unavailable(why)
            }
            Op::Regions { frame } => self.frames[*frame].regions = Load::Unavailable(why),
            Op::Fields { frame } => self.frames[*frame].fields = Load::Unavailable(why),
            Op::Histogram { frame } => self.frames[*frame].histogram = Load::Unavailable(why),
            Op::Evidence {
                excerpt, record, ..
            } => {
                if let Mode::Evidence {
                    record: shown,
                    data,
                    excerpt: shown_excerpt,
                    ..
                } = &mut self.mode
                    && shown == record
                {
                    if *excerpt {
                        *shown_excerpt = Load::Unavailable(why);
                    } else {
                        *data = Load::Unavailable(why);
                    }
                }
            }
            _ => {}
        }
    }

    fn fail(&mut self, op: &Op, payload: &Value) {
        let message = fault(payload);
        match op {
            Op::Fields { .. } | Op::Histogram { .. } | Op::Regions { .. } | Op::Evidence { .. } => {
                self.unload(op, &message)
            }
            Op::Records { .. } => {
                self.unload(op, &message);
                self.notice = message;
            }
            Op::Overview { .. } => {}
            // A failed search, drill, values page or save commits nothing and has no
            // follow-up queued; the panes queued behind it still load.
            _ => self.notice = message,
        }
    }

    fn apply(&mut self, op: Op, result: &Value) {
        match op {
            Op::Overview { frame } => {
                self.frames[frame].members = Some(pair(&result["members"]));
                if result["activity"].is_object() {
                    self.activity = Some(result["activity"].clone());
                }
            }
            Op::Records {
                frame,
                index,
                cursor,
                select_last,
            } => {
                let rows: Vec<Row> = result["rows"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|row| self.row(row))
                    .collect();
                let mut cursors = match &self.frames[frame].page {
                    Load::Ready(page) => page.cursors.clone(),
                    _ => Vec::new(),
                };
                cursors.truncate(index);
                cursors.push(cursor);
                let last = rows.len().saturating_sub(1);
                let f = &mut self.frames[frame];
                f.selected = if select_last { last } else { 0 };
                f.page = Load::Ready(Page {
                    rows,
                    index,
                    cursors,
                    next: result["page"]["next_cursor"].as_str().map(str::to_string),
                    total: pair(&result["page"]["total"]),
                });
            }
            Op::Regions { frame } => {
                let parse = |rows: &Value| -> Vec<RegionRow> {
                    rows.as_array()
                        .into_iter()
                        .flatten()
                        .map(|row| {
                            let (originals, occurrences) = pair(row);
                            RegionRow {
                                states: self
                                    .named
                                    .iter()
                                    .take(REGION_SETS)
                                    .enumerate()
                                    .filter_map(|(index, (name, _))| {
                                        Some((index, row["membership"][name].as_str()?.to_string()))
                                    })
                                    .collect(),
                                originals,
                                occurrences,
                            }
                        })
                        .collect()
                };
                let known = parse(&result["regions"]);
                let unknown = parse(&result["unknown_regions"]);
                self.frames[frame].regions = Load::Ready((known, unknown));
            }
            Op::Fields { frame } => {
                self.frames[frame].fields = match result["status"].as_str() {
                    Some("ready") => Load::Ready(
                        result["fields"]
                            .as_array()
                            .into_iter()
                            .flatten()
                            .map(|field| FieldRow {
                                name: field["field"].as_str().unwrap_or("").to_string(),
                                distinct: field["distinct_values"]["count"].as_u64().unwrap_or(0),
                                exact: field["distinct_values"]["exact"] == true,
                                with_value: pair(&field["with_value"]),
                                missing: pair(&field["missing"]),
                            })
                            .collect(),
                    ),
                    Some("projection_missing") => Load::Unavailable(
                        "No field is projected: open the source with `projections`.".into(),
                    ),
                    other => Load::Unavailable(format!("fields: {}", other.unwrap_or("?"))),
                };
            }
            Op::Histogram { frame } => {
                self.frames[frame].histogram = match result["status"].as_str() {
                    Some("ready") => {
                        let time = &result["time"];
                        Load::Ready(Histogram {
                            bucket_seconds: time["bucket_seconds"].as_i64().unwrap_or(3_600),
                            buckets: time["buckets"]
                                .as_array()
                                .into_iter()
                                .flatten()
                                .filter_map(|bucket| {
                                    let start = bucket["start"].as_str()?.to_string();
                                    Some(Bucket {
                                        start_seconds: utc_seconds(&start)?,
                                        start,
                                        originals: bucket["distinct_originals"]
                                            .as_u64()
                                            .unwrap_or(0),
                                        occurrences: bucket["occurrences"].as_u64().unwrap_or(0),
                                    })
                                })
                                .collect(),
                            with_time: pair(&time["with_time"]),
                            missing: pair(&time["missing_time"]),
                            invalid: pair(&time["invalid_time"]),
                            timezone: time["timezone"].as_str().unwrap_or("UTC").to_string(),
                        })
                    }
                    Some("projection_missing") => Load::Unavailable(
                        "No timestamp is projected: open the source with a `time` projection."
                            .into(),
                    ),
                    Some("time_unavailable") => Load::Unavailable(
                        "No occurrence has a valid timestamp; by archive entry instead (not \
                         time): press Tab to the fields and open `source`."
                            .into(),
                    ),
                    other => Load::Unavailable(format!("histogram: {}", other.unwrap_or("?"))),
                };
            }
            Op::Values { frame, field } => {
                let mut items: Vec<Item> = result["field"]["values"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|value| {
                        let text = value["value"].as_str()?.to_string();
                        let (originals, occurrences) = pair(value);
                        Some(Item {
                            label: format!(
                                "{:<24} {occurrences:>9} occ. {originals:>8} orig.",
                                draw::excerpt(&text, 24)
                            ),
                            action: Action::Op(self.refine(
                                frame,
                                format!("{field} = {text}"),
                                json!({"kind": "field", "field": field, "value": text}),
                            )),
                        })
                    })
                    .collect();
                let (originals, occurrences) = pair(&result["field"]["missing"]);
                if originals > 0 {
                    items.push(Item {
                        label: format!(
                            "{:<24} {occurrences:>9} occ. {originals:>8} orig.",
                            "(missing)"
                        ),
                        action: Action::Op(self.refine(
                            frame,
                            format!("{field} missing"),
                            json!({"kind": "field", "field": field, "missing": true}),
                        )),
                    });
                }
                let total = result["page"]["values_total"].as_u64().unwrap_or(0);
                let ranking = if result["field"]["ranking_exact"] == false {
                    ", ranked among retained values"
                } else {
                    ""
                };
                // A menu opens only where the analyst asked for it.
                if !self.waiting_on(frame) {
                    self.notice = format!(
                        "The values of {} are ready; Enter on the field opens them.",
                        draw::safe(&field)
                    );
                    return;
                }
                self.mode = Mode::Menu {
                    title: format!(
                        "{} · {} of {total} values by occurrences{ranking}",
                        draw::safe(&field),
                        items.len()
                    ),
                    items,
                    selected: 0,
                };
            }
            Op::Evidence {
                record, excerpt, ..
            } => {
                if let Mode::Evidence {
                    record: shown,
                    data,
                    excerpt: shown_excerpt,
                    ..
                } = &mut self.mode
                    && *shown == record
                {
                    if excerpt {
                        *shown_excerpt = Load::Ready(result.clone());
                    } else {
                        *data = Load::Ready(result.clone());
                    }
                }
            }
            Op::Select { from, title, .. } => {
                let Some(set) = result["set"].as_str() else {
                    self.notice = "the search returned no set".into();
                    return;
                };
                let parent = &self.frames[from];
                let args = if parent.kind == Kind::Snapshot {
                    json!({"v": 1, "mode": "view", "snapshot": self.snapshot,
                           "sets": {"s": set}, "expr": "s"})
                } else {
                    json!({"v": 1, "mode": "view", "snapshot": self.snapshot,
                           "scope": parent.handle, "sets": {"p": parent.handle, "s": set},
                           "expr": {"intersection": ["p", "s"]}})
                };
                self.queue.push_front(Op::Child {
                    from,
                    title,
                    tool: "triager_compose",
                    args,
                    key: "view",
                });
            }
            Op::Child {
                from, title, key, ..
            } => {
                let Some(handle) = result[key].as_str() else {
                    self.notice = format!("the result names no `{key}`");
                    return;
                };
                // The analyst may have gone on while the job ran: then the view is kept
                // and announced, and nothing switches under them.
                let follow = self.waiting_on(from);
                if let Some(known) = self.frames.iter().position(|f| f.handle == handle) {
                    if follow {
                        self.go(known);
                    } else {
                        self.announce(known);
                    }
                    return;
                }
                if self.frames.len() >= MAX_FRAMES {
                    self.notice = format!("This session keeps at most {MAX_FRAMES} views.");
                    return;
                }
                let mut frame = Frame::new(handle.to_string(), title, Some(from));
                let counts = &result["counts"];
                if counts.is_object() {
                    let occurrences = &result["occurrence_counts"];
                    frame.reduction = Some(Reduction {
                        originals: (
                            counts["in"].as_u64().unwrap_or(0),
                            counts["parent"].as_u64().unwrap_or(0),
                        ),
                        occurrences: (
                            occurrences["in"].as_u64().unwrap_or(0),
                            occurrences["parent"].as_u64().unwrap_or(0),
                        ),
                        unknown: counts["unknown"].as_u64().unwrap_or(0),
                    });
                    frame.members = Some((
                        counts["in"].as_u64().unwrap_or(0),
                        occurrences["in"].as_u64().unwrap_or(0),
                    ));
                }
                frame.matching = result["matching_occurrences"]["in"].as_u64();
                self.frames.push(frame);
                let index = self.frames.len() - 1;
                if follow {
                    self.go(index);
                } else {
                    self.announce(index);
                }
            }
            Op::Save { .. } => {
                let receipt = result["receipt"].as_str().unwrap_or("?");
                self.notice = format!(
                    "Saved receipt {receipt} ({} originals, {} occurrences retained); reopen \
                     it with l2d tui --receipt {receipt}",
                    result["retained"]["distinct_originals"], result["retained"]["occurrences"]
                );
                self.trace(json!({"event": "saved", "receipt": receipt}));
            }
        }
    }

    fn row(&self, row: &Value) -> Row {
        let locator = &row["locator"];
        Row {
            id: row["id"].as_str().unwrap_or("").to_string(),
            locator: match (locator["entry"].as_str(), locator["line"].as_u64()) {
                (Some(entry), Some(line)) => format!("{}:{line}", draw::safe(entry)),
                (Some(entry), None) => draw::safe(entry),
                _ => String::new(),
            },
            occurrences: row["occurrences"].as_u64().unwrap_or(0),
            states: self
                .named
                .iter()
                .map(|(name, _)| match row["membership"][name].as_str() {
                    Some("in") => '+',
                    Some("out") => '-',
                    Some(_) => '?',
                    None => ' ',
                })
                .collect(),
            allocation: row["allocation"]["pool"].as_str().map(|pool| {
                let outside = if row["allocation"]["outside_candidate"] == true {
                    "*"
                } else {
                    ""
                };
                format!(
                    "{} #{}{outside}",
                    draw::excerpt(pool, 10),
                    row["allocation"]["slot"]
                )
            }),
        }
    }

    fn refine(&self, frame: usize, title: String, action: Value) -> Op {
        Op::Child {
            from: frame,
            title,
            tool: "triager_refine",
            args: json!({"v": 1, "set": self.frames[frame].handle, "action": action}),
            key: "view",
        }
    }

    /// Whether the analyst is still where a result of `from` was asked for: on that
    /// view, browsing (not typing a search, a budget or reading evidence or a menu).
    fn waiting_on(&self, from: usize) -> bool {
        self.active == from && matches!(self.mode, Mode::Browse)
    }

    /// Announces a view that ended after the analyst moved on.
    fn announce(&mut self, frame: usize) {
        self.notice = format!(
            "View ready: {} (#{frame}); v lists it.",
            draw::excerpt(&self.frames[frame].title, 48)
        );
        let handle = self.frames[frame].handle.clone();
        self.trace(json!({"event": "ready", "handle": handle}));
    }

    fn go(&mut self, frame: usize) {
        self.active = frame;
        self.mode = Mode::Browse;
        self.load(frame);
        let handle = self.frames[frame].handle.clone();
        self.trace(json!({"event": "view", "handle": handle}));
    }

    /// The region of the named sets in `states` as a child view of the active one.
    fn region(&mut self, states: &[(usize, &str)]) -> Result<(), String> {
        if states.iter().any(|(index, _)| *index >= self.named.len()) {
            return Err("This hunt names fewer sets; see the regions table (Tab).".into());
        }
        let from = self.active;
        let parent = &self.frames[from];
        if parent.kind == Kind::Final {
            return Err("Already allocated: press b to return to the candidate.".into());
        }
        let mut sets = Map::new();
        let selectors: Vec<Value> = states
            .iter()
            .map(|(index, state)| {
                let (name, handle) = &self.named[*index];
                sets.insert(name.clone(), json!(handle));
                json!({"state": {"set": name, "is": state}})
            })
            .collect();
        let expr = if selectors.len() == 1 {
            selectors.into_iter().next().expect("one")
        } else {
            json!({"intersection": selectors})
        };
        let title = states
            .iter()
            .map(|(index, state)| {
                let word = match *state {
                    "in" => "in",
                    "out" => "out",
                    _ => "not evaluated",
                };
                format!("{} {word}", self.named[*index].0)
            })
            .collect::<Vec<_>>()
            .join(" · ");
        let mut args = json!({"v": 1, "mode": "view", "snapshot": self.snapshot,
                              "sets": sets, "expr": expr});
        if parent.kind != Kind::Snapshot {
            args["scope"] = json!(parent.handle);
        }
        self.queue.push_back(Op::Child {
            from,
            title,
            tool: "triager_compose",
            args,
            key: "view",
        });
        Ok(())
    }

    fn allocate(&mut self, input: &str) -> Result<(), String> {
        let numbers = input
            .split_whitespace()
            .map(str::parse::<u64>)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| "Enter slot counts as whole numbers.")?;
        let (general, unfamiliar, audit, seed) = match numbers.as_slice() {
            [g, u, a] => (*g, *u, *a, 5),
            [g, u, a, s] => (*g, *u, *a, *s),
            _ => {
                return Err(
                    "Enter general, unfamiliar and audit slots and an optional seed, \
                            for example 18 4 2 5."
                        .into(),
                );
            }
        };
        let frame = self.frame();
        if frame.kind != Kind::View {
            return Err("Reduce to a view first: allocation fills slots from a view.".into());
        }
        let d = frame.handle.clone();
        let total = general + unfamiliar + audit;
        let c = self
            .named
            .iter()
            .find(|(name, _)| name == "c")
            .map(|(_, handle)| handle.clone());
        let (sets, expr, pools) = match c {
            Some(c) => {
                let mut pools = Vec::new();
                for (name, limit, from, order) in [
                    (
                        "general",
                        general,
                        json!({"intersection": ["d", "c"]}),
                        json!(["score_desc", "record_id_ascending"]),
                    ),
                    (
                        "unfamiliar",
                        unfamiliar,
                        json!({"intersection": ["d", {"state": {"set": "c", "is": "not_evaluated"}}]}),
                        json!(["record_id_ascending"]),
                    ),
                    (
                        "audit",
                        audit,
                        json!({"difference": ["d", "c"]}),
                        json!(["seeded_hash", "record_id_ascending"]),
                    ),
                ] {
                    if limit == 0 {
                        continue;
                    }
                    let mut pool = json!({"name": name, "limit": limit, "from": from,
                                          "order": order});
                    if name == "audit" {
                        pool["seed"] = json!(seed);
                    }
                    pools.push(pool);
                }
                (
                    json!({"d": d, "c": c}),
                    json!({"intersection": ["d", "c"]}),
                    pools,
                )
            }
            None if unfamiliar == 0 && audit == 0 => (
                json!({"d": d}),
                json!("d"),
                vec![json!({"name": "general", "limit": general, "from": "d",
                            "order": ["score_desc", "record_id_ascending"]})],
            ),
            None => {
                return Err("Unfamiliar and audit slots need a set named c.".into());
            }
        };
        let args = json!({"v": 1, "mode": "allocate", "snapshot": self.snapshot, "sets": sets,
                          "expr": expr, "budget": {"total": total,
                          "duplicate_rule": "first_pool_wins", "shortfall": "leave_unfilled",
                          "pools": pools}});
        self.queue.push_back(Op::Child {
            from: self.active,
            title: format!("Analyst work set {general}+{unfamiliar}+{audit}"),
            tool: "triager_compose",
            args,
            key: "set",
        });
        self.mode = Mode::Browse;
        Ok(())
    }

    /// Handles a key; returns whether the workbench ends.
    pub fn press(&mut self, key: KeyEvent) -> bool {
        if !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
            return false;
        }
        self.trace(json!({"event": "key", "code": format!("{:?}", key.code)}));
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            return true;
        }
        match self.key(key) {
            Ok(quit) => quit,
            Err(error) => {
                self.notice = error;
                false
            }
        }
    }

    fn key(&mut self, key: KeyEvent) -> Result<bool, String> {
        let code = key.code;
        if code == KeyCode::Esc && self.cancel() {
            return Ok(false);
        }
        if !matches!(self.mode, Mode::Browse) && code == KeyCode::Esc {
            self.mode = Mode::Browse;
            return Ok(false);
        }
        let active = self.active;
        match &mut self.mode {
            Mode::Search => {
                let frame = &mut self.frames[active];
                match code {
                    KeyCode::Tab => frame.syntax = (frame.syntax + 1) % SYNTAXES.len(),
                    KeyCode::Enter if frame.draft.trim().is_empty() => {
                        return Err("Enter a search.".into());
                    }
                    KeyCode::Enter => {
                        let query = frame.draft.clone();
                        let syntax = SYNTAXES[frame.syntax];
                        self.queue.push_back(Op::Select {
                            from: active,
                            title: format!("search {syntax} {query}"),
                            syntax,
                            query,
                        });
                        self.mode = Mode::Browse;
                    }
                    _ => crate::ui::edit(&mut frame.draft, key),
                }
            }
            Mode::Menu {
                items, selected, ..
            } => match code {
                KeyCode::Enter if !items.is_empty() => {
                    let action = items[*selected].action.clone();
                    self.mode = Mode::Browse;
                    match action {
                        Action::Op(op) => self.queue.push_back(op),
                        Action::Go(frame) => self.go(frame),
                    }
                }
                KeyCode::Char('q' | 'b') => self.mode = Mode::Browse,
                _ => {
                    if let Some(by) = crate::ui::motion(code) {
                        *selected = selected
                            .saturating_add_signed(by)
                            .min(items.len().saturating_sub(1));
                    }
                }
            },
            Mode::Evidence {
                record,
                excerpt,
                scroll,
                ..
            } => match code {
                KeyCode::Char('x') if matches!(excerpt, Load::Idle | Load::Unavailable(_)) => {
                    *excerpt = Load::Loading;
                    let record = record.clone();
                    self.queue.push_back(Op::Evidence {
                        frame: active,
                        record,
                        excerpt: true,
                    });
                }
                KeyCode::Enter | KeyCode::Char('b' | 'q') => self.mode = Mode::Browse,
                _ => {
                    if let Some(by) = crate::ui::motion(code) {
                        *scroll = usize::from(*scroll)
                            .saturating_add_signed(by)
                            .min(usize::from(u16::MAX)) as u16;
                    }
                }
            },
            Mode::Allocate(input) => match code {
                KeyCode::Enter => {
                    let input = input.clone();
                    self.allocate(&input)?;
                }
                _ => crate::ui::edit(input, key),
            },
            Mode::Help(scroll) => {
                if let Some(by) = crate::ui::motion(code) {
                    *scroll = usize::from(*scroll)
                        .saturating_add_signed(by)
                        .min(usize::from(u16::MAX)) as u16;
                } else {
                    self.mode = Mode::Browse;
                }
            }
            Mode::Browse => return self.browse(code),
        }
        Ok(false)
    }

    fn browse(&mut self, code: KeyCode) -> Result<bool, String> {
        let region: Option<&[(usize, &str)]> = match code {
            KeyCode::Char('1') => Some(&[(0, "in"), (1, "out")]),
            KeyCode::Char('2') => Some(&[(0, "in"), (1, "in")]),
            KeyCode::Char('3') => Some(&[(0, "out"), (1, "in")]),
            KeyCode::Char('4') => Some(&[(0, "out"), (1, "out")]),
            KeyCode::Char('i') => Some(&[(2, "in")]),
            KeyCode::Char('o') => Some(&[(2, "out")]),
            KeyCode::Char('u') => Some(&[(2, "not_evaluated")]),
            _ => None,
        };
        if let Some(states) = region {
            self.region(states)?;
            return Ok(false);
        }
        let final_view = self.frame().kind == Kind::Final;
        match code {
            KeyCode::Char('q') => return Ok(true),
            KeyCode::Char('?') => self.mode = Mode::Help(0),
            KeyCode::Char('/') if final_view => {
                return Err("Already allocated: press b to return to the candidate.".into());
            }
            KeyCode::Char('/') => {
                self.mode = Mode::Search;
                self.notice.clear();
            }
            KeyCode::Char('v') => {
                let items = self
                    .frames
                    .iter()
                    .enumerate()
                    .map(|(index, frame)| Item {
                        label: format!(
                            "{index:>3} {}{}",
                            draw::excerpt(&frame.title, 50),
                            frame
                                .parent
                                .map_or(String::new(), |parent| format!("  (from #{parent})"))
                        ),
                        action: Action::Go(index),
                    })
                    .collect();
                self.mode = Mode::Menu {
                    title: "Views of this session".into(),
                    items,
                    selected: self.active,
                };
            }
            KeyCode::Char('t') => self.time_menu()?,
            KeyCode::Char('r') => self.reload(),
            KeyCode::Char('w') => {
                // The status line shows the counters once the notice is gone.
                self.notice.clear();
                self.queue.push_back(Op::Overview { frame: 0 });
            }
            KeyCode::Char('f') if final_view => {
                return Err("Already allocated: press b to return to the candidate.".into());
            }
            KeyCode::Char('f') => {
                self.mode = Mode::Allocate("2 1 1 5".into());
                self.notice.clear();
            }
            KeyCode::Char('s') => self.queue.push_back(Op::Save { frame: self.active }),
            KeyCode::Char('b') | KeyCode::Backspace | KeyCode::Esc => match self.frame().parent {
                Some(parent) => {
                    self.go(parent);
                    self.notice.clear();
                }
                None => self.notice = "At the investigation root.".into(),
            },
            KeyCode::Tab if self.panes.fields.width > 0 => {
                self.focus = match self.focus {
                    Focus::Results => Focus::Regions,
                    Focus::Regions => Focus::Fields,
                    Focus::Fields => Focus::Results,
                };
            }
            KeyCode::Enter => match self.focus {
                Focus::Results => self.open_evidence()?,
                Focus::Regions => {
                    let rows = self.region_rows();
                    let states: Vec<(usize, String)> = rows
                        .get(self.region_index)
                        .ok_or("No region is selected.")?
                        .states
                        .clone();
                    let states: Vec<(usize, &str)> =
                        states.iter().map(|(i, s)| (*i, s.as_str())).collect();
                    self.region(&states)?;
                }
                Focus::Fields => {
                    let field = self
                        .frame()
                        .fields
                        .ready()
                        .and_then(|fields| fields.get(self.field_index))
                        .map(|field| field.name.clone())
                        .ok_or("No field is selected.")?;
                    self.queue.push_back(Op::Values {
                        frame: self.active,
                        field,
                    });
                }
            },
            KeyCode::Char('[') => self.sidebar = self.sidebar.saturating_sub(2).max(20),
            KeyCode::Char(']') => self.sidebar = (self.sidebar + 2).min(68),
            KeyCode::Char('-') => self.search_rows = self.search_rows.saturating_sub(1).max(3),
            KeyCode::Char('+') => self.search_rows = (self.search_rows + 1).min(10),
            _ => {
                if let Some(by) = crate::ui::motion(code) {
                    match self.focus {
                        Focus::Results => self.move_rows(by),
                        Focus::Regions => {
                            let last = self.region_rows().len().saturating_sub(1);
                            self.region_index =
                                self.region_index.saturating_add_signed(by).min(last);
                        }
                        Focus::Fields => {
                            let last = self
                                .frame()
                                .fields
                                .ready()
                                .map_or(0, |fields| fields.len().saturating_sub(1));
                            self.field_index = self.field_index.saturating_add_signed(by).min(last);
                        }
                    }
                }
            }
        }
        Ok(false)
    }

    /// The regions of the active view: fully evaluated first, then with unknowns.
    fn region_rows(&self) -> Vec<RegionRow> {
        match self.frame().regions.ready() {
            Some((known, unknown)) => known.iter().chain(unknown).cloned().collect(),
            None => Vec::new(),
        }
    }

    fn time_menu(&mut self) -> Result<(), String> {
        let frame = self.active;
        let histogram = match &self.frames[frame].histogram {
            Load::Ready(histogram) => histogram.clone(),
            Load::Unavailable(why) => return Err(why.clone()),
            _ => return Err("The histogram is still loading.".into()),
        };
        let mut items: Vec<Item> = histogram
            .buckets
            .iter()
            .map(|bucket| {
                let end = utc_text(bucket.start_seconds + histogram.bucket_seconds);
                Item {
                    label: format!(
                        "{}  {:>9} occ. {:>8} orig.",
                        bucket.start, bucket.occurrences, bucket.originals
                    ),
                    action: Action::Op(self.refine(
                        frame,
                        format!("time {} +{}s", bucket.start, histogram.bucket_seconds),
                        json!({"kind": "time", "start": bucket.start, "end": end}),
                    )),
                }
            })
            .collect();
        let (originals, occurrences) = (
            histogram.missing.0 + histogram.invalid.0,
            histogram.missing.1 + histogram.invalid.1,
        );
        if occurrences > 0 {
            items.push(Item {
                label: format!("no valid time          {occurrences:>9} occ. {originals:>8} orig."),
                action: Action::Op(self.refine(
                    frame,
                    "no valid time".into(),
                    json!({"kind": "time", "no_time": true}),
                )),
            });
        }
        self.mode = Mode::Menu {
            title: format!(
                "Time buckets of {} s ({}), occurrences with their originals",
                histogram.bucket_seconds, histogram.timezone
            ),
            items,
            selected: 0,
        };
        Ok(())
    }

    fn open_evidence(&mut self) -> Result<(), String> {
        let frame = self.frame();
        let row = frame
            .page
            .ready()
            .and_then(|page| page.rows.get(frame.selected))
            .ok_or("No original is selected.")?;
        let record = row.id.clone();
        self.mode = Mode::Evidence {
            record: record.clone(),
            data: Load::Loading,
            excerpt: Load::Idle,
            scroll: 0,
        };
        self.queue.push_back(Op::Evidence {
            frame: self.active,
            record,
            excerpt: false,
        });
        Ok(())
    }

    fn move_rows(&mut self, by: isize) {
        let active = self.active;
        let Load::Ready(page) = &self.frames[active].page else {
            return;
        };
        let selected = self.frames[active].selected;
        let last = page.rows.len().saturating_sub(1);
        let wanted = selected as isize + by.clamp(-1_000_000, 1_000_000);
        if wanted > last as isize && by != isize::MAX && page.next.is_some() && selected == last {
            let cursor = page.next.clone();
            let index = page.index + 1;
            self.queue.push_back(Op::Records {
                frame: active,
                index,
                cursor,
                select_last: false,
            });
            return;
        }
        if wanted < 0 && by != isize::MIN && page.index > 0 && selected == 0 {
            let index = page.index - 1;
            let cursor = page.cursors.get(index).cloned().flatten();
            self.queue.push_back(Op::Records {
                frame: active,
                index,
                cursor,
                select_last: true,
            });
            return;
        }
        self.frames[active].selected = selected.saturating_add_signed(by).min(last);
    }

    fn mouse(&mut self, event: MouseEvent) {
        if !matches!(self.mode, Mode::Browse) {
            return;
        }
        let point = Position::new(event.column, event.row);
        let panes = self.panes;
        match event.kind {
            MouseEventKind::Down(MouseButton::Left)
                if panes.search.contains(point) && event.row + 1 == panes.search.bottom() =>
            {
                self.drag = Some(Divider::Search)
            }
            MouseEventKind::Down(MouseButton::Left)
                if panes.fields.contains(point) && event.column + 1 == panes.fields.right() =>
            {
                self.drag = Some(Divider::Fields)
            }
            MouseEventKind::Drag(MouseButton::Left) => match self.drag {
                Some(Divider::Fields) => {
                    self.sidebar = (event.column.saturating_sub(panes.fields.x) + 1).clamp(20, 68)
                }
                Some(Divider::Search) => {
                    self.search_rows = (event.row.saturating_sub(panes.search.y) + 1).clamp(3, 10)
                }
                None => {}
            },
            MouseEventKind::Up(_) => self.drag = None,
            MouseEventKind::ScrollDown => self.move_rows(3),
            MouseEventKind::ScrollUp => self.move_rows(-3),
            MouseEventKind::Down(MouseButton::Left) if panes.results.contains(point) => {
                self.focus = Focus::Results;
                let first = panes.results.y + 2;
                if event.row >= first && event.row + 1 < panes.results.bottom() {
                    let visible = usize::from(panes.results.height.saturating_sub(3)).max(1);
                    let frame = &mut self.frames[self.active];
                    let rows = frame.page.ready().map_or(0, |page| page.rows.len());
                    let index = frame.selected / visible * visible + usize::from(event.row - first);
                    if index < rows {
                        frame.selected = index;
                    }
                }
            }
            _ => {}
        }
    }

    /// What the workbench shows, as JSON: the active view, its counts, its first page
    /// and the regions of the named sets. Runs the calls it needs to its end.
    pub fn summary(&mut self) -> Value {
        self.settle();
        let frame = self.frame();
        let page = frame.page.ready();
        json!({
            "origin": self.origin,
            "snapshot": self.snapshot,
            "named": self.named_map(),
            "view": {"handle": frame.handle, "title": frame.title,
                     "kind": format!("{:?}", frame.kind).to_lowercase()},
            "members": frame.members.map(|(o, c)| json!({"distinct_originals": o, "occurrences": c})),
            "page": page.map(|page| json!({
                "total": {"distinct_originals": page.total.0, "occurrences": page.total.1},
                "rows": page.rows.iter().map(|row| json!({
                    "id": row.id, "occurrences": row.occurrences,
                    "membership": row.states.iter().collect::<String>(),
                    "allocation": row.allocation,
                })).collect::<Vec<_>>(),
                "more": page.next.is_some(),
            })),
            "regions": frame.regions.ready().map(|(known, unknown)| {
                known.iter().chain(unknown).map(|region| json!({
                    "membership": region.states.iter().map(|(i, s)| (self.named[*i].0.clone(), json!(s))).collect::<Map<String, Value>>(),
                    "distinct_originals": region.originals, "occurrences": region.occurrences,
                })).collect::<Vec<_>>()
            }),
            "activity": self.activity,
            "notice": self.notice,
        })
    }

    /// Draws the workbench into a frame of `width` × `height` cells and returns its
    /// text: for tests and for checking a layout without a terminal.
    pub fn render_text(&mut self, width: u16, height: u16) -> String {
        let mut terminal = Terminal::new(ratatui::backend::TestBackend::new(width, height))
            .expect("a test backend");
        terminal
            .draw(|frame| draw::screen(frame, self))
            .expect("drawing into a test backend");
        let buffer = terminal.backend().buffer();
        (0..height)
            .map(|y| {
                (0..width)
                    .map(|x| buffer[(x, y)].symbol())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// The trace event of one drawn frame.
    fn frame_drawn(&mut self, area: Rect, draw_us: u64) {
        self.last_draw_us = draw_us;
        let layout_us = self.solved_us.take();
        let (sidebar, search_rows) = (self.panes.fields.width, self.panes.search.height);
        self.trace(
            json!({"event": "frame", "width": area.width, "height": area.height,
                          "sidebar": sidebar, "search_rows": search_rows,
                          "layout_us": layout_us, "draw_us": draw_us}),
        );
    }
}

/// Runs the workbench in the terminal until the analyst quits. The caller's terminal
/// modes are restored on a normal end, an error and an unwind.
pub fn run(mut bench: Workbench) -> Result<(), String> {
    if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        return Err(
            "The workbench requires an interactive terminal on stdin and stdout (use \
                    ssh -t for a remote command)."
                .into(),
        );
    }
    let _guard = crate::ui::RawTerminal::capture().map_err(|e| e.to_string())?;
    enable_raw_mode().map_err(|e| e.to_string())?;
    let mut stdout = io::stdout();
    // Not every terminal starts the alternate screen blank; what the recipe printed
    // before must not show through cells the first frame leaves empty. A plain clear
    // asks the terminal nothing (ratatui's own clear reads the cursor position, which
    // a terminal at the far end of SSH may not answer).
    execute!(
        stdout,
        EnterAlternateScreen,
        EnableMouseCapture,
        crossterm::terminal::Clear(crossterm::terminal::ClearType::All)
    )
    .map_err(|e| e.to_string())?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout)).map_err(|e| e.to_string())?;
    loop {
        let started = Instant::now();
        let completed = terminal
            .draw(|frame| draw::screen(frame, &mut bench))
            .map_err(|e| e.to_string())?;
        bench.frame_drawn(completed.area, started.elapsed().as_micros() as u64);
        // Calls are made between frames, one at a time, so input stays responsive.
        let wait = if let Some(poll) = bench.poll_interval() {
            poll
        } else if bench.queue.is_empty() {
            Duration::from_secs(3_600)
        } else {
            Duration::ZERO
        };
        if event::poll(wait).map_err(|e| e.to_string())? {
            match event::read().map_err(|e| e.to_string())? {
                Event::Key(key) => {
                    if bench.press(key) {
                        break;
                    }
                }
                Event::Mouse(mouse) if mouse.kind != MouseEventKind::Moved => bench.mouse(mouse),
                Event::Resize(width, height) => {
                    bench.trace(json!({"event": "resize", "width": width, "height": height}));
                }
                _ => {}
            }
            continue;
        }
        bench.tick();
    }
    bench.trace(json!({"event": "exit"}));
    match bench.trace_error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utc_text_and_seconds_round_trip() {
        for text in [
            "1970-01-01T00:00:00Z",
            "2026-01-01T00:00:00Z",
            "2026-02-28T23:59:59Z",
            "2024-02-29T12:00:00Z",
            "2026-12-31T23:00:00Z",
        ] {
            let seconds = utc_seconds(text).unwrap();
            assert_eq!(utc_text(seconds), text);
        }
        assert_eq!(utc_seconds("2026-01-01T01:00:00Z"), Some(1_767_229_200));
        assert_eq!(utc_seconds("2026-01-01 01:00:00"), None);
    }

    #[test]
    fn ctrl_h_deletes_and_other_chords_are_not_text() {
        let mut text = "2 1 1".to_string();
        let key = |code, modifiers| KeyEvent::new(code, modifiers);
        crate::ui::edit(&mut text, key(KeyCode::Char('h'), KeyModifiers::CONTROL));
        crate::ui::edit(&mut text, key(KeyCode::Char('x'), KeyModifiers::CONTROL));
        crate::ui::edit(&mut text, key(KeyCode::Char('f'), KeyModifiers::ALT));
        assert_eq!(text, "2 1 ");
        crate::ui::edit(&mut text, key(KeyCode::Backspace, KeyModifiers::NONE));
        crate::ui::edit(&mut text, key(KeyCode::Char('h'), KeyModifiers::NONE));
        crate::ui::edit(&mut text, key(KeyCode::Char('H'), KeyModifiers::SHIFT));
        assert_eq!(text, "2 1hH");
    }

    #[test]
    fn a_recipe_start_names_its_sets_and_opens_at_the_last_view() {
        let bound = vec![
            ("S".to_string(), json!("snap-1")),
            ("A".to_string(), json!("set-a")),
            ("B".to_string(), json!("set-b")),
            ("D".to_string(), json!("view-d")),
            ("n".to_string(), json!(3)),
        ];
        let start = Start::from_bindings(&bound).unwrap();
        assert_eq!(start.snapshot, "snap-1");
        assert_eq!(
            start.named,
            [("a".into(), "set-a".into()), ("b".into(), "set-b".into())]
        );
        assert_eq!(start.views, [("D".to_string(), "view-d".to_string())]);
        assert_eq!(start.active.as_deref(), Some("view-d"));
        assert!(Start::from_bindings(&bound[1..2]).is_err());
    }
}
