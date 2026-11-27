//! Drawing the workbench: every pane reads what earlier calls returned and nothing
//! else, so drawing makes no call.

use std::time::Instant;

use ratatui::{
    Frame,
    layout::{Constraint, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Row as TableRow, Table, Wrap},
};
use serde_json::Value;

use super::{Focus, Kind, Load, Mode, SYNTAXES, WORK, Workbench};
use crate::ui::layout::Panes;

pub(super) use crate::ui::draw::{excerpt, safe};

/// What a pane whose job the analyst cancelled shows.
const CANCELLED: &str = "cancelled · r loads it";

/// The columns the results pane keeps beside the fields pane.
const MIN_RESULTS: u16 = 58;

const ACCENT: Color = Color::Cyan;
const MUTED: Color = Color::DarkGray;
const WARN: Color = Color::Yellow;

fn style(color: Color) -> Style {
    Style::default().fg(color)
}

fn block(title: String, focus: bool) -> Block<'static> {
    Block::default()
        .borders(Borders::ALL)
        .title(format!(" {title} "))
        .border_style(style(if focus { ACCENT } else { MUTED }))
}

/// `1,234,567`.
fn grouped(value: u64) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            out.push(',');
        }
        out.push(digit);
    }
    out
}

fn share(part: u64, whole: u64) -> String {
    if whole == 0 {
        "share undefined".into()
    } else {
        format!("{:.1}%", part as f64 * 100.0 / whole as f64)
    }
}

fn plural(count: u64, one: &str, many: &str) -> String {
    format!("{} {}", grouped(count), if count == 1 { one } else { many })
}

pub(super) fn screen(frame: &mut Frame, bench: &mut Workbench) {
    let area = frame.area();
    if area.width < 32 || area.height < 8 {
        bench.panes = Panes::default();
        bench.geometry = None;
        frame.render_widget(
            Paragraph::new("Triager · enlarge the terminal to 32×8 or press q to exit.")
                .wrap(Wrap { trim: false }),
            area,
        );
        return;
    }
    // The results keep room for a record handle, its counts and membership columns: a
    // fields pane wider than that leaves is drawn narrower, and the analyst's width
    // comes back when the terminal is wide again.
    let width = bench
        .sidebar
        .min(area.width.saturating_sub(MIN_RESULTS))
        .max(20);
    let geometry = (area, width, bench.search_rows);
    if bench.geometry != Some(geometry) {
        let started = Instant::now();
        let Ok(panes) = Panes::solve(area, width, bench.search_rows) else {
            frame.render_widget("Unable to lay out the panes. Press q to exit.", area);
            return;
        };
        bench.last_layout_us = started.elapsed().as_micros() as u64;
        bench.solved_us = Some(bench.last_layout_us);
        bench.panes = panes;
        bench.geometry = Some(geometry);
        if panes.fields.width == 0 && bench.focus != Focus::Results {
            bench.focus = Focus::Results;
        }
    }
    let p = bench.panes;
    header(frame, bench, p.header);
    search(frame, bench, p.search);
    if p.time.height > 0 {
        timeline(frame, bench, p.time);
    }
    if p.fields.width > 0 {
        sidebar(frame, bench, p.fields);
    }
    results(frame, bench, p.results);
    status(frame, bench, p.status);
    match &bench.mode {
        Mode::Menu {
            title,
            items,
            selected,
        } => {
            let lines: Vec<String> = items.iter().map(|item| item.label.clone()).collect();
            list_dialog(
                frame,
                area,
                title,
                &lines,
                *selected,
                "Enter opens · Esc closes",
            );
        }
        Mode::Evidence {
            record,
            data,
            excerpt,
            scroll,
        } => {
            let lines = evidence_lines(bench, record, data, excerpt);
            let dialog = dialog_area(area, 100, area.height.saturating_sub(2));
            frame.render_widget(Clear, dialog);
            frame.render_widget(
                Paragraph::new(lines)
                    .wrap(Wrap { trim: false })
                    .scroll((*scroll, 0))
                    .block(block(
                        format!("Evidence {} · x excerpt · Esc back", safe(record)),
                        true,
                    )),
                dialog,
            );
        }
        Mode::Allocate(input) => {
            let dialog = dialog_area(area, 72, 7);
            frame.render_widget(Clear, dialog);
            let c = if bench.named.iter().any(|(name, _)| name == "c") {
                "general from view ∩ c by score, unfamiliar from c not evaluated, audit from \
                 view − c by the policy's seeded key"
            } else {
                "general from the view by score (unfamiliar and audit need a set named c)"
            };
            frame.render_widget(
                Paragraph::new(vec![
                    Line::from(format!("Slots: general unfamiliar audit [seed]: {input}▏")),
                    Line::from(Span::styled(c, style(MUTED))),
                    Line::from(Span::styled(
                        "Enter allocates once · Esc cancels",
                        style(MUTED),
                    )),
                ])
                .wrap(Wrap { trim: false })
                .block(block("Allocate final work set".into(), true)),
                dialog,
            );
        }
        Mode::Help(scroll) => {
            let dialog = dialog_area(area, 84, area.height.saturating_sub(2));
            frame.render_widget(Clear, dialog);
            frame.render_widget(
                Paragraph::new(HELP.lines().map(Line::from).collect::<Vec<_>>())
                    .scroll((*scroll, 0))
                    .wrap(Wrap { trim: false })
                    .block(block("Keys · any other key closes".into(), true)),
                dialog,
            );
        }
        Mode::Browse | Mode::Search => {}
    }
}

const HELP: &str = "\
/            search inside this view (Tab: glob, literal, regex; Enter runs)
1 2 3 4      a in·b out, a and b, b in·a out, neither (the first two named sets)
i o u        c in, c out, c not evaluated (the third named set)
Tab          focus results, regions, fields; Enter drills the selected region or field
Enter        evidence of the selected original; x in the evidence reads its excerpt
t            time buckets of the histogram; Enter drills one
v            every view of this session; b, Backspace or Esc goes back to the parent
f            allocate the final work set once; s saves a receipt of the view
w            refresh the work counters (archive scans, rows scored, encoded, GenAI)
r            load again what was cancelled or refused in this view
[ ]          narrow or widen the fields pane; - + shrink or grow the search pane
Mouse        drag a pane edge, click a row, scroll the results
Esc          cancels a running job first; nothing it did is committed
q            quit (the terminal is restored)

Counts are distinct originals and occurrences; a share names its parent.
Drawing, resizing and paging make no new archive read, scoring or model call.";

fn breadcrumb(bench: &Workbench) -> String {
    let mut titles = Vec::new();
    let mut at = Some(bench.active);
    while let Some(index) = at {
        titles.push(excerpt(&bench.frames[index].title, 28));
        at = bench.frames[index].parent;
        if titles.len() > 6 {
            titles.push("…".into());
            break;
        }
    }
    titles.reverse();
    titles.join(" > ")
}

fn header(frame: &mut Frame, bench: &Workbench, area: Rect) {
    let f = bench.frame();
    let counts = match (f.reduction, f.members) {
        (Some(r), _) => {
            let mut text = format!(
                "{}/{} originals {} · {}/{} occurrences",
                grouped(r.originals.0),
                grouped(r.originals.1),
                share(r.originals.0, r.originals.1),
                grouped(r.occurrences.0),
                grouped(r.occurrences.1)
            );
            if r.unknown > 0 {
                text.push_str(&format!(" · {} unknown", grouped(r.unknown)));
            }
            if let Some(matching) = f.matching {
                text.push_str(&format!(" · {} matching occ.", grouped(matching)));
            }
            text
        }
        (None, Some((originals, occurrences))) => format!(
            "{} · {}",
            plural(originals, "original", "originals"),
            plural(occurrences, "occurrence", "occurrences")
        ),
        (None, None) => "counting…".into(),
    };
    // The counts first: a narrow terminal clips the path, not the denominators.
    let line = Line::from(vec![
        Span::styled("Triager ", style(ACCENT).add_modifier(Modifier::BOLD)),
        Span::styled(format!("{counts}  "), style(MUTED)),
        Span::raw(breadcrumb(bench)),
    ]);
    frame.render_widget(Paragraph::new(line), area);
}

fn search(frame: &mut Frame, bench: &Workbench, area: Rect) {
    let f = bench.frame();
    let editing = matches!(bench.mode, Mode::Search);
    let title = if editing {
        format!(
            "Search [{}] · Tab syntax · Enter runs · Esc leaves",
            SYNTAXES[f.syntax]
        )
    } else {
        format!("Search [{}]", SYNTAXES[f.syntax])
    };
    let text = if editing {
        Line::from(format!("{}▏", safe(&f.draft)))
    } else if f.draft.is_empty() {
        Line::from(Span::styled(
            "/ searches the originals of this view (a filter selection, then the overlap)",
            style(MUTED),
        ))
    } else {
        Line::from(safe(&f.draft))
    };
    frame.render_widget(
        Paragraph::new(text)
            .wrap(Wrap { trim: false })
            .block(block(title, editing)),
        area,
    );
}

fn timeline(frame: &mut Frame, bench: &Workbench, area: Rect) {
    let inner_width = usize::from(area.width.saturating_sub(2)).max(1);
    let (title, lines) = match &bench.frame().histogram {
        Load::Ready(h) if !h.buckets.is_empty() => {
            let first = h.buckets[0].start_seconds;
            let last = h.buckets[h.buckets.len() - 1].start_seconds;
            let slots = ((last - first) / h.bucket_seconds.max(1) + 1).max(1) as usize;
            let width = inner_width.min(slots).max(1);
            let mut columns = vec![0u64; width];
            for bucket in &h.buckets {
                let slot = ((bucket.start_seconds - first) / h.bucket_seconds.max(1)) as usize;
                columns[(slot * width / slots).min(width - 1)] += bucket.occurrences;
            }
            let peak = columns.iter().copied().max().unwrap_or(1).max(1);
            const BARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
            let bars: String = columns
                .iter()
                .map(|count| {
                    if *count == 0 {
                        ' '
                    } else {
                        BARS[((count * 7) / peak) as usize]
                    }
                })
                .collect();
            let end = super::utc_text(last + h.bucket_seconds);
            (
                format!(
                    "Time {} · {} s buckets · {} occ. with time · missing {} · invalid {}",
                    h.timezone,
                    h.bucket_seconds,
                    grouped(h.with_time.1),
                    grouped(h.missing.1),
                    grouped(h.invalid.1)
                ),
                vec![
                    Line::from(Span::styled(bars, style(ACCENT))),
                    Line::from(Span::styled(
                        format!("{} … {end} · t drills a bucket", h.buckets[0].start),
                        style(MUTED),
                    )),
                ],
            )
        }
        Load::Ready(h) => (
            format!("Time {} · no bucket in this view", h.timezone),
            vec![Line::from(format!(
                "missing {} · invalid {} occurrences",
                grouped(h.missing.1),
                grouped(h.invalid.1)
            ))],
        ),
        Load::Unavailable(why) => ("Time".into(), vec![Line::from(safe(why))]),
        Load::Cancelled => ("Time".into(), vec![Line::from(CANCELLED)]),
        Load::Loading | Load::Idle => ("Time".into(), vec![Line::from("loading…")]),
    };
    frame.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: true })
            .block(block(title, false)),
        area,
    );
}

fn sidebar(frame: &mut Frame, bench: &Workbench, area: Rect) {
    let f = bench.frame();
    let mut lines: Vec<Line> = Vec::new();
    let mut focus_line = 0usize;
    let names: Vec<&str> = bench
        .named
        .iter()
        .take(super::REGION_SETS)
        .map(|(name, _)| name.as_str())
        .collect();
    lines.push(Line::from(Span::styled(
        format!("Regions of {} · orig. · occ.", names.join(" ")),
        style(MUTED),
    )));
    match &f.regions {
        Load::Ready((known, unknown)) => {
            for (index, region) in known.iter().chain(unknown).enumerate() {
                let marks: String = region
                    .states
                    .iter()
                    .map(|(set, state)| {
                        let mark = match state.as_str() {
                            "in" => '+',
                            "out" => '-',
                            _ => '?',
                        };
                        format!("{}{mark} ", excerpt(&bench.named[*set].0, 6))
                    })
                    .collect();
                let selected = bench.focus == Focus::Regions && index == bench.region_index;
                if selected {
                    focus_line = lines.len();
                }
                let text = format!(
                    "{marks:<12}{:>9} {:>9}",
                    grouped(region.originals),
                    grouped(region.occurrences)
                );
                lines.push(Line::from(Span::styled(
                    text,
                    if selected {
                        Style::default().add_modifier(Modifier::REVERSED)
                    } else if index >= known.len() {
                        style(WARN)
                    } else {
                        Style::default()
                    },
                )));
            }
        }
        Load::Unavailable(why) => lines.push(Line::from(safe(why))),
        Load::Cancelled => lines.push(Line::from(CANCELLED)),
        Load::Idle if bench.named.is_empty() => {
            lines.push(Line::from("no named sets in this hunt"))
        }
        _ => lines.push(Line::from("loading…")),
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "Fields · values · with value",
        style(MUTED),
    )));
    match &f.fields {
        Load::Ready(fields) => {
            for (index, field) in fields.iter().enumerate() {
                let selected = bench.focus == Focus::Fields && index == bench.field_index;
                if selected {
                    focus_line = lines.len();
                }
                let exact = if field.exact { "" } else { "+" };
                let text = format!(
                    "{:<12}{:>7}{exact} {:>9}{}",
                    excerpt(&field.name, 12),
                    grouped(field.distinct),
                    grouped(field.with_value.1),
                    if field.missing.1 > 0 {
                        format!(" ({} missing)", grouped(field.missing.1))
                    } else {
                        String::new()
                    }
                );
                lines.push(Line::from(Span::styled(
                    text,
                    if selected {
                        Style::default().add_modifier(Modifier::REVERSED)
                    } else {
                        Style::default()
                    },
                )));
            }
        }
        Load::Unavailable(why) => lines.push(Line::from(safe(why))),
        Load::Cancelled => lines.push(Line::from(CANCELLED)),
        _ => lines.push(Line::from("loading…")),
    }
    let visible = usize::from(area.height.saturating_sub(2)).max(1);
    let scroll = focus_line.saturating_sub(visible.saturating_sub(1));
    let focused = matches!(bench.focus, Focus::Regions | Focus::Fields);
    frame.render_widget(
        Paragraph::new(lines)
            .scroll((scroll.min(usize::from(u16::MAX)) as u16, 0))
            .block(block("Regions & fields · Tab".into(), focused)),
        area,
    );
}

fn results(frame: &mut Frame, bench: &Workbench, area: Rect) {
    let f = bench.frame();
    let focused = bench.focus == Focus::Results && matches!(bench.mode, Mode::Browse);
    let page = match &f.page {
        Load::Ready(page) => page,
        Load::Unavailable(why) => {
            frame.render_widget(
                Paragraph::new(safe(why))
                    .wrap(Wrap { trim: false })
                    .block(block("Results".into(), focused)),
                area,
            );
            return;
        }
        Load::Cancelled => {
            frame.render_widget(
                Paragraph::new(CANCELLED).block(block("Results".into(), focused)),
                area,
            );
            return;
        }
        _ => {
            frame.render_widget(
                Paragraph::new("loading…").block(block("Results".into(), focused)),
                area,
            );
            return;
        }
    };
    let first = page.index * super::PAGE_ROWS as usize;
    let title = if page.rows.is_empty() {
        "Results · no original in this view".to_string()
    } else {
        format!(
            "Results {}–{} of {} · {}",
            grouped(first as u64 + 1),
            grouped((first + page.rows.len()) as u64),
            plural(page.total.0, "original", "originals"),
            plural(page.total.1, "occurrence", "occurrences")
        )
    };
    let visible = usize::from(area.height.saturating_sub(3)).max(1);
    let offset = f.selected / visible * visible;
    let named: Vec<String> = bench
        .named
        .iter()
        .map(|(name, _)| excerpt(name, 3))
        .collect();
    let final_view = f.kind == Kind::Final;
    let mut header_cells = vec!["Record".to_string(), "Locator".into(), "Occ.".into()];
    header_cells.extend(named.iter().cloned());
    if final_view {
        header_cells.push("Slot (* outside)".into());
    }
    let rows: Vec<TableRow> = page
        .rows
        .iter()
        .enumerate()
        .skip(offset)
        .take(visible)
        .map(|(index, row)| {
            let mut cells = vec![
                row.id.clone(),
                row.locator.clone(),
                grouped(row.occurrences),
            ];
            cells.extend(row.states.iter().map(char::to_string));
            if final_view {
                cells.push(row.allocation.clone().unwrap_or_default());
            }
            let style = if index == f.selected && focused {
                Style::default().add_modifier(Modifier::REVERSED)
            } else if index == f.selected {
                Style::default().add_modifier(Modifier::UNDERLINED)
            } else {
                Style::default()
            };
            TableRow::new(cells).style(style)
        })
        .collect();
    let mut widths = vec![
        Constraint::Length(18),
        Constraint::Min(6),
        Constraint::Length(6),
    ];
    widths.extend(named.iter().map(|_| Constraint::Length(3)));
    if final_view {
        widths.push(Constraint::Length(16));
    }
    frame.render_widget(
        Table::new(rows, widths)
            .header(TableRow::new(header_cells).style(style(MUTED)))
            .block(block(title, focused)),
        area,
    );
}

fn work(activity: &Option<Value>) -> String {
    match activity {
        Some(activity) => {
            let get = |key: &str| activity[key].as_u64().unwrap_or(0);
            format!(
                "work: scans {} · unpacked {} · scored {} · encoded {} · genai {} · excerpt bytes {}",
                grouped(get(WORK[0])),
                grouped(get(WORK[1])),
                grouped(get(WORK[2])),
                grouped(get(WORK[3])),
                grouped(get(WORK[4])),
                grouped(get(WORK[5]))
            )
        }
        None => "work: w to read".into(),
    }
}

fn status(frame: &mut Frame, bench: &Workbench, area: Rect) {
    let first = match &bench.pending {
        Some(pending) => Line::from(Span::styled(
            format!(
                "Running {} · {:.1} s · {} · Esc cancels",
                pending.op.label(),
                pending.started.elapsed().as_secs_f64(),
                if pending.cancel {
                    "cancel requested"
                } else {
                    "job"
                }
            ),
            style(WARN),
        )),
        None if !bench.queue.is_empty() => Line::from(Span::styled("Working…", style(WARN))),
        None if !bench.notice.is_empty() => {
            Line::from(Span::styled(excerpt(&bench.notice, 600), style(WARN)))
        }
        None => Line::from(work(&bench.activity)),
    };
    let timing = format!(
        "layout {} µs · draw {} µs · call {}",
        bench.last_layout_us,
        bench.last_draw_us,
        bench
            .last_call_ms
            .map_or("–".into(), |ms| format!("{ms:.1} ms"))
    );
    let second = Line::from(vec![
        Span::styled(
            format!(
                "{} · {} · ",
                excerpt(&bench.snapshot, 20),
                excerpt(&bench.origin, 24)
            ),
            style(MUTED),
        ),
        Span::styled(timing, style(MUTED)),
        Span::styled(" · ? keys · q quit", style(MUTED)),
    ]);
    frame.render_widget(Paragraph::new(vec![first, second]), area);
}

fn dialog_area(area: Rect, width: u16, height: u16) -> Rect {
    let width = width.min(area.width.saturating_sub(2)).max(10);
    let height = height.min(area.height.saturating_sub(2)).max(3);
    Rect::new(
        area.x + (area.width.saturating_sub(width)) / 2,
        area.y + (area.height.saturating_sub(height)) / 2,
        width,
        height,
    )
}

fn list_dialog(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    lines: &[String],
    selected: usize,
    hint: &str,
) {
    let height = (lines.len() as u16)
        .saturating_add(3)
        .min(area.height.saturating_sub(2));
    let dialog = dialog_area(area, 90, height);
    let visible = usize::from(dialog.height.saturating_sub(3)).max(1);
    let offset = selected / visible * visible;
    let mut shown: Vec<Line> = lines
        .iter()
        .enumerate()
        .skip(offset)
        .take(visible)
        .map(|(index, line)| {
            Line::from(Span::styled(
                line.clone(),
                if index == selected {
                    Style::default().add_modifier(Modifier::REVERSED)
                } else {
                    Style::default()
                },
            ))
        })
        .collect();
    if lines.is_empty() {
        shown.push(Line::from("nothing to choose"));
    }
    shown.push(Line::from(Span::styled(hint.to_string(), style(MUTED))));
    frame.render_widget(Clear, dialog);
    frame.render_widget(
        Paragraph::new(shown).block(block(safe(title), true)),
        dialog,
    );
}

fn evidence_lines(
    bench: &Workbench,
    record: &str,
    data: &Load<Value>,
    excerpt_load: &Load<Value>,
) -> Vec<Line<'static>> {
    let mut lines = vec![Line::from(Span::styled(
        safe(record),
        Style::default().add_modifier(Modifier::BOLD),
    ))];
    let evidence = match data {
        Load::Ready(evidence) => evidence,
        Load::Unavailable(why) => {
            lines.push(Line::from(safe(why)));
            return lines;
        }
        _ => {
            lines.push(Line::from("loading…"));
            return lines;
        }
    };
    let found = &evidence["record"];
    lines.push(Line::from(format!(
        "First locator  {}:{}",
        safe(found["locator"]["entry"].as_str().unwrap_or("?")),
        found["locator"]["line"]
    )));
    lines.push(Line::from(format!(
        "Occurrences    {}",
        found["occurrences"]
            .as_u64()
            .or_else(|| found["occurrence_list"]["listed"].as_u64())
            .map_or("?".into(), grouped)
    )));
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "Why it is here (memberships and producers)",
        style(ACCENT),
    )));
    for item in evidence["explanation"].as_array().into_iter().flatten() {
        let name = item["set"].as_str().unwrap_or("?");
        let name = if name == "handle" { "this view" } else { name };
        let mut text = format!(
            "  {:<10} {:<14} {}",
            excerpt(name, 10),
            item["state"].as_str().unwrap_or("?"),
            item["producer"]["kind"].as_str().unwrap_or("")
        );
        if let Some(score) = item["score"].as_f64() {
            text.push_str(&format!(" · score {score:.4}"));
        }
        if let (Some(rank), Some(among)) = (item["rank"].as_u64(), item["ranked_among"].as_u64()) {
            text.push_str(&format!(" · rank {} of {}", grouped(rank), grouped(among)));
        }
        if let Some(matched) = item["matched"].as_bool() {
            text.push_str(if matched {
                " · matched"
            } else {
                " · not matched"
            });
        }
        for key in ["reason", "route", "outcome", "distance"] {
            if let Some(value) = item.get(key).filter(|value| !value.is_null()) {
                let shown = value
                    .as_str()
                    .map_or_else(|| value.to_string(), str::to_string);
                text.push_str(&format!(" · {key} {}", excerpt(&shown, 80)));
            }
        }
        if let Some(allocation) = item.get("allocation").filter(|value| value.is_object()) {
            text.push_str(&format!(" · {}", excerpt(&allocation.to_string(), 160)));
        }
        lines.push(Line::from(text));
    }
    let list = &found["occurrence_list"];
    if let Some(rows) = list["rows"].as_array() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!(
                "Occurrences ({} listed{})",
                list["listed"],
                if list["clipped"] == true {
                    ", clipped"
                } else {
                    ""
                }
            ),
            style(ACCENT),
        )));
        for row in rows.iter().take(25) {
            let in_view = match row["in_view"].as_bool() {
                Some(true) => "in view ",
                Some(false) => "elsewhere",
                None => "",
            };
            lines.push(Line::from(format!(
                "  {in_view} {}:{} {}",
                safe(row["entry"].as_str().unwrap_or("?")),
                row["line"],
                safe(row["time"].as_str().unwrap_or(""))
            )));
        }
    }
    lines.push(Line::from(""));
    match excerpt_load {
        Load::Ready(value) => {
            lines.push(Line::from(Span::styled(
                format!(
                    "Excerpt ({} of {} bytes{}; controls escaped)",
                    value["excerpt"]["bytes"],
                    value["excerpt"]["source_bytes"],
                    if value["excerpt"]["clipped"] == true {
                        ", clipped"
                    } else {
                        ""
                    }
                ),
                style(ACCENT),
            )));
            lines.push(Line::from(
                value["excerpt"]["text"]
                    .as_str()
                    .map_or(String::new(), safe),
            ));
        }
        Load::Loading => lines.push(Line::from("Reading the excerpt… Esc cancels")),
        Load::Unavailable(why) => lines.push(Line::from(format!("Excerpt: {}", safe(why)))),
        Load::Idle | Load::Cancelled => {
            let estimate = &evidence["excerpt_estimate"];
            lines.push(Line::from(Span::styled(
                format!(
                    "x reads the excerpt: {}",
                    estimate["requires"]
                        .as_str()
                        .unwrap_or("one read of the source")
                ),
                style(MUTED),
            )));
        }
    }
    let _ = bench;
    lines
}
