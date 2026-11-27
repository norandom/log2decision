use std::fmt::Write as _;

use ratatui::{
    Frame,
    layout::{Constraint, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Row, Table, Wrap},
};

use super::{App, Menu, Mode, Panes, REGIONS};
use crate::flow::{CProducer, Counts, Recommendation, Region, SearchKind};

#[cfg(test)]
mod tests;

const ACCENT: Color = Color::Cyan;
const MUTED: Color = Color::DarkGray;
fn style(color: Color) -> Style {
    Style::default().fg(color)
}
fn block(title: &str, focus: bool) -> Block<'_> {
    Block::default()
        .borders(Borders::ALL)
        .title(format!(" {title} "))
        .border_style(style(if focus { ACCENT } else { MUTED }))
}

// Escape terminal commands and bidi overrides in source material, including titles and filenames.
pub(crate) fn safe(value: &str) -> String {
    let mut s = String::new();
    for ch in value.chars() {
        if l2d_core::terminal_text::is_invisible_or_control(ch) {
            s.extend(ch.escape_default());
        } else {
            s.push(ch);
        }
    }
    s
}

pub(crate) fn excerpt(value: &str, limit: usize) -> String {
    let end = value
        .char_indices()
        .nth(limit)
        .map_or(value.len(), |(i, _)| i);
    let mut s = safe(&value[..end]);
    if end < value.len() {
        s.push('…');
    }
    s
}

fn label(kind: &SearchKind) -> &'static str {
    match kind {
        SearchKind::Spl => "SPL",
        SearchKind::Literal => "literal",
        SearchKind::Glob => "glob",
        SearchKind::Regex => "regex",
    }
}

// Both report adapters project the saved diagnostic MarkSet ratio into A.
fn score_labels(source: CProducer) -> (&'static str, &'static str) {
    match source {
        CProducer::Fixture => ("Score", "Score"),
        CProducer::Unavailable | CProducer::PolicyInvestigateOrReview => {
            ("Diag.", "Diagnostic MarkSet")
        }
    }
}

fn scope(count: &Counts, known: bool) -> String {
    let originals = if count.originals == 1 {
        "original"
    } else {
        "originals"
    };
    if known {
        let occurrences = if count.occurrences == 1 {
            "occurrence"
        } else {
            "occurrences"
        };
        format!(
            "{} {originals} · {} {occurrences}",
            count.originals, count.occurrences
        )
    } else {
        format!("{} {originals} · occurrences unavailable", count.originals)
    }
}

pub(super) fn screen(frame: &mut Frame, app: &mut App) {
    if let Mode::Raw { browser, .. } = &mut app.mode {
        browser.draw(frame, &app.flow);
        return;
    }
    let area = frame.area();
    if area.width < 32 || area.height < 8 {
        app.fields_focused = false;
        app.panes = Panes::default();
        app.geometry = None;
        frame.render_widget(
            Paragraph::new("Triager · enlarge terminal to 32×8 or press q to exit.")
                .wrap(Wrap { trim: false }),
            area,
        );
        return;
    }
    let geometry = (area, app.sidebar, app.search_rows);
    if app.geometry != Some(geometry) {
        let Ok(panes) = Panes::solve(area, app.sidebar, app.search_rows) else {
            frame.render_widget("Unable to lay out terminal panes. Press q to exit.", area);
            return;
        };
        app.panes = panes;
        app.geometry = Some(geometry);
    }
    let p = app.panes;
    if p.fields.width == 0 || app.flow.view().is_final() {
        app.fields_focused = false;
    }
    let flow = &app.flow;
    let view = flow.view();
    let dataset = flow.dataset();
    frame.render_widget(
        Line::from(vec![
            Span::styled(
                " TRIAGER ",
                style(Color::Black).bg(ACCENT).add_modifier(Modifier::BOLD),
            ),
            Span::raw(format!(
                "  {}",
                excerpt(&dataset.title, usize::from(area.width))
            )),
        ]),
        p.header,
    );

    let local = app.local();
    let editing = matches!(app.mode, Mode::Search);
    let parent_filter = flow.root_search();
    let applied_search = match &view.step {
        Some(crate::flow::Step::Search(search)) => Some(search),
        _ => None,
    };
    let search_title = if view.is_final() {
        "Final work set · allocation"
    } else if editing {
        "Search in current view · Tab syntax · Enter apply · Esc keep draft"
    } else if applied_search.is_some() && local.draft.is_empty() {
        "Applied search · / narrow these results"
    } else {
        "Search in current view  /"
    };
    let search = if let Some(crate::flow::Step::Allocate(budget)) = &view.step {
        format!(
            "General {} · unfamiliar {} · audit {} · seed {}",
            budget.general, budget.unfamiliar, budget.audit, budget.seed
        )
    } else if editing || !local.draft.is_empty() {
        format!(
            "{}  {}{}",
            label(&super::search_kind(local.syntax)),
            safe(&local.draft),
            if editing { "▏" } else { "" }
        )
    } else if let Some(search) = applied_search {
        format!("{}  {}", label(&search.kind), safe(&search.query))
    } else {
        format!(
            "A  top {} by {}    B  {}  {}",
            flow.top_k(),
            score_labels(dataset.c_producer).1,
            label(&parent_filter.kind),
            safe(&parent_filter.query)
        )
    };
    let editor = Paragraph::new(search)
        .wrap(Wrap { trim: false })
        .block(block(search_title, editing));
    let scroll = if editing {
        editor
            .line_count(p.search.width)
            .saturating_sub(usize::from(p.search.height))
    } else {
        0
    };
    frame.render_widget(editor.scroll((scroll as u16, 0)), p.search);
    if p.time.height > 0 {
        timeline(frame, app, p.time);
    }
    if p.fields.width > 0 {
        fields(frame, app, p.fields);
    }
    results(frame, app, p.results);

    let parent = view.parent.map(|id| &flow.views()[id]).unwrap_or(view);
    let root = &flow.views()[0];
    let status = if app.notice.is_empty() {
        let mut text = format!(
            "#{} {}  ·  parent #{} {}/{} originals  ·  universe {}/{} originals",
            view.id,
            if view.is_final() {
                "final"
            } else {
                "candidate"
            },
            parent.id,
            view.counts.originals,
            parent.counts.originals,
            view.counts.originals,
            root.counts.originals
        );
        if dataset.occurrences_known {
            let _ = write!(
                text,
                " · parent {}/{} occ",
                view.counts.occurrences, parent.counts.occurrences
            );
        }
        if view.is_final() {
            let _ = write!(
                text,
                " · {} deferred",
                parent
                    .counts
                    .originals
                    .saturating_sub(view.counts.originals)
            );
        }
        text
    } else {
        safe(&app.notice)
    };
    let hints = if view.is_final() {
        "r raw logs   Enter evidence   b candidate   v branches   s save   ? keys   q quit"
    } else if area.width < 78 {
        "? keys   r raw logs   1 A−B   c coverage   Enter evidence   b back   q quit"
    } else {
        "? keys  / search  r raw logs  Enter evidence  h host  t hour  b back  v branches  f allocate  s save  q quit"
    };
    frame.render_widget(
        Paragraph::new(vec![Line::styled(status, style(ACCENT)), Line::raw(hints)]),
        p.status,
    );
    match &app.mode {
        Mode::Browse | Mode::Search | Mode::Raw { .. } => {}
        Mode::Menu(kind, selected) => menu(frame, app, *kind, *selected),
        Mode::Evidence { scroll, lines } => pane(
            frame,
            "Record evidence  ·  r raw logs  ·  ↑/↓ scroll  ·  Esc close",
            lines.clone(),
            *scroll,
        ),
        Mode::Help(scroll) => {
            let lines = [
                "1  A minus B       2  overlap       3  B minus A       4  neither",
                "c  C coverage      i  recommended   o  assessed pass   u  unknown",
                "h  host            t  UTC hour      /  search in current view",
                "Search in SPL, literal, glob or regex. Tab changes syntax. Enter applies.",
                "Try (rejected OR declined) score>=0.5 | sort - score | head 5 in SPL.",
                "Terms match whole tokens as in Splunk: fail* finds failed. Fields: user=bob.",
                "Pipe to search, where, regex, rex, head, tail, sort, dedup or fields.",
                "Up / Down or k / j select. PgUp / PgDn page. Home / End jump.",
                "Tab switches results / regions. Enter opens evidence or drills.",
                "r opens raw logs in current view order. Esc returns to the original selection.",
                "Esc or b goes back. v opens any retained branch.",
                "f allocates named pools. Edit slot counts, then Enter to confirm.",
                "s saves a receipt. S chooses another path. Ctrl+u clears input.",
                "[ / ] resize fields. - / + resize search. Drag borders with a mouse.",
                "Run the same command with --session PATH to reopen a receipt.",
                "q quits from the workbench. Esc closes help.",
            ];
            pane(
                frame,
                "Keyboard  ↑/↓ scroll · Esc close",
                lines.iter().map(|s| Line::raw(*s)).collect(),
                *scroll,
            );
        }
        Mode::Allocate(input) => {
            pane(
                frame,
                "Allocate final work set",
                vec![
                    Line::raw(format!(
                        "Source #{} · {}",
                        view.id,
                        scope(&view.counts, dataset.occurrences_known)
                    )),
                    Line::raw(
                        "New analyst allocation. Slots in order: general, unfamiliar, audit.",
                    ),
                    Line::styled(format!("{}▏", safe(input)), style(ACCENT)),
                    Line::raw(
                        "General selects C recommended by score. Unfamiliar reserves C unknown, including unavailable input. Audit samples C pass with seed 5. Shortfalls stay open.",
                    ),
                    Line::raw("Enter confirms one allocation. Ctrl+u clears. Esc cancels."),
                    Line::styled(safe(&app.notice), style(Color::Yellow)),
                ],
                0,
            );
        }
        Mode::SaveAs(input) => pane(
            frame,
            "Save receipt as",
            vec![
                Line::raw("Receipt path · Enter saves · Esc cancels"),
                Line::styled(format!("{}▏", safe(input)), style(ACCENT)),
                Line::styled(safe(&app.notice), style(Color::Yellow)),
            ],
            0,
        ),
    }
}

fn fields(frame: &mut Frame, app: &App, area: Rect) {
    let view = app.flow.view();
    let mut lines = vec![Line::styled("  Membership       originals", style(MUTED))];
    for (i, &(key, region, name)) in REGIONS.iter().enumerate() {
        let n = view.regions.get(&region).map_or(0, |c| c.originals);
        let color = if app.fields_focused && app.region_index == i {
            Color::Yellow
        } else if region == Region::Unknown {
            Color::LightYellow
        } else {
            Color::Reset
        };
        let key = if view.is_final() { ' ' } else { key };
        lines.push(Line::styled(
            format!("{key}  {name:<16} {n:>5}"),
            style(color),
        ));
    }
    if app.flow.dataset().occurrences_known {
        let title = if view.is_final() {
            "   Hosts"
        } else {
            "h  Hosts"
        };
        lines.extend([Line::raw(""), Line::styled(title, style(ACCENT))]);
        for (host, count) in view
            .hosts
            .iter()
            .take(usize::from(area.height.saturating_sub(12)))
        {
            lines.push(Line::raw(format!(
                "   {} · {}",
                excerpt(host.as_deref().unwrap_or("(missing)"), 17),
                count.occurrences
            )));
        }
    }
    frame.render_widget(
        Paragraph::new(lines).block(block("Regions & fields", app.fields_focused)),
        area,
    );
}

fn timeline(frame: &mut Frame, app: &App, area: Rect) {
    let view = app.flow.view();
    let max_cells = usize::from(area.width.saturating_sub(2)) / 16;
    let buckets = view.hours.iter().take(max_cells).collect::<Vec<_>>();
    let text = if !app.flow.dataset().occurrences_known {
        vec![Line::raw(
            "Occurrence timestamps unavailable in this source.",
        )]
    } else if buckets.is_empty() {
        vec![Line::raw("No matching occurrences.")]
    } else {
        let peak = buckets
            .iter()
            .map(|(_, c)| c.occurrences)
            .max()
            .unwrap_or(1)
            .max(1);
        let mut bars = String::new();
        let mut labels = String::new();
        for (hour, count) in buckets {
            let n = (u128::from(count.occurrences) * 7).div_ceil(u128::from(peak)) as usize;
            let _ = write!(
                bars,
                "{:<16}",
                format!("{} {}", "▮".repeat(n), count.occurrences)
            );
            let shown = hour.as_deref().unwrap_or("missing");
            let hour = shown.strip_suffix(":00Z").unwrap_or(shown);
            let _ = write!(labels, "{:<16}", excerpt(hour, 14));
        }
        vec![Line::styled(bars, style(ACCENT)), Line::raw(labels)]
    };
    let title = if app.flow.dataset().occurrences_known {
        let key = if view.is_final() { "" } else { "  t" };
        format!(
            "UTC hour · occurrences{key} · showing {} of {} buckets",
            max_cells.min(view.hours.len()),
            view.hours.len()
        )
    } else {
        "UTC hour · occurrences unavailable".into()
    };
    frame.render_widget(Paragraph::new(text).block(block(&title, false)), area);
}

fn results(frame: &mut Frame, app: &App, area: Rect) {
    let flow = &app.flow;
    let view = flow.view();
    let n = view.members.len();
    let page = usize::from(area.height.saturating_sub(3)).max(1);
    let selected = app.local().selected;
    let start = selected / page * page;
    let title = format!(
        "{} · {}",
        scope(&view.counts, flow.dataset().occurrences_known),
        excerpt(&view.title, usize::from(area.width))
    );
    let edge = block(&title, !app.fields_focused);
    if n == 0 {
        frame.render_widget(
            Paragraph::new("No matching originals. Press b to revisit the parent.")
                .wrap(Wrap { trim: false })
                .block(edge),
            area,
        );
        return;
    }
    let compact = area.width < 72;
    let id_width = if compact { 5 } else { 13 };
    let mut widths = vec![
        Constraint::Fill(1),
        Constraint::Length(id_width),
        Constraint::Length(5),
        Constraint::Length(4),
    ];
    let mut headings = vec![
        "Raw log (_raw)",
        if compact { "ID" } else { "Original ID" },
        score_labels(flow.dataset().c_producer).0,
        "AB C",
    ];
    let show_counts = !compact && flow.dataset().occurrences_known && !view.is_final();
    if view.is_final() {
        headings.push("Pool");
        widths.push(Constraint::Length(12));
    } else if show_counts {
        headings.push("Occ.");
        let digits = view.counts.occurrences.to_string().len() as u16;
        widths.push(Constraint::Length(digits.max(4)));
    }
    let rows = view
        .members
        .iter()
        .skip(start)
        .take(page)
        .enumerate()
        .map(|(i, member)| {
            let record = &flow.dataset().records[member.record];
            let (a, b, c) = flow.membership(member.record);
            let membership = format!(
                "{}{} {}",
                if a { 'A' } else { '-' },
                if b { 'B' } else { '-' },
                match c {
                    Recommendation::In => '+',
                    Recommendation::Out => '-',
                    Recommendation::Unknown => '?',
                }
            );
            let mut cells = vec![
                excerpt(&record.text, usize::from(area.width)),
                excerpt(&record.id, usize::from(id_width - 1)),
                format!("{:.3}", record.score),
                membership,
            ];
            if let Some(pool) = member.allocation {
                cells.push(format!("[{}]", pool.label()));
            } else if show_counts {
                cells.push(member.occurrences.to_string());
            }
            let row = Row::new(cells);
            if start + i == selected {
                row.style(style(Color::Black).bg(Color::Cyan))
            } else {
                row
            }
        });
    let table = Table::new(rows, widths)
        .header(Row::new(headings).style(style(MUTED)))
        .block(edge)
        .column_spacing(1);
    frame.render_widget(table, area);
}

fn dialog_area(area: Rect) -> Rect {
    let width = area.width.saturating_sub(4).min(98);
    let height = area.height.saturating_sub(4).min(25);
    Rect::new(
        area.x + (area.width - width) / 2,
        area.y + (area.height - height) / 2,
        width,
        height,
    )
}

fn pane(frame: &mut Frame, title: &str, lines: Vec<Line<'_>>, scroll: u16) {
    let area = dialog_area(frame.area());
    frame.render_widget(Clear, area);
    let body = Paragraph::new(lines)
        .block(block(title, true))
        .wrap(Wrap { trim: false });
    // For near-empty details, scrolling never moves all content off screen.
    let max = body
        .line_count(area.width)
        .saturating_sub(usize::from(area.height));
    frame.render_widget(
        body.scroll((scroll.min(u16::try_from(max).unwrap_or(u16::MAX)), 0)),
        area,
    );
}

fn menu(frame: &mut Frame, app: &App, kind: Menu, selected: usize) {
    let area = dialog_area(frame.area());
    let n = usize::from(area.height.saturating_sub(4)).max(1);
    let start = selected / n * n;
    let view = app.flow.view();
    let total = app.menu_len(kind);
    let mut lines = vec![Line::raw(format!(
        "Up / Down · Enter drills · Esc closes    {}/{}",
        (selected + 1).min(total),
        total
    ))];
    let title = match kind {
        Menu::Hosts | Menu::Hours => {
            let facet = if kind == Menu::Hosts {
                &view.hosts
            } else {
                &view.hours
            };
            for (key, counts) in facet.iter().skip(start).take(n) {
                lines.push(Line::raw(format!(
                    "{}   {}",
                    safe(key.as_deref().unwrap_or("(missing)")),
                    scope(counts, true)
                )));
            }
            if kind == Menu::Hosts {
                "Choose host"
            } else {
                "Choose UTC hour"
            }
        }
        Menu::Branches => {
            for v in app.flow.views().iter().skip(start).take(n) {
                let parent = v
                    .parent
                    .map_or_else(|| "root".into(), |id| format!("from #{id}"));
                lines.push(Line::raw(format!(
                    "#{} · {} · {} · {} originals",
                    v.id,
                    safe(&v.title),
                    parent,
                    v.counts.originals
                )));
            }
            "Retained branches"
        }
        Menu::Coverage => {
            for (key, region, name) in REGIONS[4..].iter() {
                if let Some(count) = view.regions.get(region) {
                    lines.push(Line::raw(format!(
                        "{key}  {name}    {}",
                        scope(count, app.flow.dataset().occurrences_known)
                    )));
                }
            }
            "Prepared C coverage"
        }
    };
    if let Some(line) = lines.get_mut(1 + selected - start) {
        line.style = style(Color::Black).bg(ACCENT);
    }
    pane(frame, title, lines, 0);
}

// Inspect once on opening the record. A resize or a scroll only repaints this bounded projection.
pub(super) fn evidence_lines(app: &App) -> Vec<Line<'static>> {
    const LOCATION_PREVIEW: usize = 24;
    let flow = &app.flow;
    let view = flow.view();
    let Some(member) = view.members.get(app.local().selected) else {
        return vec![];
    };
    let record = &flow.dataset().records[member.record];
    let (a, b, c) = flow.membership(member.record);
    let membership = match c {
        Recommendation::In => "recommended",
        Recommendation::Out => "pass",
        Recommendation::Unknown => "unknown",
    };
    let mut lines = vec![
        Line::styled(excerpt(&record.id, 1024), style(ACCENT)),
        Line::raw(format!(
            "{} {:.4}   A {}   B {}   C {}",
            score_labels(flow.dataset().c_producer).1,
            record.score,
            if a { "in" } else { "out" },
            if b { "in" } else { "out" },
            membership
        )),
        Line::raw(format!("Reason  {}", excerpt(&record.reason, 2048))),
        Line::raw(format!("Original  {}", excerpt(&record.text, 2048))),
    ];
    if let Some(pool) = &member.allocation {
        lines.push(Line::styled(
            format!("Allocated pool  {}", pool.label()),
            style(ACCENT),
        ));
    }
    if flow.dataset().occurrences_known {
        lines.push(Line::raw(format!(
            "Matching occurrences  {}",
            member.occurrences
        )));
        let mut locations = flow.locations(member.record);
        for e in locations.by_ref().take(LOCATION_PREVIEW) {
            lines.push(Line::raw(format!(
                "  {}  ·  {}  ×{}",
                excerpt(e.host.as_deref().unwrap_or("(missing host)"), 160),
                excerpt(e.time.as_deref().unwrap_or("(missing time)"), 64),
                e.count
            )));
        }
        if locations.next().is_some() {
            lines.push(Line::raw(
                "Showing the first 24 matching locations. The source retains the full list.",
            ));
        }
    } else {
        lines.push(Line::raw(
            "Occurrence counts and locations unavailable in this source.",
        ));
    }
    lines.push(Line::raw(format!(
        "Provenance  {}",
        excerpt(&flow.dataset().provenance, 2048)
    )));
    lines.push(Line::raw(format!(
        "Source revision  {}",
        excerpt(&flow.dataset().source_revision, 512)
    )));
    lines
}
