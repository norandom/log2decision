//! Read retained record text in view order. Only the current chunk and its neighbours'
//! one-line excerpts are formatted; navigating never needs an index of the whole log.
use std::ops::Range;

use crossterm::event::KeyCode;
use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Paragraph},
};

use super::draw::excerpt;
use crate::flow::Workflow;

const CHUNK_BYTES: usize = 1024;

fn chunk_range(text: &str, chunk: usize) -> Range<usize> {
    let boundary =
        |n: usize| text.floor_char_boundary(n.saturating_mul(CHUNK_BYTES).min(text.len()));
    boundary(chunk)..boundary(chunk.saturating_add(1))
}

// A chunk or physical line may start partway through a grapheme. Ratatui drops
// standalone zero-width graphemes, so represent their code points explicitly.
fn push_printable(text: &mut String, part: &str) {
    if part.is_ascii() {
        text.push_str(part);
        return;
    }
    let span = Span::raw(part);
    for g in span.styled_graphemes(Style::default()) {
        if Span::raw(g.symbol).width() == 0 {
            for ch in g.symbol.chars() {
                text.extend(ch.escape_unicode());
            }
        } else {
            text.push_str(g.symbol);
        }
    }
}

fn display_chunk(chunk: &str) -> String {
    let mut text = String::with_capacity(chunk.len());
    let mut start = 0;
    for (index, ch) in chunk.char_indices() {
        if ch != '\\' && !l2d_core::terminal_text::is_invisible_or_control(ch) {
            continue;
        }
        push_printable(&mut text, &chunk[start..index]);
        match ch {
            '\n' => text.push('\n'),
            '\\' => text.push_str(r"\\"),
            _ => text.extend(ch.escape_default()),
        }
        start = index + ch.len_utf8();
    }
    push_printable(&mut text, &chunk[start..]);
    text
}

/// Preserve physical line breaks and literal spaces. Wrap by terminal grapheme width
/// before drawing so the widget cannot drop whitespace at a soft line boundary.
struct TextPage {
    text: String,
    rows: Vec<Range<usize>>,
}

impl TextPage {
    fn new(chunk: &str, width: u16) -> Self {
        let text = display_chunk(chunk);
        let width = usize::from(width.max(2));
        let mut rows = Vec::new();
        let mut offset = 0;
        for line in text.split('\n') {
            let mut start = offset;
            let mut cells = 0;
            let span = Span::raw(line);
            for g in span.styled_graphemes(Style::default()) {
                let size = Span::raw(g.symbol).width();
                if cells > 0 && cells + size > width {
                    rows.push(start..offset);
                    start = offset;
                    cells = 0;
                }
                offset += g.symbol.len();
                cells += size;
            }
            rows.push(start..offset);
            offset += 1; // the newline between source lines
        }
        Self { text, rows }
    }

    fn top(&self, anchor: usize, height: usize) -> usize {
        self.rows
            .partition_point(|row| row.start <= anchor)
            .saturating_sub(1)
            .min(self.rows.len().saturating_sub(height))
    }
}

pub(super) struct RawBrowser {
    selected: usize,
    chunk: usize,
    // Offset in the escaped chunk, stable when a resized terminal rewraps the text.
    anchor: usize,
    body: Rect,
}

impl RawBrowser {
    pub(super) fn new(selected: usize) -> Self {
        Self {
            selected,
            chunk: 0,
            anchor: 0,
            body: Rect::default(),
        }
    }

    pub(super) fn handle(&mut self, key: KeyCode, flow: &Workflow) {
        let members = &flow.view().members;
        let Some(member) = members.get(self.selected) else {
            return;
        };
        let last = members.len() - 1;
        let selected = match key {
            KeyCode::Up | KeyCode::Char('k') => self.selected.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => (self.selected + 1).min(last),
            KeyCode::Home => 0,
            KeyCode::End => last,
            _ => {
                self.move_text(key, &flow.dataset().records[member.record].text);
                return;
            }
        };
        if self.selected != selected {
            self.selected = selected;
            self.chunk = 0;
            self.anchor = 0;
        }
    }

    fn move_text(&mut self, key: KeyCode, record: &str) {
        let last = record.len().saturating_sub(1) / CHUNK_BYTES;
        match key {
            KeyCode::Right if self.chunk < last => {
                self.chunk += 1;
                self.anchor = 0;
            }
            KeyCode::Left if self.chunk > 0 => {
                self.chunk -= 1;
                self.anchor = 0;
            }
            KeyCode::PageUp | KeyCode::PageDown => {
                let page = TextPage::new(&record[chunk_range(record, self.chunk)], self.body.width);
                let height = usize::from(self.body.height.max(1));
                let top = page.top(self.anchor, height);
                let bottom = page.rows.len().saturating_sub(height);
                if key == KeyCode::PageDown && top == bottom && self.chunk < last {
                    self.chunk += 1;
                    self.anchor = 0;
                } else if key == KeyCode::PageUp && top == 0 && self.chunk > 0 {
                    self.chunk -= 1;
                    self.anchor = usize::MAX;
                } else {
                    let next = if key == KeyCode::PageUp {
                        top.saturating_sub(height)
                    } else {
                        (top + height).min(bottom)
                    };
                    self.anchor = page.rows[next].start;
                }
            }
            _ => {}
        }
    }

    pub(super) fn draw(&mut self, frame: &mut Frame<'_>, flow: &Workflow) {
        let area = frame.area();
        if area.width < 32 || area.height < 8 {
            frame.render_widget(
                Paragraph::new("Raw logs · enlarge terminal · Esc back"),
                area,
            );
            return;
        }
        let view = flow.view();
        let count = view.members.len();
        let Some(member) = view.members.get(self.selected) else {
            frame.render_widget(Paragraph::new("No originals in this view."), area);
            return;
        };
        let record = &flow.dataset().records[member.record];
        let context_rows = if area.height >= 15 {
            count.min(5) as u16
        } else {
            0
        };
        let [header, context, detail, status, keys] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Length(context_rows),
            Constraint::Min(3),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .areas(area);
        let accent = Style::default().fg(Color::Cyan);
        frame.render_widget(
            Line::from(vec![
                Span::styled(" Raw logs ", accent.add_modifier(Modifier::BOLD)),
                Span::raw(format!(
                    "· #{} {}",
                    view.id,
                    excerpt(&view.title, usize::from(area.width))
                )),
            ]),
            header,
        );

        let slots = usize::from(context.height).min(count);
        let start = self
            .selected
            .saturating_sub(slots / 2)
            .min(count.saturating_sub(slots));
        let lines: Vec<_> = view.members[start..start + slots]
            .iter()
            .enumerate()
            .map(|(i, member)| {
                let row = &flow.dataset().records[member.record];
                let line = Line::raw(format!(
                    "  {}   {}",
                    excerpt(&row.id, 14),
                    excerpt(&row.text, usize::from(context.width))
                ));
                if start + i == self.selected {
                    line.style(Style::default().fg(Color::Black).bg(Color::Cyan))
                } else {
                    line
                }
            })
            .collect();
        frame.render_widget(Paragraph::new(lines), context);

        let border = Block::bordered().border_style(accent).title(format!(
            " Original {} of {} · {} ",
            self.selected + 1,
            count,
            excerpt(&record.id, usize::from(area.width))
        ));
        self.body = border.inner(detail);
        frame.render_widget(border, detail);
        let range = chunk_range(&record.text, self.chunk);
        let page = TextPage::new(&record.text[range.clone()], self.body.width);
        let height = usize::from(self.body.height);
        let top = page.top(self.anchor, height);
        let visible: Vec<_> = page
            .rows
            .iter()
            .skip(top)
            .take(height)
            .map(|row| Line::raw(&page.text[row.clone()]))
            .collect();
        frame.render_widget(Paragraph::new(visible), self.body);
        let location = if record.text.is_empty() {
            "Empty retained text".to_owned()
        } else {
            format!(
                "Bytes {}–{} of {}",
                range.start + 1,
                range.end,
                record.text.len()
            )
        };
        let mut status_text = format!(
            "{} · rows {}–{} of {}",
            location,
            top + 1,
            (top + height).min(page.rows.len()),
            page.rows.len()
        );
        let escape_hint = " · escapes shown";
        if status_text.chars().count() + escape_hint.chars().count() <= usize::from(status.width) {
            status_text.push_str(escape_hint);
        }
        frame.render_widget(Line::styled(status_text, accent), status);
        let legend = if area.width >= 68 {
            "↑/↓ or j/k original   PgUp/PgDn text   ←/→ chunk   Esc back"
        } else {
            "j/k log  PgUp/Dn  ←/→  Esc back"
        };
        frame.render_widget(Line::raw(legend), keys);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{buffer::Buffer, widgets::Widget};

    fn assert_page_renders(chunk: &str, expected: &[&str]) {
        let expected = Buffer::with_lines(expected.iter().copied());
        let area = expected.area;
        let page = TextPage::new(chunk, area.width);
        assert_eq!(page.rows.len(), usize::from(area.height));
        let visible: Vec<_> = page
            .rows
            .iter()
            .map(|row| Line::raw(&page.text[row.clone()]))
            .collect();
        let mut actual = Buffer::empty(area);
        Paragraph::new(visible).render(area, &mut actual);
        assert_eq!(actual, expected);
    }

    #[test]
    fn orphan_marks_render_at_the_start_of_a_record_line_and_chunk() {
        assert_page_renders(
            "\u{301}first\n\u{327}\u{301}next\ne\u{301}tail",
            &[r"\u{301}first", r"\u{327}\u{301}next", "e\u{301}tail"],
        );
        let record = format!("{}e\u{301}tail", "x".repeat(CHUNK_BYTES - 1));
        let range = chunk_range(&record, 1);
        assert_eq!(range, CHUNK_BYTES..CHUNK_BYTES + 6);
        assert_page_renders(&record[range], &[r"\u{301}tail"]);
    }

    #[test]
    fn escaped_controls_and_literal_backslashes_render_distinctly() {
        assert_page_renders("\t\\t", &[r"\t\\t"]);
        assert_page_renders("\\t\t", &[r"\\t\t"]);
    }

    #[test]
    fn chunks_partition_multibyte_text_without_rescanning_it() {
        let message = format!(
            "{}💾{}界end",
            "x".repeat(CHUNK_BYTES - 1),
            "é".repeat(CHUNK_BYTES)
        );
        let chunks: Vec<_> = (0..message.len().div_ceil(CHUNK_BYTES))
            .map(|i| &message[chunk_range(&message, i)])
            .collect();
        assert_eq!(chunks.concat(), message);
        assert_eq!(chunks[0].len(), CHUNK_BYTES - 1);
        assert!(chunks.iter().all(|c| c.len() <= CHUNK_BYTES + 3));
    }

    #[test]
    fn wrapping_preserves_whitespace_newlines_wide_characters_and_escapes() {
        let page = TextPage::new("ab  cd\n\n界e\u{301}z\n\t\x1b\u{202e}", 4);
        let rows: Vec<_> = page.rows.iter().map(|r| &page.text[r.clone()]).collect();
        assert_eq!(&rows[..4], &["ab  ", "cd", "", "界e\u{301}z"]);
        assert_eq!(rows[4..].concat(), "\\t\\u{1b}\\u{202e}");
    }

    #[test]
    fn page_keys_reach_all_rows_and_late_chunks_in_both_directions() {
        let text = format!("{}TAIL", "line\n".repeat(CHUNK_BYTES));
        let mut browser = RawBrowser::new(0);
        browser.body = Rect::new(0, 0, 12, 5);
        let mut seen_tail = false;
        for _ in 0..400 {
            let page = TextPage::new(&text[chunk_range(&text, browser.chunk)], browser.body.width);
            let top = page.top(browser.anchor, 5);
            seen_tail |= page
                .rows
                .iter()
                .skip(top)
                .take(5)
                .any(|r| page.text[r.clone()].contains("TAIL"));
            browser.move_text(KeyCode::PageDown, &text);
        }
        assert!(seen_tail);
        let last = browser.chunk;
        browser.move_text(KeyCode::Left, &text);
        assert_eq!(browser.chunk + 1, last);
        browser.move_text(KeyCode::Right, &text);
        assert_eq!(browser.chunk, last);
        for _ in 0..400 {
            browser.move_text(KeyCode::PageUp, &text);
        }
        assert_eq!((browser.chunk, browser.anchor), (0, 0));
    }

    #[test]
    fn resizing_keeps_the_anchor_in_the_visible_text() {
        let text = "a".repeat(CHUNK_BYTES);
        let mut browser = RawBrowser::new(0);
        browser.body = Rect::new(0, 0, 18, 6);
        browser.move_text(KeyCode::PageDown, &text);
        assert_eq!(browser.anchor, 108);
        for width in [9, 31, 80] {
            let page = TextPage::new(&text, width);
            let row = page.top(browser.anchor, 6);
            assert!(page.rows[row].contains(&browser.anchor));
        }
    }
}
