//! Keyboard and terminal adapter over the same immutable views used for receipts.
mod draw;
mod layout;
mod raw;

use std::{
    collections::HashMap,
    io::{self, IsTerminal},
    path::{Path, PathBuf},
};

use crossterm::{
    cursor::Show,
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyEventKind,
        KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
    },
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, enable_raw_mode},
};
use ratatui::{Terminal, backend::CrosstermBackend, layout::Position};

use crate::flow::{Budget, Region, Search, SearchKind, Step, Workflow};
use layout::Panes;
use raw::RawBrowser;

const REGIONS: [(char, Region, &str); 7] = [
    ('1', Region::AOnly, "A minus B"),
    ('2', Region::Both, "A and B"),
    ('3', Region::BOnly, "B minus A"),
    ('4', Region::Neither, "Neither A nor B"),
    ('i', Region::Recommended, "C recommended"),
    ('o', Region::Pass, "C pass"),
    ('u', Region::Unknown, "C unknown"),
];

#[derive(Clone, Copy, PartialEq)]
enum Menu {
    Hosts,
    Hours,
    Branches,
    Coverage,
}

enum Mode {
    Browse,
    Search,
    Menu(Menu, usize),
    Evidence {
        scroll: u16,
        lines: Vec<ratatui::text::Line<'static>>,
    },
    Raw {
        browser: RawBrowser,
        evidence_scroll: Option<u16>,
    },
    Help(u16),
    Allocate(String),
    SaveAs(String),
}

#[derive(Default)]
struct ViewFrame {
    selected: usize,
    draft: String,
    syntax: usize,
}

#[derive(Clone, Copy)]
enum Divider {
    Fields,
    Search,
}

struct App {
    flow: Workflow,
    mode: Mode,
    frames: HashMap<usize, ViewFrame>,
    session: Option<PathBuf>,
    revision: Option<String>,
    notice: String,
    sidebar: u16,
    search_rows: u16,
    panes: Panes,
    geometry: Option<(ratatui::layout::Rect, u16, u16)>,
    drag: Option<Divider>,
    fields_focused: bool,
    region_index: usize,
}

impl App {
    fn new(flow: Workflow, session: Option<&Path>) -> Self {
        let mut frames = HashMap::new();
        frames.insert(flow.view().id, ViewFrame::default());
        Self {
            flow,
            frames,
            mode: Mode::Browse,
            session: session.map(Path::to_owned),
            revision: None,
            notice: String::new(),
            sidebar: 30,
            search_rows: 3,
            panes: Panes::default(),
            geometry: None,
            drag: None,
            fields_focused: false,
            region_index: 0,
        }
    }

    fn local(&self) -> &ViewFrame {
        &self.frames[&self.flow.view().id]
    }
    fn local_mut(&mut self) -> &mut ViewFrame {
        self.frames.entry(self.flow.view().id).or_default()
    }

    fn drill(&mut self, step: Step) -> Result<(), String> {
        self.flow.drill(step)?;
        self.local_mut();
        self.mode = Mode::Browse;
        self.notice.clear();
        Ok(())
    }

    fn go_to(&mut self, view: usize) -> Result<(), String> {
        self.flow.goto(view)?;
        self.local_mut();
        self.mode = Mode::Browse;
        self.notice.clear();
        Ok(())
    }

    fn menu_len(&self, kind: Menu) -> usize {
        match kind {
            Menu::Hosts => self.flow.view().hosts.len(),
            Menu::Hours => self.flow.view().hours.len(),
            Menu::Branches => self.flow.views().len(),
            Menu::Coverage => 3,
        }
    }

    fn open_menu(&mut self, menu: Menu) -> Result<(), String> {
        if matches!(menu, Menu::Hosts | Menu::Hours) && !self.flow.dataset().occurrences_known {
            return Err("Occurrence locations are unavailable in this source.".into());
        }
        let selected = if menu == Menu::Branches {
            self.flow.view().id
        } else {
            0
        };
        self.mode = Mode::Menu(menu, selected);
        Ok(())
    }

    fn choose(&mut self, menu: Menu, index: usize) -> Result<(), String> {
        let view = self.flow.view();
        let step = match menu {
            Menu::Branches => return self.go_to(index),
            Menu::Coverage => Step::Region(REGIONS[4 + index].1),
            Menu::Hosts => Step::Host(
                view.hosts
                    .keys()
                    .nth(index)
                    .cloned()
                    .ok_or("There are no host values in this view.")?,
            ),
            Menu::Hours => Step::Hour(
                view.hours
                    .keys()
                    .nth(index)
                    .cloned()
                    .ok_or("There are no hour values in this view.")?,
            ),
        };
        self.drill(step)
    }

    fn move_rows(&mut self, by: isize) {
        let last = self.flow.view().members.len().saturating_sub(1);
        let frame = self.local_mut();
        frame.selected = frame.selected.saturating_add_signed(by).min(last);
    }

    fn save(&mut self, path: PathBuf, expected: Option<String>) -> Result<(), String> {
        let revision = crate::session::save(&path, &self.flow.receipt(), expected.as_deref())?;
        self.revision = Some(revision);
        self.notice = format!("Saved {} to {}", self.flow.view().title, path.display());
        self.session = Some(path);
        self.mode = Mode::Browse;
        Ok(())
    }

    // User input errors stay on screen and keep the current immutable view alive.
    fn handle(&mut self, key: KeyEvent) -> bool {
        if !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
            return false;
        }
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
        if !matches!(self.mode, Mode::Browse | Mode::Raw { .. }) && code == KeyCode::Esc {
            self.mode = Mode::Browse;
            self.notice.clear();
            return Ok(false);
        }
        match &mut self.mode {
            Mode::Search => {
                let local = self.local_mut();
                match code {
                    KeyCode::Tab => local.syntax = (local.syntax + 1) % SEARCH_MODES.len(),
                    KeyCode::Enter => {
                        let search = Search {
                            kind: search_kind(local.syntax),
                            query: local.draft.clone(),
                        };
                        self.drill(Step::Search(search))?;
                    }
                    _ => edit(&mut local.draft, key),
                }
            }
            Mode::Menu(menu, selected) => {
                let (menu, mut selected) = (*menu, *selected);
                let len = self.menu_len(menu);
                match code {
                    KeyCode::Enter if len > 0 => self.choose(menu, selected)?,
                    KeyCode::Char('i' | 'o' | 'u') if menu == Menu::Coverage => {
                        let idx = match code {
                            KeyCode::Char('i') => 0,
                            KeyCode::Char('o') => 1,
                            _ => 2,
                        };
                        self.choose(menu, idx)?;
                    }
                    _ => {
                        if let Some(by) = motion(code) {
                            selected = selected
                                .saturating_add_signed(by)
                                .min(len.saturating_sub(1));
                            self.mode = Mode::Menu(menu, selected);
                        }
                    }
                }
            }
            Mode::Allocate(input) if code == KeyCode::Enter => {
                let numbers = input
                    .split_whitespace()
                    .map(str::parse::<usize>)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| "Enter three nonnegative slot counts.")?;
                let [general, unfamiliar, audit] = numbers.as_slice() else {
                    return Err(
                        "Enter slots for general, unfamiliar and audit, for example 2 1 1.".into(),
                    );
                };
                general
                    .checked_add(*unfamiliar)
                    .and_then(|n| n.checked_add(*audit))
                    .ok_or("The total exceeds the supported budget.")?;
                self.drill(Step::Allocate(Budget {
                    general: *general,
                    unfamiliar: *unfamiliar,
                    audit: *audit,
                    ..Default::default()
                }))?;
            }
            Mode::SaveAs(input) if code == KeyCode::Enter => {
                if input.trim().is_empty() {
                    return Err("Enter a receipt path.".into());
                }
                let path = PathBuf::from(&*input);
                self.save(path, None)?;
            }
            Mode::Allocate(input) | Mode::SaveAs(input) => edit(input, key),
            Mode::Evidence { scroll, .. } => {
                if code == KeyCode::Char('r') {
                    let evidence_scroll = Some(*scroll);
                    self.mode = Mode::Raw {
                        browser: RawBrowser::new(self.local().selected),
                        evidence_scroll,
                    };
                } else if matches!(code, KeyCode::Enter | KeyCode::Char('b' | 'q')) {
                    self.mode = Mode::Browse;
                } else if let Some(by) = motion(code) {
                    *scroll = usize::from(*scroll)
                        .saturating_add_signed(by)
                        .min(u16::MAX.into()) as u16;
                }
            }
            Mode::Raw {
                browser,
                evidence_scroll,
            } => {
                if matches!(code, KeyCode::Esc | KeyCode::Char('b' | 'q' | 'r')) {
                    self.mode = match *evidence_scroll {
                        Some(scroll) => Mode::Evidence {
                            scroll,
                            lines: draw::evidence_lines(self),
                        },
                        None => Mode::Browse,
                    };
                } else {
                    browser.handle(code, &self.flow);
                }
            }
            Mode::Help(scroll) => {
                if let Some(by) = motion(code) {
                    *scroll = usize::from(*scroll)
                        .saturating_add_signed(by)
                        .min(u16::MAX.into()) as u16;
                } else {
                    self.mode = Mode::Browse;
                }
            }
            Mode::Browse => return self.browse(code),
        }
        Ok(false)
    }

    fn browse(&mut self, code: KeyCode) -> Result<bool, String> {
        let region = match code {
            KeyCode::Char(ch) => REGIONS.iter().find(|r| r.0 == ch),
            _ => None,
        };
        if self.flow.view().is_final()
            && (region.is_some() || matches!(code, KeyCode::Char('f' | '/' | 'h' | 't' | 'c')))
        {
            return Err("Already allocated. Press b to return to the candidate and change its budget or filters.".into());
        }
        if let Some((_, region, _)) = region {
            self.drill(Step::Region(*region))?;
            return Ok(false);
        }
        match code {
            KeyCode::Char('q') => return Ok(true),
            KeyCode::Char('r') => {
                if self.flow.view().members.is_empty() {
                    return Err("No originals in this view.".into());
                }
                self.mode = Mode::Raw {
                    browser: RawBrowser::new(self.local().selected),
                    evidence_scroll: None,
                };
                self.notice.clear();
            }
            KeyCode::Char('?') => self.mode = Mode::Help(0),
            KeyCode::Char('/') => {
                self.mode = Mode::Search;
                self.notice.clear();
            }
            KeyCode::Char('h') => self.open_menu(Menu::Hosts)?,
            KeyCode::Char('t') => self.open_menu(Menu::Hours)?,
            KeyCode::Char('c') => self.open_menu(Menu::Coverage)?,
            KeyCode::Char('v') => self.open_menu(Menu::Branches)?,
            KeyCode::Esc | KeyCode::Backspace | KeyCode::Char('b') => {
                if self.flow.back() {
                    self.local_mut();
                    self.notice.clear();
                } else {
                    self.notice = "At the investigation root.".into();
                }
            }
            KeyCode::Char('f') => {
                self.mode = Mode::Allocate("2 1 1".into());
                self.notice.clear();
            }
            KeyCode::Char(ch @ ('s' | 'S')) => match (&self.session, ch) {
                (Some(path), 's') => self.save(path.clone(), self.revision.clone())?,
                _ => {
                    self.mode = Mode::SaveAs(String::new());
                    self.notice.clear();
                }
            },
            KeyCode::Tab if self.panes.fields.width > 0 && !self.flow.view().is_final() => {
                self.fields_focused = !self.fields_focused
            }
            KeyCode::Enter if self.fields_focused && !self.flow.view().is_final() => {
                self.drill(Step::Region(REGIONS[self.region_index].1))?
            }
            KeyCode::Enter if !self.flow.view().members.is_empty() => {
                self.mode = Mode::Evidence {
                    scroll: 0,
                    lines: draw::evidence_lines(self),
                }
            }
            KeyCode::Char('[') => self.sidebar = self.sidebar.saturating_sub(2).max(20),
            KeyCode::Char(']') => self.sidebar = (self.sidebar + 2).min(68),
            KeyCode::Char('-') => self.search_rows = self.search_rows.saturating_sub(1).max(3),
            KeyCode::Char('+') => self.search_rows = (self.search_rows + 1).min(10),
            _ => {
                if let Some(by) = motion(code) {
                    if self.fields_focused {
                        self.region_index = self.region_index.saturating_add_signed(by).min(6);
                    } else {
                        self.move_rows(by);
                    }
                }
            }
        }
        Ok(false)
    }

    fn mouse(&mut self, event: MouseEvent) {
        if let Mode::Raw { browser, .. } = &mut self.mode {
            match event.kind {
                MouseEventKind::ScrollDown => browser.handle(KeyCode::Down, &self.flow),
                MouseEventKind::ScrollUp => browser.handle(KeyCode::Up, &self.flow),
                _ => {}
            }
            return;
        }
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
                self.fields_focused = false;
                let first_row = panes.results.y + 2;
                if event.row >= first_row && event.row + 1 < panes.results.bottom() {
                    let page = usize::from(panes.results.height.saturating_sub(3)).max(1);
                    let index =
                        self.local().selected / page * page + usize::from(event.row - first_row);
                    if index < self.flow.view().members.len() {
                        self.local_mut().selected = index;
                    }
                }
            }
            _ => {}
        }
    }
}

fn motion(code: KeyCode) -> Option<isize> {
    Some(match code {
        KeyCode::Down | KeyCode::Char('j') => 1,
        KeyCode::Up | KeyCode::Char('k') => -1,
        KeyCode::PageDown => 10,
        KeyCode::PageUp => -10,
        KeyCode::Home => isize::MIN,
        KeyCode::End => isize::MAX,
        _ => return None,
    })
}

const SEARCH_MODES: [SearchKind; 4] = [
    SearchKind::Spl,
    SearchKind::Literal,
    SearchKind::Glob,
    SearchKind::Regex,
];

fn search_kind(index: usize) -> SearchKind {
    SEARCH_MODES[index]
}

fn edit(input: &mut String, key: KeyEvent) {
    match key.code {
        KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => input.clear(),
        KeyCode::Backspace => {
            input.pop();
        }
        KeyCode::Char(ch) if !ch.is_control() && input.len() + ch.len_utf8() <= 4096 => {
            input.push(ch)
        }
        _ => {}
    }
}

// A failed initialization, draw, read or unwind still releases terminal ownership.
struct RawTerminal {
    #[cfg(not(windows))]
    restore_cooked: bool,
    #[cfg(windows)]
    console: [(std::os::windows::io::RawHandle, u32); 2],
}

impl RawTerminal {
    fn capture() -> io::Result<Self> {
        #[cfg(windows)]
        {
            use std::os::windows::io::AsRawHandle;
            use windows_sys::Win32::System::Console::GetConsoleMode;
            let mut console = [
                (io::stdin().as_raw_handle(), 0),
                (io::stdout().as_raw_handle(), 0),
            ];
            for (handle, mode) in &mut console {
                // The standard handles are borrowed for the process lifetime; mode is writable.
                if unsafe { GetConsoleMode(*handle, mode) } == 0 {
                    return Err(io::Error::last_os_error());
                }
            }
            Ok(Self { console })
        }
        #[cfg(not(windows))]
        {
            Ok(Self {
                restore_cooked: !crossterm::terminal::is_raw_mode_enabled()?,
            })
        }
    }
}
impl Drop for RawTerminal {
    fn drop(&mut self) {
        let _ = execute!(
            io::stdout(),
            DisableMouseCapture,
            LeaveAlternateScreen,
            Show
        );
        #[cfg(windows)]
        for &(handle, mode) in &self.console {
            // Reapply every caller-owned flag, including echo, mouse input and VT processing.
            unsafe { windows_sys::Win32::System::Console::SetConsoleMode(handle, mode) };
        }
        #[cfg(not(windows))]
        if self.restore_cooked {
            let _ = crossterm::terminal::disable_raw_mode();
        }
    }
}

pub fn run(
    workflow: Workflow,
    session: Option<&Path>,
    revision: Option<String>,
) -> Result<(), String> {
    if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        return Err("The workbench requires an interactive terminal on stdin and stdout (use ssh -t for a remote command).".into());
    }
    let _guard = RawTerminal::capture().map_err(|e| e.to_string())?;
    enable_raw_mode().map_err(|e| e.to_string())?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture).map_err(|e| e.to_string())?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout)).map_err(|e| e.to_string())?;
    let mut app = App::new(workflow, session);
    app.revision = revision;
    'workbench: loop {
        terminal
            .draw(|frame| draw::screen(frame, &mut app))
            .map_err(|e| e.to_string())?;
        loop {
            match event::read().map_err(|e| e.to_string())? {
                Event::Key(key)
                    if matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) =>
                {
                    if app.handle(key) {
                        break 'workbench;
                    }
                    break;
                }
                Event::Mouse(mouse) if mouse.kind != MouseEventKind::Moved => {
                    app.mouse(mouse);
                    break;
                }
                Event::Resize(_, _) => break,
                _ => {}
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;

    fn demo() -> App {
        let flow = Workflow::new(
            crate::input::demo(),
            8,
            Search {
                kind: SearchKind::Glob,
                query: "*login*".into(),
            },
        )
        .unwrap();
        App::new(flow, None)
    }

    fn press(app: &mut App, code: KeyCode) {
        assert!(!app.handle(KeyEvent::new(code, KeyModifiers::NONE)));
    }

    fn snapshot(app: &mut App, width: u16, height: u16) -> String {
        let mut term = Terminal::new(TestBackend::new(width, height)).unwrap();
        term.draw(|f| draw::screen(f, app)).unwrap();
        let buf = term.backend().buffer();
        (0..height)
            .map(|y| (0..width).map(|x| buf[(x, y)].symbol()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn keyboard_walkthrough_keeps_parent_denominators_and_scopes_evidence() {
        let mut app = demo();
        press(&mut app, KeyCode::Char('1'));
        let d = snapshot(&mut app, 120, 32);
        assert!(d.contains("6 originals · 11 occurrences"));
        assert!(d.contains("6/12 originals"));
        press(&mut app, KeyCode::Char('i'));
        let c = snapshot(&mut app, 120, 32);
        assert!(c.contains("3/6 originals") && c.contains("3/12 originals"));
        press(&mut app, KeyCode::Char('h'));
        let host_index = app
            .flow
            .view()
            .hosts
            .keys()
            .position(|k| k.as_deref() == Some("edge-01"))
            .unwrap();
        for _ in 0..host_index {
            press(&mut app, KeyCode::Down);
        }
        press(&mut app, KeyCode::Enter);
        press(&mut app, KeyCode::Enter);
        let evidence = snapshot(&mut app, 120, 32);
        assert!(evidence.contains("Matching occurrences  2"));
        assert!(evidence.contains("edge-01") && !evidence.contains("idp-01"));
        press(&mut app, KeyCode::Esc);
        press(&mut app, KeyCode::Char('b'));
        press(&mut app, KeyCode::Char('b'));
        press(&mut app, KeyCode::Char('u'));
        press(&mut app, KeyCode::Enter);
        let abstention = snapshot(&mut app, 120, 32);
        assert!(abstention.contains("R08") && abstention.contains("abstained"));
    }

    #[test]
    fn raw_navigation_restores_record_selection_evidence_scroll_and_receipt() {
        let mut app = demo();
        press(&mut app, KeyCode::Char('1'));
        press(&mut app, KeyCode::Down);
        let original_receipt = serde_json::to_value(app.flow.receipt()).unwrap();

        press(&mut app, KeyCode::Enter);
        press(&mut app, KeyCode::Down);
        press(&mut app, KeyCode::Char('r'));
        assert!(snapshot(&mut app, 80, 24).contains("Original 2 of 6"));
        press(&mut app, KeyCode::Down);
        assert!(snapshot(&mut app, 80, 24).contains("Original 3 of 6"));
        press(&mut app, KeyCode::Esc);
        let Mode::Evidence { scroll, lines } = &app.mode else {
            panic!("raw browser should return to the evidence that opened it");
        };
        assert_eq!(*scroll, 1);
        assert_eq!(lines[0].to_string(), "R04");
        press(&mut app, KeyCode::Esc);
        press(&mut app, KeyCode::Char('r'));
        press(&mut app, KeyCode::End);
        press(&mut app, KeyCode::Char('b'));
        assert!(matches!(app.mode, Mode::Browse));
        assert_eq!(app.local().selected, 1);
        assert_eq!(
            serde_json::to_value(app.flow.receipt()).unwrap(),
            original_receipt
        );
    }

    #[test]
    fn a_second_allocation_keeps_the_final_work_set_and_pool_attribution() {
        let mut app = demo();
        for key in [KeyCode::Char('1'), KeyCode::Char('f'), KeyCode::Enter] {
            press(&mut app, key);
        }
        let original = serde_json::to_value(app.flow.receipt()).unwrap();
        assert!(
            app.flow
                .view()
                .members
                .iter()
                .all(|m| m.allocation.is_some())
        );
        let screen = snapshot(&mut app, 120, 32);
        assert!(screen.contains("b candidate") && !screen.contains("f allocate"));
        for key in ['f', 'f', '1', '/', 'h', 't', 'c'] {
            press(&mut app, KeyCode::Char(key));
            assert!(matches!(app.mode, Mode::Browse));
            assert!(app.notice.contains("Already allocated"));
        }
        press(&mut app, KeyCode::Enter);
        assert!(snapshot(&mut app, 120, 32).contains("Allocated pool"));
        assert_eq!(original, serde_json::to_value(app.flow.receipt()).unwrap());
        press(&mut app, KeyCode::Esc);
        press(&mut app, KeyCode::Char('b'));
        press(&mut app, KeyCode::Char('f'));
        assert!(matches!(app.mode, Mode::Allocate(_)));
    }

    #[test]
    fn report_forms_label_the_diagnostic_markset_even_when_the_source_title_is_clipped() {
        use crate::flow::{CProducer, Recommendation};
        for producer in [CProducer::Unavailable, CProducer::PolicyInvestigateOrReview] {
            let mut data = crate::input::demo();
            data.c_producer = producer;
            data.title = "long source name ".repeat(12);
            for record in &mut data.records {
                record.recommendation = Recommendation::Unknown;
            }
            let flow = Workflow::new(
                data,
                8,
                Search {
                    kind: SearchKind::Glob,
                    query: "*login*".into(),
                },
            )
            .unwrap();
            let mut app = App::new(flow, None);
            assert!(snapshot(&mut app, 62, 18).contains("Diag."));
            press(&mut app, KeyCode::Enter);
            let evidence = snapshot(&mut app, 62, 18);
            assert!(evidence.contains("Diagnostic MarkSet") && evidence.contains("C unknown"));
        }
    }

    #[test]
    fn record_preview_bounds_untrusted_text_and_location_projection() {
        let mut data = crate::input::demo();
        let item = &mut data.records[0];
        let location = item.occurrences[0].clone();
        item.text = "\u{1b}[31m".repeat(3000);
        item.occurrences = vec![location; 100];
        let flow = Workflow::new(
            data,
            8,
            Search {
                kind: SearchKind::Glob,
                query: "*login*".into(),
            },
        )
        .unwrap();
        let mut app = App::new(flow, None);
        let idx = app
            .flow
            .view()
            .members
            .iter()
            .position(|m| m.record == 0)
            .unwrap();
        app.local_mut().selected = idx;
        press(&mut app, KeyCode::Enter);
        let Mode::Evidence { lines, .. } = &app.mode else {
            panic!("expected evidence")
        };
        let preview = lines
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(preview.contains("first 24 matching locations"));
        assert!(preview.contains("\\u{1b}") && !preview.contains('\u{1b}'));
        assert!(preview.len() < 12_000);
        assert!(lines.len() < 36);
    }

    #[test]
    fn search_drafts_selection_and_resizes_are_local_to_their_view() {
        let mut app = demo();
        press(&mut app, KeyCode::Char('1'));
        press(&mut app, KeyCode::Down);
        press(&mut app, KeyCode::Char('/'));
        for c in "no matching fixture".chars() {
            press(&mut app, KeyCode::Char(c));
        }
        press(&mut app, KeyCode::Esc);
        let id = app.flow.view().id;
        let views = app.flow.views().len();
        snapshot(&mut app, 120, 32);
        let previous = app.panes;
        press(&mut app, KeyCode::Char(']'));
        press(&mut app, KeyCode::Char('+'));
        snapshot(&mut app, 120, 32);
        assert!(
            app.panes.fields.width > previous.fields.width
                && app.panes.search.height > previous.search.height
        );
        assert_eq!(app.panes.fields.right(), app.panes.results.x);
        assert_eq!(app.flow.views().len(), views);
        press(&mut app, KeyCode::Char('/'));
        press(&mut app, KeyCode::Enter);
        let empty = snapshot(&mut app, 80, 24);
        assert!(empty.contains("No matching originals"));
        press(&mut app, KeyCode::Char('r'));
        assert!(matches!(app.mode, Mode::Browse));
        assert_eq!(app.notice, "No originals in this view.");
        press(&mut app, KeyCode::Char('b'));
        let narrow = snapshot(&mut app, 62, 18);
        assert!(narrow.contains("no matching fixture"));
        assert_eq!(app.flow.view().id, id);
        assert_eq!(app.local().selected, 1);
        assert_eq!(app.panes.fields.width, 0);
    }
}
