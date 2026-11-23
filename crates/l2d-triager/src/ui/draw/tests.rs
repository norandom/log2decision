use ratatui::{Terminal, backend::TestBackend};

use super::*;
use crate::flow::{Search, Step, Workflow};

fn render(app: &mut App, width: u16) -> String {
    let mut terminal = Terminal::new(TestBackend::new(width, 24)).unwrap();
    terminal.draw(|frame| screen(frame, app)).unwrap();
    let buffer = terminal.backend().buffer();
    (0..24)
        .map(|y| {
            (0..width)
                .map(|x| buffer[(x, y)].symbol())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn searched_rows_lead_with_raw_text_at_both_terminal_sizes() {
    let flow = Workflow::new(
        crate::input::demo(),
        8,
        Search {
            kind: SearchKind::Glob,
            query: "*login*".into(),
        },
    )
    .unwrap();
    let mut app = App::new(flow, None);
    app.drill(Step::Search(Search {
        kind: SearchKind::Spl,
        query: "declined OR rejected".into(),
    }))
    .unwrap();
    for width in [80, 120] {
        let output = render(&mut app, width);
        assert!(output.contains("SPL  declined OR rejected"));
        assert!(output.contains("Raw log (_raw)"));
        let row = output.lines().find(|line| line.contains("R04")).unwrap();
        assert!(
            row.split("R04")
                .next()
                .unwrap()
                .contains("session verification declined")
        );
        assert_eq!(output.contains("Occ."), width == 120);
    }

    // A long expression still shows the insertion point in a one-line editor.
    app.mode = Mode::Search;
    app.local_mut().draft = format!("{}id=R04", "login OR ".repeat(20));
    assert!(render(&mut app, 80).contains("id=R04▏"));
}
