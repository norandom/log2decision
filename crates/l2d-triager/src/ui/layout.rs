use ratatui::layout::Rect;
use taffy::prelude::{AvailableSpace, FlexDirection, Size, Style, TaffyTree, auto, length};

#[derive(Clone, Copy, Default)]
pub(crate) struct Panes {
    pub header: Rect,
    pub search: Rect,
    pub time: Rect,
    pub fields: Rect,
    pub results: Rect,
    pub status: Rect,
}

impl Panes {
    pub fn solve(area: Rect, sidebar: u16, search_rows: u16) -> Result<Self, taffy::TaffyError> {
        let width = area.width;
        let height = area.height;
        let search_rows = search_rows.min(height.saturating_sub(6));
        let sidebar = if width >= 78 {
            sidebar.min(width / 2)
        } else {
            0
        };
        let time_rows = if height >= 18 { 4 } else { 0 };
        let mut tree: TaffyTree<()> = TaffyTree::new();
        let row = |rows: u16| Style {
            size: Size {
                width: auto(),
                height: length(f32::from(rows)),
            },
            flex_shrink: 0.0,
            ..Default::default()
        };
        let header = tree.new_leaf(row(1))?;
        let search = tree.new_leaf(row(search_rows))?;
        let time = tree.new_leaf(row(time_rows))?;
        let fields = tree.new_leaf(Style {
            size: Size {
                width: length(f32::from(sidebar)),
                height: auto(),
            },
            flex_shrink: 0.0,
            ..Default::default()
        })?;
        let flex = Style {
            flex_grow: 1.0,
            ..Default::default()
        };
        let results = tree.new_leaf(flex.clone())?;
        let main = tree.new_with_children(flex, &[fields, results])?;
        let status = tree.new_leaf(row(2))?;
        let root = tree.new_with_children(
            Style {
                flex_direction: FlexDirection::Column,
                size: Size {
                    width: length(f32::from(width)),
                    height: length(f32::from(height)),
                },
                ..Default::default()
            },
            &[header, search, time, main, status],
        )?;
        tree.compute_layout(
            root,
            Size {
                width: AvailableSpace::Definite(f32::from(width)),
                height: AvailableSpace::Definite(f32::from(height)),
            },
        )?;
        let rect = |node, parent: Rect| -> Result<Rect, taffy::TaffyError> {
            let layout = tree.layout(node)?;
            let x = layout.location.x.round() as u16;
            let y = layout.location.y.round() as u16;
            let right = (layout.location.x + layout.size.width).round() as u16;
            let bottom = (layout.location.y + layout.size.height).round() as u16;
            Ok(Rect::new(
                parent.x.saturating_add(x),
                parent.y.saturating_add(y),
                right.saturating_sub(x),
                bottom.saturating_sub(y),
            )
            .intersection(parent))
        };
        let body = rect(main, area)?;
        Ok(Self {
            header: rect(header, area)?,
            search: rect(search, area)?,
            time: rect(time, area)?,
            fields: rect(fields, body)?,
            results: rect(results, body)?,
            status: rect(status, area)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Task 3.2: the solved edges become clipped cell rectangles whose shared edges
    /// meet on the same cell, at every frame size and pane setting the keys reach. The
    /// rectangles are the ones drawing and hit testing use.
    #[test]
    fn shared_pane_edges_meet_on_one_cell_at_every_size() {
        for width in 32..=220u16 {
            for height in [8u16, 9, 12, 17, 18, 24, 36, 60, 120] {
                for (sidebar, search) in [(20u16, 3u16), (34, 3), (38, 4), (50, 10), (68, 10)] {
                    let area = Rect::new(0, 0, width, height);
                    let p = Panes::solve(area, sidebar, search).unwrap();
                    let all = [p.header, p.search, p.time, p.fields, p.results, p.status];
                    for pane in all {
                        assert_eq!(pane.intersection(area), pane, "{width}x{height}: {pane:?}");
                    }
                    assert_eq!(p.header.y, 0);
                    assert_eq!(p.search.y, p.header.bottom());
                    assert_eq!(p.time.y, p.search.bottom());
                    assert_eq!(p.results.y, p.time.bottom(), "{width}x{height}");
                    assert_eq!(p.status.y, p.results.bottom(), "{width}x{height}");
                    assert_eq!(p.status.bottom(), height, "{width}x{height}");
                    assert_eq!(p.results.right(), width);
                    if p.fields.width > 0 {
                        assert_eq!(p.fields.y, p.results.y);
                        assert_eq!(p.fields.height, p.results.height);
                        assert_eq!(p.fields.right(), p.results.x, "{width}x{height}");
                        assert!(p.fields.width <= width / 2);
                    } else {
                        assert_eq!(p.results.x, 0);
                        assert!(width < 78);
                    }
                    let covered: u32 = all.iter().map(|pane| pane.area()).sum();
                    assert_eq!(covered, u32::from(width) * u32::from(height));
                }
            }
        }
    }
}
