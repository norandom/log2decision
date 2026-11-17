use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
struct Counting;
static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
fn grew(n: usize) {
    let now = CURRENT.fetch_add(n, Ordering::Relaxed) + n;
    PEAK.fetch_max(now, Ordering::Relaxed);
}
// SAFETY: all allocations are forwarded unchanged to System; atomics only count bytes.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            grew(l.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(l) };
        if !p.is_null() {
            grew(l.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) };
        CURRENT.fetch_sub(l.size(), Ordering::Relaxed);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, l, n) };
        if !q.is_null() {
            if n >= l.size() {
                grew(n - l.size());
            } else {
                CURRENT.fetch_sub(l.size() - n, Ordering::Relaxed);
            }
        }
        q
    }
}
#[global_allocator]
static ALLOCATOR: Counting = Counting;
use l2d_core::{
    InputSummary, L2dError, Label, LogRecord, LogsFrame, MarkQuery, MarkSet, MemoryBudget,
    MemorySettings,
};
use l2d_decide::close_enough::{MarkHistory, ReachSettings, group_marks};
use l2d_decide::exploration::{CharacterTrigram, ExplorationEncoder, Representation};
fn budget() -> MemoryBudget {
    MemoryBudget::observe_with(&MemorySettings::default(), || Ok(8 * 1024 * 1024 * 1024)).unwrap()
}
struct Failure(usize);
impl ExplorationEncoder for Failure {
    fn profile(&self) -> &str {
        CharacterTrigram.profile()
    }
    fn scratch_bytes(&self) -> u64 {
        65536
    }
    fn unsupported_reason(&self) -> &str {
        "failure"
    }
    fn encode(&mut self, text: &str) -> Result<Option<Representation>, L2dError> {
        self.0 += 1;
        if self.0 == 2 {
            Err(L2dError::Refused {
                reason: "injected encoder failure".into(),
            })
        } else {
            CharacterTrigram.encode(text)
        }
    }
}
#[test]
fn whole_input_history_and_group_output_are_admitted_before_copies_and_errors_restore_hold() {
    let rows: Vec<_> = (0..500)
        .map(|i| LogRecord {
            text: format!("record={i} {}", "αβγ/".repeat(80)),
            label: Label::Unknown,
            group: None,
            first_position: i,
            occurrences: 1,
        })
        .collect();
    let logs = LogsFrame::from_records(&rows).unwrap();
    drop(rows);
    let summary = InputSummary {
        source_path: "memory-fixture.log".into(),
        records: 500,
        distinct_texts: 500,
        content_sha256: "a".repeat(64),
        skipped: vec![],
        label_conflicts: 0,
    };
    let mut b = budget();
    b.hold(logs.frame().estimated_size() as u64);
    let mut marks = MarkSet::mark(
        "memory",
        Label::Bad,
        MarkQuery::Phrase {
            phrase: "record".into(),
        },
        &logs,
        &summary,
        &b,
    )
    .unwrap();
    let before = CURRENT.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let prior = b.held_bytes();
    let history = MarkHistory::from_input(&logs, &summary, &marks, None, &mut b).unwrap();
    let s = ReachSettings {
        radius: 1.,
        top_k: 10,
        minimum_labelled_support: 2,
        expected_profile: CharacterTrigram.profile().into(),
    };
    let grouping = group_marks(&history, &s, &mut CharacterTrigram, &mut b).unwrap();
    let json = serde_json::to_value(&grouping).unwrap();
    let bytes = serde_json::to_vec(&json).unwrap();
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(before);
    let admitted = b.held_bytes() - prior;
    println!(
        "MarkSet input/group heap growth={peak}; reserved={admitted}; JSON={}",
        bytes.len()
    );
    assert!((peak as u64) < admitted);
    assert_eq!(grouping.groups[0].size, 500);
    grouping.deselect(0, &mut marks, None, &mut b).unwrap();
    assert_eq!(marks.selected_count(), 0);
    let held = b.held_bytes();
    let mut encoder = Failure(0);
    assert!(group_marks(&history, &s, &mut encoder, &mut b).is_err());
    assert_eq!(encoder.0, 2);
    assert_eq!(b.held_bytes(), held);
    b.hold(b.usable_bytes() - 1);
    let held = b.held_bytes();
    let mut encoder = Failure(0);
    assert!(group_marks(&history, &s, &mut encoder, &mut b).is_err());
    assert_eq!(encoder.0, 0);
    assert_eq!(b.held_bytes(), held);
    assert!(MarkHistory::from_input(&logs, &summary, &marks, None, &mut b).is_err());
    assert_eq!(b.held_bytes(), held);
}
