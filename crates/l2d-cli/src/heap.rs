//! A heap counter for measurements, compiled only with the cargo feature `heap-report`
//! (off by default; the shipped binary has no counting allocator).
//!
//! With the feature on, every allocation is counted beside the system allocator, and
//! when the environment variable `L2D_HEAP_REPORT` is set the binary prints one more
//! line on standard error before it exits: `heap peak: <bytes> bytes`. That is the
//! largest number of bytes that were allocated at one time over the whole run, which is
//! what the memory limit of requirement 12.9 is about, less the 32 MiB the limit keeps
//! for the tool itself (code, stacks, the allocator's own bookkeeping).
//!
//! Build and run, in a target directory of its own so that the ordinary binary is not
//! replaced:
//!
//! ```text
//! cargo build --release -j 4 -p l2d-cli --features heap-report --target-dir D:\BuildArtifacts\log2decision\heap-report
//! $env:L2D_HEAP_REPORT = "1"
//! D:\BuildArtifacts\log2decision\heap-report\release\l2d.exe triage --input ... --baseline ...
//! ```

use std::alloc::{GlobalAlloc, Layout, System};
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grew(bytes: usize) {
    let now = CURRENT.fetch_add(bytes, Ordering::Relaxed) + bytes;
    PEAK.fetch_max(now, Ordering::Relaxed);
}

// SAFETY: every call is forwarded unchanged to the system allocator; the counters are
// only bookkeeping beside it.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            grew(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            grew(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
        CURRENT.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let moved = unsafe { System.realloc(pointer, layout, new_size) };
        if !moved.is_null() {
            if new_size >= layout.size() {
                grew(new_size - layout.size());
            } else {
                CURRENT.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            }
        }
        moved
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Prints the heap peak of the run on standard error when `L2D_HEAP_REPORT` is set.
pub fn report() {
    if std::env::var_os("L2D_HEAP_REPORT").is_some() {
        let _ = writeln!(
            std::io::stderr(),
            "heap peak: {} bytes",
            PEAK.load(Ordering::Relaxed)
        );
    }
}
