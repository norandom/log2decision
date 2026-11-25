//! Heap counting for the unit tests of this crate: a counting global allocator around
//! the system allocator, with counters per thread, so tests that run side by side do
//! not see each other. A test measures what one call allocates at its peak and
//! compares it with what that call asks the memory budget for.
//!
//! The counters are const-initialized cells, which never allocate themselves.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static CURRENT: Cell<usize> = const { Cell::new(0) };
    static PEAK: Cell<usize> = const { Cell::new(0) };
}

fn grew(bytes: usize) {
    let _ = CURRENT.try_with(|c| {
        let now = c.get() + bytes;
        c.set(now);
        let _ = PEAK.try_with(|p| p.set(p.get().max(now)));
    });
}

fn shrank(bytes: usize) {
    let _ = CURRENT.try_with(|c| c.set(c.get().saturating_sub(bytes)));
}

pub struct Counting;

// SAFETY: every call is forwarded unchanged to the system allocator; the
// counters are bookkeeping beside it.
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
        shrank(layout.size());
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let moved = unsafe { System.realloc(pointer, layout, size) };
        if !moved.is_null() {
            if size >= layout.size() {
                grew(size - layout.size());
            } else {
                shrank(layout.size() - size);
            }
        }
        moved
    }
}

/// The heap growth of `run` at its peak, on this thread.
pub fn measure<T>(run: impl FnOnce() -> T) -> (T, u64) {
    let before = CURRENT.with(Cell::get);
    PEAK.with(|p| p.set(before));
    let result = run();
    (result, (PEAK.with(Cell::get) - before) as u64)
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;
