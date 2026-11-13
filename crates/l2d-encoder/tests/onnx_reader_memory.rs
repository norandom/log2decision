//! Task 2.4, review round 1: what the model reader allocates does not grow with the
//! model file.
//!
//! This test binary counts heap bytes with an allocator of its own, and holds one test,
//! so that nothing else allocates while it measures.

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use common::Model;
use l2d_encoder::ModelInterface;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

struct Counting;

// SAFETY: every call is passed to the system allocator unchanged; the counters beside
// it do not touch the memory.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let moved = unsafe { System.realloc(pointer, layout, new_size) };
        if !moved.is_null() {
            LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
            let live = LIVE.fetch_add(new_size, Ordering::Relaxed) + new_size;
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        moved
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// The heap bytes `read` adds at its peak to what was live before it.
fn peak_of<T>(read: impl FnOnce() -> T) -> (T, usize) {
    let before = LIVE.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let result = read();
    (result, PEAK.load(Ordering::Relaxed).saturating_sub(before))
}

/// The ceiling the reader's documentation states for what it keeps, with room for the
/// result and a message.
const CEILING: usize = 256 * 1024;

#[test]
fn the_reader_allocates_no_more_for_a_file_of_very_many_tiny_nodes() {
    // The tiny model: a few hundred bytes of heap.
    let tiny = Model::tiny().bytes();
    let (read, small) = peak_of(|| ModelInterface::from_onnx(&tiny));
    read.unwrap();
    assert!(small < 4 * 1024, "{small} bytes for the tiny model");

    // 400,000 more nodes of a few bytes each, about 4 MiB of file: the same heap.
    let many = Model {
        extra_nodes: 400_000,
        ..Model::tiny()
    }
    .bytes();
    assert!(many.len() > 3 << 20, "{} bytes of file", many.len());
    let (read, peak) = peak_of(|| ModelInterface::from_onnx(&many));
    read.unwrap();
    println!(
        "file {} bytes, 400,005 nodes: peak heap {peak} bytes (tiny model: {small} bytes)",
        many.len()
    );
    assert!(
        peak < 4 * 1024,
        "{peak} bytes for {} bytes of file",
        many.len()
    );

    // The same number of nodes that all read the input: a refusal, and a short one.
    let readers = Model {
        extra_nodes: 400_000,
        extra_read_input: true,
        ..Model::tiny()
    }
    .bytes();
    let (read, peak) = peak_of(|| ModelInterface::from_onnx(&readers));
    let message = read.unwrap_err().to_string();
    println!("refusal of {} bytes, peak heap {peak} bytes", message.len());
    assert!(message.len() < 400, "{message}");
    assert!(peak < 4 * 1024, "{peak} bytes");

    // The most the reader keeps: 256 graph inputs of 8 dimensions each, all listed as
    // initializers too (as old exporters write them), plus the encoder's own input.
    let mut graph = Vec::new();
    for at in 0..255 {
        let name = format!("weight-{at}");
        common::bytes_field(
            &mut graph,
            11,
            &common::value_info(&name, common::FLOAT, &[1; 8]),
        );
        common::bytes_field(&mut graph, 5, &common::float_weights(&name, 1, 0));
    }
    let mut full = Vec::new();
    common::bytes_field(&mut full, 7, &graph);
    let (read, peak) = peak_of(|| ModelInterface::from_onnx(&full));
    // No input that is not an initializer: refused, after everything was kept.
    read.unwrap_err();
    println!("255 inputs of 8 dimensions: peak heap {peak} bytes");
    assert!(peak < CEILING, "{peak} bytes at the ceiling");

    // A million initializers beside 255 listed inputs and the encoder's own: the pass
    // over the initializers looks each name up once, so it is linear in the file. With
    // one comparison per initializer and listed input (255 million of them) this took
    // 4.5 s in a debug build; the bound leaves room for a busy machine.
    let mut graph = Vec::new();
    for at in 0..255 {
        let name = format!("weight-{at}");
        common::bytes_field(
            &mut graph,
            11,
            &common::value_info(&name, common::FLOAT, &[1]),
        );
        common::bytes_field(&mut graph, 5, &common::float_weights(&name, 0, 0));
    }
    common::bytes_field(
        &mut graph,
        11,
        &common::value_info("x", common::FLOAT, &[2, 3, 1, 8]),
    );
    common::bytes_field(
        &mut graph,
        12,
        &common::value_info("e", common::FLOAT, &[2, 4]),
    );
    let mut other = Vec::new();
    common::bytes_field(&mut other, 8, b"v");
    for _ in 0..1_000_000 {
        common::bytes_field(&mut graph, 5, &other);
    }
    let mut million = Vec::new();
    common::bytes_field(&mut million, 7, &graph);
    let started = std::time::Instant::now();
    let (read, peak) = peak_of(|| ModelInterface::from_onnx(&million));
    let took = started.elapsed();
    // The one real input is found among the 256; nothing reads it, which is the refusal.
    let message = read.unwrap_err().to_string();
    assert!(message.contains("the input `x` must be read"), "{message}");
    println!(
        "file {} bytes, 1,000,255 initializers, 256 listed inputs: {took:?}, peak heap {peak} \
         bytes",
        million.len()
    );
    assert!(peak < CEILING, "{peak} bytes");
    // Measured 0.9 s in a debug build on the reference host.
    assert!(took < std::time::Duration::from_secs(3), "{took:?}");
}
