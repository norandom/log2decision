//! Observe real store batches without changing query or membership semantics.
use l2d_core::patterns::PatternStore;
use l2d_core::polars::prelude::DataFrame;
use l2d_core::selections::SelectionStore;
use l2d_core::{L2dError, MarkStore, MemoryBudget};
use serde::Serialize;
use std::cell::RefCell;

#[derive(Default, Debug, Serialize)]
pub(super) struct StoreBatchReport {
    pub operation: &'static str,
    pub configured_max_rows: u64,
    pub batches: u64,
    pub rows: u64,
    pub min_delivered_rows: Option<u64>,
    pub max_delivered_rows: Option<u64>,
    pub sizing: &'static str,
}
impl StoreBatchReport {
    fn new(operation: &'static str, limit: u64) -> Self {
        Self {
            operation,
            configured_max_rows: limit,
            sizing: "actual delivered frames; byte-budget cuts and final short frame included",
            ..Default::default()
        }
    }
    fn note(&mut self, rows: u64) {
        self.batches += 1;
        self.rows += rows;
        self.min_delivered_rows = Some(self.min_delivered_rows.map_or(rows, |n| n.min(rows)));
        self.max_delivered_rows = Some(self.max_delivered_rows.map_or(rows, |n| n.max(rows)));
    }
}
pub(super) struct ObservedStore<'a> {
    pub store: &'a l2d_store::Store,
    pub selection: RefCell<StoreBatchReport>,
    pub catalogue: RefCell<StoreBatchReport>,
}
impl<'a> ObservedStore<'a> {
    pub fn new(store: &'a l2d_store::Store, limit: u64) -> Self {
        Self {
            store,
            selection: RefCell::new(StoreBatchReport::new("selection text scan", limit)),
            catalogue: RefCell::new(StoreBatchReport::new("catalogue text scan", limit)),
        }
    }
}
impl MarkStore for ObservedStore<'_> {
    fn store_name(&self) -> &str {
        self.store.store_name()
    }
    fn content_sha256(&self, b: &mut MemoryBudget) -> Result<String, L2dError> {
        MarkStore::content_sha256(self.store, b)
    }
    fn phrase_ids(
        &self,
        phrase: &str,
        id_bytes: u64,
        b: &mut MemoryBudget,
        each: &mut dyn FnMut(&str),
    ) -> Result<(), L2dError> {
        MarkStore::phrase_ids(self.store, phrase, id_bytes, b, each)
    }
    fn text_batches(
        &self,
        beside: u64,
        per_row: u64,
        b: &mut MemoryBudget,
        each: &mut dyn FnMut(&DataFrame) -> Result<u64, L2dError>,
    ) -> Result<(), L2dError> {
        MarkStore::text_batches(self.store, beside, per_row, b, &mut |frame| {
            self.selection.borrow_mut().note(frame.height() as u64);
            each(frame)
        })
    }
}
impl SelectionStore for ObservedStore<'_> {
    fn fts5_trigram_ids(
        &self,
        expression: &str,
        b: &mut MemoryBudget,
        each: &mut dyn FnMut(&str) -> Result<(), L2dError>,
    ) -> Result<u64, L2dError> {
        {
            self.selection.borrow_mut().sizing = "FTS5 index cursor; no selection frame batches";
            self.store.fts5_trigram_ids(expression, b, each)
        }
    }
}
impl PatternStore for ObservedStore<'_> {
    fn pattern_snapshot(&self, b: &mut MemoryBudget) -> Result<String, L2dError> {
        self.store.pattern_snapshot(b)
    }
    fn pattern_batches(
        &self,
        beside: u64,
        per_row: u64,
        b: &mut MemoryBudget,
        each: &mut dyn FnMut(&DataFrame) -> Result<u64, L2dError>,
    ) -> Result<(), L2dError> {
        self.store
            .pattern_batches(beside, per_row, b, &mut |frame| {
                self.catalogue.borrow_mut().note(frame.height() as u64);
                each(frame)
            })
    }
}
