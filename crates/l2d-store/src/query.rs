//! Searching a store by text, label and source, and reading its rows back
//! (requirements 5.2 and 5.5; task 1.9). The search by vector is in [`crate::vector`].
//!
//! # What can be asked
//!
//! A [`Filter`] names the logs: a label, a source, phrases the text must contain, the
//! exact text. Everything given must hold. An empty filter names every log.
//!
//! - **Phrases** ([`Filter::contains`]): the text must contain every phrase, ignoring
//!   case (defined below) unless [`Filter::case_exact`] is set, in which case byte for
//!   byte.
//! - **The exact text** ([`Filter::equals`]) is compared byte for byte, whatever
//!   [`Filter::case_exact`] says.
//! - **Label and source** are compared whole and restrict the rows in SQL, through the
//!   indexes `logs_label` and `logs_source` (requirement 5.5). When the text index is
//!   used as well it leads, and label and source are checked on the rows it yields;
//!   the two SQL indexes are then not used.
//!
//! # Who decides a match
//!
//! The comparison in this module decides, for every phrase and for the exact text:
//! each log that is left is read and compared here. The text index (FTS5 with the
//! `trigram` tokenizer) never decides; it only narrows the logs that are read. It is
//! given short pieces of the phrases, [`INDEX_PIECE_CHARS`] characters each and at most
//! [`MAX_INDEX_PIECES`] in one search: of a phrase its first characters, then its last,
//! then its middle ones, the first piece of every phrase before the second of any,
//! and no piece twice (two pieces with the same lower-case form count as one).
//! Every log that contains the phrase contains its pieces, so the index can only leave
//! too many logs, never too few. Short pieces spread over a phrase were measured to
//! leave nearly as few logs as the whole phrase (see [`INDEX_PIECE_CHARS`]), and what
//! the index takes in memory grows with every character it is given.
//!
//! A phrase of one or two characters has no piece: it is answered by reading every log
//! that the rest of the filter leaves (a scan). [`Store::explain`] names the pieces and
//! the phrases that are scanned.
//!
//! # Ignoring case
//!
//! There is one definition, applied to every phrase whatever its length and whether
//! or not the index narrowed the logs. Every character of the text and of the phrase
//! is replaced by its lower-case form, and the text must then contain the phrase,
//! character for character. The lower-case form of a character is its Unicode
//! lower-case mapping (`char::to_lowercase` of the Rust standard library) when that
//! mapping is a single character, and the character itself otherwise. So:
//!
//! - `Ä` matches `ä`, `Σ` matches `σ`, the Kelvin sign matches `k`, `ǅ` matches `ǆ`
//!   and `Ǆ`, Cherokee `Ꭰ` matches `ꭰ`;
//! - nothing is expanded or normalised: `ß` does not match `SS` (it does match `ẞ`),
//!   `ſ` does not match `s`, final `ς` does not match `σ`, a letter does not match the
//!   same letter with a combining mark;
//! - `İ` (U+0130), the one character whose lower-case mapping is two characters, stays
//!   as it is and matches only itself.
//!
//! The text index folds case by a table of its own, which differs: it folds `ς` to
//! `σ` and maps U+FFFE and U+FFFF to U+FFFD (it leaves more logs, which the comparison
//! here then drops), and it does not know the case pairs of newer scripts, Cherokee
//! among them (it would leave too few). A piece of a phrase that ignores case is
//! therefore taken only from characters for which the index is known to agree with
//! the definition above ([`index_folds_alike`]: characters without case, and the
//! letters of basic Latin, Greek and Cyrillic; a test compares the index with the
//! definition for every character of Unicode that has a case partner, so a change of
//! either table fails it). A phrase without three such characters in a row is
//! scanned. Case-exact
//! phrases and the exact text may give the index any characters: a text that contains
//! the phrase byte for byte contains the piece under any folding.
//!
//! # A phrase is text, never syntax
//!
//! Nothing a caller passes becomes part of an SQL statement: label, source and the
//! query of the text index are bound parameters. Within the query of the text index a
//! piece is written as an FTS5 string, in double quotes with every double quote of the
//! piece doubled, which is the only character that means anything inside such a
//! string; so `AND`, `OR`, `NOT`, `NEAR`, `*`, `^`, `:` and parentheses are searched for
//! as characters. No `LIKE` or `GLOB` is used, so `%` and `_` mean nothing either.
//! The comparisons done here are plain substring and equality tests.
//!
//! An empty phrase is [`StoreError::InvalidQuery`]. A phrase with a NUL character, or
//! one longer than [`crate::MAX_TEXT_BYTES`], matches no log: no stored text holds a
//! NUL or is that long. Phrases may otherwise be of any length and number.
//!
//! # Rows
//!
//! Rows come back as a Polars frame, the stored rows frame ([`stored_schema`]): the
//! columns of the rows frame that [`Store::add`] takes, with the `id` in front, in the
//! order the logs were stored. The encoder revision of every row is that of the store
//! ([`Store::identity`]) and is not repeated per row.
//!
//! # Memory (requirement 12)
//!
//! A search holds its own share of the caller's budget for as long as it runs, on top
//! of what the caller holds, and is refused before it reads anything, with the figures,
//! when that is not left ([`Explain::held_bytes`] says what it is for a filter):
//!
//! - [`SEARCH_FIXED_BYTES`] for SQLite itself;
//! - what the open store still retains above its idle level
//!   ([`Store::retained_bytes`]; only after a reopening that failed);
//! - for a search with a phrase or an exact text, [`TEXT_READ_FACTOR`] times the bytes
//!   of the longest stored text ([`Store::longest_text_bytes`]), because the texts are
//!   read one at a time to be compared, and the bytes of the phrases that ignore case,
//!   of which the search keeps a lower-case copy;
//! - for the pieces given to the text index, [`PHRASE_BYTE_BYTES`] for each of their
//!   bytes, and [`INDEX_TEXT_FACTOR`] times the bytes of the longest stored text for
//!   each of their characters: the index reads, for the log it is at, the list of all
//!   places where each trigram of a piece occurs in that log, and in a text that
//!   repeats itself that list is as long as the text.
//!
//! At the limit of [`crate::MAX_TEXT_BYTES`] (1 MiB) for the longest text, a search
//! with one phrase of twelve characters or more holds 16 MiB + 3 MiB + 1.5 MiB + 24 MiB
//! = 44.5 MiB, and the most any search holds for SQLite and the texts is 59 MiB (four
//! pieces of four characters of four bytes each). In a store whose longest text has
//! 4 KiB the same searches hold 17.6 and 24.1 MiB.
//!
//! A search ends by releasing SQLite's page cache, so an open store keeps no more
//! after a search than [`Store::retained_bytes`] says.
//!
//! Reading rows back needs all of them in memory at once, since they are returned as
//! one frame. [`Store::text_search_with`] and [`Store::vectors_with`] therefore go over
//! the matching rows twice within one read transaction: first they only count them and
//! add up what the frame will take ([`Matching::read_bytes`]: [`READ_FIXED_BYTES`]
//! once, [`READ_ROW_BYTES`] and the vector for each row, twice the bytes of its text,
//! source and group, and twice the longest text for SQLite's own copy of the text
//! being read), ask the budget for that ([`MemoryBudget::require_whole`]) and are
//! refused with the estimate and the limit when it does not fit; only then are the
//! rows read, into columns of exactly that many rows. The frame returned is the
//! caller's to count from then on.

use l2d_core::polars::prelude::{
    Column, DataFrame, DataType, Field, Float32Type, IntoColumn, ListBuilderTrait,
    ListPrimitiveChunkedBuilder, Schema, SchemaExt, StringChunkedBuilder,
};
use l2d_core::{L2dError, Label, MemoryBudget, MemorySettings};
use rusqlite::types::ToSql;

use crate::{
    EMPTY, GROUP, IDLE_BYTES, LABEL, MAX_TEXT_BYTES, OCCURRENCES, SCHEMA_VERSION, SOURCE, Store,
    StoreError, TEXT, TEXT_LONGER_THAN_INPUT, TRUNCATED, VECTOR, VectorOrigin, failed,
};

/// Column of the stored rows frame and of the hits frame: the identity of the log
/// ([`crate::log_id`]).
pub const ID: &str = "id";
/// Column of the hits frame: the squared L2 distance between the query vector and the
/// stored vector.
pub const DISTANCE: &str = "distance";
/// Column of the hits frame: which of the query vectors the hit belongs to, from 0.
pub const QUERY: &str = "query";

/// What a search holds of the memory limit for SQLite itself for as long as it runs:
/// the page cache of the connection ([`crate::CACHE_BYTES`]) and the statements.
///
/// Measured on the reference host in release mode (`csic_searches_measured` in
/// `tests/store_csic.rs`): over the 25,608 logs of HTTP CSIC 2010 with 128-dimension
/// vectors, SQLite's heap (`sqlite3_memory_highwater`) peaked at 8.4 MiB in the
/// searches without a phrase, which is the filled page cache. What the text index and
/// the reading of texts take is held on top ([`PHRASE_BYTE_BYTES`],
/// [`INDEX_TEXT_FACTOR`], [`TEXT_READ_FACTOR`]).
pub const SEARCH_FIXED_BYTES: u64 = 16 * 1024 * 1024;

/// What a search holds of the memory limit for every byte of the pieces it gives the
/// text index, whatever the stored texts are.
///
/// The index opens a reader for every trigram of a piece in every segment it consists
/// of, and each reader keeps pages of the index in memory. Measured on the reference
/// host in release mode, before the index was given pieces only: the peak of SQLite's
/// heap for one whole phrase, the start of the longest log of HTTP CSIC 2010, searched
/// in a store of its 25,608 logs (the same whether the store was filled by 1, 16 or 256
/// adds):
///
/// | Phrase, bytes | 3 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 891 |
/// |---------------|---|---|----|----|----|-----|-----|-----|-----|
/// | heap, MiB | 8.4 | 8.8 | 9.4 | 10.5 | 12.6 | 12.1 | 17.4 | 27.5 | 44.3 |
///
/// That is 40 to 70 KiB per byte on top of the page cache (8.4 MiB at most); the
/// factor is about twice the highest rate. A search now gives the index at most
/// [`MAX_INDEX_PIECES`] pieces of [`INDEX_PIECE_CHARS`] characters, 64 bytes at the
/// very most. It was measured on that one store: an index of many more segments (a
/// much larger store) was not measured.
pub const PHRASE_BYTE_BYTES: u64 = 128 * 1024;

/// How many characters of a phrase one piece for the text index has (a phrase of
/// three characters is its own piece).
///
/// Measured on the reference host (`index_piece_selectivity_measured` in
/// `tests/store_csic.rs`): 182 phrases of 24 characters cut from the logs of HTTP CSIC
/// 2010 (from the start, the middle and the end of a log), and how many of the 25,608
/// logs the index leaves to be read, by what it is given:
///
/// | Given to the index | characters | median | mean | index, ms |
/// |--------------------|------------|--------|------|-----------|
/// | the whole phrase | 24 | 346 | 4,530 | 11 |
/// | its first 4 | 4 | 4,431 | 6,861 | 0.9 |
/// | its last 4 | 4 | 14,456 | 12,812 | 1.5 |
/// | its first 8 | 8 | 609 | 5,362 | 3.6 |
/// | its last 8 | 8 | 4,609 | 11,024 | 4.1 |
/// | its first 16 | 16 | 435 | 4,676 | 7.7 |
/// | first and last 3 | 6 | 1,556 | 5,384 | 0.7 |
/// | first and last 4 | 8 | 622 | 5,163 | 1.5 |
/// | first, middle and last 3 | 9 | 495 | 4,886 | 0.9 |
/// | first, middle and last 4 | 12 | 473 | 4,751 | 1.9 |
/// | first and last 6 | 12 | 449 | 4,916 | 2.4 |
/// | first, middle and last 6 | 18 | 435 | 4,646 | 3.6 |
/// | first, middle and last 8 | 24 | 346 | 4,531 | 7.0 |
///
/// Where a piece is taken matters more than how long it is, and no single place is
/// good for every phrase; three pieces of four characters leave about as few logs as
/// sixteen characters in a row, for less memory ([`INDEX_TEXT_FACTOR`]) and less time
/// in the index.
pub const INDEX_PIECE_CHARS: usize = 4;

/// The most pieces the text index is given in one search. With several phrases every
/// phrase gets its first piece before any gets a second.
pub const MAX_INDEX_PIECES: usize = 4;

/// What a search holds of the memory limit for every character of the pieces it gives
/// the text index, per byte of the longest stored text.
///
/// For the log it is at, the index reads the list of all the places where each
/// trigram of a piece occurs in that log. In a text that repeats itself (a run of one
/// character, as an overflow payload or padding has it) a trigram occurs at every
/// place, so each list is as long as the text, and the buffer it is read into doubles
/// as it grows: a text just above a power of two takes twice its length per list.
/// Measured on the reference host in release mode (`index_piece_memory_measured` in
/// `tests/store_csic.rs`): the peak of SQLite's heap, in MiB, for pieces that occur in
/// one long text stored among 200 short ones, by the kind and length of that text and
/// by the characters given to the index:
///
/// | Long text | one piece of 3 | of 4 | of 6 | of 8 | of 16 | two of 4 | three of 4 | four of 4 |
/// |-----------|----------------|------|------|------|-------|----------|------------|-----------|
/// | one character, 64 KiB | 0.4 | 0.7 | 0.9 | 1.2 | 2.2 | 1.1 | 1.5 | 1.9 |
/// | one character, 1 MiB less 16 bytes | 2.3 | 4.3 | 6.3 | 8.3 | 16.4 | 7.3 | 10.3 | 13.3 |
/// | one character, 1 MiB | 3.3 | 7.3 | 11.3 | 15.3 | 30.4 | 13.3 | 19.3 | 25.3 |
/// | cycle of 2 characters, 1 MiB | 1.7 | 4.3 | 6.3 | 8.3 | 16.4 | 7.3 | 10.3 | 13.3 |
/// | cycle of 3 characters, 1 MiB | 1.1 | 2.4 | 3.8 | 4.8 | 8.9 | 3.9 | 5.5 | 7.0 |
/// | cycle of 16 characters, 1 MiB | 0.4 | 0.7 | 1.1 | 1.5 | 3.1 | 1.1 | 1.5 | 1.9 |
/// | one two-byte character, 1 MiB | 1.7 | 3.7 | 5.8 | 7.8 | 15.3 | 6.8 | 9.8 | 12.8 |
/// | log-like, 1 MiB | 0.3 | 0.3 | 0.4 | 0.4 | 0.8 | 0.4 | 0.5 | 0.5 |
/// | random ASCII, 1 MiB | 0.2 | 0.2 | 0.2 | 0.3 | 0.3 | 0.2 | 0.2 | 0.2 |
///
/// The worst case is a run of one single-byte character of exactly the longest length
/// a store takes (sixteen bytes less halve it): about 2 MiB for each trigram and 3 MiB
/// more for each piece, that is 7.3 MiB for one piece of four characters, 19.3 MiB
/// for three (what one long phrase is given) and 25.3 MiB for four (the most). Twice
/// the longest text for every character given covers that (8, 24 and 32 MiB: 0.91,
/// 0.80 and 0.79 used); the fixed share and [`PHRASE_BYTE_BYTES`] are held beside it,
/// and the test asserts for every search that SQLite stays within what was held.
/// A piece longer than four characters is never given; the columns for 6, 8 and 16
/// show what it would cost. Longer texts were measured before the limit of 1 MiB,
/// with the same rates up to 8 MiB (58 MiB for one piece of 8 characters).
pub const INDEX_TEXT_FACTOR: u64 = 2;

/// What a search holds of the memory limit, per byte of the longest stored text, when
/// it reads texts to compare them: once for SQLite's copy of the text being read, and
/// twice for the lower-case copy made of it for phrases that ignore case, which is
/// the most that copy can have ([`fold`]).
pub const TEXT_READ_FACTOR: u64 = 3;

/// What the budget is asked for once for a frame of rows read back, whatever its rows:
/// the first buffers of its string columns (8 KiB each) and its validity maps.
pub const READ_FIXED_BYTES: u64 = 64 * 1024;

/// The fixed part of what the budget is asked for per row that is read back: the views
/// of the five string columns (16 bytes each), the 64 characters of the id, and the
/// fixed-width values. The other part is the vector and twice the bytes of text,
/// source and group.
pub const READ_ROW_BYTES: u64 = 256;

/// How many columns [`row_columns`] selects.
const ROW_COLUMN_COUNT: usize = 9;

/// The columns of a stored row as they are selected from a file of the format `format`,
/// in the order [`RowsBuilder::push`] reads them; `logs.vector` follows where the vector
/// is read. A file of format 1 has no columns for the flags `text_longer_than_input`
/// and `empty`: they are selected as absent values, which is what such a store knows
/// of them.
pub(crate) fn row_columns(format: u32) -> String {
    let flags = if format >= SCHEMA_VERSION {
        "logs.text_longer_than_input, logs.empty"
    } else {
        "NULL, NULL"
    };
    format!(
        "logs.id, logs.text, logs.source, logs.occurrences, logs.label, logs.group_key, \
         logs.truncated, {flags}"
    )
}

/// Which logs a search or an export is about. Everything that is given must hold; the
/// default names every log.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Filter {
    /// Only logs with this label.
    pub label: Option<Label>,
    /// Only logs from this source, compared whole and byte for byte.
    pub source: Option<String>,
    /// Phrases the text must contain, all of them. Case is ignored (see the module
    /// description) unless `case_exact` is set. A phrase is text: no character in it
    /// has a special meaning.
    pub contains: Vec<String>,
    /// Whether the phrases of `contains` must be contained byte for byte.
    pub case_exact: bool,
    /// The whole text, byte for byte.
    pub equals: Option<String>,
}

/// How a search finds its logs ([`Store::explain`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Explain {
    /// Whether the text index narrows the logs.
    pub text_index: bool,
    /// The pieces of the phrases (and of the exact text) that the text index is given.
    pub index_pieces: Vec<String>,
    /// The phrases (and the exact text) of which the text index is given no piece:
    /// they are answered by reading every log the rest of the filter leaves.
    pub scanned_phrases: Vec<String>,
    /// The bytes of the longest stored text, which the share below is sized by; 0 when
    /// the search reads no text.
    pub longest_text_bytes: u64,
    /// What a search with this filter holds of the budget for SQLite and for reading
    /// texts for as long as it runs (see the module description).
    pub held_bytes: u64,
    /// SQLite's own description of how it reads the rows (`EXPLAIN QUERY PLAN`), one
    /// step per entry.
    pub steps: Vec<String>,
}

/// The logs a filter names, counted without reading them back
/// ([`Store::matching_with`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Matching {
    /// How many logs match (at most the limit given).
    pub rows: u64,
    /// What reading them back asks of the budget, beside what the search holds
    /// ([`Explain::held_bytes`]).
    pub read_bytes: u64,
}

/// The column names and types of the stored rows frame, in order: `id`, then the
/// columns of the rows frame ([`crate::rows_schema`]).
///
/// Unlike in the rows frame that is added, `text_longer_than_input` and `empty` can be
/// absent here: a store of format 1 never recorded them, and an absent value means
/// unknown, not false.
pub fn stored_schema() -> Schema {
    let mut fields = vec![Field::new(ID.into(), DataType::String)];
    fields.extend(crate::rows_schema().iter_fields());
    Schema::from_iter(fields)
}

/// The column names and types of the hits frame of [`Store::nearest`], in order:
/// `query`, `distance`, then the columns of the stored rows frame without the vector.
pub fn hits_schema() -> Schema {
    let mut fields = vec![
        Field::new(QUERY.into(), DataType::UInt32),
        Field::new(DISTANCE.into(), DataType::Float32),
    ];
    fields.extend(
        stored_schema()
            .iter_fields()
            .filter(|field| field.name().as_str() != VECTOR),
    );
    Schema::from_iter(fields)
}

/// What a stored row is estimated to take in a frame.
pub(crate) fn row_bytes(text: u64, source: u64, group: u64, vector: u64) -> u64 {
    READ_ROW_BYTES + 2 * (text + source + group) + vector
}

/// The lower-case form of a character as "ignoring case" means it (see the module
/// description): its Unicode lower-case mapping when that is one character, the
/// character itself otherwise.
pub fn fold_char(character: char) -> char {
    let mut lower = character.to_lowercase();
    match (lower.next(), lower.next()) {
        (Some(one), None) => one,
        _ => character,
    }
}

/// `text` with every character in its lower-case form ([`fold_char`]).
///
/// The copy is allocated once, at exactly its length (counted in a first pass), and
/// never grows. It has at most twice the bytes of `text`: an ASCII character stays one
/// byte, and any other character has two bytes or more and its lower-case form four
/// at most. (Today the most is one and a half times: `Ⱥ`, U+023A, two bytes, becomes
/// U+2C65, three.)
pub fn fold(text: &str) -> String {
    if text.is_ascii() {
        return text.to_ascii_lowercase();
    }
    let bytes: usize = text
        .chars()
        .map(|character| fold_char(character).len_utf8())
        .sum();
    let mut lower = String::with_capacity(bytes);
    for character in text.chars() {
        lower.push(fold_char(character));
    }
    lower
}

/// Whether the text index is known to fold `character` as [`fold_char`] does, for
/// every character with the same lower-case form: only such characters are given to
/// the index for a phrase that ignores case.
///
/// Two kinds of character qualify:
///
/// - a character without case, that is one the standard library maps to itself in
///   both directions: only the character itself has its lower-case form, so a text
///   that matches holds that very character, whatever the index does with it;
/// - a letter of the blocks whose case pairs are old and complete: Basic Latin,
///   Latin-1 Supplement and Latin Extended-A (below U+0180), the Greek letters U+0386
///   to U+03CE, and the Cyrillic letters U+0400 to U+045F.
///
/// Other cased letters are left out. The index's table was made for an older version
/// of Unicode than the standard library's and lacks the case pairs added since: whole
/// scripts (Cherokee, Georgian Mtavruli, Osage, Adlam) and single letters inside old
/// blocks (`ƛ`, ten IPA letters such as `ɡ` and `ɪ`, `Ϳ`, Cyrillic U+0528 to U+052F).
/// `the_text_index_never_narrows_what_ignoring_case_accepts` in `tests/search.rs`
/// checks every character of Unicode that has a case partner against the index.
pub fn index_folds_alike(character: char) -> bool {
    fn itself(mut mapped: impl Iterator<Item = char>, character: char) -> bool {
        mapped.next() == Some(character) && mapped.next().is_none()
    }
    let caseless =
        itself(character.to_lowercase(), character) && itself(character.to_uppercase(), character);
    caseless
        || character < '\u{0180}'
        || ('\u{0386}'..='\u{03CE}').contains(&character)
        || ('\u{0400}'..='\u{045F}').contains(&character)
}

/// A piece as a string of the query language of the text index.
fn quoted(piece: &str) -> String {
    let mut quoted = String::with_capacity(piece.len() + 2);
    quoted.push('"');
    for character in piece.chars() {
        if character == '"' {
            quoted.push('"');
        }
        quoted.push(character);
    }
    quoted.push('"');
    quoted
}

/// The pieces of `phrase` the text index may be given, the most useful first: its
/// first [`INDEX_PIECE_CHARS`] characters, its last, the ones in its middle. With
/// `any_character` unset, only pieces of characters the index folds as the comparison
/// does ([`index_folds_alike`]). Empty for a phrase of fewer than three characters or
/// without such a piece.
fn pieces_of(phrase: &str, any_character: bool) -> Vec<&str> {
    let characters = phrase.chars().count();
    if characters < 3 {
        return Vec::new();
    }
    let width = INDEX_PIECE_CHARS.min(characters);
    let centre = (characters - width) / 2;
    // One pass, without a list of the characters (a phrase may be as long as a text):
    // the byte offsets of the last `width` characters, and how many usable characters
    // end at the one the pass is at. A piece is its first character's number and its
    // bytes.
    let mut starts = [0usize; INDEX_PIECE_CHARS];
    let mut run = 0usize;
    let mut first: Option<(usize, &str)> = None;
    let mut last: Option<(usize, &str)> = None;
    let mut middle: Option<(usize, &str)> = None;
    for (number, (at, character)) in phrase.char_indices().enumerate() {
        starts[number % width] = at;
        run = if any_character || index_folds_alike(character) {
            run + 1
        } else {
            0
        };
        if run < width {
            continue;
        }
        let begins = number + 1 - width;
        let piece = (
            begins,
            &phrase[starts[begins % width]..at + character.len_utf8()],
        );
        first = first.or(Some(piece));
        last = Some(piece);
        if middle.is_none_or(|(other, _)| begins.abs_diff(centre) < other.abs_diff(centre)) {
            middle = Some(piece);
        }
    }
    let mut chosen: Vec<(usize, &str)> = Vec::new();
    for piece in [first, last, middle].into_iter().flatten() {
        if !chosen.iter().any(|(begins, _)| *begins == piece.0) {
            chosen.push(piece);
        }
    }
    chosen.into_iter().map(|(_, piece)| piece).collect()
}
/// A filter in the form the search uses.
#[derive(Debug)]
pub(crate) struct Compiled<'a> {
    /// A phrase that no stored text can hold: nothing matches.
    pub nothing: bool,
    /// The query of the text index, if it narrows the logs.
    match_query: Option<String>,
    /// The pieces in `match_query`.
    pieces: Vec<String>,
    label: Option<&'static str>,
    source: Option<String>,
    /// Phrases the text must contain byte for byte.
    exact: Vec<&'a str>,
    /// Lower-cased phrases the lower-cased text must contain: the search's own copies.
    folded: Vec<String>,
    equals: Option<&'a str>,
    /// The phrases of which the index has no piece.
    scanned: Vec<&'a str>,
}

impl<'a> Compiled<'a> {
    pub(crate) fn new(filter: &'a Filter) -> Result<Self, StoreError> {
        let mut compiled = Self {
            nothing: false,
            match_query: None,
            pieces: Vec::new(),
            label: filter.label.map(Label::as_str),
            source: filter.source.clone(),
            exact: Vec::new(),
            folded: Vec::new(),
            equals: None,
            scanned: Vec::new(),
        };
        let impossible =
            |phrase: &str| phrase.contains('\0') || phrase.len() as u64 > MAX_TEXT_BYTES;
        // The pieces each phrase offers the index, most useful first.
        let mut offered: Vec<(&'a str, Vec<&'a str>)> = Vec::new();
        for phrase in &filter.contains {
            if phrase.is_empty() {
                return Err(StoreError::InvalidQuery {
                    detail: "a phrase to search for is empty; every text contains the empty \
                             phrase, so leave it out"
                        .to_string(),
                });
            }
            if impossible(phrase) {
                compiled.nothing = true;
            } else if filter.case_exact {
                compiled.exact.push(phrase);
                offered.push((phrase, pieces_of(phrase, true)));
            } else {
                compiled.folded.push(fold(phrase));
                offered.push((phrase, pieces_of(phrase, false)));
            }
        }
        if let Some(text) = &filter.equals {
            if impossible(text) {
                compiled.nothing = true;
            } else {
                compiled.equals = Some(text);
                offered.push((text, pieces_of(text, true)));
            }
        }

        // The first piece of every phrase, then the second of every phrase, and so on.
        // A piece whose lower-case form was given already (by this phrase or another)
        // would narrow nothing and cost a slot and memory: the phrase offers its next.
        let mut given = vec![false; offered.len()];
        let mut next = vec![0usize; offered.len()];
        let mut seen: Vec<String> = Vec::new();
        for _ in 0..3 {
            for (index, (_, pieces)) in offered.iter().enumerate() {
                while let Some(piece) = pieces.get(next[index]) {
                    next[index] += 1;
                    let lower = fold(piece);
                    if seen.contains(&lower) {
                        // The index narrows for this phrase through that piece.
                        given[index] = true;
                        continue;
                    }
                    if compiled.pieces.len() < MAX_INDEX_PIECES {
                        compiled.pieces.push((*piece).to_string());
                        seen.push(lower);
                        given[index] = true;
                    }
                    break;
                }
            }
        }
        for ((phrase, _), given) in offered.iter().zip(given) {
            if !given {
                compiled.scanned.push(phrase);
            }
        }
        if !compiled.pieces.is_empty() {
            let terms: Vec<String> = compiled.pieces.iter().map(|piece| quoted(piece)).collect();
            compiled.match_query = Some(terms.join(" AND "));
        }
        Ok(compiled)
    }

    /// Whether the search reads the texts to compare them: with any phrase or an exact
    /// text.
    pub(crate) fn reads_text(&self) -> bool {
        !self.exact.is_empty() || !self.folded.is_empty() || self.equals.is_some()
    }

    /// The statement that yields `columns` of the logs that the text index, the label
    /// and the source leave, in the order they were stored. Its parameters are
    /// [`Compiled::parameters`].
    ///
    /// With the text index, the index leads (`CROSS JOIN` keeps SQLite from turning the
    /// join round) and yields its rows in the order of their row ids, which is the
    /// order of storing; without it, the rows of a label or a source come out of the
    /// SQL index in that order too. So no statement sorts, and none keeps more than the
    /// row it is at.
    pub(crate) fn sql(&self, columns: &str) -> String {
        let mut sql = format!("SELECT {columns} FROM ");
        let mut conditions = Vec::new();
        if self.match_query.is_some() {
            sql.push_str("logs_fts CROSS JOIN logs ON logs.seq = logs_fts.rowid");
            conditions.push("logs_fts MATCH ?");
        } else {
            sql.push_str("logs");
        }
        if self.label.is_some() {
            conditions.push("logs.label = ?");
        }
        if self.source.is_some() {
            conditions.push("logs.source = ?");
        }
        if !conditions.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&conditions.join(" AND "));
        }
        sql.push_str(if self.match_query.is_some() {
            " ORDER BY logs_fts.rowid"
        } else {
            " ORDER BY logs.seq"
        });
        sql
    }

    /// The values of the parameters of [`Compiled::sql`], in order.
    pub(crate) fn parameters(&self) -> Vec<&dyn ToSql> {
        let mut parameters: Vec<&dyn ToSql> = Vec::new();
        if let Some(query) = &self.match_query {
            parameters.push(query);
        }
        if let Some(label) = &self.label {
            parameters.push(label);
        }
        if let Some(source) = &self.source {
            parameters.push(source);
        }
        parameters
    }

    /// Whether `text` holds every phrase and is the exact text, where those are asked.
    fn holds(&self, text: &str) -> bool {
        self.equals.is_none_or(|wanted| wanted == text)
            && self.exact.iter().all(|phrase| text.contains(phrase))
            && (self.folded.is_empty() || {
                let lower = fold(text);
                self.folded
                    .iter()
                    .all(|phrase| lower.contains(phrase.as_str()))
            })
    }

    /// Whether the row the statement is at passes the comparison done here. `bytes` is
    /// the column with the length of the text and `text` the column with the text,
    /// which is read only when it has to be compared (and not when its length already
    /// rules the exact text out).
    pub(crate) fn accepts(
        &self,
        row: &rusqlite::Row<'_>,
        bytes: usize,
        text: usize,
    ) -> rusqlite::Result<bool> {
        if !self.reads_text() {
            return Ok(true);
        }
        if let Some(wanted) = self.equals {
            let length: u64 = row.get(bytes)?;
            if wanted.len() as u64 != length {
                return Ok(false);
            }
        }
        Ok(self.holds(row.get_ref(text)?.as_str()?))
    }
}

/// What a search holds of the budget, and what for.
pub(crate) struct Share {
    pub bytes: u64,
    /// The parts of `bytes`, for a refusal.
    pub parts: String,
    /// The longest stored text, if the share depends on it.
    pub longest_text_bytes: u64,
}

/// Builds the columns of stored rows from the rows of a statement that selects
/// [`row_columns`], and the vector after them where it is wanted.
pub(crate) struct RowsBuilder {
    id: StringChunkedBuilder,
    text: StringChunkedBuilder,
    source: StringChunkedBuilder,
    occurrences: Vec<u32>,
    label: StringChunkedBuilder,
    group: StringChunkedBuilder,
    truncated: Vec<bool>,
    /// Absent for the rows of a store of format 1.
    text_longer_than_input: Vec<Option<bool>>,
    /// Absent for the rows of a store of format 1.
    empty: Vec<Option<bool>>,
    vector: Option<ListPrimitiveChunkedBuilder<Float32Type>>,
    /// The vector of the row being read; reused from row to row.
    values: Vec<f32>,
    dimension: usize,
}

impl RowsBuilder {
    /// For exactly `rows` rows, so that no column grows while it is filled.
    pub(crate) fn new(rows: usize, dimension: usize, with_vectors: bool) -> Self {
        Self {
            id: StringChunkedBuilder::new(ID.into(), rows),
            text: StringChunkedBuilder::new(TEXT.into(), rows),
            source: StringChunkedBuilder::new(SOURCE.into(), rows),
            occurrences: Vec::with_capacity(rows),
            label: StringChunkedBuilder::new(LABEL.into(), rows),
            group: StringChunkedBuilder::new(GROUP.into(), rows),
            truncated: Vec::with_capacity(rows),
            text_longer_than_input: Vec::with_capacity(rows),
            empty: Vec::with_capacity(rows),
            vector: with_vectors.then(|| {
                ListPrimitiveChunkedBuilder::new(
                    VECTOR.into(),
                    rows,
                    rows * dimension,
                    DataType::Float32,
                )
            }),
            values: Vec::with_capacity(dimension),
            dimension,
        }
    }

    /// Appends the row `row` is at; its [`row_columns`] start at column `first`.
    pub(crate) fn push(&mut self, row: &rusqlite::Row<'_>, first: usize) -> rusqlite::Result<()> {
        let text = |column: usize| -> rusqlite::Result<&str> {
            row.get_ref(first + column)?
                .as_str()
                .map_err(rusqlite::Error::from)
        };
        self.id.append_value(text(0)?);
        self.text.append_value(text(1)?);
        self.source.append_value(text(2)?);
        self.occurrences.push(row.get(first + 3)?);
        self.label.append_value(text(4)?);
        match row.get_ref(first + 5)?.as_str_or_null()? {
            Some(group) => self.group.append_value(group),
            None => self.group.append_null(),
        }
        self.truncated.push(row.get(first + 6)?);
        self.text_longer_than_input.push(row.get(first + 7)?);
        self.empty.push(row.get(first + 8)?);
        if let Some(vector) = &mut self.vector {
            let blob = row.get_ref(first + ROW_COLUMN_COUNT)?.as_blob()?;
            self.values.clear();
            decode(blob, self.dimension, &mut self.values)?;
            vector.append_slice(&self.values);
        }
        Ok(())
    }

    /// The columns, in the order of the stored rows frame.
    pub(crate) fn finish(mut self) -> Vec<Column> {
        let mut columns = vec![
            self.id.finish().into_column(),
            self.text.finish().into_column(),
            self.source.finish().into_column(),
            Column::new(OCCURRENCES.into(), self.occurrences),
            self.label.finish().into_column(),
            self.group.finish().into_column(),
            Column::new(TRUNCATED.into(), self.truncated),
            Column::new(TEXT_LONGER_THAN_INPUT.into(), self.text_longer_than_input),
            Column::new(EMPTY.into(), self.empty),
        ];
        if let Some(vector) = &mut self.vector {
            columns.push(vector.finish().into_column());
        }
        columns
    }
}

/// The stored form of a vector (little-endian 32-bit floats) as numbers, appended to
/// `values`.
pub(crate) fn decode(blob: &[u8], dimension: usize, values: &mut Vec<f32>) -> rusqlite::Result<()> {
    if blob.len() != 4 * dimension {
        return Err(rusqlite::Error::FromSqlConversionFailure(
            0,
            rusqlite::types::Type::Blob,
            format!(
                "a stored vector has {} bytes; the store's dimension {dimension} makes {}",
                blob.len(),
                4 * dimension
            )
            .into(),
        ));
    }
    values.extend(
        blob.as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes)),
    );
    Ok(())
}

impl Store {
    pub(crate) fn failure(&self, error: rusqlite::Error) -> StoreError {
        failed("searching the store", self.path(), error)
    }

    /// What a search with `compiled` holds of the budget (see the module description).
    pub(crate) fn share(&self, compiled: &Compiled) -> Result<Share, StoreError> {
        let mut bytes = SEARCH_FIXED_BYTES;
        let mut parts = format!("{SEARCH_FIXED_BYTES} bytes for the database itself");
        let retained = self.retained_bytes().saturating_sub(IDLE_BYTES);
        if retained > 0 {
            bytes += retained;
            parts.push_str(&format!(
                ", {retained} bytes that the open store still keeps of a long text it indexed \
                 (it could not be opened again after that add)"
            ));
        }
        let mut longest = 0;
        if compiled.reads_text() && !compiled.nothing {
            longest = self.longest_text_bytes()?;
            let copies: u64 = compiled
                .folded
                .iter()
                .map(|phrase| phrase.len() as u64)
                .sum();
            let reading = TEXT_READ_FACTOR * longest + copies;
            bytes += reading;
            parts.push_str(&format!(
                ", {reading} bytes for reading the stored texts one at a time to compare them \
                 ({TEXT_READ_FACTOR} for each of the {longest} bytes of the longest stored \
                 text, and {copies} for the lower-case copies of the phrases)"
            ));
            if !compiled.pieces.is_empty() {
                let piece_bytes: u64 = compiled.pieces.iter().map(|piece| piece.len() as u64).sum();
                let characters: u64 = compiled
                    .pieces
                    .iter()
                    .map(|piece| piece.chars().count() as u64)
                    .sum();
                let index =
                    PHRASE_BYTE_BYTES * piece_bytes + INDEX_TEXT_FACTOR * characters * longest;
                bytes += index;
                parts.push_str(&format!(
                    ", {index} bytes for the text index, which is given {} pieces of the \
                     phrases, {characters} characters in {piece_bytes} bytes \
                     ({PHRASE_BYTE_BYTES} for each byte, and {INDEX_TEXT_FACTOR} for each \
                     character and each of the {longest} bytes of the longest stored text)",
                    compiled.pieces.len()
                ));
            }
        }
        Ok(Share {
            bytes,
            parts,
            longest_text_bytes: longest,
        })
    }

    /// Runs a search: in one read transaction, after checking that the file is still
    /// this store's ([`Store::require_same_file`]), with the search's share and
    /// `more` bytes (described by `more_parts`) held of the budget on top of what the
    /// caller holds. Refused, before a row is read, when that is not left. Whatever the
    /// outcome, SQLite's page cache and the budget are given back afterwards. `run`
    /// gets the budget and the bytes held.
    pub(crate) fn search<T>(
        &self,
        compiled: &Compiled,
        more: u64,
        more_parts: &str,
        budget: &mut MemoryBudget,
        run: impl FnOnce(&mut MemoryBudget, u64) -> Result<T, StoreError>,
    ) -> Result<T, StoreError> {
        let left = budget.left_bytes();
        // What the caller holds already (its frames) stays counted during the search.
        let held = budget.usable_bytes() - left;
        let result = (|| {
            let transaction = self
                .connection
                .unchecked_transaction()
                .map_err(|error| self.failure(error))?;
            self.require_same_file(&transaction)?;
            let share = self.share(compiled)?;
            let own = share.bytes.saturating_add(more);
            if left < own {
                return Err(StoreError::Memory(L2dError::Refused {
                    reason: format!(
                        "searching the store {} needs {own} bytes before it reads a row: \
                         {}{more_parts}; but {left} bytes are left of the memory limit of {} \
                         bytes; nothing was read",
                        self.name(),
                        share.parts,
                        budget.limit_bytes()
                    ),
                }));
            }
            budget.hold(held + own);
            // The transaction ends when this returns: the rows counted are the rows read.
            run(budget, own)
        })();
        let _ = self.connection.execute_batch("PRAGMA shrink_memory");
        budget.hold(held);
        result
    }

    /// Counts the logs `compiled` names, up to `limit`, and adds up what reading them
    /// back takes.
    fn measure(&self, compiled: &Compiled, limit: usize) -> Result<Matching, StoreError> {
        let mut matching = Matching {
            rows: 0,
            read_bytes: READ_FIXED_BYTES,
        };
        if compiled.nothing || limit == 0 {
            return Ok(matching);
        }
        let vector = 4 * u64::from(self.identity().dimension) + 8;
        let mut columns = String::from(
            "octet_length(logs.text), octet_length(logs.source), \
             ifnull(octet_length(logs.group_key), 0)",
        );
        if compiled.reads_text() {
            columns.push_str(", logs.text");
        }
        let fail = |error| self.failure(error);
        let mut statement = self
            .connection
            .prepare(&compiled.sql(&columns))
            .map_err(fail)?;
        let mut rows = statement
            .query(compiled.parameters().as_slice())
            .map_err(fail)?;
        let mut longest = 0u64;
        while let Some(row) = rows.next().map_err(fail)? {
            if !compiled.accepts(row, 0, 3).map_err(fail)? {
                continue;
            }
            let text: u64 = row.get(0).map_err(fail)?;
            let source: u64 = row.get(1).map_err(fail)?;
            let group: u64 = row.get(2).map_err(fail)?;
            longest = longest.max(text);
            matching.read_bytes += row_bytes(text, source, group, vector);
            matching.rows += 1;
            if matching.rows == limit as u64 {
                break;
            }
        }
        // SQLite's copy of the text of the row that is being read back.
        matching.read_bytes += TEXT_READ_FACTOR * longest;
        Ok(matching)
    }

    /// Reads the first `rows` logs `compiled` names into a stored rows frame.
    fn load(&self, compiled: &Compiled, rows: u64) -> Result<DataFrame, StoreError> {
        let dimension = usize::from(self.identity().dimension);
        let mut builder = RowsBuilder::new(rows as usize, dimension, true);
        if rows > 0 {
            let fail = |error| self.failure(error);
            let columns = format!(
                "octet_length(logs.text), {}, logs.vector",
                row_columns(self.format())
            );
            let mut statement = self
                .connection
                .prepare(&compiled.sql(&columns))
                .map_err(fail)?;
            let mut cursor = statement
                .query(compiled.parameters().as_slice())
                .map_err(fail)?;
            let mut read = 0u64;
            while read < rows
                && let Some(row) = cursor.next().map_err(fail)?
            {
                if compiled.accepts(row, 0, 2).map_err(fail)? {
                    builder.push(row, 1).map_err(fail)?;
                    read += 1;
                }
            }
            if read != rows {
                return Err(failed(
                    "searching the store",
                    self.path(),
                    format!(
                        "{rows} logs matched when they were counted, {read} when they were read"
                    ),
                ));
            }
        }
        DataFrame::new(builder.finish())
            .map_err(|error| failed("searching the store", self.path(), error))
    }

    /// Counts, asks the budget, reads: the common part of the text search and the
    /// export.
    fn read_back(
        &self,
        filter: &Filter,
        limit: usize,
        budget: &mut MemoryBudget,
    ) -> Result<DataFrame, StoreError> {
        let compiled = Compiled::new(filter)?;
        self.search(&compiled, 0, "", budget, |budget, _| {
            let matching = self.measure(&compiled, limit)?;
            budget
                .require_whole(
                    matching.read_bytes,
                    &format!(
                        "reading the {} matching logs of the store {} back",
                        matching.rows,
                        self.name()
                    ),
                )
                .map_err(StoreError::Memory)?;
            self.load(&compiled, matching.rows)
        })
    }

    /// [`Store::text_search_with`] under a budget observed now with the default
    /// settings, in which nothing is held: for callers that hold nothing that matters.
    pub fn text_search(&self, filter: &Filter, limit: usize) -> Result<DataFrame, StoreError> {
        let mut budget =
            MemoryBudget::observe(&MemorySettings::default()).map_err(StoreError::Memory)?;
        self.text_search_with(filter, limit, &mut budget)
    }

    /// The first `limit` logs that `filter` names, in the order they were stored, as a
    /// stored rows frame ([`stored_schema`]): requirement 5.2, and 5.5 for the label and
    /// the source. See the module description for what a filter means, who decides a
    /// match, how phrases are kept from being read as syntax, and for the memory.
    ///
    /// - An empty phrase is [`StoreError::InvalidQuery`].
    /// - When the search's own share ([`Explain::held_bytes`]) is not left of the
    ///   budget, or the matching rows do not fit, the search is refused
    ///   ([`StoreError::Memory`]) before a row is read back, with the memory needed and
    ///   the limit. A smaller `limit` or a narrower filter needs less for the rows.
    /// - When the file no longer describes this open store, [`StoreError::Failed`],
    ///   naming what it was opened as and what the file says now.
    ///
    /// Nothing in the store is changed. The budget holds afterwards what it held
    /// before.
    ///
    /// The rows carry their vectors, as every stored rows frame does, and no origin is
    /// asked here: a search by phrase compares no vector. A caller that goes on to
    /// compare the vectors of these rows with others calls [`Store::require_origin`]
    /// first, or reads them with [`Store::vectors_with`], which does.
    pub fn text_search_with(
        &self,
        filter: &Filter,
        limit: usize,
        budget: &mut MemoryBudget,
    ) -> Result<DataFrame, StoreError> {
        self.read_back(filter, limit, budget)
    }

    /// [`Store::vectors_with`] under a budget observed now with the default settings,
    /// in which nothing is held.
    pub fn vectors(
        &self,
        origin: VectorOrigin<'_>,
        filter: &Filter,
    ) -> Result<DataFrame, StoreError> {
        let mut budget =
            MemoryBudget::observe(&MemorySettings::default()).map_err(StoreError::Memory)?;
        self.vectors_with(origin, filter, &mut budget)
    }

    /// Every log that `filter` names with its vector, in the order they were stored, as
    /// a stored rows frame: the export for the assessment of a vector set and for the
    /// shift test. The vectors come back bit for bit as they were added.
    ///
    /// The vectors leave the store here to be compared outside it, so the caller says
    /// which vectors it expects, as for a search by vector: `origin` is the encoder
    /// identity and the runtime release of the vectors it will compare these with (or
    /// [`VectorOrigin::without_release`]). Another identity than the store's is
    /// [`StoreError::IdentityMismatch`], another release
    /// [`StoreError::ReleaseMismatch`], both before anything is read.
    ///
    /// The whole result is in memory at once, so it is counted first and refused with
    /// the estimate and the limit when it does not fit (requirement 12.5); otherwise
    /// as [`Store::text_search_with`] without a limit.
    pub fn vectors_with(
        &self,
        origin: VectorOrigin<'_>,
        filter: &Filter,
        budget: &mut MemoryBudget,
    ) -> Result<DataFrame, StoreError> {
        self.require_origin(origin)?;
        self.read_back(filter, usize::MAX, budget)
    }

    /// How many logs `filter` names, up to `limit`, and what reading them back would
    /// ask of the budget, without reading them back. Errors as
    /// [`Store::text_search_with`], except that the rows themselves need not fit.
    pub fn matching_with(
        &self,
        filter: &Filter,
        limit: usize,
        budget: &mut MemoryBudget,
    ) -> Result<Matching, StoreError> {
        let compiled = Compiled::new(filter)?;
        self.search(&compiled, 0, "", budget, |_, _| {
            self.measure(&compiled, limit)
        })
    }

    /// How a search with `filter` finds its logs: the pieces of the phrases that the
    /// text index is given, the phrases that are answered by a scan, what the search
    /// holds of the budget, and SQLite's description of how it reads the rows (which
    /// names the SQL index of a label or source restriction).
    pub fn explain(&self, filter: &Filter) -> Result<Explain, StoreError> {
        self.require_same_file(&self.connection)?;
        let compiled = Compiled::new(filter)?;
        let share = self.share(&compiled)?;
        let fail = |error| self.failure(error);
        let sql = format!(
            "EXPLAIN QUERY PLAN {}",
            compiled.sql("logs.seq, logs.vector")
        );
        let mut statement = self.connection.prepare(&sql).map_err(fail)?;
        let steps = statement
            .query_map(compiled.parameters().as_slice(), |row| row.get(3))
            .map_err(fail)?
            .collect::<rusqlite::Result<Vec<String>>>()
            .map_err(fail)?;
        drop(statement);
        Ok(Explain {
            text_index: compiled.match_query.is_some(),
            index_pieces: compiled.pieces,
            scanned_phrases: compiled
                .scanned
                .iter()
                .map(|phrase| phrase.to_string())
                .collect(),
            longest_text_bytes: share.longest_text_bytes,
            held_bytes: share.bytes,
            steps,
        })
    }
}
