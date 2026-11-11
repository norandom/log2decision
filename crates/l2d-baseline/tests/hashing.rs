//! Character n-gram hashing (task 1.5; requirements 2.1, 2.4, 2.6).
//!
//! The pinned bucket indices and signs in this file were derived outside Rust, with
//! arbitrary-precision integer arithmetic, from the published FNV-1a 64-bit constants:
//!
//! | n-gram | FNV-1a 64            | low 32 bits mod 65,536 | one bits in high 32 | sign |
//! |--------|----------------------|------------------------|---------------------|------|
//! | `a`    | `0xaf63dc4c8601ec8c` | 60556                  | 18                  | `+`  |
//! | `b`    | `0xaf63df4c8601f1a5` | 61861                  | 20                  | `+`  |
//! | `ab`   | `0x089c4407b545986a` | 39018                  | 10                  | `+`  |
//! | `aa`   | `0x089c4307b54596b7` | 38583                  | 11                  | `-`  |
//! | `aaa`  | `0xe71cbc19053f4da2` | 19874                  | 17                  | `-`  |
//! | `G`    | `0xaf63fa4c86021f86` | 8070                   | 19                  | `-`  |
//! | `E`    | `0xaf63f84c86021c20` | 7200                   | 18                  | `+`  |
//! | `T`    | `0xaf64094c86023903` | 14595                  | 14                  | `+`  |
//! | `GE`   | `0x09021307b59bf859` | 63577                  | 9                   | `-`  |
//! | `ET`   | `0x08fb5e07b596591c` | 22812                  | 16                  | `+`  |
//! | `GET`  | `0xea7e631998067e17` | 32279                  | 18                  | `+`  |

use std::collections::BTreeMap;
use std::process::Command;

use l2d_baseline::hashing::{
    CharHasher, DEFAULT_BUCKETS, DEFAULT_NGRAM_MAX, HashingError, HashingSettings, MAX_BUCKETS,
    RoutingMode, SparseRow, fnv1a_64,
};
use l2d_core::adapters::csic2010::split_routing;

const GET_TEXT: &str =
    "GET http://localhost:8080/tienda1/publico/anadir.jsp?id=2&nombre=Jam%F3n+Ib%E9rico&precio=85";
const POST_TEXT: &str = "POST http://localhost:8080/tienda1/publico/autenticar.jsp\nmodo=entrar&login=bob&pwd=' OR '1'='1";
const NO_PAYLOAD_TEXT: &str = "GET http://localhost:8080/tienda1/index.jsp";
/// Layout of the mini fixture: a space, not a line feed, before the body.
const FIXTURE_POST_TEXT: &str =
    "POST http://localhost:8080/tienda1/publico/autenticar.jsp modo=entrar&login=hugo&pwd=clave191";
const ODD_TEXT: &str = "\u{0}\u{1}\t\r\u{7f}\u{80}\u{ff}é\u{fffd}😀 ñ\u{0}\u{0}";

fn hasher(routing: RoutingMode) -> CharHasher {
    CharHasher::new(HashingSettings {
        routing,
        ..HashingSettings::default()
    })
    .expect("default settings with a routing mode are valid")
}

fn sample_texts() -> Vec<String> {
    vec![
        "a".to_string(),
        "ab".to_string(),
        "aaa".to_string(),
        GET_TEXT.to_string(),
        POST_TEXT.to_string(),
        NO_PAYLOAD_TEXT.to_string(),
        FIXTURE_POST_TEXT.to_string(),
        ODD_TEXT.to_string(),
        "?".to_string(),
        "\n".to_string(),
        "a?".to_string(),
        "?a".to_string(),
        "x?y?z\nw".to_string(),
        long_text(),
    ]
}

fn long_text() -> String {
    let mut text = String::from("POST http://localhost:8080/t/a.jsp\n");
    let mut state: u32 = 12345;
    while text.len() < 200_000 {
        // A fixed linear congruential sequence: varied bytes without randomness.
        state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        let c = char::from_u32(0x20 + (state >> 16) % 0x250).unwrap_or('?');
        text.push(if c == '\n' || c == '?' { '_' } else { c });
    }
    text
}

/// The feature rule written a second time, on an ordered map, for comparison.
fn reference_counts(
    bytes: &[u8],
    buckets: u32,
    ngram_max: usize,
    offset: u32,
    counts: &mut BTreeMap<u32, i64>,
) {
    for n in 1..=ngram_max {
        if bytes.len() < n {
            break;
        }
        for start in 0..=bytes.len() - n {
            let mut hash: u64 = 14_695_981_039_346_656_037;
            for &byte in &bytes[start..start + n] {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(1_099_511_628_211);
            }
            let bucket = ((hash % (1u64 << 32)) % buckets as u64) as u32;
            let sign = if (hash >> 32).count_ones() & 1 == 0 {
                1
            } else {
                -1
            };
            *counts.entry(bucket + offset).or_insert(0) += sign;
        }
    }
}

fn reference_row(counts: &BTreeMap<u32, i64>) -> (Vec<u32>, Vec<f64>) {
    let kept: Vec<(u32, i64)> = counts
        .iter()
        .filter(|(_, count)| **count != 0)
        .map(|(index, count)| (*index, *count))
        .collect();
    let squares: i64 = kept.iter().map(|(_, count)| count * count).sum();
    let norm = (squares as f64).sqrt();
    (
        kept.iter().map(|(index, _)| *index).collect(),
        kept.iter().map(|(_, count)| *count as f64 / norm).collect(),
    )
}

fn reference(text: &str, settings: HashingSettings) -> (Vec<u32>, Vec<f64>) {
    let mut counts = BTreeMap::new();
    let n = settings.ngram_max as usize;
    let split = split_routing(text);
    match settings.routing {
        RoutingMode::Keep => reference_counts(text.as_bytes(), settings.buckets, n, 0, &mut counts),
        RoutingMode::Strip => reference_counts(
            split.payload.as_bytes(),
            settings.buckets,
            n,
            0,
            &mut counts,
        ),
        RoutingMode::Isolate => {
            reference_counts(
                split.payload.as_bytes(),
                settings.buckets,
                n,
                0,
                &mut counts,
            );
            reference_counts(
                split.routing.as_bytes(),
                settings.buckets,
                n,
                settings.buckets,
                &mut counts,
            );
        }
    }
    reference_row(&counts)
}

fn bits(row: &SparseRow) -> Vec<u64> {
    row.values().iter().map(|value| value.to_bits()).collect()
}

fn assert_row(row: &SparseRow, indices: &[u32], values: &[f64]) {
    assert_eq!(row.indices(), indices);
    let expected: Vec<u64> = values.iter().map(|value| value.to_bits()).collect();
    assert_eq!(bits(row), expected, "values differ: {:?}", row.values());
}

fn squared_norm(row: &SparseRow) -> f64 {
    row.values().iter().map(|value| value * value).sum()
}

#[test]
fn fnv1a_64_matches_published_vectors() {
    assert_eq!(fnv1a_64(b""), 0xcbf2_9ce4_8422_2325);
    assert_eq!(fnv1a_64(b"a"), 0xaf63_dc4c_8601_ec8c);
    assert_eq!(fnv1a_64(b"foobar"), 0x8594_4171_f739_67e8);
}

#[test]
fn defaults_are_the_design_feature_rule() {
    let settings = HashingSettings::default();
    assert_eq!(settings.buckets, 65_536);
    assert_eq!(settings.ngram_max, 3);
    assert_eq!(settings.routing, RoutingMode::Keep);
    assert_eq!(DEFAULT_BUCKETS, 65_536);
    assert_eq!(DEFAULT_NGRAM_MAX, 3);
    assert_eq!(CharHasher::default().settings(), settings);
    assert_eq!(CharHasher::default().dimension(), 65_536);
    assert_eq!(CharHasher::default().routing_offset(), None);
}

#[test]
fn pinned_row_of_a_single_byte() {
    assert_row(&CharHasher::default().row("a"), &[60_556], &[1.0]);
}

#[test]
fn pinned_row_of_two_bytes() {
    // `ab` (+), `a` (+), `b` (+): three buckets with one n-gram each.
    let value = 1.0 / 3f64.sqrt();
    assert_row(
        &CharHasher::default().row("ab"),
        &[39_018, 60_556, 61_861],
        &[value, value, value],
    );
}

#[test]
fn pinned_row_with_repeated_and_negative_features() {
    // `aaa` once (-), `aa` twice (-), `a` three times (+): squares sum to 14.
    let norm = 14f64.sqrt();
    assert_row(
        &CharHasher::default().row("aaa"),
        &[19_874, 38_583, 60_556],
        &[-1.0 / norm, -2.0 / norm, 3.0 / norm],
    );
}

#[test]
fn pinned_row_of_three_distinct_bytes() {
    // Ascending: E (+), G (-), T (+), ET (+), GET (+), GE (-).
    let value = 1.0 / 6f64.sqrt();
    assert_row(
        &CharHasher::default().row("GET"),
        &[7_200, 8_070, 14_595, 22_812, 32_279, 63_577],
        &[value, -value, value, value, value, -value],
    );
}

#[test]
fn pinned_rows_of_the_routing_modes() {
    // Routing `a`, payload `b`; the `?` belongs to neither.
    assert_row(&hasher(RoutingMode::Strip).row("a?b"), &[61_861], &[1.0]);
    let value = 1.0 / 2f64.sqrt();
    assert_row(
        &hasher(RoutingMode::Isolate).row("a?b"),
        &[61_861, 65_536 + 60_556],
        &[value, value],
    );
    // The same byte on both sides stays two features.
    assert_row(
        &hasher(RoutingMode::Isolate).row("a\na"),
        &[60_556, 65_536 + 60_556],
        &[value, value],
    );
    // Keep hashes the delimiter and the n-grams across it too.
    let keep = hasher(RoutingMode::Keep).row("a?b");
    assert_eq!(keep.len(), 6);
    assert!(keep.indices().contains(&60_556) && keep.indices().contains(&61_861));
}

#[test]
fn rows_match_the_reference_rule_in_every_mode() {
    for routing in [RoutingMode::Keep, RoutingMode::Strip, RoutingMode::Isolate] {
        for (buckets, ngram_max) in [(65_536, 3), (1_000, 3), (65_536, 1), (7, 5), (1 << 30, 2)] {
            let settings = HashingSettings {
                buckets,
                ngram_max,
                routing,
            };
            let hasher = CharHasher::new(settings).expect("valid settings");
            for text in sample_texts() {
                let (indices, values) = reference(&text, settings);
                let row = hasher.row(&text);
                assert_eq!(row.indices(), indices, "{settings:?}");
                let expected: Vec<u64> = values.iter().map(|value| value.to_bits()).collect();
                assert_eq!(bits(&row), expected, "{settings:?}");
            }
        }
    }
}

#[test]
fn rows_are_sorted_unique_nonzero_and_of_unit_norm() {
    let mut non_empty = 0;
    for routing in [RoutingMode::Keep, RoutingMode::Strip, RoutingMode::Isolate] {
        let hasher = hasher(routing);
        for text in sample_texts() {
            let row = hasher.row(&text);
            assert_eq!(row.indices().len(), row.values().len());
            assert_eq!(row.len(), row.indices().len());
            assert!(row.indices().windows(2).all(|pair| pair[0] < pair[1]));
            assert!(
                row.indices()
                    .iter()
                    .all(|index| *index < hasher.dimension())
            );
            assert!(
                row.values()
                    .iter()
                    .all(|value| value.is_finite() && *value != 0.0)
            );
            let pairs: Vec<(u32, f64)> = row.iter().collect();
            assert_eq!(pairs.len(), row.len());
            if row.is_empty() {
                continue;
            }
            non_empty += 1;
            assert!(
                (squared_norm(&row) - 1.0).abs() < 1e-12,
                "{routing:?} {}",
                squared_norm(&row)
            );
        }
    }
    assert!(non_empty >= 30, "only {non_empty} non-empty rows checked");
}

#[test]
fn an_empty_text_gives_an_empty_row() {
    for routing in [RoutingMode::Keep, RoutingMode::Strip, RoutingMode::Isolate] {
        let row = hasher(routing).row("");
        assert!(row.is_empty());
        assert_eq!(row, SparseRow::default());
    }
    assert!(!CharHasher::default().row("a").is_empty());
}

#[test]
fn features_that_cancel_are_dropped() {
    // One bucket: `G` is negative and `E` positive, so the unigrams of `GE` cancel.
    let unigrams = CharHasher::new(HashingSettings {
        buckets: 1,
        ngram_max: 1,
        routing: RoutingMode::Keep,
    })
    .expect("valid settings");
    assert!(unigrams.row("GE").is_empty());
    assert_row(&unigrams.row("GET"), &[0], &[1.0]);
    assert_row(&unigrams.row("G"), &[0], &[-1.0]);
}

#[test]
fn bucket_count_and_ngram_length_are_configurable() {
    let small = CharHasher::new(HashingSettings {
        buckets: 1_000,
        ..HashingSettings::default()
    })
    .expect("valid settings");
    // 0x8601ec8c mod 1000 = 36.
    assert_row(&small.row("a"), &[36], &[1.0]);
    assert_eq!(small.dimension(), 1_000);

    let unigrams = CharHasher::new(HashingSettings {
        ngram_max: 1,
        ..HashingSettings::default()
    })
    .expect("valid settings");
    let value = 1.0 / 2f64.sqrt();
    assert_row(&unigrams.row("ab"), &[60_556, 61_861], &[value, value]);
}

#[test]
fn unusable_settings_are_refused() {
    let with = |buckets, ngram_max| {
        CharHasher::new(HashingSettings {
            buckets,
            ngram_max,
            routing: RoutingMode::Isolate,
        })
    };
    assert_eq!(
        with(0, 3),
        Err(HashingError::BucketsOutOfRange { buckets: 0 })
    );
    assert_eq!(
        with(MAX_BUCKETS + 1, 3),
        Err(HashingError::BucketsOutOfRange {
            buckets: MAX_BUCKETS + 1
        })
    );
    assert_eq!(with(65_536, 0), Err(HashingError::NgramMaxZero));
    let largest = with(MAX_BUCKETS, 3).expect("the largest bucket count is valid");
    assert_eq!(largest.dimension(), 1 << 31);
    assert_eq!(largest.routing_offset(), Some(1 << 30));
    assert!(
        HashingError::BucketsOutOfRange { buckets: 0 }
            .to_string()
            .contains('0')
    );
}

#[test]
fn strip_hashes_the_payload_only() {
    let keep = hasher(RoutingMode::Keep);
    let strip = hasher(RoutingMode::Strip);
    for text in [GET_TEXT, POST_TEXT] {
        let split = split_routing(text);
        assert!(!split.routing.is_empty() && !split.payload.is_empty());
        let row = strip.row(text);
        assert!(!row.is_empty());
        // Exactly the row of the payload on its own ...
        assert_eq!(row, keep.row(split.payload));
        // ... whatever the routing part is.
        for delimiter in ['?', '\n'] {
            for routing in ["", "DELETE /other/path.jsp", "PUT http://example.org/zzz"] {
                let other = format!("{routing}{delimiter}{}", split.payload);
                assert_eq!(strip.row(&other), row, "{routing:?}");
            }
        }
        assert_ne!(row, keep.row(text));
    }
}

#[test]
fn a_text_without_delimiter_is_routing_only() {
    let strip = hasher(RoutingMode::Strip);
    let isolate = hasher(RoutingMode::Isolate);
    for text in [NO_PAYLOAD_TEXT, FIXTURE_POST_TEXT, "plain log line"] {
        assert!(strip.row(text).is_empty(), "{text}");
        let row = isolate.row(text);
        assert!(!row.is_empty());
        assert!(row.indices().iter().all(|index| *index >= 65_536));
        // The routing half is the plain row moved up by the bucket count.
        let plain = hasher(RoutingMode::Keep).row(text);
        let moved: Vec<u32> = plain.indices().iter().map(|index| index + 65_536).collect();
        assert_eq!(row.indices(), moved);
        assert_eq!(bits(&row), bits(&plain));
    }
}

#[test]
fn isolate_keeps_routing_and_payload_in_disjoint_index_ranges() {
    let keep = hasher(RoutingMode::Keep);
    let strip = hasher(RoutingMode::Strip);
    let isolate = hasher(RoutingMode::Isolate);
    assert_eq!(isolate.dimension(), 131_072);
    assert_eq!(isolate.routing_offset(), Some(65_536));
    assert_eq!(strip.routing_offset(), None);

    for text in [GET_TEXT, POST_TEXT] {
        let split = split_routing(text);
        let row = isolate.row(text);
        let payload: Vec<(u32, f64)> = row.iter().filter(|(index, _)| *index < 65_536).collect();
        let routing: Vec<(u32, f64)> = row.iter().filter(|(index, _)| *index >= 65_536).collect();
        assert!(!payload.is_empty() && !routing.is_empty());

        // Payload half: the indices of the stripped row.
        let stripped = strip.row(text);
        let payload_indices: Vec<u32> = payload.iter().map(|(index, _)| *index).collect();
        assert_eq!(payload_indices, stripped.indices());
        // Routing half: the indices of the routing part's own row, moved up.
        let alone = keep.row(split.routing);
        let routing_indices: Vec<u32> = routing.iter().map(|(index, _)| *index - 65_536).collect();
        assert_eq!(routing_indices, alone.indices());

        // Zeroing the routing half and renormalizing gives the stripped row.
        let norm = payload
            .iter()
            .map(|(_, value)| value * value)
            .sum::<f64>()
            .sqrt();
        for ((_, value), expected) in payload.iter().zip(stripped.values()) {
            assert!((value / norm - expected).abs() < 1e-12);
        }

        // Another endpoint changes the routing half only.
        let moved = isolate.row(&format!("PUT http://example.org/zzz?{}", split.payload));
        let moved_payload: Vec<u32> = moved
            .indices()
            .iter()
            .copied()
            .filter(|index| *index < 65_536)
            .collect();
        assert_eq!(moved_payload, payload_indices);
    }
}

#[test]
fn keep_hashes_the_whole_text() {
    let keep = hasher(RoutingMode::Keep);
    for text in [GET_TEXT, POST_TEXT] {
        let (indices, _) = reference(text, HashingSettings::default());
        assert_eq!(keep.row(text).indices(), indices);
        // The method unigram `G` (8070) or `P` contributes, unlike with strip.
        assert!(keep.row(text).len() > hasher(RoutingMode::Strip).row(text).len());
    }
}

#[test]
fn bytes_outside_ascii_and_control_bytes_are_hashed_as_bytes() {
    let row = CharHasher::default().row(ODD_TEXT);
    assert!(!row.is_empty());
    assert!((squared_norm(&row) - 1.0).abs() < 1e-12);
    // `é` is the two bytes C3 A9: two unigrams and one bigram, no feature for the char.
    let accent = CharHasher::default().row("é");
    assert_eq!(accent.len(), 3);
    // FNV-1a 64 of C3 A9 is 0x0ac21707b7181e01: bucket 7681.
    assert!(accent.indices().contains(&7_681));
    // Case and neighbouring bytes are distinct features.
    assert_ne!(
        CharHasher::default().row("select"),
        CharHasher::default().row("SELECT")
    );
}

#[test]
fn a_long_text_gives_a_stable_unit_row() {
    let text = long_text();
    assert!(text.len() >= 200_000);
    for routing in [RoutingMode::Keep, RoutingMode::Strip, RoutingMode::Isolate] {
        let hasher = hasher(routing);
        let first = hasher.row(&text);
        let second = hasher.row(&text);
        assert!(first.len() > 1_000);
        assert_eq!(first.indices(), second.indices());
        assert_eq!(bits(&first), bits(&second));
        assert!((squared_norm(&first) - 1.0).abs() < 1e-12);
    }
}

// ---------------------------------------------------------------------------------------
// Bit-identical rows across processes.
// ---------------------------------------------------------------------------------------

const PROBE_VARIABLE: &str = "L2D_BASELINE_HASHING_PROBE";
const PROBE_MARKER: &str = "L2D_HASHING_ROW ";

/// Every sample row in every mode as text: indices and the bit patterns of the values.
fn probe_lines() -> Vec<String> {
    let mut lines = Vec::new();
    for routing in [RoutingMode::Keep, RoutingMode::Strip, RoutingMode::Isolate] {
        let hasher = hasher(routing);
        for (position, text) in sample_texts().iter().enumerate() {
            let row = hasher.row(text);
            let entries: Vec<String> = row
                .iter()
                .map(|(index, value)| format!("{index:x}:{:016x}", value.to_bits()))
                .collect();
            lines.push(format!("{routing:?} {position} [{}]", entries.join(",")));
        }
    }
    lines
}

/// Prints the probe rows when started as a child of
/// [`rows_are_bit_identical_across_processes`]; does nothing in a normal test run.
#[test]
fn process_probe() {
    if std::env::var_os(PROBE_VARIABLE).is_none() {
        return;
    }
    println!();
    for line in probe_lines() {
        println!("{PROBE_MARKER}{line}");
    }
}

fn rows_of_a_child_process() -> Vec<String> {
    let binary = std::env::current_exe().expect("the test binary has a path");
    let output = Command::new(binary)
        .args([
            "process_probe",
            "--exact",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(PROBE_VARIABLE, "1")
        .output()
        .expect("the test binary starts a second time");
    assert!(output.status.success(), "child failed: {output:?}");
    String::from_utf8(output.stdout)
        .expect("probe output is text")
        .lines()
        .filter_map(|line| {
            line.find(PROBE_MARKER)
                .map(|at| line[at + PROBE_MARKER.len()..].trim_end().to_string())
        })
        .collect()
}

#[test]
fn rows_are_bit_identical_across_processes() {
    let own = probe_lines();
    assert_eq!(own.len(), 3 * sample_texts().len());
    // The rows carry content: most lines hold at least one entry.
    assert!(own.iter().filter(|line| line.contains(':')).count() >= 30);
    for run in ["first", "second"] {
        let child = rows_of_a_child_process();
        assert_eq!(child.len(), own.len(), "{run} child process: line count");
        // Compared line by line so a failure names the row instead of printing all rows.
        for (position, (theirs, ours)) in child.iter().zip(&own).enumerate() {
            assert!(
                theirs == ours,
                "{run} child process differs from this process in line {position}"
            );
        }
    }
}
