//! Triager task 2.1 through the service: pattern, neighbour and recommendation
//! drill-down (requirements 2.2, 2.3, 2.4, 3.2, 4.4 and 6.2).
//!
//! What the service returns is compared with what the project's own operations give
//! when they are called directly on the same archive: the archive search, the pattern
//! catalogue, its refinement, the exemplar retrieval and the saved policy report.
//! The issued-batch protocol is driven by the test as its model client.

mod support;

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

use l2d_core::patterns::refinement::PatternRefinement;
use l2d_core::patterns::{PatternCatalogue, PatternSettings};
use l2d_core::selections::{Member, SearchQuery, SearchSettings, SelectionSet};
use l2d_core::{L2dError, MemoryBudget, ZipInputSettings};
use l2d_decide::exploration::{
    self, CharacterTrigram, ExplorationEncoder, ExploreSettings, Representation,
};
use l2d_triager::service::{DisclosureProfile, EncoderSource, Encoders};
use serde_json::{Value, json};
use support::{Host, HostOptions, Orig, auth_entries, count, oracle, record_id};

fn zip_settings() -> ZipInputSettings {
    ZipInputSettings {
        entries: vec!["*.log".into()],
        ..ZipInputSettings::default()
    }
}

fn budget() -> MemoryBudget {
    support::budget(&HostOptions::default())
}

/// The refinement `l2d discover search` builds for `query` over `archive`, made with
/// the project's operations and nothing of the service.
fn native_refinement(archive: &Path, query: SearchQuery, work: &Path) -> PatternRefinement {
    let mut budget = budget();
    let zip = zip_settings();
    let (selection, _) = SelectionSet::search_archive(
        "triager",
        query,
        SearchSettings::default(),
        archive,
        &zip,
        &mut budget,
    )
    .unwrap();
    let catalogue = PatternCatalogue::build_archive(
        &selection,
        archive,
        &zip,
        &PatternSettings::default(),
        work,
        &mut budget,
    )
    .unwrap();
    PatternRefinement::from_catalogue(&catalogue, work, &mut budget).unwrap()
}

fn positions(members: &[Member]) -> Vec<String> {
    members
        .iter()
        .map(|member| match member {
            Member::Position(position) => record_id(*position),
            Member::StoreId(id) => panic!("a store member {id}"),
        })
        .collect()
}

/// The record IDs of the rows of `refinement` in `pattern`.
fn members_of(refinement: &PatternRefinement, pattern: usize) -> Vec<String> {
    let mut ids: Vec<String> = refinement
        .rows
        .iter()
        .filter(|row| row.pattern == Some(pattern))
        .map(|row| positions(std::slice::from_ref(&row.member)).remove(0))
        .collect();
    ids.sort();
    ids
}

fn record_ids(host: &Host, handle: &str) -> Vec<String> {
    host.all_records(handle, json!({}), 100)
        .iter()
        .map(|row| row["id"].as_str().unwrap().to_string())
        .collect()
}

fn refine(host: &Host, target: (&str, &str), action: Value) -> Value {
    host.ok(
        "triager_refine",
        json!({"v": 1, target.0: target.1, "action": action}),
    )
}

/// Every pattern of a refinement, read page by page.
fn all_patterns(host: &Host, refinement: &str, limit: u64) -> Vec<Value> {
    let mut rows = Vec::new();
    let mut after: Option<u64> = None;
    loop {
        let mut action = json!({"kind": "list", "limit": limit});
        if let Some(after) = after {
            action["after"] = json!(after);
        }
        let page = refine(host, ("refinement", refinement), action);
        rows.extend(page["patterns"].as_array().unwrap().iter().cloned());
        match page["page"]["next_after"].as_u64() {
            Some(next) => after = Some(next),
            None => return rows,
        }
    }
}

fn pattern_set(host: &Host, scope: &str, refinement: &str, pattern: u64) -> Value {
    host.ok(
        "triager_select",
        json!({"v": 1, "scope": scope, "producer": {
            "kind": "filter", "syntax": "pattern", "refinement": refinement, "pattern": pattern}}),
    )
}

/// Requirement 2.4: the patterns of a selection are the project's catalogue of its
/// members; a split and a merge are new revisions with their parent, and the earlier
/// revision reads as before; a pattern is a selection with exact membership, its
/// exceptions out and everything else not evaluated; a preview counts without
/// changing anything.
#[test]
fn patterns_split_merge_and_preview_bind_the_projects_refinement_to_immutable_sets() {
    let entries = auth_entries();
    let universe = oracle(&entries, ".log");
    let host = Host::new(&entries, HostOptions::default());
    let archive = host.dir.path().join("private.zip");
    let snapshot = host.open("private")["snapshot"]
        .as_str()
        .unwrap()
        .to_string();
    let b = host.select_filter(&snapshot, "glob", "*login*");
    let b_set = b["set"].as_str().unwrap();
    let b_members = count(&b, "counts", "in");
    let expected: Vec<&Orig> = universe
        .iter()
        .filter(|original| original.text.contains("login"))
        .collect();
    assert_eq!(b_members, expected.len() as u64);

    let native_dir = host.dir.path().join("native");
    let native = native_refinement(&archive, SearchQuery::glob("*login*"), &native_dir);
    assert_eq!(native.rows.len() as u64, b_members);
    assert!(native.groups.len() >= 2, "{}", native.groups.len());

    let before = host.activity(&snapshot);
    let p = refine(&host, ("set", b_set), json!({"kind": "patterns"}));
    let p_handle = p["refinement"].as_str().unwrap().to_string();
    assert_eq!(p["revision"], native.revision.as_str(), "{p}");
    assert_eq!(p["groups"], native.groups.len() as u64);
    assert_eq!(p["set"], b_set);
    assert_eq!(p["parent"], Value::Null);
    assert_eq!(p["members"]["distinct_originals"], b_members);
    assert_eq!(p["cost"]["archive_scans"], 2, "{p}");
    let after = host.activity(&snapshot);
    assert_eq!(
        after["archive_scans"].as_u64().unwrap(),
        before["archive_scans"].as_u64().unwrap() + 2
    );
    // Asked again, the refinement is the same and nothing is read.
    let again = refine(&host, ("set", b_set), json!({"kind": "patterns"}));
    assert_eq!(again, p);
    assert_eq!(
        host.activity(&snapshot)["archive_scans"],
        after["archive_scans"]
    );

    // Every pattern, page by page, with the project's counts; without a projection
    // no text of the source is shown.
    let listed = all_patterns(&host, &p_handle, 3);
    assert_eq!(listed.len(), native.groups.len());
    let mut grouped = 0;
    for (row, group) in listed.iter().zip(&native.groups) {
        assert_eq!(row["pattern"], group.id as u64);
        assert_eq!(row["members"], group.counts.members);
        assert_eq!(row["occurrences"], group.counts.occurrences);
        assert_eq!(row["distinct_texts"], group.counts.distinct_texts);
        assert!(row.get("template").is_none(), "{row}");
        grouped += group.counts.members;
    }
    assert_eq!(
        grouped + p["ungrouped"]["members"].as_u64().unwrap(),
        b_members
    );
    let templates = refine(
        &host,
        ("refinement", &p_handle),
        json!({"kind": "list", "limit": 2, "projection": ["template"]}),
    );
    assert_eq!(
        templates["patterns"][0]["template"],
        native.groups[0].template.as_str()
    );

    // A pattern as a selection: exact members in, the other members of B out, and
    // every original outside B not evaluated.
    for group in &native.groups {
        let set = pattern_set(&host, &snapshot, &p_handle, group.id as u64);
        let handle = set["set"].as_str().unwrap();
        assert_eq!(record_ids(&host, handle), members_of(&native, group.id));
        assert_eq!(count(&set, "counts", "in"), group.counts.members);
        assert_eq!(
            count(&set, "counts", "out"),
            b_members - group.counts.members
        );
        assert_eq!(
            count(&set, "counts", "unknown"),
            universe.len() as u64 - b_members
        );
        assert_eq!(set["cost"]["archive_scans"], 0, "{set}");
        assert_eq!(
            set["producer"]["refinement_revision"],
            native.revision.as_str()
        );
    }

    // Split the largest pattern, as the project's refinement splits it.
    let largest = native
        .groups
        .iter()
        .max_by_key(|group| (group.counts.members, std::cmp::Reverse(group.id)))
        .unwrap();
    let regex = "user=u0";
    let native_split = native.split(largest.id, regex, &budget()).unwrap();
    let split = refine(
        &host,
        ("refinement", &p_handle),
        json!({"kind": "split", "pattern": largest.id, "regex": regex}),
    );
    let split_handle = split["refinement"].as_str().unwrap().to_string();
    assert_ne!(split_handle, p_handle);
    assert_eq!(split["parent"], p_handle.as_str());
    assert_eq!(split["revision"], native_split.revision.as_str());
    assert_eq!(split["groups"], native_split.groups.len() as u64);
    let made = split["changed"]["new_pattern"].as_u64().unwrap();
    let kept = members_of(&native_split, largest.id);
    let off = members_of(&native_split, made as usize);
    assert!(!kept.is_empty() && !off.is_empty());
    assert_eq!(split["changed"]["kept"]["members"], kept.len() as u64);
    assert_eq!(split["changed"]["split_off"]["members"], off.len() as u64);
    assert_eq!(
        split["changed"]["before"]["members"],
        (kept.len() + off.len()) as u64
    );
    let off_set = pattern_set(&host, &snapshot, &split_handle, made);
    assert_eq!(record_ids(&host, off_set["set"].as_str().unwrap()), off);
    // The parent revision is unchanged: the same groups, and the same members under
    // the handle its pattern had before the split.
    assert_eq!(all_patterns(&host, &p_handle, 100), listed);
    let unsplit = pattern_set(&host, &snapshot, &p_handle, largest.id as u64);
    assert_eq!(
        record_ids(&host, unsplit["set"].as_str().unwrap()),
        members_of(&native, largest.id)
    );
    // The same split again is the same revision.
    let same = refine(
        &host,
        ("refinement", &p_handle),
        json!({"kind": "split", "pattern": largest.id, "regex": regex}),
    );
    assert_eq!(same["refinement"], split_handle.as_str());

    // Merging the two halves again gives the members of the pattern before.
    let native_merged = native_split
        .merge(&[largest.id, made as usize], &budget())
        .unwrap();
    let merged = refine(
        &host,
        ("refinement", &split_handle),
        json!({"kind": "merge", "patterns": [largest.id, made]}),
    );
    assert_eq!(merged["revision"], native_merged.revision.as_str());
    assert_eq!(merged["parent"], split_handle.as_str());
    assert_eq!(
        merged["changed"]["merged"]["members"],
        largest.counts.members
    );
    let merged_set = pattern_set(
        &host,
        &snapshot,
        merged["refinement"].as_str().unwrap(),
        largest.id as u64,
    );
    assert_eq!(
        record_ids(&host, merged_set["set"].as_str().unwrap()),
        members_of(&native, largest.id)
    );

    // A preview of a given regex counts what the project's preview lists.
    let native_preview = native
        .preview_query(largest.id, "attempt=[0-3]", &budget())
        .unwrap();
    let preview = refine(
        &host,
        ("refinement", &p_handle),
        json!({"kind": "preview", "pattern": largest.id, "regex": "attempt=[0-3]"}),
    );
    let shown = &preview["preview"];
    for (field, members) in [
        ("member_matches", &native_preview.member_matches),
        ("member_misses", &native_preview.member_misses),
        ("background_matches", &native_preview.background_matches),
        ("background_misses", &native_preview.background_misses),
    ] {
        assert_eq!(shown[field]["count"], members.len() as u64, "{field}");
        let first: Vec<String> = positions(members).into_iter().take(10).collect();
        assert_eq!(shown[field]["first"], json!(first), "{field}");
    }
    assert_eq!(shown["regex_proposed"], false);
    assert!(
        preview.get("refinement").is_none(),
        "a preview makes nothing"
    );
    // Without a regex the project's proposed query is previewed, or refused as the
    // project refuses it.
    for group in &native.groups {
        let action = json!({"kind": "preview", "pattern": group.id});
        match native.proposed_regex(group.id, &budget()) {
            Ok(proposed) => {
                let preview = refine(&host, ("refinement", &p_handle), action);
                assert_eq!(preview["preview"]["regex"], proposed.as_str());
                assert_eq!(preview["preview"]["regex_proposed"], true);
                assert_eq!(
                    preview["preview"]["member_misses"]["count"], 0,
                    "a proposed query matches every member: {preview}"
                );
            }
            Err(_) => {
                let (code, message) = host.err(
                    "triager_refine",
                    json!({"v": 1, "refinement": p_handle, "action": action}),
                );
                assert_eq!(code, "refused", "{message}");
            }
        }
    }

    // What is not a filter over a whole snapshot has no catalogue, and says why.
    let a = host.select_top(&snapshot, 10);
    let (code, message) = host.err(
        "triager_refine",
        json!({"v": 1, "set": a["set"], "action": {"kind": "patterns"}}),
    );
    assert_eq!(code, "refused");
    assert!(message.contains("another producer"), "{message}");
    let inside = host.select_filter(a["set"].as_str().unwrap(), "glob", "*login*");
    let (code, message) = host.err(
        "triager_refine",
        json!({"v": 1, "set": inside["set"], "action": {"kind": "patterns"}}),
    );
    assert_eq!(code, "refused");
    assert!(message.contains("inside another set"), "{message}");
    let (code, _) = host.err(
        "triager_select",
        json!({"v": 1, "scope": snapshot, "producer": {
            "kind": "filter", "syntax": "pattern", "refinement": p_handle, "pattern": 9_999}}),
    );
    assert_eq!(code, "unknown_pattern");
    // A region is composed, and a field or time drill (task 3.1) works on a set, not a
    // refinement.
    for (action, expected) in [
        ("region", "unsupported_operation"),
        ("field", "invalid_arguments"),
        ("time", "invalid_arguments"),
    ] {
        let (code, message) = host.err(
            "triager_refine",
            json!({"v": 1, "refinement": p_handle, "action": {"kind": action}}),
        );
        assert_eq!(code, expected, "{message}");
    }
    // A pattern inside a view: outside the view nothing is evaluated.
    let in_view = pattern_set(
        &host,
        a["set"].as_str().unwrap(),
        &p_handle,
        largest.id as u64,
    );
    assert_eq!(count(&in_view, "counts", "parent"), 10);
    let top: BTreeSet<String> = record_ids(&host, a["set"].as_str().unwrap())
        .into_iter()
        .collect();
    let both: Vec<String> = members_of(&native, largest.id)
        .into_iter()
        .filter(|id| top.contains(id))
        .collect();
    assert_eq!(record_ids(&host, in_view["set"].as_str().unwrap()), both);
}

/// A deterministic encoder for the vector route: eight counts of byte classes.
struct ByteClasses;

impl ExplorationEncoder for ByteClasses {
    fn profile(&self) -> &str {
        "test-byte-classes-v1"
    }

    fn encode(&mut self, text: &str) -> Result<Option<Representation>, L2dError> {
        // A text that names the page 7 has no representation.
        if text.is_empty() || text.contains("/page/7 ") {
            return Ok(None);
        }
        let mut vector = vec![1.0f32; 8];
        for byte in text.bytes() {
            vector[(byte % 8) as usize] += 1.0;
        }
        Ok(Some(Representation::Vector(vector)))
    }

    fn unsupported_reason(&self) -> &str {
        "the test encoder does not represent this text"
    }

    fn scratch_bytes(&self) -> u64 {
        4096
    }
}

struct TestEncoders;

impl EncoderSource for TestEncoders {
    fn session_bytes(&self) -> u64 {
        64 * 1024
    }

    fn profile(&self) -> &str {
        "test-byte-classes-v1"
    }

    fn open(&self, budget: &mut MemoryBudget) -> Result<Box<dyn ExplorationEncoder>, L2dError> {
        budget.require_whole(self.session_bytes(), "the test encoder")?;
        Ok(Box::new(ByteClasses))
    }
}

/// Requirements 2.4 and 6.2: the neighbours of an exemplar are the project's direct
/// exemplar retrieval, within the refinement or with an explicit wider target, and
/// become a selection: neighbours in, the other members that were considered out,
/// members the encoder cannot represent and everything outside not evaluated.
#[test]
fn neighbours_are_the_projects_retrieval_within_a_view_and_in_an_explicit_wider_scope() {
    let entries = auth_entries();
    let universe = oracle(&entries, ".log");
    let options = HostOptions {
        encoder: Some(Encoders(Arc::new(TestEncoders))),
        ..HostOptions::default()
    };
    let host = Host::new(&entries, options);
    let archive = host.dir.path().join("private.zip");
    let snapshot = host.open("private")["snapshot"]
        .as_str()
        .unwrap()
        .to_string();
    let b = host.select_filter(&snapshot, "literal", "login failed user=u00");
    let wide = host.select_filter(&snapshot, "regex", "user=|/page/");
    let p = refine(
        &host,
        ("set", b["set"].as_str().unwrap()),
        json!({"kind": "patterns"}),
    );
    let pw = refine(
        &host,
        ("set", wide["set"].as_str().unwrap()),
        json!({"kind": "patterns"}),
    );
    let (p, pw) = (
        p["refinement"].as_str().unwrap().to_string(),
        pw["refinement"].as_str().unwrap().to_string(),
    );
    let seed = native_refinement(
        &archive,
        SearchQuery::literal("login failed user=u00"),
        &host.dir.path().join("native-seed"),
    );
    let target = native_refinement(
        &archive,
        SearchQuery::regex("user=|/page/"),
        &host.dir.path().join("native-wide"),
    );
    let exemplar = positions(std::slice::from_ref(&seed.rows[3].member)).remove(0);
    let b_members = count(&b, "counts", "in");

    let check = |method: &str,
                 within: &str,
                 with_target: bool,
                 encoder: &mut dyn ExplorationEncoder,
                 radius: f64| {
        let settings = ExploreSettings {
            exemplar_row: 3,
            scope: match within {
                "pattern" => exploration::Scope::WithinPattern,
                "other_patterns" => exploration::Scope::AcrossPatterns,
                _ => exploration::Scope::AllPatterns,
            },
            top_k: 12,
            radius,
            expected_profile: None,
        };
        let report = exploration::explore(
            &seed,
            with_target.then_some(&target),
            &settings,
            encoder,
            &mut budget(),
        )
        .unwrap();
        let mut action = json!({"kind": "neighbours", "exemplar": exemplar, "method": method,
                                "within": within, "top_k": 12, "radius": radius});
        if with_target {
            action["target"] = json!(pw);
        }
        let found = refine(&host, ("refinement", &p), action.clone());
        let handle = found["set"].as_str().unwrap().to_string();
        let mut expected: BTreeMap<String, f64> = report
            .neighbours
            .iter()
            .map(|neighbour| {
                (
                    positions(std::slice::from_ref(&neighbour.row.member)).remove(0),
                    neighbour.distance.unwrap(),
                )
            })
            .collect();
        assert_eq!(expected.len(), report.neighbours.len());
        assert!(
            !expected.is_empty(),
            "{method} {within}: no neighbour to compare"
        );
        expected.insert(exemplar.clone(), 0.0);
        assert_eq!(
            record_ids(&host, &handle),
            expected.keys().cloned().collect::<Vec<_>>(),
            "{method} {within} target {with_target}"
        );
        let considered: BTreeSet<&Member> = seed
            .rows
            .iter()
            .chain(with_target.then_some(&target.rows).into_iter().flatten())
            .map(|row| &row.member)
            .collect();
        let unsupported = report.unsupported.len() as u64;
        assert_eq!(count(&found, "counts", "in"), expected.len() as u64);
        assert_eq!(
            count(&found, "counts", "out"),
            considered.len() as u64 - expected.len() as u64 - unsupported
        );
        assert_eq!(
            count(&found, "counts", "unknown"),
            universe.len() as u64 - considered.len() as u64 + unsupported
        );
        assert_eq!(found["neighbours"]["eligible"], report.eligible);
        assert_eq!(found["neighbours"]["not_represented"], unsupported);
        assert_eq!(found["producer"]["profile"], report.profile.as_str());
        assert_eq!(found["cost"]["archive_scans"], 0);
        // The distance of a neighbour is its evidence.
        let (id, distance) = expected.iter().find(|(id, _)| **id != exemplar).unwrap();
        let evidence = host.ok(
            "triager_inspect",
            json!({"v": 1, "handle": handle, "view": "evidence", "record": id}),
        );
        let shown = evidence["explanation"][0]["distance"].as_f64().unwrap();
        assert!(
            (shown - distance).abs() < 1e-6,
            "{shown} against {distance}"
        );
        // The same search again is the same selection.
        let again = refine(&host, ("refinement", &p), action);
        assert_eq!(again["set"], handle.as_str());
        (found, report)
    };

    let (lexical, _) = check(
        "character_trigram",
        "all",
        false,
        &mut CharacterTrigram,
        0.6,
    );
    assert_eq!(lexical["cost"]["rows_encoded"], 0);
    assert_eq!(
        count(&lexical, "counts", "unknown"),
        universe.len() as u64 - b_members
    );
    check(
        "character_trigram",
        "pattern",
        false,
        &mut CharacterTrigram,
        0.6,
    );
    // Back to the wider scope, named explicitly: neighbours outside the seed search.
    let (wider, report) = check("character_trigram", "all", true, &mut CharacterTrigram, 0.6);
    assert!(
        report
            .neighbours
            .iter()
            .any(|neighbour| neighbour.outside_saved_search),
        "the wider search found nothing outside the seed"
    );
    assert_ne!(wider["set"], lexical["set"]);
    assert_eq!(wider["producer"]["target"], pw.as_str());

    // The vector route: every eligible member is encoded once, and a member without
    // a representation is reported as not covered, not as out.
    let rows_before = host.activity(&snapshot)["rows_encoded"].as_u64().unwrap();
    let (vector, report) = check("vector", "all", true, &mut ByteClasses, 0.05);
    assert!(!report.unsupported.is_empty(), "no member without a vector");
    assert_eq!(vector["cost"]["rows_encoded"], report.eligible + 1);
    assert_eq!(
        host.activity(&snapshot)["rows_encoded"].as_u64().unwrap(),
        rows_before + report.eligible + 1
    );
    let unsupported = positions(std::slice::from_ref(
        &report.unsupported[0].member.row.member,
    ))
    .remove(0);
    let evidence = host.ok(
        "triager_inspect",
        json!({"v": 1, "handle": snapshot, "view": "evidence", "record": unsupported,
               "sets": {"n": vector["set"]}}),
    );
    assert_eq!(evidence["membership"]["n"]["state"], "not_evaluated");
    assert_eq!(evidence["explanation"][0]["reason"], "not_represented");

    // A record that is no member of the refined selection is no exemplar.
    let outsider = universe
        .iter()
        .find(|original| !original.text.contains("login failed"))
        .unwrap();
    let (code, _) = host.err(
        "triager_refine",
        json!({"v": 1, "refinement": p, "action": {
            "kind": "neighbours", "exemplar": outsider.id(), "method": "character_trigram",
            "top_k": 3, "radius": 0.5}}),
    );
    assert_eq!(code, "record_not_in_scope");
    // Without an encoder the vector route is refused by name, before any work.
    let plain = Host::new(&entries, HostOptions::default());
    let snapshot = plain.open("private")["snapshot"]
        .as_str()
        .unwrap()
        .to_string();
    let b = plain.select_filter(&snapshot, "literal", "login failed");
    let p = refine(
        &plain,
        ("set", b["set"].as_str().unwrap()),
        json!({"kind": "patterns"}),
    );
    let (code, message) = plain.err(
        "triager_refine",
        json!({"v": 1, "refinement": p["refinement"], "action": {
            "kind": "neighbours", "exemplar": exemplar, "method": "vector",
            "top_k": 3, "radius": 0.5}}),
    );
    assert_eq!(code, "unsupported_operation", "{message}");
}

/// A saved policy report over the host's private archive as `l2d triage` writes one
/// whose exploration profile is incompatible: every decision is routed to review as
/// unsupported, and the budget allocates the first three by ID.
fn write_policy(host: &Host, opened: &Value, decided: &[&Orig], scorer_revision: &str) {
    let path = host.dir.path().join("private.zip");
    let path = path.to_string_lossy();
    let mut rows: Vec<(String, &Orig)> = decided
        .iter()
        .map(|original| (l2d_store::log_id(&path, &original.text), *original))
        .collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    let decisions: Vec<Value> = rows
        .iter()
        .enumerate()
        .map(|(index, (id, original))| {
            json!({
                "input": {"id": id, "text": original.text, "group": null},
                "score": {"kind": "mark-set", "revision": scorer_revision,
                          "comparison_class_assumed": true,
                          "diagnostic_ratio": 0.75, "actionable_ratio": null},
                "vector": null, "queue": "Review", "reason": "Unsupported",
                "explanation": "incompatible exploration profile; actionable ratio withheld",
                "allocation": match index { 0 => json!("UnknownConflict"),
                                            1 | 2 => json!("General"), _ => Value::Null },
            })
        })
        .collect();
    let interpretation = serde_json::to_value(l2d_core::marks::InputInterpretation::archive(
        &zip_settings(),
    ))
    .unwrap();
    let report = json!({
        "format": 1,
        "query_source": {
            "path": path,
            "content_sha256": opened["source"]["content_sha256"],
            "archive_entries": ["*.log"],
            "interpretation": interpretation,
        },
        "policy": {
            "version": "test-v1", "selection_provenance": "synthetic test", "low": 0.2, "high": 0.8,
            "corroboration": 1,
            "reach": {"radius": 0.2, "top_k": 2, "minimum_labelled_support": 1,
                      "expected_profile": "expected"},
            "require_calibration": false, "review_budget": 4, "unknown_slots": 1,
            "audit_slots": 1, "audit_seed": 5
        },
        "eligibility": l2d_decide::policy::ELIGIBILITY,
        "allocation_rule": l2d_decide::policy::ALLOCATION,
        "scorer_revision": scorer_revision, "evidence_revision": "history-v1",
        "actual_profile": "incompatible",
        "calibration": "Missing",
        "qualification": "Exploratory; qualification incomplete. Policy mechanics and local support do not establish deployment quality.",
        "decisions": decisions,
        "unknown": {"configured": 1, "selected": 1, "unused": 0},
        "general": {"configured": 2, "selected": 2, "unused": 0},
        "audit": {"configured": 1, "selected": 0, "unused": 1},
        "deferred": decided.len() - 3
    });
    std::fs::write(
        host.dir.path().join("policy-v1.json"),
        serde_json::to_vec(&report).unwrap(),
    )
    .unwrap();
}

/// Requirements 2.2, 2.3 and 3.2: C from the saved Venn policy keeps who recommended,
/// what was assessed, the predicate, the native route, reason and allocation, tells
/// an assessed negative from a record outside the report, and shows its dependence
/// on the scorer of A when both use that scorer.
#[test]
fn c_from_a_saved_venn_policy_keeps_routes_reasons_coverage_and_the_scorer_it_depends_on() {
    let entries = auth_entries();
    let universe = oracle(&entries, ".log");
    let options = HostOptions {
        policies: vec!["v1".into()],
        ..HostOptions::default()
    };
    let host = Host::new(&entries, options);
    let opened = host.open("private");
    let snapshot = opened["snapshot"].as_str().unwrap().to_string();
    let decided: Vec<&Orig> = universe.iter().step_by(7).collect();
    // The policy scored with the revision of the scorer `sum`.
    write_policy(&host, &opened, &decided, "rev-1");
    let a = host.select_top(&snapshot, 40);
    let b = host.select_filter(&snapshot, "glob", "*login*");
    let venn = |scope: &str, membership: &str| {
        host.ok(
            "triager_select",
            json!({"v": 1, "scope": scope, "producer": {
                "kind": "recommend", "method": "venn", "policy": "v1", "membership": membership}}),
        )
    };
    let before = host.activity(&snapshot);
    let c = venn(&snapshot, "investigate_or_review");
    let c_set = c["set"].as_str().unwrap();
    // Nothing of the source is read and nothing is scored or encoded for it.
    assert_eq!(
        host.activity(&snapshot)["archive_scans"],
        before["archive_scans"]
    );
    assert_eq!(
        host.activity(&snapshot)["rows_scored"],
        before["rows_scored"]
    );
    assert_eq!(c["cost"]["archive_scans"], 0);
    let mut decided_ids: Vec<String> = decided.iter().map(|original| original.id()).collect();
    decided_ids.sort();
    assert_eq!(record_ids(&host, c_set), decided_ids);
    assert_eq!(count(&c, "counts", "in"), decided.len() as u64);
    assert_eq!(count(&c, "counts", "out"), 0);
    assert_eq!(
        count(&c, "counts", "unknown"),
        (universe.len() - decided.len()) as u64
    );
    assert_eq!(
        c["coverage"]["reasons"][0]["reason"],
        "outside_producer_coverage"
    );
    assert_eq!(c["producer"]["method"], "venn");
    assert_eq!(c["producer"]["policy"], "v1");
    assert_eq!(c["producer"]["policy_version"], "test-v1");
    assert_eq!(c["producer"]["membership"], "investigate_or_review");
    assert_eq!(c["producer"]["calibration"]["state"], "missing");
    assert_eq!(c["producer"]["depends_on"][0]["revision"], "rev-1");
    assert_eq!(c["producer"]["depends_on"][1]["revision"], "history-v1");
    assert_eq!(c["policy"]["routes"]["review"], decided.len() as u64);
    assert_eq!(c["policy"]["reasons"]["unsupported"], decided.len() as u64);
    assert_eq!(c["policy"]["allocations"]["unknown_conflict"], 1);
    assert_eq!(c["policy"]["allocations"]["general"], 2);
    assert_eq!(
        c["policy"]["allocations"]["deferred"],
        decided.len() as u64 - 3
    );
    // The dependence on the scorer of A is stated where C is made and where A and C
    // are read together.
    let read = host.ok(
        "triager_inspect",
        json!({"v": 1, "handle": c_set, "view": "overview"}),
    );
    assert_eq!(read["dependencies"][0]["set"], a["set"]);
    assert_eq!(read["dependencies"][0]["model"], "sum");
    let regions = host.ok(
        "triager_inspect",
        json!({"v": 1, "handle": snapshot, "view": "regions",
               "sets": {"a": a["set"], "b": b["set"], "c": c_set}}),
    );
    assert_eq!(
        regions["dependencies"].as_array().unwrap().len(),
        1,
        "{regions}"
    );
    assert_eq!(regions["dependencies"][0]["set"], "c");
    assert_eq!(regions["dependencies"][0]["depends_on"], "a");
    assert_eq!(regions["regions"].as_array().unwrap().len(), 8);
    let in_c: u64 = regions["regions"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|region| region["membership"]["c"] == "in")
        .map(|region| region["distinct_originals"].as_u64().unwrap())
        .sum();
    assert_eq!(in_c, decided.len() as u64);
    assert_eq!(
        regions["unknown"]["distinct_originals"],
        (universe.len() - decided.len()) as u64
    );
    let apart = host.ok(
        "triager_inspect",
        json!({"v": 1, "handle": snapshot, "view": "regions", "sets": {"b": b["set"], "c": c_set}}),
    );
    assert_eq!(apart["dependencies"], json!([]));

    // Why a record is in C: the native route, reason and allocation, as a suggestion.
    let path = host.dir.path().join("private.zip");
    let mut by_native: Vec<(String, &Orig)> = decided
        .iter()
        .map(|original| {
            (
                l2d_store::log_id(&path.to_string_lossy(), &original.text),
                *original,
            )
        })
        .collect();
    by_native.sort_by(|a, b| a.0.cmp(&b.0));
    for (index, expected) in [(0, "unknown_conflict"), (1, "general"), (2, "general")] {
        let evidence = host.ok(
            "triager_inspect",
            json!({"v": 1, "handle": snapshot, "view": "evidence",
                   "record": by_native[index].1.id(), "sets": {"a": a["set"], "c": c_set}}),
        );
        let explained = &evidence["explanation"][1];
        assert_eq!(explained["state"], "in");
        assert_eq!(explained["venn"]["route"], "review");
        assert_eq!(explained["venn"]["reason"], "unsupported");
        assert_eq!(explained["venn"]["allocation"], expected);
        assert_eq!(explained["venn"]["actionable_ratio"], "withheld");
        assert_eq!(explained["venn"]["comparison_class"], "assumed");
        assert!(
            explained["venn"]["kind"]
                .as_str()
                .unwrap()
                .contains("not a reviewed outcome"),
            "{explained}"
        );
        assert_eq!(evidence["dependencies"][0]["depends_on"], "a");
        assert_eq!(evidence["record"]["native_id"], by_native[index].0.as_str());
    }
    let deferred = host.ok(
        "triager_inspect",
        json!({"v": 1, "handle": snapshot, "view": "evidence",
               "record": by_native[5].1.id(), "sets": {"c": c_set}}),
    );
    assert_eq!(
        deferred["explanation"][0]["venn"]["allocation"],
        Value::Null
    );
    assert_eq!(
        deferred["explanation"][0]["venn"]["deferred_by_policy"],
        true
    );
    let outside = universe
        .iter()
        .find(|original| !decided.contains(original))
        .unwrap();
    let unassessed = host.ok(
        "triager_inspect",
        json!({"v": 1, "handle": snapshot, "view": "evidence",
               "record": outside.id(), "sets": {"c": c_set}}),
    );
    assert_eq!(unassessed["explanation"][0]["state"], "not_evaluated");
    assert_eq!(
        unassessed["explanation"][0]["reason"],
        "outside_producer_coverage"
    );
    assert!(unassessed["explanation"][0].get("venn").is_none());

    // Another predicate is another set: review is an assessed negative of
    // `investigate`, told apart from a record the policy did not decide.
    let strict = venn(&snapshot, "investigate");
    assert_ne!(strict["set"], c["set"]);
    assert_eq!(count(&strict, "counts", "in"), 0);
    assert_eq!(count(&strict, "counts", "out"), decided.len() as u64);
    assert_eq!(
        count(&strict, "counts", "unknown"),
        (universe.len() - decided.len()) as u64
    );
    // `in - not_evaluated` stays unknown: A minus C keeps what C did not assess apart.
    let view = host.ok(
        "triager_compose",
        json!({"v": 1, "mode": "view", "snapshot": snapshot,
               "sets": {"a": a["set"], "c": strict["set"]}, "expr": {"difference": ["a", "c"]}}),
    );
    let a_ids: BTreeSet<String> = record_ids(&host, a["set"].as_str().unwrap())
        .into_iter()
        .collect();
    let assessed_in_a = decided_ids.iter().filter(|id| a_ids.contains(*id)).count() as u64;
    assert_eq!(count(&view, "counts", "in"), assessed_in_a);
    assert_eq!(count(&view, "counts", "unknown"), 40 - assessed_in_a);
    // Inside B the policy fills coverage only for members of B.
    let within = venn(b["set"].as_str().unwrap(), "investigate_or_review");
    let b_ids: BTreeSet<String> = record_ids(&host, b["set"].as_str().unwrap())
        .into_iter()
        .collect();
    let expected: Vec<String> = decided_ids
        .iter()
        .filter(|id| b_ids.contains(*id))
        .cloned()
        .collect();
    assert_eq!(record_ids(&host, within["set"].as_str().unwrap()), expected);
    assert_eq!(count(&within, "counts", "parent"), b_ids.len() as u64);

    // A report that does not validate, a report of another source and an unknown
    // name are refused; the path of the host's file is never shown.
    let (code, _) = host.err(
        "triager_select",
        json!({"v": 1, "scope": snapshot, "producer": {
            "kind": "recommend", "method": "venn", "policy": "v9",
            "membership": "investigate"}}),
    );
    assert_eq!(code, "unknown_policy");
    let report = host.dir.path().join("policy-v1.json");
    let saved = std::fs::read_to_string(&report).unwrap();
    std::fs::write(&report, saved.replace("\"deferred\":", "\"deferred\":1")).unwrap();
    let (code, message) = host.err(
        "triager_select",
        json!({"v": 1, "scope": snapshot, "producer": {
            "kind": "recommend", "method": "venn", "policy": "v1",
            "membership": "investigate_or_review"}}),
    );
    assert_eq!(code, "refused", "{message}");
    assert!(message.contains("validation"), "{message}");
    assert!(!message.contains("policy-v1.json"), "{message}");
    let other = host.ok(
        "triager_open",
        json!({"v": 1, "source": {"ref": "private"}}),
    );
    std::fs::write(&report, &saved).unwrap();
    let (code, message) = host.err(
        "triager_select",
        json!({"v": 1, "scope": other["snapshot"], "producer": {
            "kind": "recommend", "method": "venn", "policy": "v1",
            "membership": "investigate_or_review"}}),
    );
    assert_eq!(code, "refused", "{message}");
    assert!(message.contains("entry patterns"), "{message}");
    // The committed C still reads as it was made.
    assert_eq!(record_ids(&host, c_set), decided_ids);
}

fn issued(host: &Host, scope: &str, mut producer: Value) -> Value {
    producer["kind"] = json!("recommend");
    producer["method"] = json!("issued_batch");
    host.ok(
        "triager_select",
        json!({"v": 1, "scope": scope, "producer": producer}),
    )
}

fn issued_err(host: &Host, scope: &str, mut producer: Value) -> (String, String) {
    producer["kind"] = json!("recommend");
    producer["method"] = json!("issued_batch");
    host.err(
        "triager_select",
        json!({"v": 1, "scope": scope, "producer": producer}),
    )
}

fn batch_ids(batch: &Value) -> Vec<String> {
    batch["records"]
        .as_array()
        .unwrap()
        .iter()
        .map(|record| record["id"].as_str().unwrap().to_string())
        .collect()
}

/// Requirements 2.2, 3.2, 4.4 and 6.2: the issued-batch protocol on an explicit
/// public fixture with the test as its model client: bounded batches under the
/// evidence allowance, a partial submission, a later revision, and duplicate, foreign,
/// stale and mismatching submissions refused whole.
#[test]
fn issued_batches_take_partial_and_later_submissions_and_refuse_duplicate_and_foreign_ones() {
    let entries = auth_entries();
    let universe = oracle(&entries, ".log");
    let options = HostOptions {
        profile: DisclosureProfile::PublicDevelopment,
        evidence_allowance_bytes: 64 * 1024,
        ..HostOptions::default()
    };
    let host = Host::new(&entries, options);
    let snapshot = host.open("public-fixture")["snapshot"]
        .as_str()
        .unwrap()
        .to_string();
    let a = host.select_filter(&snapshot, "literal", "user=");
    let b = host.select_filter(&snapshot, "literal", "login ok");
    let d = host.ok(
        "triager_compose",
        json!({"v": 1, "mode": "view", "snapshot": snapshot,
               "sets": {"a": a["set"], "b": b["set"]}, "expr": {"difference": ["a", "b"]}}),
    );
    let d_view = d["view"].as_str().unwrap().to_string();
    let d_ids = record_ids(&host, &d_view);
    let d_members = d_ids.len() as u64;
    assert!(d_members >= 20, "{d_members}");
    let text_of = |id: &str| -> &str {
        &universe
            .iter()
            .find(|original| original.id() == id)
            .unwrap()
            .text
    };

    // A batch: the next members of D by record ID, with their excerpts, counted
    // against the allowance.
    let before = host.activity(&snapshot);
    let g1 = issued(
        &host,
        &d_view,
        json!({"action": "issue", "limit": 8, "projection": ["excerpt"]}),
    );
    let g1_ids = batch_ids(&g1);
    assert_eq!(g1_ids, d_ids[..8]);
    assert_eq!(g1["revision"], 1);
    assert_eq!(g1["issued"]["not_issued"], d_members - 8);
    let mut disclosed = 0;
    for record in g1["records"].as_array().unwrap() {
        let text = text_of(record["id"].as_str().unwrap());
        assert_eq!(record["excerpt"], text);
        disclosed += text.len() as u64;
    }
    assert_eq!(g1["disclosure"]["excerpt_bytes"], disclosed);
    assert_eq!(g1["disclosure"]["source_class"], "public_fixture");
    assert_eq!(
        g1["disclosure"]["remaining_evidence_bytes"],
        64 * 1024 - disclosed
    );
    assert_eq!(g1["cost"]["archive_scans"], 1);
    let after = host.activity(&snapshot);
    assert_eq!(after["genai_rows_issued"], 8);
    assert_eq!(after["rows_scored"], before["rows_scored"]);
    let g1_handle = g1["batch"].as_str().unwrap().to_string();

    let submit = |batch: &str, revision: u64, model: &str, items: Value| {
        json!({"action": "submit", "batch": batch, "expected_revision": revision,
               "model": model, "prompt_revision": "p1", "items": items})
    };
    let item = |id: &str, outcome: &str| {
        json!({"id": id, "outcome": outcome,
                                                 "reason": format!("{outcome} by the test model")})
    };

    // Refused whole, before anything is taken: an ID twice, an ID of another batch,
    // an ID outside D, an outcome that is none, a revision that is not the batch's.
    let (code, _) = issued_err(
        &host,
        &d_view,
        submit(
            &g1_handle,
            1,
            "dev/model@1",
            json!([item(&g1_ids[0], "recommend"), item(&g1_ids[0], "pass")]),
        ),
    );
    assert_eq!(code, "duplicate_item");
    let (code, message) = issued_err(
        &host,
        &d_view,
        submit(
            &g1_handle,
            1,
            "dev/model@1",
            json!([item(&g1_ids[0], "recommend"), item(&d_ids[9], "recommend")]),
        ),
    );
    assert_eq!(code, "foreign_item", "{message}");
    assert!(message.contains(&d_ids[9]), "{message}");
    let outside_d = universe
        .iter()
        .find(|original| !d_ids.contains(&original.id()))
        .unwrap();
    let (code, _) = issued_err(
        &host,
        &d_view,
        submit(
            &g1_handle,
            1,
            "dev/model@1",
            json!([item(&outside_d.id(), "recommend")]),
        ),
    );
    assert_eq!(code, "foreign_item");
    let (code, _) = issued_err(
        &host,
        &d_view,
        submit(
            &g1_handle,
            1,
            "dev/model@1",
            json!([item(&g1_ids[0], "maybe")]),
        ),
    );
    assert_eq!(code, "invalid_arguments");
    let (code, _) = issued_err(
        &host,
        &d_view,
        submit(
            &g1_handle,
            2,
            "dev/model@1",
            json!([item(&g1_ids[0], "recommend")]),
        ),
    );
    assert_eq!(code, "stale_revision");
    // A submission fills coverage only in the scope its batch was issued for.
    let (code, _) = issued_err(
        &host,
        a["set"].as_str().unwrap(),
        submit(
            &g1_handle,
            1,
            "dev/model@1",
            json!([item(&g1_ids[0], "recommend")]),
        ),
    );
    assert_eq!(code, "run_mismatch");
    assert_eq!(host.activity(&snapshot)["genai_rows_submitted"], 0);
    let (code, _) = issued_err(&host, &d_view, json!({"action": "collect"}));
    assert_eq!(code, "refused", "nothing was submitted yet");

    // A partial submission: five of eight judged, three omitted.
    let partial = submit(
        &g1_handle,
        1,
        "dev/model@1",
        json!([
            item(&g1_ids[0], "recommend"),
            item(&g1_ids[1], "recommend"),
            item(&g1_ids[2], "pass"),
            item(&g1_ids[3], "abstain"),
            item(&g1_ids[4], "recommend")
        ]),
    );
    let c1 = issued(&host, &d_view, partial.clone());
    let c1_set = c1["set"].as_str().unwrap().to_string();
    assert_eq!(count(&c1, "counts", "parent"), d_members);
    assert_eq!(count(&c1, "counts", "in"), 3);
    assert_eq!(count(&c1, "counts", "out"), 1);
    assert_eq!(count(&c1, "counts", "unknown"), d_members - 4);
    assert_eq!(c1["coverage"]["reasons"][0]["reason"], "abstained");
    assert_eq!(c1["coverage"]["reasons"][0]["distinct_originals"], 1);
    assert_eq!(c1["run_coverage"]["omitted"], 3);
    assert_eq!(c1["run_coverage"]["not_issued"], d_members - 8);
    assert_eq!(c1["producer"]["model"]["name"], "dev/model@1");
    assert_eq!(c1["producer"]["model"]["identity"], "caller_reported");
    assert_eq!(c1["producer"]["prompt_revision"], "p1");
    assert_eq!(c1["accepted"]["batch_revision"], 2);
    assert_eq!(host.activity(&snapshot)["genai_rows_submitted"], 5);
    assert_eq!(
        record_ids(&host, &c1_set),
        [g1_ids[0].clone(), g1_ids[1].clone(), g1_ids[4].clone()]
    );
    // The same payload again is a retry: the same selection, nothing taken twice.
    let retry = issued(&host, &d_view, partial);
    assert_eq!(retry["set"], c1_set.as_str());
    assert_eq!(host.activity(&snapshot)["genai_rows_submitted"], 5);
    // Another model for the same run is refused.
    let (code, _) = issued_err(
        &host,
        &d_view,
        submit(
            &g1_handle,
            2,
            "other/model@2",
            json!([item(&g1_ids[5], "pass")]),
        ),
    );
    assert_eq!(code, "run_mismatch");

    // A later revision judges what was omitted, and judges one record differently:
    // that record is not evaluated from then on.
    let c2 = issued(
        &host,
        &d_view,
        submit(
            &g1_handle,
            2,
            "dev/model@1",
            json!([
                item(&g1_ids[5], "recommend"),
                item(&g1_ids[6], "pass"),
                item(&g1_ids[7], "pass"),
                item(&g1_ids[1], "pass")
            ]),
        ),
    );
    let c2_set = c2["set"].as_str().unwrap().to_string();
    assert_ne!(c2_set, c1_set);
    assert_eq!(count(&c2, "counts", "in"), 3);
    assert_eq!(count(&c2, "counts", "out"), 3);
    assert_eq!(c2["run_coverage"]["conflicting"], 1);
    assert_eq!(c2["run_coverage"]["omitted"], 0);
    assert_eq!(
        record_ids(&host, &c2_set),
        [g1_ids[0].clone(), g1_ids[4].clone(), g1_ids[5].clone()]
    );
    // The earlier revision is unchanged.
    assert_eq!(
        record_ids(&host, &c1_set),
        [g1_ids[0].clone(), g1_ids[1].clone(), g1_ids[4].clone()]
    );
    let explain = |set: &str, id: &str| {
        host.ok(
            "triager_inspect",
            json!({"v": 1, "handle": d_view, "view": "evidence", "record": id,
                   "sets": {"c": set}}),
        )["explanation"][1]
            .clone()
    };
    let conflicting = explain(&c2_set, &g1_ids[1]);
    assert_eq!(conflicting["state"], "not_evaluated");
    assert_eq!(
        conflicting["recommendation"]["outcome"],
        "conflicting_submissions"
    );
    assert_eq!(
        conflicting["recommendation"]["judgments"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    let earlier = explain(&c1_set, &g1_ids[1]);
    assert_eq!(earlier["state"], "in");
    assert_eq!(earlier["recommendation"]["outcome"], "recommend");
    assert_eq!(
        earlier["recommendation"]["judgments"][0]["reason"],
        "recommend by the test model"
    );
    // A suggestion is not a review, and says so (requirement 3.2).
    assert!(
        earlier["recommendation"]["kind"]
            .as_str()
            .unwrap()
            .contains("not a reviewed outcome")
    );
    assert_eq!(explain(&c1_set, &g1_ids[3])["reason"], "abstained");
    assert_eq!(
        explain(&c1_set, &g1_ids[6])["recommendation"]["outcome"],
        "omitted"
    );
    assert_eq!(
        explain(&c1_set, &d_ids[12])["recommendation"]["outcome"],
        "not_issued"
    );

    // The next batch goes on after the first; the run is collected into one set.
    let g2 = issued(
        &host,
        &d_view,
        json!({"action": "issue", "limit": 6, "projection": ["excerpt"]}),
    );
    let g2_ids = batch_ids(&g2);
    assert_eq!(g2_ids, d_ids[8..14]);
    assert_ne!(g2["batch"], g1["batch"]);
    let c3 = issued(
        &host,
        &d_view,
        submit(
            g2["batch"].as_str().unwrap(),
            1,
            "dev/model@1",
            json!([item(&g2_ids[0], "recommend"), item(&g2_ids[1], "pass")]),
        ),
    );
    let collected = issued(&host, &d_view, json!({"action": "collect"}));
    assert_eq!(
        collected["set"], c3["set"],
        "collecting adds nothing to the last submission"
    );
    assert_eq!(count(&collected, "counts", "in"), 4);
    assert_eq!(count(&collected, "counts", "out"), 4);
    assert_eq!(collected["run_coverage"]["issued"], 14);
    assert_eq!(collected["run_coverage"]["not_issued"], d_members - 14);
    assert_eq!(collected["producer"]["batches"], 2);
    assert_eq!(collected["producer"]["submissions"], 3);
    let reference = collected["producer"]["submissions_ref"].as_str().unwrap();
    let bound = issued(
        &host,
        &d_view,
        json!({"action": "collect", "submissions_ref": reference}),
    );
    assert_eq!(bound["set"], collected["set"]);
    let (code, _) = issued_err(
        &host,
        &d_view,
        json!({"action": "collect",
               "submissions_ref": c1["producer"]["submissions_ref"]}),
    );
    assert_eq!(code, "stale_revision");
    // C in the set algebra: what the model did not judge stays unknown in D ∩ C.
    let reduced = host.ok(
        "triager_compose",
        json!({"v": 1, "mode": "view", "snapshot": snapshot, "scope": d_view,
               "sets": {"c": collected["set"]}, "expr": "c"}),
    );
    assert_eq!(count(&reduced, "counts", "in"), 4);
    assert_eq!(count(&reduced, "counts", "unknown"), d_members - 8);

    // Bounds of the protocol.
    let (code, _) = issued_err(&host, &snapshot, json!({"action": "issue"}));
    assert_eq!(
        code, "invalid_arguments",
        "a whole snapshot is no candidate scope"
    );
    let (code, _) = issued_err(
        &host,
        &d_view,
        json!({"action": "issue", "run": "other", "limit": 101}),
    );
    assert_eq!(code, "invalid_arguments");
    let (code, _) = issued_err(&host, &d_view, json!({"action": "collect", "run": "none"}));
    assert_eq!(code, "unknown_run");
    let (code, message) = host.err(
        "triager_select",
        json!({"v": 1, "scope": d_view, "producer": {"kind": "recommend", "method": "local_model"}}),
    );
    assert_eq!(code, "unsupported_operation", "{message}");
    // A run without excerpts discloses nothing and reads no source.
    let scans = host.activity(&snapshot)["archive_scans"].clone();
    let ids_only = issued(
        &host,
        &d_view,
        json!({"action": "issue", "run": "ids", "limit": 100}),
    );
    assert_eq!(batch_ids(&ids_only), d_ids);
    assert_eq!(ids_only["disclosure"]["excerpt_bytes"], 0);
    assert_eq!(host.activity(&snapshot)["archive_scans"], scans);
    let (code, _) = issued_err(&host, &d_view, json!({"action": "issue", "run": "ids"}));
    assert_eq!(code, "run_exhausted");
}

/// Requirement 4.4: under the public-development profile a private source cannot
/// reach an issued batch or a pattern template, and batches stop at the evidence
/// allowance of the session; under the local profile the local client is issued the
/// private source.
#[test]
fn private_evidence_cannot_reach_issued_batches_and_the_allowance_bounds_them() {
    let entries = auth_entries();
    let options = HostOptions {
        profile: DisclosureProfile::PublicDevelopment,
        evidence_allowance_bytes: 100,
        ..HostOptions::default()
    };
    let host = Host::new(&entries, options);
    // The private source is not opened by reference or by path: no handle of it
    // exists that a batch could be issued for.
    let mine = host.dir.path().join("allowed").join("mine.zip");
    for source in [
        json!({"ref": "private"}),
        json!({"path": mine.to_string_lossy()}),
    ] {
        let (code, message) = host.err("triager_open", json!({"v": 1, "source": source}));
        assert_eq!(code, "disclosure_refused", "{message}");
    }
    assert!(!host.work().join("snapshot-1").exists());
    let snapshot = host.open("public-fixture")["snapshot"]
        .as_str()
        .unwrap()
        .to_string();
    let b = host.select_filter(&snapshot, "literal", "login failed");
    let b_set = b["set"].as_str().unwrap();
    // The allowance of 100 bytes holds three excerpts of the fixture, not eight.
    let batch = issued(
        &host,
        b_set,
        json!({"action": "issue", "limit": 8, "projection": ["excerpt"]}),
    );
    let records = batch["records"].as_array().unwrap();
    let disclosed: u64 = records
        .iter()
        .map(|record| record["excerpt"].as_str().unwrap().len() as u64)
        .sum();
    assert!(
        !records.is_empty() && records.len() < 8,
        "{}",
        records.len()
    );
    assert!(disclosed <= 100);
    assert_eq!(batch["clipping"]["rows_by_allowance"], true);
    assert_eq!(
        batch["disclosure"]["remaining_evidence_bytes"],
        100 - disclosed
    );
    let (code, message) = issued_err(
        &host,
        b_set,
        json!({"action": "issue", "limit": 8, "projection": ["excerpt"]}),
    );
    assert_eq!(code, "disclosure_refused", "{message}");
    // Templates and proposed queries are text of the source and count the same way.
    let p = refine(&host, ("set", b_set), json!({"kind": "patterns"}));
    let (code, _) = host.err(
        "triager_refine",
        json!({"v": 1, "refinement": p["refinement"],
               "action": {"kind": "list", "projection": ["template"]}}),
    );
    assert_eq!(code, "disclosure_refused");
    // Counts, handles and memberships stay available.
    let listed = refine(
        &host,
        ("refinement", p["refinement"].as_str().unwrap()),
        json!({"kind": "list"}),
    );
    assert!(
        listed["patterns"]
            .as_array()
            .unwrap()
            .iter()
            .all(|row| row.get("template").is_none())
    );
    let ids = issued(
        &host,
        b_set,
        json!({"action": "issue", "run": "ids", "limit": 5}),
    );
    assert_eq!(ids["records"].as_array().unwrap().len(), 5);

    // The local profile: the client is the local analyst or a local model, and the
    // private source is issued to it without an allowance.
    let local = Host::new(&entries, HostOptions::default());
    let snapshot = local.open("private")["snapshot"]
        .as_str()
        .unwrap()
        .to_string();
    let b = local.select_filter(&snapshot, "literal", "login failed");
    let batch = issued(
        &local,
        b["set"].as_str().unwrap(),
        json!({"action": "issue", "limit": 8, "projection": ["excerpt"]}),
    );
    assert_eq!(batch["records"].as_array().unwrap().len(), 8);
    assert_eq!(batch["disclosure"]["profile"], "private_local");
    assert_eq!(batch["disclosure"]["source_class"], "private");
    assert_eq!(batch["disclosure"]["remaining_evidence_bytes"], Value::Null);
}
