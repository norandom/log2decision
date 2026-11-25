use super::*;
use crate::flow::{Occurrence, Recommendation};
use l2d_core::MemorySettings;

fn budget() -> MemoryBudget {
    MemoryBudget::observe_with(&MemorySettings::default(), || Ok(4_u64 << 30)).unwrap()
}

fn record(id: &str, text: &str, score: f64) -> Record {
    Record {
        id: id.into(),
        text: text.into(),
        score,
        recommendation: Recommendation::In,
        reason: "policy".into(),
        occurrences: vec![],
    }
}

fn members(records: &[Record]) -> Vec<Member> {
    (0..records.len())
        .map(|record| Member {
            record,
            occurrences: 1,
            allocation: None,
        })
        .collect()
}

/// IDs selected by `query`, in pipeline order.
fn run(query: &str, records: &[Record]) -> Vec<String> {
    let budget = budget();
    let query = Query::parse(query, &budget).unwrap_or_else(|e| panic!("{query}: {e}"));
    query
        .select(records, members(records), None, 0, &budget)
        .unwrap()
        .into_iter()
        .map(|m| records[m.record].id.clone())
        .collect()
}

fn hits(query: &str, text: &str) -> bool {
    !run(query, &[record("R1", text, 0.75)]).is_empty()
}

#[test]
fn bare_terms_follow_splunk_segmentation() {
    for (query, text, expected) in [
        ("fail", "login failed for guest", false),
        ("fail*", "login failed for guest", true),
        ("*ail", "mail queue", true),
        ("error", "error_code=7", true),
        ("user", "username=bob", false),
        ("0.0.1", "from 10.0.0.1 port 22", true),
        ("10.0.0.1", "src=10.0.0.1,dst=x", true),
        ("10.0.0", "src=10.0.0.1", true),
        ("/admin", "GET x/admin/login", true),
        ("LOGIN", "Login denied", true),
        (r#""locked out""#, "account LOCKED OUT now", true),
        (r#""lock""#, "account locked", false),
        ("*", "", true),
        ("a*z", "a.b-z", true),
        ("a*z", "a b z", false),
        (r"literal\*", "a literal* b", true),
        (r"literal\*", "a literally", false),
        ("?", "what? now", true),
    ] {
        assert_eq!(hits(query, text), expected, "{query} in {text:?}");
    }
}

#[test]
fn term_and_case_directives_narrow_matching() {
    assert!(hits("TERM(10.0.0.1)", "from 10.0.0.1 port"));
    assert!(!hits("TERM(10.0.0.1)", "src=10.0.0.1 port"));
    assert!(!hits("TERM(10.0.0)", "from 10.0.0.1 port"));
    assert!(hits("CASE(Error)", "Error: disk"));
    assert!(!hits("CASE(Error)", "error: disk"));
    assert!(hits(r#"CASE("Disk Full")"#, "warn Disk Full"));
}

#[test]
fn boolean_precedence_and_pipe_searches_filter_the_previous_results() {
    let q = r#"search alpha OR bravo charlie | search NOT "locked out""#;
    for (text, expected) in [
        ("alpha", false),
        ("bravo", false),
        ("alpha CHARLIE", true),
        ("bravo charlie", true),
        ("alpha charlie LOCKED OUT", false),
    ] {
        assert_eq!(hits(q, text), expected, "{text}");
    }
    let q = r#"alpha OR (bravo AND charlie) | search "a|b" OR id=R1"#;
    assert!(hits(q, "alpha"));
    assert!(!hits(q, "bravo"));
    assert!(hits("zero | search alpha OR bravo", "zero bravo"));
    assert!(!hits("zero | search alpha OR bravo", "bravo"));
    assert!(hits("alpha ```ignored | stats``` bravo", "alpha bravo"));
    assert!(hits(r"a\|b", "x a|b y"));
}

#[test]
fn fields_use_built_ins_then_automatic_key_value_extraction() {
    let text = r#"login user=bob status=200 msg="disk full" path=/a/b,next=1"#;
    for (query, expected) in [
        ("user=bob", true),
        ("user=BOB", true),
        ("user=b*", true),
        ("USER=bob", false),
        ("status>=200 status<300", true),
        ("status=200.0", true),
        ("status>3", true),
        (r#"msg="disk full""#, true),
        ("path=/a/b", true),
        ("next=1", true),
        ("missing=x", false),
        ("missing!=x", false),
        ("NOT missing=x", true),
        ("missing=*", false),
        ("user=*", true),
        ("user!=alice", true),
        (r#"user IN (alice, "bob")"#, true),
        ("user IN (al*, carol)", false),
        ("id=r1 score>=0.75 c=RECOMMENDED reason=pol*", true),
        ("score>0.75", false),
        (r#"_raw="login*full*""#, true),
        ("linecount=1", true),
        (r#"punct="_*=*""#, true),
    ] {
        assert_eq!(hits(query, text), expected, "{query}");
    }
}

#[test]
fn where_evaluates_spl_expressions_with_case_sensitive_strings() {
    let text = "src=10.1.2.3 user=Bob status=404 bytes=1500";
    for (query, expected) in [
        ("* | where status>=400 AND status<500", true),
        (r#"* | where user="Bob""#, true),
        (r#"* | where user="bob""#, false),
        ("* | where user=Bob", false),
        (r#"* | where lower(user)="bob" AND len(user)==3"#, true),
        (r#"* | where like(user, "B%") AND user LIKE "_ob""#, true),
        (r#"* | where match(_raw, "status=4\d\d")"#, true),
        (r#"* | where cidrmatch("10.0.0.0/8", src)"#, true),
        (r#"* | where cidrmatch("192.168.0.0/16", src)"#, false),
        (r#"* | where searchmatch("user=bob status=404")"#, true),
        (
            r#"* | where tonumber("0x1F", 16) = 31 AND bytes/1000 = 1.5"#,
            true,
        ),
        (r#"* | where if(score>0.5, "hi", "lo") == "hi""#, true),
        (
            r#"* | where case(status=200, "ok", status=404, "missing") = "missing""#,
            true,
        ),
        (
            "* | where isnull(nothing) AND isnotnull(user) AND isnum(status)",
            true,
        ),
        (r#"* | where coalesce(nothing, user) . "!" = "Bob!""#, true),
        (
            r#"* | where substr(user, 2) = "ob" AND substr(user, -2, 1) = "o""#,
            true,
        ),
        (
            r#"* | where replace(user, "(B)ob", "\1ill") = "Bill""#,
            true,
        ),
        (
            "* | where round(score * 10) = 8 AND abs(-2) = 2 AND max(1, status) = 404",
            true,
        ),
        ("* | where nothing > 1 OR NOT nothing = 1", false),
        ("* | where status != 404 XOR bytes = 1500", true),
        (
            r#"* | where typeof(status) = "Number" AND typeof(user) = "String""#,
            true,
        ),
        (
            r#"* | where trim("  x ") = "x" AND in(status, 200, 404)"#,
            true,
        ),
    ] {
        assert_eq!(hits(query, text), expected, "{query}");
    }
}

#[test]
fn regex_rex_and_fields_shape_later_stages() {
    let text = "GET /login?user=bob from 10.0.0.5 agent=Mozilla";
    for (query, expected) in [
        (r#"* | regex _raw="user=\w+""#, true),
        (r#"* | regex "USER=""#, false),
        (r#"* | regex agent!="curl""#, true),
        (r#"* | rex "from (?<ip>[\d.]+)" | search ip=10.0.0.5"#, true),
        (
            r#"* | rex "from (?<ip>[\d.]+)" | where cidrmatch("10.0.0.0/24", ip)"#,
            true,
        ),
        (
            r#"* | rex field=agent "(?<browser>Moz\w+)" | search browser=mozilla"#,
            true,
        ),
        (r#"* | rex "(?<agent>curl)" | search agent=Mozilla"#, true),
        (r#"* | rex "(?<agent>GET)" | search agent=GET"#, true),
        ("* | fields - agent | search agent=*", false),
        ("* | fields id | search agent=*", false),
        ("* | fields id | search _raw=*login*", true),
        ("* | fields - user* | search user=bob", false),
        (
            r#"* | fields - ip | rex "from (?<ip>[\d.]+)" | search ip=10.0.0.5"#,
            true,
        ),
    ] {
        assert_eq!(hits(query, text), expected, "{query}");
    }
}

fn table() -> Vec<Record> {
    [
        ("A", "user=carol n=10 level=warn", 0.2),
        ("B", "user=alice n=9 level=error", 0.9),
        ("C", "user=bob n=100 level=warn", 0.5),
        ("D", "user=alice n=2", 0.7),
        ("E", "anonymous n=x", 0.9),
    ]
    .into_iter()
    .map(|(id, text, score)| record(id, text, score))
    .collect()
}

#[test]
fn ordering_commands_reorder_and_truncate_originals() {
    let t = table();
    for (query, expected) in [
        ("* | sort - score", "B E D C A"),
        ("* | sort -score, +id", "B E D C A"),
        ("* | sort score desc", "B E D C A"),
        ("* | sort n", "D B A C E"),
        ("* | sort num(n)", "D B A C E"),
        ("* | sort str(n)", "A C D B E"),
        // Descending reverses numbers before text; missing values stay last.
        ("* | sort - n", "E C A B D"),
        ("* | sort - level", "A C B D E"),
        ("* | sort level, id", "B A C D E"),
        ("* | sort 2 -score", "B E"),
        ("* | sort limit=1 id", "A"),
        ("* | head 2", "A B"),
        ("* | head", "A B C D E"),
        ("* | head (score < 0.8)", "A"),
        ("* | head (score < 0.8) keeplast=true", "A B"),
        ("* | head (level=\"warn\") null=true", "A"),
        ("* | tail 2", "E D"),
        ("* | dedup user", "A B C"),
        ("* | dedup user keepempty=true", "A B C E"),
        ("* | dedup 2 user", "A B C D"),
        ("* | dedup level consecutive=true", "A B C"),
        ("* | dedup user sortby - score", "B C A"),
        ("* | sort -score | dedup user | head 2", "B C"),
        // Non-numeric values compare as text, as in Splunk: "x" > "5".
        ("n>5 | sort -n", "E C A B"),
        ("n>5 n!=x | sort -n", "C A B"),
    ] {
        assert_eq!(run(query, &t).join(" "), expected, "{query}");
    }
}

#[test]
fn occurrence_filters_clear_groups_and_ordering_refuses_them() {
    let mut r = record("R1", "login user=bob", 0.5);
    let at = |host: &str, time: &str| Occurrence {
        host: Some(host.into()),
        time: Some(time.into()),
        count: 2,
    };
    r.occurrences = vec![
        at("edge-01", "2026-11-18T10:00:00Z"),
        at("edge-02", "2026-11-18T11:30:00Z"),
        Occurrence {
            host: None,
            time: None,
            count: 1,
        },
    ];
    let records = vec![r, record("R2", "other", 0.1)];
    let budget = budget();
    let offsets = [0, 3, 3];
    for (query, bits, kept) in [
        ("host=edge-*", 0b011, 1),
        ("NOT host=edge-01", 0b110, 1),
        ("host!=edge-01", 0b010, 1),
        ("earliest=11/18/2026:11:00:00", 0b010, 1),
        ("latest=2026-11-18T11:00:00Z login", 0b001, 1),
        // 10:00Z is 1794996000 and 11:30Z is 1795001400.
        ("earliest=1795000000 latest=1795003000", 0b010, 1),
        ("earliest=1794996000 latest=1794996001", 0b001, 1),
        ("* | where _time >= 1795001400", 0b010, 1),
        ("host=edge-02 | head 0", 0b000, 0),
        ("host=nowhere", 0b000, 0),
    ] {
        let q = Query::parse(query, &budget).unwrap();
        assert!(q.uses_occurrence(), "{query}");
        let mut mask = [0b111_u64];
        let selected = q
            .select(
                &records,
                vec![Member {
                    record: 0,
                    occurrences: 5,
                    allocation: None,
                }],
                Some(Occurrences {
                    mask: &mut mask,
                    offsets: &offsets,
                }),
                0,
                &budget,
            )
            .unwrap();
        assert_eq!((mask[0], selected.len()), (bits, kept), "{query}");
    }
    let q = Query::parse("host=edge-01", &budget).unwrap();
    assert!(
        q.select(&records, members(&records), None, 0, &budget)
            .is_err()
    );
    for query in [
        "* | sort host",
        "* | dedup _time",
        "* | head (host=\"a\")",
        r#"* | rex field=host "(?<x>.)""#,
    ] {
        let error = Query::parse(query, &budget).err().unwrap();
        assert!(error.contains("occurrence"), "{query}: {error}");
    }
}

#[test]
fn predicates_compile_once_within_the_budget() {
    let budget = budget();
    let q = Query::parse(r#"a | where match(_raw, "x") | rex "(?<u>\w+)""#, &budget).unwrap();
    assert!(q.is_predicate());
    assert!(!q.uses_occurrence());
    assert_eq!(q.reserved_bytes(), PARSER_BYTES + 2 * REGEX_HELD_BYTES);
    assert!(!Query::parse("a | head 1", &budget).unwrap().is_predicate());
    assert!(
        Query::parse("a | sort id", &budget)
            .unwrap()
            .run_bytes(1000, 100)
            > 1000 * 64
    );
}

#[test]
fn rejects_invalid_and_unsupported_queries_before_scanning() {
    let budget = budget();
    for text in [
        "",
        "search",
        "a OR",
        "a AND",
        "NOT",
        "(broken",
        "a)",
        "id=",
        "a | stats count",
        "a | eval x=1",
        "a | table id",
        "a | frobnicate",
        "a | search",
        "a || b",
        "| head 5",
        "x XOR y",
        "double==x",
        r#""unterminated"#,
        "()",
        "search | search foo",
        "[search foo]",
        "`macro`",
        "a ``` unclosed",
        "TERM(",
        "user IN ()",
        "user IN alice",
        "earliest=-24h",
        "earliest=@d",
        "earliest=yesterday",
        "a OR earliest=0",
        "NOT latest=0",
        "a | where",
        "a | where now() > 0",
        "a | where frob(1)",
        r#"a | where match(_raw, user)"#,
        "a | where x LIKE y",
        "a | where (1",
        "a | regex",
        r#"a | regex "[""#,
        r#"a | rex "no groups""#,
        r#"a | rex mode=sed "s/a/b/""#,
        r#"a | rex max_match=0 "(?<x>a)""#,
        r#"a | rex "(?<host>a)""#,
        "a | head x",
        "a | head 1 2",
        "a | tail -1",
        "a | sort",
        "a | sort num(",
        "a | dedup",
        "a | dedup keepevents=true x",
        "a | fields",
    ] {
        assert!(Query::parse(text, &budget).is_err(), "{text}");
    }
    for text in [
        "NOT ".repeat(MAX_DEPTH_FOR_TESTS + 1) + "id=x",
        "a ".repeat(300),
        "a".repeat(MAX_BYTES + 1),
        "a".to_owned() + &" | head 1".repeat(MAX_STAGES),
        "a".to_owned() + &r#" | regex "x""#.repeat(MAX_REGEXES + 1),
    ] {
        assert!(Query::parse(&text, &budget).is_err());
    }
}

#[test]
fn adversarial_arguments_neither_panic_nor_overflow() {
    for (query, expected) in [
        (r#"* | where substr(_raw, 2, 1e300) = "ogin x""#, true),
        (r#"* | where substr(_raw, -1e300, 2) = "lo""#, true),
        (r#"* | where substr(_raw, 1e300) = """#, true),
        (r#"* | where substr(_raw, -2) = " x""#, true),
        (r#"* | where substr(_raw, 1, -1) = "x""#, false),
        ("* | where isnull(round(score, 1e300))", true),
        ("* | where tonumber(\"zz\", 1e300) = 1", false),
        (
            "* | head 18446744073709551615 | tail 18446744073709551615",
            true,
        ),
        ("* | sort 0 id | dedup 18446744073709551615 id", true),
    ] {
        assert_eq!(hits(query, "login x"), expected, "{query}");
    }
    // A trailing escaped backslash in a quoted regex and a search phrase.
    assert!(hits(r#"* | regex _raw="a\\""#, r"x a\ y"));
    assert!(hits(r#"* | regex _raw="\"q\"""#, r#"say "q" now"#));
    assert!(hits(r#""a\\b""#, r"x a\b y"));
    // A leading star only tries segment starts, so a long blob stays linear.
    let blob = "a".repeat(200_000);
    let started = std::time::Instant::now();
    assert!(!hits("*a*a*a*a*b", &blob));
    assert!(hits("*a*a*a*a", &blob));
    assert!(started.elapsed().as_secs() < 5, "{:?}", started.elapsed());
}

#[test]
fn sort_auto_orders_numbers_then_ip_addresses_then_text() {
    let records: Vec<Record> = [
        ("A", "src=10.0.0.10"),
        ("B", "src=host-b"),
        ("C", "src=2.0.0.1"),
        ("D", "src=10.0.0.9"),
        ("E", "src=7"),
        ("F", "src=::1"),
    ]
    .into_iter()
    .map(|(id, text)| record(id, text, 0.5))
    .collect();
    assert_eq!(run("* | sort src", &records).join(" "), "E C D A F B");
    assert_eq!(run("* | sort str(src)", &records).join(" "), "A D C E F B");
    assert_eq!(run("* | sort - ip(src)", &records).join(" "), "F A D C B E");
}

#[test]
fn ordering_commands_clear_the_occurrences_of_dropped_originals() {
    let budget = budget();
    let mut records = table();
    for (i, r) in records.iter_mut().enumerate() {
        r.occurrences = vec![Occurrence {
            host: Some(format!("h{i}")),
            time: None,
            count: 1,
        }];
    }
    let offsets: Vec<usize> = (0..=records.len()).collect();
    for (query, bits) in [
        ("host=h* | dedup user", 0b00111),
        ("host=h* | sort limit=2 - score", 0b10010),
        ("host=h* | tail 2", 0b11000),
        ("host=h* | head 1", 0b00001),
        ("host!=h2 | dedup level consecutive=true", 0b00011),
        ("host=h* level=warn", 0b00101),
    ] {
        let q = Query::parse(query, &budget).unwrap();
        let mut mask = [0b11111_u64];
        let selected = q
            .select(
                &records,
                members(&records),
                Some(Occurrences {
                    mask: &mut mask,
                    offsets: &offsets,
                }),
                0,
                &budget,
            )
            .unwrap();
        assert_eq!(mask[0], bits, "{query}");
        assert_eq!(selected.len(), bits.count_ones() as usize, "{query}");
    }
}

#[test]
fn string_growth_is_estimated_and_refused_before_evaluation() {
    let budget = budget();
    let text = "x".repeat(100_000);
    let records = vec![record("R1", &text, 0.5)];
    let insert = "0123456789".repeat(10);
    let q = Query::parse(
        &format!(
            r#"* | where len(replace(replace(_raw, "(?:)", "{insert}"), "(?:)", "{insert}")) > 0"#
        ),
        &budget,
    )
    .unwrap();
    // Each replace may insert its replacement at every position.
    assert!(q.run_bytes(1, 100_000) > 1_000_000_000);
    let small = MemoryBudget::observe_with(
        &MemorySettings {
            memory_limit_bytes: Some(l2d_core::memory::PROCESS_BASELINE_BYTES + 64 * 1024 * 1024),
            ..Default::default()
        },
        || Ok(4_u64 << 30),
    )
    .unwrap();
    let error = q
        .select(&records, members(&records), None, 0, &small)
        .unwrap_err();
    assert!(error.contains("search pipeline"), "{error}");
    // What the caller holds beside the run counts against the same limit.
    let plain = Query::parse("* | sort id", &small).unwrap();
    assert!(
        plain
            .select(&records, members(&records), None, 0, &small)
            .is_ok()
    );
    assert!(
        plain
            .select(&records, members(&records), None, 64 * 1024 * 1024, &small)
            .is_err()
    );
}
// Heap counting for this test thread only: see `crate::test_heap`.
use crate::test_heap as heap;

#[test]
fn pipelines_allocate_within_their_admitted_estimate() {
    let budget = budget();
    let records: Vec<Record> = (0..2_000)
        .map(|i| {
            let text = format!(
                "GET /a?q={} user=u{} src=10.0.{}.{} status={} ¿¡«»‹›§¶†‡ msg=\"{}\"",
                "%".repeat(i % 40),
                i % 50,
                i % 7,
                i % 250,
                200 + i % 5,
                "word ".repeat(i % 60),
            );
            record(&format!("R{i:05}"), &text, (i % 97) as f64 / 97.0)
        })
        .collect();
    let longest = records.iter().map(|r| r.text.len()).max().unwrap() as u64;
    for query in [
        "* | sort - status, punct, - score",
        "* | dedup 2 punct user sortby - score",
        r#"* | rex "src=(?<ip>[\d.]+)" | where cidrmatch("10.0.0.0/16", ip) AND lower(user) . upper(msg) != "" | sort ip(ip)"#,
        r#"* | where len(replace(_raw, "u", "UUUU")) > 0 AND substr(msg, 2, 40) != "x""#,
        r#"status>=202 | regex _raw="q=%+" | head (score < 0.9) keeplast=true | tail 500"#,
        "* | where tostring(score * 1e300) != \"\" OR max(_raw, msg) = id",
    ] {
        let q = Query::parse(query, &budget).unwrap();
        let input = members(&records);
        let estimate = q.run_bytes(input.len() as u64, longest) + q.reserved_bytes();
        let (selected, peak) =
            heap::measure(|| q.select(&records, input, None, 0, &budget).unwrap());
        assert!(!selected.is_empty(), "{query}");
        assert!(
            peak <= estimate,
            "{query}: measured {peak} bytes, admitted {estimate}"
        );
    }
}
const MAX_DEPTH_FOR_TESTS: usize = search::MAX_DEPTH;
