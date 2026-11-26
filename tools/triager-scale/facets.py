"""The field, time and re-scan legs of the triager scale gate (task 3.1) through MCP.

A leg starts `l2d mcp` on the 1.3 fixture, opens it with occurrence projections of five
`key=value` fields, and then:

- warm facet/time: at least 30 interactions over the committed projection - the fields
  of the snapshot, pages of values (a high-cardinality field among them), the histogram
  (`time_unavailable` on this fixture, whose records carry no timestamp: its time bins
  are its entries, served as the labelled alternative), drills into an entry, a user
  below it and a host, the fields and histogram of the drilled views, a page of records
  and the evidence of a repeated original with every occurrence. Each is checked against
  the oracle (`oracle.py facets`), and the server's work counters must not move;
- explicit re-scan: a field the projection does not declare returns its missing
  coverage and the scan it takes, nothing runs until the named request is called, the
  re-scan is one admitted read and its values reconcile with the oracle; the same
  archive opened with another reader setting is another snapshot, which no projection
  supplies.

    python facets.py leg streamed --l2d L2D --project P --fixture F --oracle O/facets.json \
        --time-fixture T --time-oracle O/time.json --out O

Round 2 of the review adds the timestamped fixture (`generate.py --timestamps`), opened
twice: with its records' leading timestamp (histogram at the automatic width and by day,
a bucket drill, its hourly histogram and users, a user drill below it, a drill into the
records without a time) and with its entries' days as the time of every occurrence
(a repeated original falls on several days). Fresh drills and replays of views that
exist are timed apart. On the 1.3 fixture the declared timestamp is missing everywhere:
its histogram is `time_unavailable` with the entries as the labelled alternative.

Standard library only; `run.py` gives the client, the legs' settings and the hashes.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from run import LEGS, WORK_COUNTERS, Failed, Server, check, harness_hashes, sha256_of

FIELDS = ("event", "host", "src", "svc", "user")
# The 1.3 fixture: five fields and the leading timestamp, which its records do not have.
PROJECTIONS = {"fields": [{"name": name, "from": "kv"} for name in FIELDS],
               "time": {"from": "text"}}
# The timestamped fixture: its leading timestamps, and, as another snapshot of it, the
# days its entries are named by.
STAMPED = {"fields": [{"name": "user", "from": "kv"}], "time": {"from": "text"}}
DAYS = {"time": {"from": "entry", "capture": r"(\d{4}-\d{2}-\d{2})\.log$", "format": "date"}}
DAY = "logs/day-03.log"


def facets_hashes() -> dict:
    here = Path(__file__).resolve().parent
    hashes = harness_hashes()
    hashes["facets.py"] = sha256_of(here / "facets.py")
    return hashes


def pair(value):
    return {"distinct_originals": value["distinct_originals"], "occurrences": value["occurrences"]}


def field_of(summary, name):
    for field in summary["fields"]:
        if field["field"] == name:
            return field
    raise Failed(f"no field {name} in the fields view")


def rows_of(page):
    return [{"value": row["value"], **pair(row)} for row in page["field"]["values"]]


def counters(server, snapshot):
    activity = server.overview(snapshot)["activity"]
    return {name: activity[name] for name in WORK_COUNTERS}


def buckets_of(time):
    return [{"start": row["start"], **pair(row)} for row in time["buckets"]]


def warm(server, snapshot, oracle, timings, stamped, days, time_oracle):
    """The warm facet/time leg: at least 30 interactions over three snapshots, all
    checked. `snapshot` is the 1.3 fixture (a declared timestamp missing everywhere),
    `stamped` the timestamped fixture with the records' leading time and `days` the
    same archive with the entries' days as the time of every occurrence."""
    phase = {"round": 0}

    def timed(kind, name, arguments, label):
        payload = server.ok(name, arguments, label)
        timings.setdefault(kind, []).append(server.last["ms_to_result"])
        return payload

    def inspect(handle, view, label, **extra):
        return timed("view", "triager_inspect",
                     {"v": 1, "handle": handle, "view": view, **extra}, label)

    def drill(handle, action, label):
        # The first round makes every view; later rounds name views that exist.
        kind = "drill_fresh" if phase["round"] == 0 else "drill_replay"
        return timed(kind, "triager_refine", {"v": 1, "set": handle, "action": action}, label)

    def time_leg():
        """The histogram, bucket, nested field and no-time drills of the timestamped
        fixture, and the entries' days."""
        text = time_oracle["text"]
        auto = inspect(stamped, "histogram", "time histogram")
        check(auto["status"], "ready", "time histogram status")
        time = auto["time"]
        check(time["bucket_seconds"], text["auto"]["bucket_seconds"], "automatic width")
        check(time["range"], text["auto"]["range"], "range")
        check(time["observed"], text["auto"]["observed"], "observed")
        check(buckets_of(time), text["auto"]["buckets"], "buckets")
        for part in ("with_time", "missing_time", "invalid_time"):
            check(pair(time[part]), text[part], part)
        daily = inspect(stamped, "histogram", "time histogram day", bucket_seconds=86_400)
        check(buckets_of(daily["time"]), text["day"]["buckets"], "day buckets")
        expected = time_oracle["drill_bucket"]
        bucket = drill(stamped, {"kind": "time", "start": expected["start"],
                                 "end": expected["end"]}, "drill bucket")
        check(bucket["counts"]["in"], expected["counts_in"], "bucket in")
        check(bucket["occurrence_counts"]["in"], expected["occurrence_counts_in"],
              "bucket occurrences")
        check(bucket["matching_occurrences"]["in"], expected["matching_in"], "bucket matching")
        check(bucket["matching_occurrences"]["parent"], expected["matching_parent"],
              "bucket parent")
        hourly = inspect(bucket["view"], "histogram", "bucket hourly", bucket_seconds=3_600)
        check(buckets_of(hourly["time"]), expected["hourly"], "hourly buckets of the bucket")
        users = inspect(bucket["view"], "fields", "bucket values user", field="user")
        check(rows_of(users), expected["user_page"], "user page of the bucket")
        expected = time_oracle["drill_bucket_user"]
        nested = drill(bucket["view"], {"kind": "field", "field": "user",
                                        "value": expected["user"]}, "drill bucket user")
        check(nested["counts"]["in"], expected["counts_in"], "bucket user in")
        check(nested["counts"]["parent"], expected["counts_parent"], "bucket user parent")
        check(nested["matching_occurrences"]["in"], expected["matching_in"], "bucket user matching")
        check(nested["matching_occurrences"]["parent"], expected["matching_parent"],
              "bucket user matching parent")
        untimed = drill(stamped, {"kind": "time", "no_time": True}, "drill no time")
        check(untimed["counts"]["in"], time_oracle["drill_no_time"]["counts_in"], "no time in")
        check(untimed["matching_occurrences"]["in"], time_oracle["drill_no_time"]["matching_in"],
              "no time matching")
        # The entries' days: a repeated original falls on several days.
        expected = time_oracle["entry_days"]
        entry_days = inspect(days, "histogram", "days histogram", bucket_seconds=86_400)
        check(entry_days["time"]["range"], expected["range"], "days range")
        check(buckets_of(entry_days["time"]), expected["buckets"], "days buckets")
        check(pair(entry_days["time"]["missing_time"]), expected["missing_time"], "days missing")
        spread = sum(row["distinct_originals"] for row in entry_days["time"]["buckets"])
        if spread - entry_days["time"]["with_time"]["distinct_originals"] != expected[
                "extra_day_counts"]:
            # The originals read on several days are counted once a day.
            raise Failed(f"day buckets count {spread} originals and "
                         f"{entry_days['time']['with_time']['distinct_originals']} have a day; "
                         f"{expected['extra_day_counts']} more days are expected")
        expected = time_oracle["drill_day"]
        day = drill(days, {"kind": "time", "start": expected["start"], "end": expected["end"]},
                    "drill entry day")
        check(day["counts"]["in"], expected["counts_in"], "entry day in")
        check(day["matching_occurrences"]["in"], expected["matching_in"], "entry day matching")
        check(day["occurrence_counts"]["in"], expected["occurrence_counts_in"],
              "entry day occurrences")
        return bucket["view"]

    before = counters(server, snapshot or stamped)
    checked = []
    bucket_view = None
    for round_ in range(3):
        phase["round"] = round_
        if stamped is not None:
            bucket_view = time_leg()
        if snapshot is None:
            checked.append(round_)
            continue
        summary = inspect(snapshot, "fields", "fields")
        check(pair(summary["population"]["counted"]),
              {"distinct_originals": oracle["counts"]["distinct_originals"],
               "occurrences": oracle["counts"]["occurrences"]}, "fields counted")
        for name in FIELDS:
            field, expected = field_of(summary, name), oracle["fields"][name]
            for part in ("with_value", "missing", "invalid", "overflow"):
                check(pair(field[part]), expected[part], f"{name} {part}")
            check(pair(field["types"]["number"]), expected["types"]["number"], f"{name} numbers")
            check(field["distinct_values"]["count"], expected["distinct_values"], f"{name} distinct")
            check(field["distinct_values"]["exact"], expected["exact"], f"{name} exact")
        check(field_of(summary, "source")["distinct_values"]["count"],
              oracle["source_field"]["distinct_values"], "source distinct")
        for name in ("user", "host", "src"):
            page = inspect(snapshot, "fields", f"values {name}", field=name)
            check(rows_of(page), oracle["pages"][name]["rows"], f"{name} page")
            check(page["page"]["values_total"], oracle["pages"][name]["values_total"],
                  f"{name} values")
        # A second page through the cursor.
        first = inspect(snapshot, "fields", "values user p1", field="user", limit=10)
        second = inspect(snapshot, "fields", "values user p2", field="user", limit=10,
                         cursor=first["page"]["next_cursor"])
        check(rows_of(first) + rows_of(second), oracle["pages"]["user"]["rows"][:20],
              "user pages")
        histogram = inspect(snapshot, "histogram", "histogram", limit=100)
        check(histogram["status"], "time_unavailable", "histogram status")
        # The declared timestamp is missing on every occurrence of the 1.3 fixture.
        check(histogram["time"]["declared"]["from"], "text", "declared timestamp")
        check(pair(histogram["time"]["missing_time"]),
              {"distinct_originals": oracle["counts"]["distinct_originals"],
               "occurrences": oracle["counts"]["occurrences"]}, "missing everywhere")
        check([{"entry": row["entry"], "entry_index": row["entry_index"], **pair(row)}
               for row in histogram["alternative"]["rows"]], oracle["entries"],
              "entries alternative")
        # The time bin of the fixture: one entry.
        day = drill(snapshot, {"kind": "field", "field": "source", "value": DAY}, "drill day")
        expected = oracle["drill_day"]
        check(day["counts"]["in"], expected["counts_in"], "day in")
        check(day["occurrence_counts"]["in"], expected["occurrence_counts_in"], "day occurrences")
        check(day["matching_occurrences"]["in"], expected["matching_in"], "day matching")
        check(day["matching_occurrences"]["parent"], expected["matching_parent"], "day parent")
        users = inspect(day["view"], "fields", "day values user", field="user")
        check(rows_of(users), expected["user_page"]["rows"], "day user page")
        check(pair(users["population"]["counted"]), expected["user_summary_counted"], "day counted")
        day_histogram = inspect(day["view"], "histogram", "day histogram")
        check([row["entry"] for row in day_histogram["alternative"]["rows"]], [DAY],
              "day alternative")
        nested = drill(day["view"], {"kind": "field", "field": "user",
                                     "value": oracle["drill_day_user"]["user"]}, "drill user")
        expected = oracle["drill_day_user"]
        check(nested["counts"]["in"], expected["counts_in"], "nested in")
        check(nested["counts"]["parent"], expected["counts_parent"], "nested parent")
        check(nested["matching_occurrences"]["in"], expected["matching_in"], "nested matching")
        check(nested["matching_occurrences"]["parent"], expected["matching_parent"],
              "nested matching parent")
        host = drill(snapshot, {"kind": "field", "field": "host", "value": "h042"}, "drill host")
        check(host["counts"]["in"], oracle["drill_host"]["counts_in"], "host in")
        check(host["matching_occurrences"]["in"], oracle["drill_host"]["matching_in"],
              "host matching")
        records = inspect(day["view"], "records", "day records", limit=25)
        check(len(records["rows"]), 25, "a page of the day")
        evidence = inspect(day["view"], "evidence", "evidence", record=oracle["evidence"]["id"])
        listed = evidence["record"]["occurrence_list"]
        check(listed["total"], oracle["evidence"]["total"], "occurrences of the original")
        check(listed["in_view"], oracle["evidence"]["in_view"], "occurrences in the day")
        check([row["entry"] for row in listed["rows"]],
              oracle["evidence"]["entries"][:len(listed["rows"])], "occurrence entries")
        checked.append(round_)
    after = counters(server, snapshot or stamped)
    check(after, before, "work counters over the warm leg")
    if snapshot is None:
        return {"rounds": len(checked), "counters": after, "bucket": bucket_view}
    return {"rounds": len(checked), "counters": after,
            "views": day["view"], "nested": nested["view"]}


def rescan(server, snapshot, oracle, fixture, timings):
    """The explicit re-scan leg."""
    before = counters(server, snapshot)
    wanted = server.ok("triager_inspect", {"v": 1, "handle": snapshot, "view": "fields",
                                           "field": "status"}, "missing status")
    check(wanted["status"], "projection_missing", "status of a field not projected")
    check(wanted["missing"]["fields"], ["status"], "missing fields")
    check(wanted["missing"]["coverage"]["not_projected"], oracle["counts"]["occurrences"],
          "missing coverage")
    estimate = wanted["estimate"]
    check(estimate["records_to_scan"], oracle["counts"]["records"], "records to scan")
    check(estimate["occurrences_to_project"], oracle["counts"]["occurrences"], "occurrences")
    check(estimate["rows_scored"], 0, "no scoring")
    check(counters(server, snapshot), before, "asking scheduled nothing")
    started = time.perf_counter()
    rescanned = server.ok("triager_open", wanted["rescan"]["args"], "rescan")
    seconds = round(time.perf_counter() - started, 2)
    check(rescanned["cost"]["archive_scans"], 1, "one read")
    check(rescanned["cost"]["rows_scored"], 0, "no scoring in the re-scan")
    after = counters(server, snapshot)
    check(after["archive_scans"], before["archive_scans"] + 1, "the re-scan is one scan")
    status = server.ok("triager_inspect", {"v": 1, "handle": snapshot, "view": "fields",
                                           "field": "status"}, "values status")
    check(rows_of(status), oracle["rescan"]["page"]["rows"], "status values")
    check(pair(status["field"]["missing"]), oracle["rescan"]["summary"]["missing"],
          "status missing")
    again = server.ok("triager_open", wanted["rescan"]["args"], "rescan again")
    check(again["projection"], rescanned["projection"], "the same projection")
    check(counters(server, snapshot), after, "the same request scans nothing")
    # The declarations of the first projection are kept.
    users = server.ok("triager_inspect", {"v": 1, "handle": snapshot, "view": "fields",
                                          "field": "user"}, "values user after")
    check(rows_of(users), oracle["pages"]["user"]["rows"], "user page after the re-scan")
    # Another reader setting is another snapshot, which no projection supplies.
    other = server.ok("triager_open", {"v": 1, "source": {
        "ref": "fixture", "entries": ["*.log"], "max_record_bytes": 4000}}, "open other")
    if other["snapshot"] == snapshot:
        raise Failed("another reader setting gave the same snapshot")
    missing = server.ok("triager_inspect", {"v": 1, "handle": other["snapshot"],
                                            "view": "fields"}, "other fields")
    check(missing["status"], "projection_missing", "another snapshot has no projection")
    return {"seconds": seconds, "cost": rescanned["cost"], "projection": rescanned["projection"],
            "previous": rescanned["previous"], "estimate": estimate,
            "other_snapshot": other["snapshot"], "counters": counters(server, snapshot)}


def percentiles(values):
    values = sorted(values)
    if not values:
        return None
    rank = lambda share: values[min(len(values) - 1, max(0, round(share * len(values)) - 1))]
    return {"n": len(values), "p50_ms": statistics.median(values), "p95_ms": rank(0.95),
            "max_ms": values[-1]}


def leg(args):
    oracle = json.loads(args.oracle.read_text())
    time_oracle = json.loads(args.time_oracle.read_text())
    global_args, mcp_args = LEGS[args.name]
    out = args.out / f"facets-{args.name}-{args.run}"
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    harness = facets_hashes()
    report = {"leg": args.name, "run": args.run, "global": global_args, "mcp": mcp_args,
              "l2d": str(args.l2d), "l2d_sha256": sha256_of(args.l2d),
              "fixture_sha256": sha256_of(args.fixture / "fixture.zip"),
              "time_fixture_sha256": sha256_of(args.time_fixture / "fixture.zip"),
              "time_oracle_sha256": sha256_of(args.time_oracle),
              "harness_sha256": harness, "oracle_sha256": sha256_of(args.oracle),
              "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "passed": False}
    timings = {}
    servers = []

    def start(name, sources):
        server = Server(args.l2d, args.project, global_args, [
            *sources, "--work-dir", str(work / name), "--inline-wait-ms", "200", *mcp_args],
            out / f"server-{name}-stderr.txt")
        servers.append((name, server))
        return server

    source = {"ref": "fixture", "entries": ["*.log"],
              "max_record_bytes": oracle["source"]["max_record_bytes"]}
    try:
        # Server 1: the 1.3 fixture, its fields, the declared timestamp it does not have,
        # and the explicit re-scan.
        server = start("fields", ["--source", f"fixture={args.fixture / 'fixture.zip'}"])
        started = time.perf_counter()
        opened = server.ok("triager_open", {"v": 1, "source": source,
                                            "projections": PROJECTIONS}, "open projected")
        report["open_seconds"] = round(time.perf_counter() - started, 2)
        check(opened["counts"]["occurrences"], oracle["counts"]["occurrences"], "occurrences")
        check(opened["counts"]["distinct_originals"], oracle["counts"]["distinct_originals"],
              "originals")
        check(opened["projection"]["occurrences"], oracle["counts"]["occurrences"],
              "projected occurrences")
        report["open_cost"] = opened["cost"]
        report["projection"] = opened["projection"]
        snapshot = opened["snapshot"]
        report["warm"] = warm(server, snapshot, oracle, timings, None, None, time_oracle)
        report["rescan"] = rescan(server, snapshot, oracle, args.fixture, timings)
        final = server.overview(snapshot)
        report["activity"] = final["activity"]
        report["session"] = final["session"]
        # Server 2: the timestamped fixture, twice: its records' leading time, and its
        # entries' days (another entry pattern for the same entries, so another snapshot).
        server = start("time", ["--source", f"stamped={args.time_fixture / 'fixture.zip'}"])
        started = time.perf_counter()
        stamped = server.ok("triager_open", {"v": 1, "source": {**source, "ref": "stamped"},
                                             "projections": STAMPED}, "open stamped")
        report["stamped_open_seconds"] = round(time.perf_counter() - started, 2)
        check(stamped["counts"]["occurrences"], time_oracle["counts"]["occurrences"],
              "stamped occurrences")
        check(stamped["projection"]["occurrences"], time_oracle["counts"]["occurrences"],
              "stamped projected occurrences")
        report["stamped_open_cost"] = stamped["cost"]
        report["stamped_projection"] = stamped["projection"]
        started = time.perf_counter()
        days = server.ok("triager_open", {"v": 1, "source": {**source, "ref": "stamped",
                                                             "entries": ["logs/*.log"]},
                                          "projections": DAYS}, "open days")
        report["days_open_seconds"] = round(time.perf_counter() - started, 2)
        if days["snapshot"] == stamped["snapshot"]:
            raise Failed("the day snapshot is the stamped snapshot")
        report["days_projection"] = days["projection"]
        report["warm_time"] = warm(server, None, oracle, timings, stamped["snapshot"],
                                   days["snapshot"], time_oracle)
        final = server.overview(stamped["snapshot"])
        report["time_activity"] = final["activity"]
        report["time_session"] = final["session"]
        report["passed"] = True
    except Failed as failure:
        report["failure"] = str(failure)
    finally:
        report["timings"] = {kind: percentiles(values) for kind, values in timings.items()}
        report["timings"]["all_warm"] = percentiles(
            [value for kind, values in timings.items() for value in values])
        report["warm_interactions"] = sum(len(values) for values in timings.values())
        report["servers"] = {}
        for name, server in servers:
            entry = {"calls": len(server.calls),
                     "line_bytes_returned": sum(call["line_bytes"] for call in server.calls),
                     "call_log": [{key: call.get(key) for key in ("label", "ms", "ms_to_result",
                                                                  "line_bytes")}
                                  for call in server.calls]}
            try:
                entry["peak_working_set_bytes"] = server.close()
            except Exception as error:  # noqa: BLE001 - the report says how the server ended
                entry["server_end"] = str(error)
            directory = work / name
            entry["work_dir_left"] = (sorted(path.name for path in directory.iterdir())
                                      if directory.exists() else [])
            report["servers"][name] = entry
        report["server_peak_working_set_bytes"] = max(
            (entry.get("peak_working_set_bytes") or 0 for entry in report["servers"].values()),
            default=None)
        report["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        report["harness_unchanged"] = facets_hashes() == harness
        if not report["harness_unchanged"]:
            report["passed"] = False
            report["failure"] = "the harness was edited while the leg ran"
        (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({key: report.get(key) for key in (
        "leg", "run", "passed", "failure", "open_seconds", "warm_interactions",
        "server_peak_working_set_bytes")}))
    print(json.dumps(report["timings"]))
    return 0 if report["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("leg")
    run.add_argument("name", choices=["streamed", "spill", "tight"])
    run.add_argument("--l2d", required=True, type=Path)
    run.add_argument("--project", required=True, type=Path)
    run.add_argument("--fixture", required=True, type=Path)
    run.add_argument("--oracle", required=True, type=Path)
    run.add_argument("--time-fixture", required=True, type=Path,
                     help="the fixture of generate.py --timestamps")
    run.add_argument("--time-oracle", required=True, type=Path,
                     help="what oracle.py time computed of it")
    run.add_argument("--out", required=True, type=Path)
    run.add_argument("--run", default="1")
    args = parser.parse_args()
    return leg(args)


if __name__ == "__main__":
    raise SystemExit(main())
