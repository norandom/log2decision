# The triager's search language: Splunk SPL over retained originals

The SPL mode of `l2d tui` (the `/` editor, `--query`, and `--match-mode spl --filter` for set B) follows Splunk's Search Processing Language as closely as a view of originals allows. A query is a search followed by optional `|` stages. [Requirement 7](../.kiro/specs/triager/requirements.md) · [Design](../.kiro/specs/triager/design.md#search-language) · [Splunk search command](https://help.splunk.com/en/splunk-enterprise/search/search-manual/10.4/search-primer/search-command-primer)

~~~text
login fail* NOT user=guest earliest=11/18/2026:00:00:00
  | rex "from (?<ip>[\d.]+)"
  | where cidrmatch("10.0.0.0/8", ip) AND score >= 0.5
  | dedup ip sortby - score
  | head 20
~~~

## Search

| Form | Meaning |
|---|---|
| `fail`, `"locked out"` | Term or phrase in `_raw`, case-insensitive. It must start and end at a breaker, as with Splunk's segmentation: `fail` misses `failed`, `error` finds `error_code`, `0.0.1` finds `10.0.0.1`. |
| `fail*`, `*ail`, `a*z` | `*` is the only wildcard. It stays inside one major segment. `\*` and `\\` are literal. |
| `TERM(10.0.0.1)` | The whole term must be bounded by major breakers, so `src=10.0.0.1` does not match. |
| `CASE(Error)` | A case-sensitive term. |
| `a b`, `a AND b`, `a OR b`, `NOT a`, `( )` | NOT binds tightest, then OR, then AND (explicit or implicit). Operators must be uppercase. |
| `f=v`, `f!=v`, `f<v`, `f<=v`, `f>v`, `f>=v` | Field comparisons. Values are case-insensitive and may use `*`. Comparisons are numeric when both sides are numbers, otherwise byte order. A missing field fails `f=v` and `f!=v`, so `NOT f=v` keeps events without the field. |
| `f IN (a, "b c", d*)` | Any of the values. |
| `earliest=…`, `latest=…` | Bound occurrence time, with `earliest` inclusive and `latest` exclusive. Takes epoch seconds, `%m/%d/%Y:%H:%M:%S` (read as UTC) or ISO 8601 with `Z` or an offset. An occurrence time without a zone has no `_time`, so these bounds exclude it. Allowed only outside OR and NOT. |
| ```` ```comment``` ```` | Removed before parsing. |

## Fields

Field names are case-sensitive.

| Field | Value |
|---|---|
| `_raw` | The original's text |
| `id` | The original's ID |
| `score` | Set A's score |
| `c` | The recommendation: `recommended`, `passed` or `unknown` |
| `reason` | The producer's reason |
| `punct` | The first 30 punctuation characters of `_raw`, with a space as `_` |
| `linecount` | Lines in `_raw` |
| `host`, `_time` | Per occurrence. A filter on them keeps only the matching occurrences of each original |
| any other name | A `rex` capture, else the first `name=value` pair in `_raw` (a quoted value, or one ending at whitespace, `,`, `;`, `&` or `"`) |

## Commands after `|`

| Command | Supported form |
|---|---|
| `search` | As above |
| `where <eval>` | Keeps rows where the expression is true |
| `regex [f=\|f!=]"<re>"` | Keeps rows whose field (default `_raw`) matches, or does not match. Case-sensitive |
| `rex [field=f] "<re with (?<name>…)>"` | Adds named captures as fields. `max_match=1` only |
| `fields [+\|-] f1, f2*` | Keeps or removes fields for later stages. `_raw` and `_time` stay unless removed explicitly |
| `head [N\|limit=N\|(<eval>)] [keeplast=] [null=]` | Default 10 |
| `tail [N]` | Default 10. The last N rows, in reverse order |
| `sort [N\|limit=N] [+\|-]f \| num(f) \| str(f) \| ip(f) \| auto(f) … [desc]` | Default limit 10,000; `0` means no limit. Stable. In auto mode numbers sort first, then IP addresses, then text. Missing values sort last |
| `dedup [N] f1 f2 [keepempty=] [consecutive=] [sortby …]` | Keeps the first N rows per value combination |

**Eval (`where`, `head (…)`)**

- **Syntax.** Bare words are fields, `"…"` is a string and `'…'` is a field name. String comparisons are case-sensitive, so `where user=Bob` compares two fields.
- **Operators,** tightest first: `* / %`, `+ - .`, `< <= > >=`, `= == != LIKE`, `NOT`, `AND`, then `OR XOR`.
- **Functions:** `if`, `case`, `coalesce`, `isnull`, `isnotnull`, `isnum`, `isstr`, `isint`, `null`, `true`, `false`, `len`, `lower`, `upper`, `substr`, `trim`, `ltrim`, `rtrim`, `replace` (with `\1` back-references), `match`, `like`, `tonumber`, `tostring`, `abs`, `round`, `floor`, `ceil`/`ceiling`, `sqrt`, `exp`, `ln`, `log`, `pow`, `min`, `max`, `in`, `cidrmatch`, `searchmatch`, `typeof`.
- **Literal arguments.** The patterns of `match`, `replace`, `like` and `searchmatch` must be string literals.

## Differences from Splunk

| Topic | Here | Why |
|---|---|---|
| Result rows | Deduplicated originals with occurrence groups, not events. `sort`, `dedup`, `rex` and `head (…)` refuse `host` and `_time`. | The triager counts originals and occurrences separately. Filter on `host` or `_time` first. |
| Transforming commands | `stats`, `eval`, `table`, `timechart`, `top`, `rare`, `transaction`, lookups and joins are refused with the supported list. | They produce tables or new fields, which need a statistics result type. |
| Generating commands, subsearches and macros | A leading `\|`, `[ … ]` and `` `macro` `` are refused. | No index or saved-search catalogue exists. |
| Time | Absolute `earliest`/`latest` only, in UTC. `now()`, `time()`, `random()` and `relative_time()` are refused. | A saved receipt must replay identically. |
| Segmentation | The single-character default breakers. Multi-character and URL-encoded breakers (`--`, `%20`, ...) are not applied. | Approximation of `segmenters.conf`. |
| Automatic extraction | The first `key=value` per name. Built-in fields win over extracted ones. No multivalue fields. | Approximation of `KV_MODE = auto`. |
| Regular expressions | The Rust `regex` engine: no look-around or back-references inside patterns. | Bounded, linear-time matching under the memory budget. |
| `regex f!=…` with a missing field | Keeps the row. | Documented choice. |
| Limits | 8 KiB per query, 32 stages, 16 regular expressions, 256 tokens per stage and nesting depth 16. | Compilation is bounded before any scan. |
| Not confirmed against a Splunk instance | `status=200.0` equals `200` in search. `n>5` falls back to byte order for non-numeric values. Missing values sort last even with `-`. `0.0.1` matches inside `10.0.0.1`. `tonumber("0x1F", 16)` accepts the `0x` prefix. | These follow the documented rules as closely as they could be read. Treat them as approximations until they are compared with Splunk. |
| Saved receipts | A receipt saved before 18 November with an SPL query may be refused on reopen if the new term semantics change its members. | Replay compares the result digest. |
