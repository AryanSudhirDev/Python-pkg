# Developer notes

Internal documentation for contributors to the `irw` Python package.

## Sharding, and why these are lists

**Redivis caps any dataset at 1000 tables.** That is the only reason IRW spans
several datasets, and it applies to item text as well as response data. So there
are two shard lists in `src/irw/config.py`, `MAIN_REFS` and `ITEMTEXT_REFS`, and
they behave identically: declared oldest-to-newest, searched newest-first, an
unopenable shard skipped rather than fatal.

## Adding a main IRW Redivis warehouse

Main IRW response tables can span multiple Redivis datasets ("warehouses").
Within *this package*, adding another one (e.g. a 7th warehouse) means updating
`MAIN_REFS` in `src/irw/config.py` and nothing else:

```python
MAIN_REFS: ClassVar[Tuple[Tuple[str, str], ...]] = (
    ("datapages", "item_response_warehouse:as2e"),
    ("datapages", "item_response_warehouse_2:epbx"),
    ("datapages", "item_response_warehouse_3:5xaj"),
    ("datapages", "item_response_warehouse_4:980f"),
    ("datapages", "item_response_warehouse_5:3ykx"),
    ("datapages", "item_response_warehouse_6:xxxx"),  # new warehouse
)
```

Each entry is `(redivis_user, dataset_ref)`, where `dataset_ref` is the Redivis dataset slug (e.g. `item_response_warehouse_6:xxxx`).

No other code change is needed **here** -- but this package is one of three
that declare the same dataset list, and the other two are not optional. See
[Adding a warehouse everywhere else](#adding-a-warehouse-everywhere-else)
before opening a pull request. Within the package, it automatically:

- initializes every warehouse listed in `MAIN_REFS`
- lists tables from all of them (`list_tables`, `filter`, `info`, etc.)
- searches them **newest first** when fetching or downloading (`fetch`,
  `download`), so a table present in more than one warehouse resolves to its
  most recent copy. List `MAIN_REFS` oldest-to-newest; `_order_main_datasets()`
  reverses it. This matches the R package's `.irw_order_datasources()`.
- skips a warehouse it cannot open, with a logged warning, instead of failing
  every lookup. A newly created warehouse has no released version yet and is
  unreadable with a read-only token, so **publish a release on Redivis before
  shipping a new `MAIN_REFS` entry** or its tables will silently be missing. If
  no warehouse opens at all, `_init_datasets_from_refs` raises.

## Adding an item text shard

Identical in shape, on `ITEMTEXT_REFS`:

```python
ITEMTEXT_REFS: ClassVar[Tuple[Tuple[str, str], ...]] = (
    ("datapages", "irw_text:07b6"),
    ("datapages", "irw_text_2:xxxx"),  # new shard
)
```

No other code changes are needed. `_get_itemtext_datasets()` opens and orders
them, `_list_itemtext_tables()` returns the union, and `_get_itemtext_table()`
searches newest-first — so `itemtext()` keeps working for a table in any shard.

The same trap applies, and it is the one that bites: **publish a release of the
new shard on Redivis before shipping the config entry.** An unreleased dataset
is unreadable with a read-only token, and the package skips it with a logged
warning rather than an error, so its tables are simply missing and nobody sees
a failure.

Two things must ship together with this: `IRW_TEXT_DATASETS` in
`src/metadata/redivis_config.R` (the `ben-domingue/irw` repo) and
`.irw_itemtext_specs` in the R package. A config naming a shard the other two do
not is the drift recorded as `ben-domingue/irw#1733`. The full runbook lives in
`Rpkg/inst/developer/warehouses.md`.

### Verify the change

```bash
pip install pytest

# Fast, offline checks (mocked Redivis)
python -m pytest tests/test_main_refs.py tests/test_itemtext_refs.py -v

# Live Redivis checks (network + auth required)
RUN_REDIVIS_TESTS=1 python -m pytest tests/test_redivis_integration.py -v
```

After releasing or sharing the update, ask users to **restart their Python session** so in-process caches pick up the new warehouse list.

## MCP server (issue #1713)

The optional MCP server lives in `src/irw/mcp.py` and is deliberately separate
from the core API. Install it with `pip install "irw[mcp]"` on Python 3.10 or
newer. The server uses the official MCP Python SDK over stdio and registers the
eight read-only tools documented in the package README. Every response is
stamped with `irw_version` / `irw_released_at` from `current_version()`; a
manifest failure degrades to an unpinned result with a warning, never an error.

Two sources feed the tools. `PackageBackend` wraps the public `irw` API for
everything on Redivis. `GitHubSource` reads public files with no login and no
quota: the `data/` script listing and headers (`get_processing_notes`), the
per-table notes embedded in the site's `itemtext_issues.qmd` (the `rights`
object on `get_itemtext`), and `processing_notes/validator_overrides.csv`.
Both are injectable, which is how the offline tests run without a network.

Guards happen before the call that would cost something. Missing Redivis
credentials are a structured `authentication_required` error, not a hang: the
SDK's fallback is an interactive browser login that can never complete inside
a stdio server, so `PackageBackend.ensure_ready` checks for `REDIVIS_API_TOKEN`
or `~/.redivis/python_credentials` first.

`fetch_table` bounds its window on the wire: it passes `max_rows` and
`columns` to `irw.fetch()`, which forwards both to Redivis's read session, so
the rows outside the page are never sent. The `FETCH_MAX_RESPONSES` guard
(1,000,000, the same number as `llms.txt` section 3) survives only for
`wide=true` and `dedup=true`, which describe the whole table and so cannot be
expressed as a page; for those it reads `n_responses` from the catalogue and
refuses before downloading.

**Do not reimplement the package inside the adapter.** `search_tables` takes a
`filters` object, validates the names against `irw.get_filters()` and passes
it to `irw.filter()`; the tool description and the per-filter caveats are
generated from `FILTER_DESCRIPTIONS`, not `describe_filter()`, because the
latter loads the metadata tables to compute each filter's values and must
not run at server startup. The first version of the server filtered
over `list_tables()` by hand, accepted five filters where the package had
nineteen, and dropped the coverage caveats `FILTER_DESCRIPTIONS` already
carried -- which is how "no match" starts reading as "no data". If a filter is
missing, add it to `irw.filter()` (see `has_item_text`), not to the adapter.

Keep stdout clean: MCP protocol messages use stdout, while diagnostics belong
on stderr. The adapter captures human-readable output and warnings emitted by
the existing package APIs and returns warnings in the structured tool result.
Do not add OpenAI or other model-provider dependencies to this package.

Tests use a fake backend and a fake GitHub source and do not require Redivis
credentials or a network. The live machine checks in `tests/test_mcp_live.py`
follow `briefing-check/` in the site repository -- every assertion is one a
silent no-op cannot satisfy -- and are opt-in:

```bash
RUN_REDIVIS_TESTS=1 python -m pytest tests/test_mcp_live.py -v
```

They download one 72-row table and nothing else.

## Collections (issue #1633)

Labelled groupings of IRW tables — study designs (`rct`, `q_matrix`), instrument
families (`big_five`), constructs (`depression`). A table can be in several.

Two Redivis tables in `irw_meta`, registered in `config.py`'s `META_TABLES`:
`collections:va83` (the registry, one row per collection) and
`collection_members:j7rp` (long, one row per `(table, collection)`).

Public surface: `irw.collections()`, `irw.collection(name)`,
`irw.collection_members()`, and `collection=` on `irw.filter()`.

Three things to know before changing any of it:

- **`get_collections_table()` is deliberately not filtered to existing tables.**
  The registry has no `table` column; filtering would empty it.
- **Membership is joined into `_table_info()` as a *list* column**, not merged
  directly — a plain merge on a long table multiplies rows per table.
  `_apply_tag_filter` handles list values, so no new filter primitive exists.
  (It did not handle them until 2026-08-29: `pd.isna()` raises on a list, which
  made the `isinstance(row_value, (list, tuple))` branch unreachable. Fixed;
  `tests/test_collections.py` pins both the list and string paths.)
- **`n_tables` is recomputed** in `irw.collections()` from live membership
  rather than read from the published registry column, which is a build-time
  count taken before live filtering.

Adding a collection needs no change here at all — it is one line in
`src/collections/registry.csv` in the main repo. That is the point of the long
format. See `Rpkg/inst/developer/collections.md`.


## Adding a warehouse everywhere else

The dataset list is declared once per language, because there are three clients
in three runtimes with no shared build:

| repo | file | carries |
|---|---|---|
| `Python-pkg` | `src/irw/config.py` | names **+ version hashes** |
| `Rpkg` | `R/redivis-config.R` | names **+ version hashes** |
| `irw` | `metadata/redivis_config.R` | dataset **names** only |

All three must list the same datasets, in the same order for the sharded
sources (`MAIN_REFS` and `ITEMTEXT_REFS` here). The order is not cosmetic: every
client searches shards newest-first so a table resolves to its most recent copy,
and a file listing them differently would quietly resolve some tables to a stale
shard while every name still matched.

A shard added here and nowhere else is reachable from Python and invisible from
R -- and the reverse has already happened. `irw_nominal` was in both R configs
and in no Python file at all, so `source="nom"` did not exist for Python users
for months; it was found by hand rather than by anything mechanical
(`ben-domingue/irw#1733`).

The duplication is deliberate. Publishing the registry as a Redivis table was
considered and rejected: it would put a network round-trip and a bootstrap
dependency in every client's cold start, so a client that could not reach
Redivis could no longer learn its own configuration. Instead,
`irw/metadata/check_config_parity.py` compares the three on every pull request
to the `irw` repository and fails when they disagree.

**Land this package and `Rpkg` before `irw`.** The check reads both packages at
their default branch, so while a shard exists in `irw` and not yet here, it
reports a real disagreement and the `irw` pull request stays red. Merging the
two package pull requests first makes it pass. That ordering is intended, not a
limitation to work around -- during that window the configs genuinely disagree.

The full cross-repo runbook, including the Redivis and metadata steps that have
no Python side, is `Rpkg/inst/developer/warehouses.md`.

## Cutting a release

The package is on PyPI as [`irw`](https://pypi.org/project/irw/). Users install
`pip install irw`; `pip install git+https://...` still works and is now the
development install.

1. **Bump both version literals in one PR** — `pyproject.toml` `[project]
   version` and `VERSION` in `src/irw/config.py`.
   `tests/test_version_string.py` fails if they disagree, so this is one commit,
   not two.
2. Merge it, and let `tests.yml` go green on `main`.
3. `git tag vX.Y.Z && git push origin vX.Y.Z`.

The tag fires `.github/workflows/release.yml`, which builds, runs
`twine check`, publishes to PyPI, and creates the GitHub release with the
artifacts attached. There is no other step.

**Never move a tag onto an unbumped commit.** A fix that lands on `main` without
a version bump does not reach anyone: pip resolves the version, sees it already
installed and skips even under `--upgrade`. That is not hypothetical — it left
Rpkg v1.1.2 broken against `irw_meta` v21.0 for three days (`Rpkg#153`).
`.github/scripts/check_release_version.py` runs before the build and refuses a
tag that disagrees with either literal; run it locally with the tag you intend
to push if you want the check early.

**PyPI version numbers are burn-once.** A number cannot be re-uploaded, even
after the release is deleted. To rehearse without spending one, run `release`
via `workflow_dispatch` — that does everything except the upload.

**Authentication is Trusted Publishing (OIDC), not a token.** There is no
credential in this repository. PyPI is configured to trust
`itemresponsewarehouse/Python-pkg`, workflow `release.yml`, environment `pypi`
— so renaming the workflow file or the environment breaks publishing until the
publisher entry is updated at
<https://pypi.org/manage/project/irw/settings/publishing/>.
