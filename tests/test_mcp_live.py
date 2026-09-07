"""Live machine checks for the MCP server, in the style of briefing-check/.

Written for one failure mode: everything runs, nothing errors, and the
numbers are wrong. Every assertion below is one a silent no-op cannot satisfy
-- a filter must return well under the whole catalogue, a page of the largest
table in the corpus must arrive in seconds, item text must come back with
rows and rights, the processing notes must contain the fact metadata cannot.

Opt-in like the other live tests, because it needs Redivis credentials:

    RUN_REDIVIS_TESTS=1 python -m pytest tests/test_mcp_live.py -v

The response data downloaded is the smallest table in the corpus (72 rows)
and two bounded pages, so a run spends effectively no export quota -- which
is itself one of the things being checked.
"""

from __future__ import annotations

import os
import time

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_REDIVIS_TESTS") != "1",
    reason="live Redivis checks; set RUN_REDIVIS_TESTS=1",
)

# On 0.0.2 a no-op filter returned 4,229 of 4,230 tables, so "strictly fewer"
# would have passed. Same bar as briefing-check: at most 98% of the catalogue.
NOOP_FRACTION = 0.98
LARGEST_TABLE = "criticalperiod_syntax"  # 107M responses per llms.txt section 3
LARGEST_TABLE_PAGE_SECONDS = 90  # a full export of it would be ~2.7 GB
SMALLEST_TABLE = "otaki_2022_dental_adaptability"  # 72 responses
SHARD_ONE_TEXT_TABLE = "16_personalityfactors"  # item text lives in irw_text, not _2
NOTED_TEXT_TABLE = "karajko2025_ai_benefit"  # has a public note on the issues page
NOTED_SCRIPT_TABLE = "gizaw_2023_phq9"  # id fell back to the row index; see #1713


@pytest.fixture(scope="module")
def tools():
    os.environ.setdefault("TQDM_DISABLE", "1")
    from irw.mcp import IRWTools

    return IRWTools()


@pytest.fixture(scope="module")
def catalogue(tools):
    return tools.search_tables(limit=1)


def test_catalogue_arrives_and_is_stamped(catalogue):
    assert catalogue["total"] > 1000
    assert catalogue["irw_version"] is not None
    assert int(catalogue["irw_version"]) >= 352
    assert catalogue["irw_released_at"].startswith("20")


def test_filters_filter(tools, catalogue):
    total = catalogue["total"]
    counts = {
        "collection=depression": tools.search_tables(
            filters={"collection": "depression"}, limit=1
        )["total"],
        "longitudinal=True": tools.search_tables(
            filters={"longitudinal": True}, limit=1
        )["total"],
        "has_item_text=True": tools.search_tables(
            filters={"has_item_text": True}, limit=1
        )["total"],
        "n_items>=50": tools.search_tables(
            filters={"n_items": [50, None]}, limit=1
        )["total"],
        "query=phq": tools.search_tables("phq", limit=1)["total"],
    }
    for label, count in counts.items():
        assert 0 < count <= NOOP_FRACTION * total, f"{label}: {count} of {total}"
    assert len(set(counts.values())) > 1, counts


def test_every_filter_the_package_offers_actually_runs(tools):
    """The adapter advertises the package's whole filter list. If a name in
    that list raises when it is passed through, the list is a lie."""
    for name in tools.filter_names():
        described = tools.describe_filter(name)
        assert described["description"]
        values = described.get("available_values")
        probe = values[0] if isinstance(values, list) and values else 1
        tools.search_tables(filters={name: probe}, limit=1)


def test_search_does_not_silently_drop_sparse_tables(tools, catalogue):
    """irw.filter() defaults density to [0.5, 1]. A caller asking about a
    construct did not ask for sparse tables to disappear."""
    unfiltered = tools.search_tables(filters={"has_item_text": True}, limit=1)["total"]
    dense_only = tools.search_tables(
        filters={"has_item_text": True, "density": [0.5, 1]}, limit=1
    )["total"]
    assert unfiltered > dense_only, (unfiltered, dense_only)


def test_search_cards_stay_small(tools):
    """A full metadata record is ~220 tokens, so a maximum page of them was
    ~28k of the assistant's context for a list of candidates."""
    import json

    page = tools.search_tables(limit=100)
    per_card = len(json.dumps(page["tables"])) / max(len(page["tables"]), 1)
    assert per_card < 500, f"{per_card:.0f} characters per card"


def test_untagged_tables_are_marked_not_hidden(tools, catalogue):
    n_untagged = catalogue["n_untagged_in_catalogue"]
    assert 0 < n_untagged < catalogue["total"]
    page = tools.search_tables(limit=100)["tables"]
    assert {row["tagged"] for row in page} <= {True, False}
    assert any("untagged table is not a non-matching" in c for c in catalogue["caveats"])


def test_a_page_of_the_largest_table_costs_a_page(tools):
    """The whole point of the wire-level bound. criticalperiod_syntax is 107M
    responses (~2.7 GB); a page of it has to arrive in seconds, not minutes,
    and a full export would blow the wall clock long before this timeout."""
    started = time.time()
    result = tools.fetch_table(LARGEST_TABLE, limit=5, columns=["id", "item", "resp"])
    elapsed = time.time() - started
    assert result["returned"] == 5
    assert len(result["rows"][0]) == 3
    assert elapsed < LARGEST_TABLE_PAGE_SECONDS, f"took {elapsed:.0f}s"
    # It cannot claim a row count it never downloaded.
    assert result["total_rows"] is None
    assert result["total_rows_estimate"] > 100_000_000


def test_the_guard_still_fires_where_a_page_cannot_bound_the_work(tools):
    """wide and dedup are computed over the whole table, so they keep the
    catalogue pre-check that the bounded path no longer needs."""
    from irw.mcp import IRWMCPError

    started = time.time()
    with pytest.raises(IRWMCPError) as error:
        tools.fetch_table(LARGEST_TABLE, limit=1, dedup=True)
    assert error.value.code == "table_too_large"
    assert time.time() - started < 30, "a refusal must not take a download's time"


def test_fetch_of_the_smallest_table_is_real(tools):
    result = tools.fetch_table(SMALLEST_TABLE, limit=10)
    assert result["returned"] == 10
    names = [column["name"] for column in result["columns"]]
    assert {"id", "item", "resp"} <= set(names)
    # Columnar: one list per row, in column order.
    assert len(result["rows"][0]) == len(names)
    assert result["irw_version"] is not None


def test_itemtext_from_the_older_shard_is_reachable(tools):
    """Regression for the phantom-handle bug: 732 of 747 text tables live in the
    first shard and every one of them reported 'not available'."""
    result = tools.get_itemtext(SHARD_ONE_TEXT_TABLE, limit=5)
    assert result["available"] is True, result["warnings"]
    assert result["returned"] > 0
    assert "item_text" in {column["name"] for column in result["columns"]}
    rights = result["rights"]
    assert rights["response_data_license"]
    assert rights["response_data_license_field"] == "Derived License"
    assert rights["original_license"] is None
    assert "does not extend to the instrument" in rights["instrument_rights"]


def test_itemtext_public_notes_travel_with_the_text(tools):
    result = tools.get_itemtext(NOTED_TEXT_TABLE, limit=1)
    assert result["rights"]["public_notes"], "the issues page lists this table"
    assert any("public item-text note" in w for w in result["warnings"])


def test_processing_notes_carry_what_metadata_cannot(tools):
    result = tools.get_processing_notes(NOTED_SCRIPT_TABLE)
    assert result["match"] == "exact"
    header = result["scripts"][0]["header"]
    assert "row index" in header
    assert "hw_id" in header
    assert "import" not in header.split("\n")[-1]


def test_describe_citation_and_collections_return_records(tools):
    described = tools.describe_table(SMALLEST_TABLE)
    assert described["metadata"]["stats"]
    cited = tools.get_citation("bang_2023_depression")
    assert cited["available"] and cited["bibtex"][0].startswith("@")
    collections = tools.list_collections(limit=200)
    assert collections["total"] >= 20
    assert "depression" in {c.get("collection") for c in collections["collections"]}
