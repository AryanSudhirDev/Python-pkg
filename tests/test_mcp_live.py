"""Live machine checks for the MCP server, in the style of briefing-check/.

Written for one failure mode: everything runs, nothing errors, and the
numbers are wrong. Every assertion below is one a silent no-op cannot satisfy
-- a filter must return well under the whole catalogue, the size guard must
refuse the largest table without a download, item text must come back with
rows and rights, the processing notes must contain the fact metadata cannot.

Opt-in like the other live tests, because it needs Redivis credentials:

    RUN_REDIVIS_TESTS=1 python -m pytest tests/test_mcp_live.py -v

The only response table downloaded is the smallest in the corpus (72 rows),
so a run spends effectively no export quota.
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
        "collection=depression": tools.search_tables(collection="depression", limit=1)["total"],
        "longitudinal=True": tools.search_tables(longitudinal=True, limit=1)["total"],
        "has_item_text=True": tools.search_tables(has_item_text=True, limit=1)["total"],
        "query=phq": tools.search_tables("phq", limit=1)["total"],
    }
    for label, count in counts.items():
        assert 0 < count <= NOOP_FRACTION * total, f"{label}: {count} of {total}"
    assert len(set(counts.values())) > 1, counts


def test_untagged_tables_are_marked_not_hidden(tools, catalogue):
    n_untagged = catalogue["n_untagged_in_catalogue"]
    assert 0 < n_untagged < catalogue["total"]
    page = tools.search_tables(limit=100)["tables"]
    assert {row["tagged"] for row in page} <= {True, False}
    assert any("untagged table is not a non-matching" in c for c in catalogue["caveats"])


def test_size_guard_refuses_the_largest_table_without_downloading(tools):
    from irw.mcp import IRWMCPError

    started = time.time()
    with pytest.raises(IRWMCPError) as error:
        tools.fetch_table(LARGEST_TABLE, limit=1)
    assert error.value.code == "table_too_large"
    assert time.time() - started < 30, "a refusal must not take a download's time"


def test_fetch_of_the_smallest_table_is_real(tools):
    result = tools.fetch_table(SMALLEST_TABLE, limit=10)
    assert result["total_rows"] == 72
    assert result["returned"] == 10
    assert {"id", "item", "resp"} <= set(result["rows"][0])
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
