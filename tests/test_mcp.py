"""Offline tests for the optional IRW MCP adapter."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from irw.mcp import GitHubSource, IRWMCPError, IRWTools, create_server


class FakeBackend:
    def __init__(self):
        self.fetch_calls = []
        self.filter_calls = []
        self.tables = pd.DataFrame(
            {
                "name": ["alpha_depression", "beta_math", "gamma_depression"],
                "description": [
                    "Depression scale",
                    "Math assessment",
                    "Depression follow-up",
                ],
                "collections": [["depression", "instrument"], ["math"], ["depression"]],
                "variables": [
                    "id item resp cov_age",
                    "id item resp",
                    "id item resp wave",
                ],
                "license": ["CC BY", "CC0", "CC BY"],
                "longitudinal": [False, False, True],
                "has_item_text": [True, False, True],
                "n_responses": np.array([100, 200, 300], dtype=np.int64),
                "construct_type": ["Affective", None, "Affective"],
            }
        )
        big = self.tables.iloc[[1]].copy()
        big["name"] = ["huge_assessment"]
        big["n_responses"] = [50_000_000]
        self.tables = pd.concat([self.tables, big], ignore_index=True)
        self.info = {
            "alpha_depression": {
                "stats": {"n_responses": np.int64(100)},
                "tags": {"construct_name": "depression"},
            }
        }
        self.frames = {
            "alpha_depression": pd.DataFrame(
                {
                    "id": np.array([1, 2, 3], dtype=np.int64),
                    "item": ["q1", "q2", "q3"],
                    "resp": [1.0, np.nan, 3.0],
                    "when": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
                }
            ),
        }
        self.items = {
            "alpha_depression": pd.DataFrame(
                {
                    "item": ["q1", "q2"],
                    "text": ["I feel low", "I enjoy activities"],
                }
            )
        }

    def list_tables(self):
        return self.tables.copy()

    def describe_table(self, table_name):
        return self.info.get(table_name)

    def fetch_table(
        self, table_name, *, wide, dedup, max_rows=None, columns=None
    ):
        self.fetch_calls.append(
            {
                "table": table_name,
                "wide": wide,
                "dedup": dedup,
                "max_rows": max_rows,
                "columns": columns,
            }
        )
        frame = self.frames.get(table_name)
        if frame is None:
            return None
        # Stand in for Redivis: the wire honours the bounds, so a test that
        # asserts on the returned page cannot pass by accident when the
        # adapter drops back to slicing a full download.
        if columns is not None:
            frame = frame.loc[:, list(columns)]
        if max_rows is not None:
            frame = frame.iloc[:max_rows]
        return frame.copy()

    def filter_tables(self, **filters):
        self.filter_calls.append(dict(filters))
        frame = self.tables
        if "collection" in filters:
            wanted = filters["collection"]
            wanted = [wanted] if isinstance(wanted, str) else list(wanted)
            keep = frame["collections"].map(
                lambda cs: any(c in (cs or []) for c in wanted)
            )
            frame = frame[keep]
        if "longitudinal" in filters:
            frame = frame[frame["longitudinal"] == filters["longitudinal"]]
        if "license" in filters:
            frame = frame[frame["license"] == filters["license"]]
        if "construct_type" in filters:
            frame = frame[frame["construct_type"] == filters["construct_type"]]
        return pd.Series(frame["name"].tolist(), name="name", dtype=str)

    def filter_names(self):
        return [
            "n_responses",
            "n_items",
            "density",
            "var",
            "construct_type",
            "longitudinal",
            "license",
            "collection",
        ]

    def describe_filter(self, filter_name):
        return {
            "description": f"How {filter_name} works.",
            "available_values": ["a", "b"],
        }

    def itemtext(self, table_name):
        return self.items.get(table_name, "unavailable")

    def collections(self):
        return pd.DataFrame(
            {
                "collection": ["math", "depression"],
                "kind": ["construct", "construct"],
                "n_tables": pd.array([1, 2], dtype="Int64"),
            }
        )

    def citation(self, table_name):
        if table_name == "alpha_depression":
            return ["@article{alpha_depression,\n  title={Depression scale}\n}"]
        return []

    def version_stamp(self):
        return (42, "2026-09-06T14:49:33Z")


ISSUES_QMD = """---
title: "Item Text"
---
issues <- yaml.load(r"---(
- table: alpha_depression
  issue: |-
    The English in the `_translated` columns is a machine translation
    produced by IRW; treat it as a reading aid.
- table: other_table
  issue: |-
    Withdrawn on 2026-09-05: the rights holder bars redistribution.
)---")
"""

TREE_JSON = json.dumps(
    {
        "tree": [
            {"path": "data/alpha_depression.py", "type": "blob"},
            {"path": "data/DART_Brysbaert_2020.R", "type": "blob"},
            {"path": "data/README.md", "type": "blob"},
            {"path": "metadata/01_metadata.R", "type": "blob"},
        ]
    }
)

SCRIPTS = {
    "data/alpha_depression.py": (
        "#!/usr/bin/env python3\n"
        "# Source: https://example.org\n"
        "# `hw_id` collides within a wave, so id fell back to the row index.\n"
        "\n"
        "import pandas as pd\n"
        "df = pd.read_csv('x.csv')\n"
    ),
    "data/DART_Brysbaert_2020.R": "# Five sub-datasets from one paper.\nlibrary(dplyr)\n",
}

OVERRIDES_CSV = "date,tool,table,checks,reason,user\n2026-09-01,validate_irw,alpha_depression,rt_units,rt is already in seconds,bd\n"


class FakeSource(GitHubSource):
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail
        super().__init__(fetch_text=self._fetch_text)

    def _fetch_text(self, url):
        self.calls.append(url)
        if self.fail:
            raise ConnectionError("offline")
        if url.endswith("trees/main?recursive=1"):
            return TREE_JSON
        if url.endswith("itemtext_issues.qmd"):
            return ISSUES_QMD
        if url.endswith("validator_overrides.csv"):
            return OVERRIDES_CSV
        for path, text in SCRIPTS.items():
            if url.endswith(path):
                return text
        raise FileNotFoundError(url)


@pytest.fixture
def tools():
    return IRWTools(FakeBackend(), FakeSource())


def test_search_is_deterministic_and_bounded(tools):
    result = tools.search_tables("depression scale", limit=1)
    assert result["total"] == 1
    assert result["tables"][0]["name"] == "alpha_depression"
    assert result["has_more"] is False


def test_search_delegates_filtering_to_the_package(tools):
    """The filter names and semantics must be irw.filter()'s, not ours."""
    result = tools.search_tables(filters={"collection": "depression"})
    assert [row["name"] for row in result["tables"]] == [
        "alpha_depression",
        "gamma_depression",
    ]
    assert tools.backend.filter_calls[-1]["collection"] == "depression"
    assert result["filters_applied"] == {"collection": "depression"}

    result = tools.search_tables(filters={"longitudinal": True})
    assert [row["name"] for row in result["tables"]] == ["gamma_depression"]


def test_search_accepts_every_filter_the_package_offers(tools):
    """A filter list kept by hand is a list that goes stale."""
    for name in tools.backend.filter_names():
        tools.search_tables(filters={name: "x"})


def test_search_rejects_a_filter_the_package_does_not_have(tools):
    with pytest.raises(IRWMCPError) as error:
        tools.search_tables(filters={"has_item_text": True})
    assert error.value.code == "invalid_input"
    # The message has to name the real ones, or the caller just guesses again.
    assert "construct_type" in error.value.message


def test_search_opts_out_of_the_default_density_filter(tools):
    """irw.filter() defaults density to [0.5, 1] and drops sparse tables with
    only a warning. Nobody asked for that by asking about a construct."""
    tools.search_tables(filters={"construct_type": "Affective"})
    assert tools.backend.filter_calls[-1]["density"] is None


def test_search_keeps_a_density_the_caller_actually_asked_for(tools):
    tools.search_tables(filters={"density": [0.9, 1]})
    assert tools.backend.filter_calls[-1]["density"] == [0.9, 1]


def test_search_cards_are_lean_and_describe_table_still_has_everything(tools):
    """The full record is ~220 tokens; twenty of them is most of a search."""
    card = tools.search_tables()["tables"][0]
    assert "variables" not in card and "description" not in card
    assert card["name"] and "n_responses" in card and card["tagged"] is True


def test_search_query_still_matches_fields_the_card_omits(tools):
    """Leaning out the card must not lean out the search."""
    result = tools.search_tables(query="cov_age")
    assert [row["name"] for row in result["tables"]] == ["alpha_depression"]


def test_describe_filter_reports_the_packages_own_values(tools):
    result = tools.describe_filter("construct_type")
    assert result["description"] == "How construct_type works."
    assert result["available_values"] == ["a", "b"]
    with pytest.raises(IRWMCPError) as error:
        tools.describe_filter("nonsense")
    assert error.value.code == "invalid_input"


def test_search_paginates_and_sorts_without_query(tools):
    result = tools.search_tables(limit=2, offset=1)
    assert [row["name"] for row in result["tables"]] == [
        "beta_math",
        "gamma_depression",
    ]
    assert result["total"] == 4


def test_describe_suppresses_package_stdout(tools, capsys):
    result = tools.describe_table("alpha_depression")
    assert result["metadata"]["stats"]["n_responses"] == 100
    assert capsys.readouterr().out == ""


def test_fetch_is_bounded_and_json_safe(tools):
    result = tools.fetch_table("alpha_depression", limit=2, offset=1)
    assert result["returned"] == 2
    assert [column["name"] for column in result["columns"]] == [
        "id",
        "item",
        "resp",
        "when",
    ]
    # Columnar: values in column order, not a dict per row.
    assert result["rows"][0][0] == 2
    assert result["rows"][0][2] is None
    assert result["rows"][0][3] == "2026-01-02T00:00:00"
    json.dumps(result, allow_nan=False)


def test_fetch_bounds_the_window_on_the_wire_not_after_the_download(tools):
    """The assertion that matters is on what was requested. A row-count check
    passes just as well against a full download that was sliced afterwards."""
    tools.fetch_table("alpha_depression", limit=2, offset=1)
    assert tools.backend.fetch_calls[-1]["max_rows"] == 3


def test_fetch_does_not_claim_a_row_count_it_did_not_download(tools):
    result = tools.fetch_table("alpha_depression", limit=1)
    assert result["total_rows"] is None
    assert result["has_more"] is True
    assert result["total_rows_estimate"] == 100
    assert "not known from this call" in result["total_rows_note"]


def test_fetch_pushes_columns_to_the_wire(tools):
    result = tools.fetch_table("alpha_depression", columns=["id", "resp"], limit=1)
    assert tools.backend.fetch_calls[-1]["columns"] == ["id", "resp"]
    assert [column["name"] for column in result["columns"]] == ["id", "resp"]
    assert len(result["rows"][0]) == 2


def test_fetch_names_an_unknown_column_instead_of_sending_it(tools):
    with pytest.raises(IRWMCPError) as error:
        tools.fetch_table("alpha_depression", columns=["missing"])
    assert error.value.code == "invalid_input"
    assert "cov_age" in error.value.message
    assert tools.backend.fetch_calls == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"limit": 0},
        {"limit": 1001},
        {"offset": -1},
        {"wide": "yes"},
        {"columns": []},
    ],
)
def test_fetch_validates_bounds_and_types(tools, kwargs):
    with pytest.raises(IRWMCPError) as error:
        tools.fetch_table("alpha_depression", **kwargs)
    assert error.value.code == "invalid_input"


def test_missing_table_is_a_structured_error(tools):
    with pytest.raises(IRWMCPError) as error:
        tools.fetch_table("missing")
    assert error.value.code == "not_found"


def test_itemtext_is_bounded_and_carries_disclaimer(tools):
    result = tools.get_itemtext("alpha_depression", limit=1)
    assert result["available"] is True
    assert result["returned"] == 1
    assert "original source" in result["disclaimer"]
    assert any("rights" in warning for warning in result["warnings"])


def test_itemtext_unavailable_is_not_a_server_failure(tools):
    result = tools.get_itemtext("beta_math")
    assert result["available"] is False
    assert result["items"] == []
    assert result["total_items"] == 0


def test_collections_are_structured_and_paginated(tools):
    result = tools.list_collections(limit=1)
    assert result["collections"][0]["collection"] == "depression"
    assert result["total"] == 2
    assert result["has_more"] is True


def test_get_citation_returns_bibtex(tools):
    result = tools.get_citation("alpha_depression")
    assert result["available"] is True
    assert result["bibtex"][0].startswith("@article{alpha_depression")


def test_get_citation_missing_is_soft_not_an_error(tools):
    result = tools.get_citation("beta_math")
    assert result["available"] is False
    assert result["bibtex"] == []
    assert any("BibTeX" in warning for warning in result["warnings"])


def test_responses_are_stamped_with_irw_version(tools):
    assert tools.search_tables()["irw_version"] == "42"
    assert tools.list_collections()["irw_version"] == "42"
    assert tools.fetch_table("alpha_depression")["irw_version"] == "42"


def test_version_stamp_degrades_to_a_warning_when_manifest_fails():
    class BrokenManifest(FakeBackend):
        def version_stamp(self):
            return None

    result = IRWTools(BrokenManifest(), FakeSource()).search_tables()
    assert result["irw_version"] is None
    assert any("not pinned" in warning for warning in result["warnings"])


def test_missing_credentials_are_an_error_not_a_hang(monkeypatch, tmp_path):
    from irw.mcp import PackageBackend

    monkeypatch.delenv("REDIVIS_API_TOKEN", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    with pytest.raises(IRWMCPError) as error:
        PackageBackend().ensure_ready()
    assert error.value.code == "authentication_required"
    assert "REDIVIS_API_TOKEN" in error.value.message


def test_backend_ensure_ready_gates_every_call():
    class Unauthenticated(FakeBackend):
        def ensure_ready(self):
            raise IRWMCPError("authentication_required", "no credentials")

    with pytest.raises(IRWMCPError) as error:
        IRWTools(Unauthenticated()).search_tables()
    assert error.value.code == "authentication_required"


def test_server_exposes_exactly_the_seven_public_tools():
    pytest.importorskip("mcp")

    async def check():
        from mcp import Client

        async with Client(create_server(FakeBackend(), FakeSource())) as client:
            listed = await client.list_tools()
            assert {tool.name for tool in listed.tools} == {
                "search_tables",
                "describe_table",
                "fetch_table",
                "get_itemtext",
                "list_collections",
                "get_citation",
                "get_processing_notes",
            }
            assert all(tool.annotations.read_only_hint is True for tool in listed.tools)
            result = await client.call_tool("search_tables", {"query": "math"})
            assert result.is_error is False
            assert (
                result.structured_content["result"]["tables"][0]["name"] == "beta_math"
            )

            calls = [
                ("describe_table", {"table_name": "alpha_depression"}),
                ("fetch_table", {"table_name": "alpha_depression", "limit": 1}),
                ("get_itemtext", {"table_name": "alpha_depression", "limit": 1}),
                ("list_collections", {"limit": 1}),
                ("get_citation", {"table_name": "alpha_depression"}),
                ("get_processing_notes", {"table_name": "alpha_depression"}),
            ]
            for name, arguments in calls:
                result = await client.call_tool(name, arguments)
                assert result.is_error is False
                assert isinstance(result.structured_content["result"], dict)

    asyncio.run(check())


def test_stdio_entrypoint_lists_tools_without_protocol_noise():
    pytest.importorskip("mcp")
    entrypoint = Path(sys.executable).with_name("irw-mcp")
    assert entrypoint.is_file()
    code = """import asyncio
import os
from mcp import Client
from mcp.client.stdio import StdioServerParameters

async def main():
    params = StdioServerParameters(command=os.environ['IRW_MCP_ENTRYPOINT'])
    async with Client(params) as client:
        result = await client.list_tools()
        print(sorted(tool.name for tool in result.tools))

asyncio.run(main())
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "IRW_MCP_ENTRYPOINT": str(entrypoint)},
        text=True,
        capture_output=True,
        check=True,
    )
    assert "search_tables" in completed.stdout
    assert completed.stderr == ""


def test_version_stamp_carries_number_and_release_date(tools):
    result = tools.list_collections()
    assert result["irw_version"] == "42"
    assert result["irw_released_at"].startswith("2026-09-06")


def test_search_marks_untagged_tables_and_says_so(tools):
    result = tools.search_tables()
    by_name = {row["name"]: row for row in result["tables"]}
    assert by_name["alpha_depression"]["tagged"] is True
    assert by_name["beta_math"]["tagged"] is False
    assert result["n_untagged_in_catalogue"] == 2
    assert any("untagged table is not a non-matching" in c for c in result["caveats"])
    filtered = tools.search_tables(filters={"longitudinal": True})
    # The caveat text is the package's own, so it cannot drift from it.
    assert any("How longitudinal works." in c for c in filtered["caveats"])


def test_a_huge_table_can_still_be_sampled_a_page_at_a_time(tools):
    """The old guard refused these outright. A bounded page of a 50M-response
    table costs a page, and refusing it is the answer to a problem the wire
    limit already solved."""
    tools.backend.frames["huge_assessment"] = tools.backend.frames[
        "alpha_depression"
    ]
    result = tools.fetch_table("huge_assessment", limit=2)
    assert result["returned"] == 2
    assert tools.backend.fetch_calls[-1]["max_rows"] == 2


def test_the_guard_still_fires_where_the_window_cannot_bound_the_work(tools):
    """wide and dedup describe the whole table, so they cannot be paged."""
    for kwargs in ({"wide": True}, {"dedup": True}):
        with pytest.raises(IRWMCPError) as error:
            tools.fetch_table("huge_assessment", **kwargs)
        assert error.value.code == "table_too_large"
        assert "50,000,000" in error.value.message
    assert tools.backend.fetch_calls == [], "the guard must precede any download"


def test_dedup_on_a_small_table_says_it_downloaded_the_whole_thing(tools):
    result = tools.fetch_table("alpha_depression", limit=1, dedup=True)
    assert tools.backend.fetch_calls[-1]["max_rows"] is None
    assert any("whole table" in w for w in result["warnings"])


def test_fetch_of_an_uncatalogued_table_proceeds_bounded(tools):
    """No catalogue entry means no size to check -- but the window still
    bounds the download, so there is nothing to warn about."""
    tools.backend.frames["off_catalogue"] = tools.backend.frames["alpha_depression"]
    result = tools.fetch_table("off_catalogue", limit=1)
    assert result["returned"] == 1
    assert tools.backend.fetch_calls[-1]["max_rows"] == 1


def test_an_uncatalogued_table_warns_when_it_must_be_downloaded_whole(tools):
    tools.backend.frames["off_catalogue"] = tools.backend.frames["alpha_depression"]
    result = tools.fetch_table("off_catalogue", limit=1, dedup=True)
    assert any("not in the IRW catalogue" in w for w in result["warnings"])


def test_itemtext_carries_rights_licence_and_public_notes(tools):
    result = tools.get_itemtext("alpha_depression", limit=1)
    rights = result["rights"]
    assert rights["response_data_license"] == "CC BY"
    assert "not an instrument licence" in rights["instrument_rights"] or "does not extend to the instrument" in rights["instrument_rights"]
    assert len(rights["public_notes"]) == 1
    assert "machine translation" in rights["public_notes"][0]
    # The derived licence must not be passed off as the source deposit's.
    assert rights["response_data_license_field"] == "Derived License"
    assert rights["original_license"] is None
    assert "reported as null" in rights["original_license_note"]
    assert any("public item-text note" in w for w in result["warnings"])


def test_itemtext_unavailable_but_catalogued_is_flagged_as_a_fault(tools):
    # gamma_depression is flagged has_item_text=True but the fake has no text.
    result = tools.get_itemtext("gamma_depression")
    assert result["available"] is False
    assert result["rights"]["public_notes"] == []
    assert any("package or shard fault" in w for w in result["warnings"])
    # beta_math is flagged False, so silence is the truth there.
    assert not any("shard fault" in w for w in tools.get_itemtext("beta_math")["warnings"])


def test_itemtext_survives_an_unreachable_issues_page():
    result = IRWTools(FakeBackend(), FakeSource(fail=True)).get_itemtext("alpha_depression", limit=1)
    assert result["available"] is True
    assert result["rights"]["public_notes"] == []
    assert any("could not be loaded" in w for w in result["warnings"])


def test_processing_notes_exact_match_returns_the_header(tools):
    result = tools.get_processing_notes("alpha_depression")
    assert result["match"] == "exact"
    assert result["scripts"][0]["path"] == "data/alpha_depression.py"
    assert "row index" in result["scripts"][0]["header"]
    assert "import pandas" not in result["scripts"][0]["header"]
    assert result["scripts"][0]["url"].startswith("https://github.com/ben-domingue/irw/blob/main/data/")
    assert result["validator_overrides"][0]["checks"] == "rt_units"


def test_processing_notes_prefix_match_names_a_multi_table_script(tools):
    result = tools.get_processing_notes("DART_Brysbaert_2020_1")
    assert result["match"] == "prefix"
    assert result["scripts"][0]["path"] == "data/DART_Brysbaert_2020.R"
    assert any("prefix" in w for w in result["warnings"])


def test_processing_notes_missing_script_is_a_warning_not_an_error(tools):
    result = tools.get_processing_notes("nothing_like_this")
    assert result["match"] == "none"
    assert result["scripts"] == []
    assert any("No processing script" in w for w in result["warnings"])


def test_processing_notes_offline_is_a_retryable_error():
    with pytest.raises(IRWMCPError) as error:
        IRWTools(FakeBackend(), FakeSource(fail=True)).get_processing_notes("alpha_depression")
    assert error.value.code == "upstream_unavailable"
    assert error.value.retryable is True


def test_github_source_fetches_each_resource_once(tools):
    tools.get_processing_notes("alpha_depression")
    tools.get_processing_notes("alpha_depression")
    tools.get_itemtext("alpha_depression")
    tree_calls = [u for u in tools.source.calls if u.endswith("recursive=1")]
    issue_calls = [u for u in tools.source.calls if u.endswith("itemtext_issues.qmd")]
    assert len(tree_calls) == 1
    assert len(issue_calls) == 1


def test_issue_list_parser_handles_the_page_format():
    from irw.mcp import _parse_issue_list

    parsed = _parse_issue_list(ISSUES_QMD)
    assert set(parsed) == {"alpha_depression", "other_table"}
    assert parsed["other_table"] == ["Withdrawn on 2026-09-05: the rights holder bars redistribution."]


def test_script_header_stops_at_code_and_handles_docstrings():
    from irw.mcp import _script_header

    header, truncated = _script_header('"""Notes.\nMore notes.\n"""\nimport os\n')
    assert header == '"""Notes.\nMore notes.\n"""'
    assert truncated is False
    header, _ = _script_header("x <- 1\ny <- 2\n")
    assert header.startswith("x <- 1")
    header, truncated = _script_header("\n".join("# line" for _ in range(500)))
    assert truncated is True


def test_a_truncated_github_listing_is_not_reported_as_a_missing_script():
    """GitHub caps a recursive tree and says so only in `truncated`. A capped
    listing looks exactly like a repo with fewer scripts in it, which turns
    "no notes for this table" from an error into a false statement."""
    truncated = json.loads(TREE_JSON)
    truncated["truncated"] = True
    truncated["tree"] = [
        entry for entry in truncated["tree"] if "alpha_depression" not in entry["path"]
    ]

    class _Truncated(FakeSource):
        def _fetch_text(self, url):
            if url.endswith("trees/main?recursive=1"):
                return json.dumps(truncated)
            return super()._fetch_text(url)

    tools = IRWTools(FakeBackend(), _Truncated())
    result = tools.get_processing_notes("alpha_depression")
    assert result["match"] == "none"
    assert any("truncated the repository listing" in w for w in result["warnings"])


def test_a_complete_listing_does_not_claim_it_was_truncated(tools):
    result = tools.get_processing_notes("not_a_table_anywhere")
    assert result["match"] == "none"
    assert not any("truncated" in w for w in result["warnings"])


def test_the_search_tool_description_is_generated_from_the_package():
    """A hand-written filter list in the tool text is a list that goes stale
    the first time irw.filter() gains an argument."""
    from irw.mcp import _search_tables_doc

    backend = FakeBackend()
    doc = _search_tables_doc(IRWTools(backend, FakeSource()))
    for name in backend.filter_names():
        assert f"- {name}: How {name} works." in doc


def test_the_description_still_builds_when_the_catalogue_is_unreachable():
    """A server that will not start is worse than one with a terse description."""
    from irw.mcp import _search_tables_doc

    class _Broken(FakeBackend):
        def filter_names(self):
            raise ConnectionError("offline")

    doc = _search_tables_doc(IRWTools(_Broken(), FakeSource()))
    assert "describe_filter" in doc


def test_filter_results_that_are_not_a_series_still_resolve():
    """irw.filter() returns a Series today; _name_set must not depend on that."""
    from irw.mcp import _name_set

    assert _name_set(pd.Series(["A", "b"])) == {"a", "b"}
    assert _name_set(["A", None]) == {"a"}
    assert _name_set(np.array(["A"])) == {"a"}
    assert _name_set(pd.DataFrame({"name": ["A"]})) == {"a"}
    assert _name_set(None) == set()
