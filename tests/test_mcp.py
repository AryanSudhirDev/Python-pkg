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

    def fetch_table(self, table_name, *, wide, dedup):
        return self.frames.get(table_name)

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


def test_search_filters_collections_variables_and_longitudinal(tools):
    result = tools.search_tables(collection="depression", variable="cov_age")
    assert [row["name"] for row in result["tables"]] == ["alpha_depression"]

    result = tools.search_tables(longitudinal=True, has_item_text=True)
    assert [row["name"] for row in result["tables"]] == ["gamma_depression"]


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
    assert result["total_rows"] == 3
    assert result["returned"] == 2
    assert result["has_more"] is False
    assert result["rows"][0]["id"] == 2
    assert result["rows"][0]["resp"] is None
    assert result["rows"][0]["when"] == "2026-01-02T00:00:00"
    json.dumps(result, allow_nan=False)


def test_fetch_selects_columns_and_rejects_unknown_columns(tools):
    result = tools.fetch_table("alpha_depression", columns=["id", "resp"], limit=1)
    assert [column["name"] for column in result["columns"]] == ["id", "resp"]
    assert set(result["rows"][0]) == {"id", "resp"}
    with pytest.raises(IRWMCPError) as error:
        tools.fetch_table("alpha_depression", columns=["missing"])
    assert error.value.code == "invalid_input"


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
    filtered = tools.search_tables(longitudinal=True)
    assert any("cov_birthdate" in c for c in filtered["caveats"])


def test_fetch_refuses_a_table_over_the_size_guard_before_downloading():
    backend = FakeBackend()
    calls = []
    original = backend.fetch_table

    def spy(table_name, *, wide, dedup):
        calls.append(table_name)
        return original(table_name, wide=wide, dedup=dedup)

    backend.fetch_table = spy
    with pytest.raises(IRWMCPError) as error:
        IRWTools(backend, FakeSource()).fetch_table("huge_assessment")
    assert error.value.code == "table_too_large"
    assert "50,000,000" in error.value.message
    assert calls == [], "the guard must fire before any download"


def test_fetch_of_an_uncatalogued_table_warns_but_proceeds(tools):
    tools.backend.frames["off_catalogue"] = tools.backend.frames["alpha_depression"]
    result = tools.fetch_table("off_catalogue", limit=1)
    assert result["returned"] == 1
    assert any("not in the IRW catalogue" in w for w in result["warnings"])


def test_itemtext_carries_rights_licence_and_public_notes(tools):
    result = tools.get_itemtext("alpha_depression", limit=1)
    rights = result["rights"]
    assert rights["response_data_license"] == "CC BY"
    assert "not an instrument licence" in rights["instrument_rights"] or "does not extend to the instrument" in rights["instrument_rights"]
    assert len(rights["public_notes"]) == 1
    assert "machine translation" in rights["public_notes"][0]
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
