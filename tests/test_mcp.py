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

from irw.mcp import IRWMCPError, IRWTools, create_server


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
            }
        )
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


@pytest.fixture
def tools():
    return IRWTools(FakeBackend())


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
    assert result["total"] == 3


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


def test_server_exposes_exactly_the_five_public_tools():
    pytest.importorskip("mcp")

    async def check():
        from mcp import Client

        async with Client(create_server(FakeBackend())) as client:
            listed = await client.list_tools()
            assert {tool.name for tool in listed.tools} == {
                "search_tables",
                "describe_table",
                "fetch_table",
                "get_itemtext",
                "list_collections",
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
