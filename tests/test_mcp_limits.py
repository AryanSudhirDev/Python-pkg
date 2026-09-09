"""Resource limits and public calls must hold independently of credentials."""

import json
import asyncio
import pandas as pd

import pytest

from irw.mcp import IRWMCPError, IRWTools, PAYLOAD_MAX_BYTES, _bound_payload
from test_mcp import FakeBackend, FakeSource


@pytest.mark.parametrize("size", [float("inf"), -1, 3.5, True, None])
def test_invalid_catalogue_sizes_do_not_authorize_whole_table_download(size):
    backend = FakeBackend()
    backend.tables["n_responses"] = pd.Series([size] * len(backend.tables), dtype=object)
    with pytest.raises(IRWMCPError) as error:
        IRWTools(backend, FakeSource()).fetch_table("alpha_depression", wide=True)
    assert error.value.code == "table_size_unknown"
    assert backend.fetch_calls == []


def test_wire_payload_limit_and_continuation_preserve_records():
    pytest.importorskip("mcp")
    from mcp import Client
    from irw.mcp import create_server

    backend = FakeBackend()
    original = [[i, "q1", "\u00e9" * 5000] for i in range(100)]
    backend.frames["alpha_depression"] = pd.DataFrame(original, columns=["id", "item", "resp"])

    async def check():
        async with Client(create_server(backend, FakeSource())) as client:
            offset = 0
            collected = []
            while True:
                result = await client.call_tool("fetch_table", {
                    "table_name": "alpha_depression", "limit": 100, "offset": offset,
                })
                assert not result.is_error
                assert len(json.dumps(result.structured_content, ensure_ascii=False).encode()) <= PAYLOAD_MAX_BYTES
                payload = result.structured_content["result"]
                collected.extend(payload["rows"])
                if payload["next_offset"] is None:
                    break
                assert payload["next_offset"] > offset
                offset = payload["next_offset"]
            assert collected == original
    asyncio.run(check())


def test_payload_pages_preserve_complete_unicode_values():
    original = [[str(i), "\u00e9" * 5000] for i in range(100)]
    offset = 0
    collected = []
    while offset < len(original):
        result = _bound_payload({
            "rows": original[offset:], "offset": offset,
            "has_more": False, "warnings": [], "total_rows": 100,
        })
        assert len(json.dumps(result, ensure_ascii=False).encode()) <= PAYLOAD_MAX_BYTES
        collected.extend(result["rows"])
        if result["next_offset"] is None:
            break
        assert result["next_offset"] > offset
        offset = result["next_offset"]
    assert collected == original


@pytest.mark.parametrize("payload", [
    {"citation": "x" * PAYLOAD_MAX_BYTES},
    {"rows": [["x" * PAYLOAD_MAX_BYTES]], "offset": 0, "warnings": []},
])
def test_indivisible_large_result_is_an_error(payload):
    with pytest.raises(IRWMCPError) as error:
        _bound_payload(payload)
    assert error.value.code == "response_too_large"


def test_large_offset_never_downloads():
    backend = FakeBackend()
    with pytest.raises(IRWMCPError) as error:
        IRWTools(backend, FakeSource()).fetch_table("alpha_depression", offset=10000)
    assert error.value.code == "request_too_large"
    assert backend.fetch_calls == []


def test_public_notes_and_static_descriptions_do_not_require_authentication():
    class NoCredentials(FakeBackend):
        def ensure_ready(self):
            raise AssertionError("Public operation tried to authenticate")

    tools = IRWTools(NoCredentials(), FakeSource())
    assert tools.filter_names()
    assert tools._filter_description("n_items")
    result = tools.get_processing_notes("alpha_depression")
    assert result["irw_version"] is not None
    assert result["data_pinned"] is False
