"""The advertised wire schema must describe and validate each tool result."""

import asyncio

import pytest

pytest.importorskip("mcp")

from mcp import Client
from pydantic import ValidationError

from irw.mcp import create_server
from irw.mcp_models import OUTPUT_MODELS
from test_mcp import FakeBackend, FakeSource


def test_every_tool_advertises_concrete_required_result_fields():
    async def check():
        async with Client(create_server(FakeBackend(), FakeSource())) as client:
            listing = await client.list_tools()
            for tool in listing.tools:
                schema = tool.output_schema
                reference = schema["properties"]["result"]["$ref"].split("/")[-1]
                result = schema["$defs"][reference]
                assert {"source", "warnings", "retrieved_at", "data_pinned"} <= set(result["required"])
                assert result["additionalProperties"] is False
                assert len(result["properties"]) >= 10
    asyncio.run(check())


@pytest.mark.parametrize("name", list(OUTPUT_MODELS))
def test_empty_result_is_not_a_valid_tool_response(name):
    with pytest.raises(ValidationError):
        OUTPUT_MODELS[name].model_validate({"result": {}})
