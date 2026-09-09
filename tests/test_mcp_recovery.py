"""Authentication fallback and global capture must recover without leaks."""

from concurrent.futures import ThreadPoolExecutor
import sys
import time
import warnings
import asyncio
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from irw.mcp import IRWMCPError, IRWTools, PackageBackend
from test_mcp import FakeBackend, FakeSource


def test_browser_fallback_is_blocked_and_hook_is_restored():
    from redivis.common import auth

    class ReadyBackend(PackageBackend):
        def ensure_ready(self):
            pass

    tools = IRWTools(ReadyBackend(), FakeSource())
    with patch.object(auth, "perform_oauth_login") as browser_login:
        with pytest.raises(IRWMCPError) as error:
            tools._call(lambda: auth.perform_oauth_login(scope=["data.data"]))
        assert error.value.code == "authentication_required"
        browser_login.assert_not_called()
        assert auth.perform_oauth_login is browser_login
        assert tools._call(lambda: "recovered") == ("recovered", [])


def test_concurrent_instances_isolate_stdout_and_warnings():
    original_stdout = sys.stdout
    first = IRWTools(FakeBackend(), FakeSource())
    second = IRWTools(FakeBackend(), FakeSource())

    def noisy(label):
        print("private output " + label)
        warnings.warn(label, UserWarning)
        time.sleep(0.01)
        return label

    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(first._call, noisy, "first")
        b = pool.submit(second._call, noisy, "second")
        assert a.result() == ("first", ["first"])
        assert b.result() == ("second", ["second"])
    assert sys.stdout is original_stdout


def test_exception_restores_capture_and_later_call_works():
    tools = IRWTools(FakeBackend(), FakeSource())
    original_stdout = sys.stdout

    def broken():
        print("secret should not reach stdout")
        raise RuntimeError("secret should not reach client")

    with pytest.raises(IRWMCPError) as error:
        tools._call(broken)
    assert "secret" not in str(error.value)
    assert sys.stdout is original_stdout
    assert tools._call(lambda: 42) == (42, [])


def test_spawned_stdio_all_tools_errors_concurrency_and_shutdown():
    pytest.importorskip("mcp")
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters

    code = (
        "import asyncio; from irw.mcp import create_server; "
        "from test_mcp import FakeBackend, FakeSource; "
        "asyncio.run(create_server(FakeBackend(), FakeSource()).run_stdio_async())"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).parent)
    params = StdioServerParameters(command=sys.executable, args=["-c", code], env=env)

    async def check():
        async with Client(params) as client:
            listing = await client.list_tools()
            assert len(listing.tools) == 8
            calls = [
                ("search_tables", {"query": "math"}),
                ("describe_filter", {"filter_name": "n_items"}),
                ("describe_table", {"table_name": "alpha_depression"}),
                ("fetch_table", {"table_name": "alpha_depression", "limit": 1}),
                ("get_itemtext", {"table_name": "alpha_depression", "limit": 1}),
                ("list_collections", {}),
                ("get_citation", {"table_name": "alpha_depression"}),
                ("get_processing_notes", {"table_name": "alpha_depression"}),
            ]
            for name, arguments in calls:
                result = await client.call_tool(name, arguments)
                assert not result.is_error, (name, result)
                assert result.structured_content["result"]["source"] == "main"
            failed = await client.call_tool("fetch_table", {"table_name": "alpha_depression", "offset": 10000})
            assert failed.is_error
            assert "request_too_large" in str(failed.content)
            recovered = await asyncio.gather(*[
                client.call_tool("search_tables", {"query": query})
                for query in ("math", "depression", "math")
            ])
            assert all(not result.is_error for result in recovered)

    asyncio.run(asyncio.wait_for(check(), timeout=30))
