"""Local MCP server for read-only access to the Item Response Warehouse.

The MCP dependency is intentionally imported lazily.  Importing ``irw`` without
the optional MCP extra must continue to work on every Python version supported
by the core package.
"""

from __future__ import annotations

import asyncio
import base64
import datetime as dt
import io
import json
import logging
import math
import re
import sys
import threading
import warnings
from collections.abc import Mapping
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np
import pandas as pd

import irw

from .operations.list_tables import IRWMetadataUnavailable

logger = logging.getLogger(__name__)

SOURCE = "main"
SEARCH_DEFAULT_LIMIT = 20
SEARCH_MAX_LIMIT = 100
ROW_DEFAULT_LIMIT = 100
ROW_MAX_LIMIT = 1000
ITEMTEXT_MAX_LIMIT = 500
COLLECTION_DEFAULT_LIMIT = 100
COLLECTION_MAX_LIMIT = 200

ITEMTEXT_DISCLAIMER = (
    "IRW item text is reconstructed from published sources with partial human "
    "review. Verify it against the original source; availability does not grant "
    "rights to reuse an instrument. See "
    "https://itemresponsewarehouse.org/itemtext_issues.html"
)

_INTERNAL_METADATA_COLUMNS = {"name_lower", "table_lower", "bibtex"}
_METADATA_KEY_MAP = {
    "Description": "description",
    "Reference_x": "reference",
    "DOI__for_paper_": "doi",
    "URL__for_data_": "url",
    "Derived_License": "license",
    "BibTex": "bibtex",
}
_TRANSIENT_MARKERS = (
    "timeout",
    "temporar",
    "connection",
    "server error",
    "502",
    "503",
    "incomplete read",
    "remotedisconnected",
    "read timed out",
    "protocolerror",
)
_AUTH_MARKERS = (
    "authentication",
    "unauthorized",
    "credentials",
    "permission denied",
    "login required",
    "access token",
)


class IRWMCPError(RuntimeError):
    """A safe, machine-readable error returned by an MCP tool."""

    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        self.code = code
        self.message = message
        self.retryable = retryable
        payload = {
            "code": code,
            "message": message,
            "retryable": retryable,
        }
        super().__init__(json.dumps(payload, sort_keys=True))


class IRWBackend(Protocol):
    """The package calls needed by the MCP adapter."""

    def list_tables(self) -> pd.DataFrame: ...

    def describe_table(self, table_name: str) -> Any: ...

    def fetch_table(self, table_name: str, *, wide: bool, dedup: bool) -> Any: ...

    def itemtext(self, table_name: str) -> Any: ...

    def collections(self) -> pd.DataFrame: ...


class PackageBackend:
    """Default backend that delegates to the public ``irw`` API."""

    def list_tables(self) -> pd.DataFrame:
        return irw.list_tables(source=SOURCE, include_metadata=True)

    def describe_table(self, table_name: str) -> Any:
        return irw.info(table_name, source=SOURCE, return_dict=True)

    def fetch_table(self, table_name: str, *, wide: bool, dedup: bool) -> Any:
        return irw.fetch(table_name, source=SOURCE, wide=wide, dedup=dedup)

    def itemtext(self, table_name: str) -> Any:
        return irw.itemtext(table_name)

    def collections(self) -> pd.DataFrame:
        return irw.collections()


@dataclass
class _ConversionState:
    warnings: List[str] = field(default_factory=list)
    _seen: set[str] = field(default_factory=set)

    def add(self, message: str) -> None:
        message = message.strip()
        if message and message not in self._seen:
            self._seen.add(message)
            self.warnings.append(message)


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    try:
        marker = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(marker, (bool, np.bool_)) and bool(marker)


def _jsonable(value: Any, state: _ConversionState, *, path: str = "value") -> Any:
    """Convert pandas/numpy values into strict JSON-compatible values."""
    if _is_missing(value):
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _jsonable(value.item(), state, path=path)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist(), state, path=path)
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray, memoryview)):
        encoded = base64.b64encode(bytes(value)).decode("ascii")
        return f"base64:{encoded}"
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item, state, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _jsonable(item, state, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, (set, frozenset)):
        ordered = sorted(value, key=repr)
        return [
            _jsonable(item, state, path=f"{path}[{index}]")
            for index, item in enumerate(ordered)
        ]

    try:
        json.dumps(value, allow_nan=False)
        return value
    except (TypeError, ValueError):
        state.add(f"Converted unsupported value at {path} to text.")
        return str(value)


def _warning_messages(caught: List[warnings.WarningMessage]) -> List[str]:
    state = _ConversionState()
    for warning in caught:
        state.add(str(warning.message))
    return state.warnings


def _map_exception(error: Exception) -> IRWMCPError:
    if isinstance(error, IRWMCPError):
        return error

    message = str(error).casefold()
    if isinstance(error, IRWMetadataUnavailable):
        return IRWMCPError(
            "upstream_unavailable",
            "IRW metadata could not be loaded. Check network access and try again.",
            retryable=True,
        )
    if any(marker in message for marker in _AUTH_MARKERS):
        return IRWMCPError(
            "authentication_required",
            "Redivis authentication is required. Authenticate with the IRW "
            "package and retry.",
        )
    if "not found" in message or "not_found" in message:
        return IRWMCPError("not_found", "The requested IRW resource was not found.")
    if any(marker in message for marker in _TRANSIENT_MARKERS):
        return IRWMCPError(
            "upstream_unavailable",
            "The IRW data service was temporarily unavailable. Retry the request.",
            retryable=True,
        )
    return IRWMCPError(
        "upstream_error",
        "The IRW package could not complete the requested operation.",
    )


def _validate_text(value: Any, field_name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise IRWMCPError("invalid_input", f"{field_name} must be a string.")
    value = value.strip()
    if not allow_empty and not value:
        raise IRWMCPError("invalid_input", f"{field_name} must not be empty.")
    if "\x00" in value or any(ord(char) < 32 and char not in "\t\n" for char in value):
        raise IRWMCPError(
            "invalid_input", f"{field_name} contains a control character."
        )
    if len(value) > 512:
        raise IRWMCPError("invalid_input", f"{field_name} is too long.")
    return value


def _validate_table_name(table_name: Any) -> str:
    return _validate_text(table_name, "table_name")


def _validate_limit(
    value: Any, default: int, maximum: int, field_name: str = "limit"
) -> int:
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, int):
        raise IRWMCPError("invalid_input", f"{field_name} must be an integer.")
    if value < 1 or value > maximum:
        raise IRWMCPError(
            "invalid_input", f"{field_name} must be between 1 and {maximum}."
        )
    return value


def _validate_offset(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise IRWMCPError("invalid_input", "offset must be an integer.")
    if value < 0:
        raise IRWMCPError("invalid_input", "offset must be non-negative.")
    return value


def _validate_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise IRWMCPError("invalid_input", f"{field_name} must be a boolean.")
    return value


def _validate_columns(columns: Optional[List[str]]) -> Optional[List[str]]:
    if columns is None:
        return None
    if not isinstance(columns, list) or not columns:
        raise IRWMCPError(
            "invalid_input", "columns must be a non-empty list of strings."
        )
    if any(not isinstance(column, str) or not column.strip() for column in columns):
        raise IRWMCPError("invalid_input", "columns must contain non-empty strings.")
    normalized = [column.strip() for column in columns]
    if len(set(normalized)) != len(normalized):
        raise IRWMCPError("invalid_input", "columns must not contain duplicates.")
    return normalized


def _iter_values(value: Any) -> List[Any]:
    if _is_missing(value):
        return []
    if isinstance(value, Mapping):
        return [item for pair in value.items() for item in pair]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [item for nested in value for item in _iter_values(nested)]
    return [value]


def _search_text(value: Any) -> str:
    if _is_missing(value):
        return ""
    if isinstance(value, Mapping):
        return " ".join(
            f"{_search_text(key)} {_search_text(item)}" for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return " ".join(_search_text(item) for item in value)
    return str(value)


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and not _is_missing(
        value
    ):
        if value in (0, 1):
            return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "t", "yes", "y", "1"}:
            return True
        if normalized in {"false", "f", "no", "n", "0"}:
            return False
    return None


def _matches_text(value: Any, wanted: str) -> bool:
    needle = wanted.casefold()
    return any(needle in _search_text(item).casefold() for item in _iter_values(value))


def _matches_exact(value: Any, wanted: str) -> bool:
    needle = wanted.casefold()
    return any(
        _search_text(item).strip().casefold() == needle for item in _iter_values(value)
    )


def _canonical_metadata_key(key: Any) -> str:
    key_text = str(key)
    return _METADATA_KEY_MAP.get(key_text, key_text)


def _metadata_record(row: Mapping[Any, Any], state: _ConversionState) -> Dict[str, Any]:
    record: Dict[str, Any] = {}
    for key, value in row.items():
        canonical_key = _canonical_metadata_key(key)
        if canonical_key in _INTERNAL_METADATA_COLUMNS or canonical_key.startswith("_"):
            continue
        record[canonical_key] = _jsonable(
            value, state, path=f"metadata.{canonical_key}"
        )
    if "name" not in record and "table" in record:
        record["name"] = record.pop("table")
    return record


def _page_dataframe(
    frame: pd.DataFrame,
    *,
    row_key: str,
    total_key: str,
    limit: int,
    offset: int,
    columns: Optional[List[str]],
    initial_warnings: List[str],
) -> Dict[str, Any]:
    if not isinstance(frame, pd.DataFrame):
        raise IRWMCPError("serialization_error", "IRW returned a non-tabular result.")

    original_columns = list(frame.columns)
    column_names = [str(column) for column in original_columns]
    if len(set(column_names)) != len(column_names):
        raise IRWMCPError("serialization_error", "IRW returned duplicate column names.")
    column_lookup = dict(zip(column_names, original_columns))
    selected_names = column_names if columns is None else columns
    missing = [column for column in selected_names if column not in column_lookup]
    if missing:
        raise IRWMCPError(
            "invalid_input",
            f"Unknown column(s): {', '.join(missing)}.",
        )

    selected_original = [column_lookup[name] for name in selected_names]
    view = frame.loc[:, selected_original].iloc[offset : offset + limit]
    state = _ConversionState()
    for message in initial_warnings:
        state.add(message)

    rows = []
    for row_index, values in enumerate(
        view.itertuples(index=False, name=None), start=offset
    ):
        rows.append(
            {
                name: _jsonable(value, state, path=f"{row_key}[{row_index}].{name}")
                for name, value in zip(selected_names, values)
            }
        )

    schema = [
        {"name": name, "dtype": str(frame[column_lookup[name]].dtype)}
        for name in selected_names
    ]
    total = int(len(frame))
    returned = len(rows)
    return {
        row_key: rows,
        "columns": schema,
        total_key: total,
        "offset": offset,
        "limit": limit,
        "returned": returned,
        "has_more": offset + returned < total,
        "truncated": returned < total,
        "warnings": state.warnings,
    }


class IRWTools:
    """Synchronous tool implementations, separated from MCP registration."""

    def __init__(self, backend: Optional[IRWBackend] = None) -> None:
        self.backend = backend or PackageBackend()
        self._capture_lock = threading.Lock()

    def _call(
        self, callback: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Tuple[Any, List[str]]:
        """Run package code without allowing stdout to corrupt MCP stdio."""
        callback_name = getattr(callback, "__name__", callback.__class__.__name__)
        with self._capture_lock:
            with redirect_stdout(io.StringIO()) as captured_stdout:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    try:
                        value = callback(*args, **kwargs)
                    except Exception as error:
                        if captured_stdout.getvalue().strip():
                            logger.debug(
                                "Suppressed package output from %s", callback_name
                            )
                        raise _map_exception(error) from None
        if captured_stdout.getvalue().strip():
            logger.debug("Suppressed package output from %s", callback_name)
        return value, _warning_messages(caught)

    def search_tables(
        self,
        query: str = "",
        collection: Optional[str] = None,
        variable: Optional[str] = None,
        license: Optional[str] = None,
        longitudinal: Optional[bool] = None,
        has_item_text: Optional[bool] = None,
        limit: int = SEARCH_DEFAULT_LIMIT,
        offset: int = 0,
    ) -> Dict[str, Any]:
        query = _validate_text(query, "query", allow_empty=True)
        if collection is not None:
            collection = _validate_text(collection, "collection")
        if variable is not None:
            variable = _validate_text(variable, "variable")
        if license is not None:
            license = _validate_text(license, "license")
        if longitudinal is not None:
            longitudinal = _validate_bool(longitudinal, "longitudinal")
        if has_item_text is not None:
            has_item_text = _validate_bool(has_item_text, "has_item_text")
        limit = _validate_limit(limit, SEARCH_DEFAULT_LIMIT, SEARCH_MAX_LIMIT)
        offset = _validate_offset(offset)

        frame, package_warnings = self._call(self.backend.list_tables)
        if not isinstance(frame, pd.DataFrame):
            raise IRWMCPError(
                "serialization_error", "IRW catalogue was not returned as a table."
            )

        query_casefold = query.casefold()
        query_tokens = re.findall(r"\w+", query_casefold, flags=re.UNICODE)
        state = _ConversionState()
        for message in package_warnings:
            state.add(message)
        matches: List[Tuple[int, str, Dict[str, Any]]] = []
        seen_names: set[str] = set()

        for _, row in frame.iterrows():
            raw = row.to_dict()
            record = _metadata_record(raw, state)
            name = record.get("name")
            if _is_missing(name):
                continue
            name_text = str(name)
            name_key = name_text.casefold()
            if name_key in seen_names:
                continue
            seen_names.add(name_key)

            if collection is not None and not _matches_exact(
                record.get("collections"), collection
            ):
                continue
            if variable is not None and not _matches_text(
                record.get("variables"), variable
            ):
                continue
            if license is not None and not _matches_text(
                record.get("license"), license
            ):
                continue
            if (
                longitudinal is not None
                and _as_bool(record.get("longitudinal")) != longitudinal
            ):
                continue
            if (
                has_item_text is not None
                and _as_bool(record.get("has_item_text")) != has_item_text
            ):
                continue

            searchable = " ".join(
                _search_text(value) for value in raw.values()
            ).casefold()
            if query_tokens and not all(token in searchable for token in query_tokens):
                continue

            score = 0
            if query_casefold:
                if name_key == query_casefold:
                    score += 100000
                elif query_casefold in name_key:
                    score += 10000
                score += sum(100 if token in name_key else 10 for token in query_tokens)
            matches.append((score, name_key, record))

        matches.sort(key=lambda item: (-item[0], item[1]))
        total = len(matches)
        page = [record for _, _, record in matches[offset : offset + limit]]
        return {
            "source": SOURCE,
            "tables": page,
            "total": total,
            "offset": offset,
            "limit": limit,
            "has_more": offset + len(page) < total,
            "warnings": state.warnings,
        }

    def describe_table(self, table_name: str) -> Dict[str, Any]:
        table_name = _validate_table_name(table_name)
        details, package_warnings = self._call(self.backend.describe_table, table_name)
        if details is None or details == {}:
            raise IRWMCPError(
                "not_found", f"No metadata was found for table '{table_name}'."
            )
        state = _ConversionState()
        for message in package_warnings:
            state.add(message)
        metadata = _jsonable(details, state, path="metadata")
        if not isinstance(metadata, Mapping):
            metadata = {"raw": metadata}
        schema = metadata.get("schema") or metadata.get("columns")
        return {
            "source": SOURCE,
            "table": table_name,
            "metadata": dict(metadata),
            "schema": schema,
            "warnings": state.warnings,
        }

    def fetch_table(
        self,
        table_name: str,
        limit: int = ROW_DEFAULT_LIMIT,
        offset: int = 0,
        columns: Optional[List[str]] = None,
        wide: bool = False,
        dedup: bool = False,
    ) -> Dict[str, Any]:
        table_name = _validate_table_name(table_name)
        limit = _validate_limit(limit, ROW_DEFAULT_LIMIT, ROW_MAX_LIMIT)
        offset = _validate_offset(offset)
        columns = _validate_columns(columns)
        wide = _validate_bool(wide, "wide")
        dedup = _validate_bool(dedup, "dedup")

        frame, package_warnings = self._call(
            self.backend.fetch_table,
            table_name,
            wide=wide,
            dedup=dedup,
        )
        if frame is None:
            raise IRWMCPError("not_found", f"Table '{table_name}' was not found.")
        if isinstance(frame, dict):
            candidate = frame.get(table_name)
            if candidate is None:
                for key, value in frame.items():
                    if str(key).casefold() == table_name.casefold():
                        candidate = value
                        break
            frame = candidate
        if frame is None:
            raise IRWMCPError("not_found", f"Table '{table_name}' was not found.")

        payload = _page_dataframe(
            frame,
            row_key="rows",
            total_key="total_rows",
            limit=limit,
            offset=offset,
            columns=columns,
            initial_warnings=package_warnings,
        )
        payload.update(
            {"source": SOURCE, "table": table_name, "wide": wide, "dedup": dedup}
        )
        return payload

    def get_itemtext(
        self,
        table_name: str,
        limit: int = ROW_DEFAULT_LIMIT,
        offset: int = 0,
    ) -> Dict[str, Any]:
        table_name = _validate_table_name(table_name)
        limit = _validate_limit(limit, ROW_DEFAULT_LIMIT, ITEMTEXT_MAX_LIMIT)
        offset = _validate_offset(offset)

        value, package_warnings = self._call(self.backend.itemtext, table_name)
        if isinstance(value, str) or value is None:
            state = _ConversionState()
            for message in package_warnings:
                state.add(message)
            state.add(ITEMTEXT_DISCLAIMER)
            return {
                "source": SOURCE,
                "table": table_name,
                "available": False,
                "items": [],
                "columns": [],
                "total_items": 0,
                "offset": offset,
                "limit": limit,
                "returned": 0,
                "has_more": False,
                "truncated": False,
                "disclaimer": ITEMTEXT_DISCLAIMER,
                "warnings": state.warnings,
            }

        payload = _page_dataframe(
            value,
            row_key="items",
            total_key="total_items",
            limit=limit,
            offset=offset,
            columns=None,
            initial_warnings=package_warnings + [ITEMTEXT_DISCLAIMER],
        )
        payload.update(
            {
                "source": SOURCE,
                "table": table_name,
                "available": True,
                "disclaimer": ITEMTEXT_DISCLAIMER,
            }
        )
        return payload

    def list_collections(
        self,
        limit: int = COLLECTION_DEFAULT_LIMIT,
        offset: int = 0,
    ) -> Dict[str, Any]:
        limit = _validate_limit(limit, COLLECTION_DEFAULT_LIMIT, COLLECTION_MAX_LIMIT)
        offset = _validate_offset(offset)
        frame, package_warnings = self._call(self.backend.collections)
        if not isinstance(frame, pd.DataFrame):
            raise IRWMCPError(
                "serialization_error", "IRW collections were not returned as a table."
            )

        state = _ConversionState()
        for message in package_warnings:
            state.add(message)
        records = []
        for _, row in frame.iterrows():
            records.append(_metadata_record(row.to_dict(), state))
        records.sort(key=lambda record: str(record.get("collection", "")).casefold())
        total = len(records)
        page = records[offset : offset + limit]
        return {
            "source": SOURCE,
            "collections": page,
            "total": total,
            "offset": offset,
            "limit": limit,
            "has_more": offset + len(page) < total,
            "warnings": state.warnings,
        }


def create_server(backend: Optional[IRWBackend] = None) -> Any:
    """Create the MCP server; import the optional SDK only when requested."""
    try:
        from mcp.server import MCPServer
    except ImportError as error:
        raise RuntimeError(
            "The MCP extra is not installed. Use Python 3.10+ and run "
            "`python -m pip install 'irw[mcp]'`."
        ) from error

    tools = IRWTools(backend)
    server = MCPServer(
        name="IRW",
        version=str(getattr(irw, "__version__", "unknown")),
        description="Read-only access to Item Response Warehouse tables and metadata.",
    )

    from mcp.types import ToolAnnotations

    read_only = ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
    )

    @server.tool(name="search_tables", annotations=read_only, structured_output=True)
    def search_tables(
        query: str = "",
        collection: Optional[str] = None,
        variable: Optional[str] = None,
        license: Optional[str] = None,
        longitudinal: Optional[bool] = None,
        has_item_text: Optional[bool] = None,
        limit: int = SEARCH_DEFAULT_LIMIT,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Search IRW table names and metadata using deterministic matching.

        Results are paginated with a default limit of 20 and a maximum of 100.
        Use structured filters for collections, variables, licenses, longitudinal
        studies, and item-text availability.
        """
        return tools.search_tables(
            query,
            collection,
            variable,
            license,
            longitudinal,
            has_item_text,
            limit,
            offset,
        )

    @server.tool(name="describe_table", annotations=read_only, structured_output=True)
    def describe_table(table_name: str) -> Dict[str, Any]:
        """Return metadata and statistics for one IRW table without fetching rows."""
        return tools.describe_table(table_name)

    @server.tool(name="fetch_table", annotations=read_only, structured_output=True)
    def fetch_table(
        table_name: str,
        limit: int = ROW_DEFAULT_LIMIT,
        offset: int = 0,
        columns: Optional[List[str]] = None,
        wide: bool = False,
        dedup: bool = False,
    ) -> Dict[str, Any]:
        """Fetch a bounded page of rows from one IRW table.

        The default page is 100 rows and the maximum is 1,000. Use offset for
        subsequent pages and columns to reduce the response size.
        """
        return tools.fetch_table(table_name, limit, offset, columns, wide, dedup)

    @server.tool(name="get_itemtext", annotations=read_only, structured_output=True)
    def get_itemtext(
        table_name: str,
        limit: int = ROW_DEFAULT_LIMIT,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Return a bounded page of item-level text and its research-use disclaimer."""
        return tools.get_itemtext(table_name, limit, offset)

    @server.tool(name="list_collections", annotations=read_only, structured_output=True)
    def list_collections(
        limit: int = COLLECTION_DEFAULT_LIMIT,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List IRW's labelled collections and their metadata."""
        return tools.list_collections(limit, offset)

    return server


def main() -> None:
    """Run the local stdio server for an MCP host."""
    try:
        server = create_server()
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error
    asyncio.run(server.run_stdio_async())


__all__ = [
    "IRWBackend",
    "IRWMCPError",
    "IRWTools",
    "PackageBackend",
    "create_server",
    "main",
]


if __name__ == "__main__":
    main()
