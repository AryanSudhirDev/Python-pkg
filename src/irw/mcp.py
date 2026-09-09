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
import os
import re
import sys
import threading
import warnings
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np
import pandas as pd

import irw

from .operations.list_tables import IRWMetadataUnavailable
from .operations.filter import InvalidFilterValue, validate_filter_value
from .utils.redivis.tables import _classify_error, _sanitize_error

logger = logging.getLogger(__name__)
_PACKAGE_CALL_LOCK = threading.RLock()

SOURCE = "main"
# The card returned per search hit. The full metadata record runs ~220 tokens,
# so a default page of 20 spent ~5.5k of the assistant's context and a maximum
# page ~28k -- most of it the `variables` string and the bibliography, which
# nobody reads twenty at a time. describe_table still returns everything.
SEARCH_CARD_FIELDS = (
    "name",
    "construct_type",
    "n_responses",
    "n_participants",
    "n_items",
    "n_categories",
    "density",
    "longitudinal",
    "license",
    "has_item_text",
    "collections",
)

SEARCH_DEFAULT_LIMIT = 20
SEARCH_MAX_LIMIT = 100
ROW_DEFAULT_LIMIT = 100
ROW_MAX_LIMIT = 1000
FETCH_MAX_WINDOW = 10_000
PAYLOAD_MAX_BYTES = 256 * 1024
ITEMTEXT_MAX_LIMIT = 500
COLLECTION_DEFAULT_LIMIT = 100
COLLECTION_MAX_LIMIT = 200

# The size guard from llms.txt section 3: the corpus is mostly small, and the
# export risk sits in the ~164 tables at a million responses or more. The same
# number in both places, so the briefing and the server cannot drift apart.
FETCH_MAX_RESPONSES = 1_000_000

IRW_REPO = "ben-domingue/irw"
IRW_REPO_TREE_URL = f"https://api.github.com/repos/{IRW_REPO}/git/trees/main?recursive=1"
IRW_REPO_RAW_URL = f"https://raw.githubusercontent.com/{IRW_REPO}/main/"
IRW_REPO_BLOB_URL = f"https://github.com/{IRW_REPO}/blob/main/"
ITEMTEXT_ISSUES_QMD_URL = (
    "https://raw.githubusercontent.com/datapages/irw/main/itemtext_issues.qmd"
)
ITEMTEXT_ISSUES_PAGE = "https://itemresponsewarehouse.org/itemtext_issues.html"
PROCESSING_NOTES_MAX_LINES = 120
PROCESSING_NOTES_MAX_CHARS = 8000
PUBLIC_SOURCE_MAX_BYTES = 8 * 1024 * 1024

# IRW records two licences per table: the Original License of the source
# deposit and the Derived License IRW redistributes its own extract under.
# Only the derived one reaches this server -- the biblio table the package
# reads carries `Derived_License` and nothing else, while `Original License`
# lives in the data dictionary and is not exported. So the field is reported
# as null with its provenance stated, rather than the derived licence being
# quietly passed off as the answer to a question about the source.
ORIGINAL_LICENSE_NOTE = (
    "IRW records an Original License for the source deposit separately from "
    "the Derived License above, but it is not carried in the metadata this "
    "server can read, so it is reported as null rather than guessed. A "
    "restrictive derived licence does not imply a restrictive original one, "
    "or the reverse. Check the table's entry in the IRW data dictionary "
    "before relying on either."
)

INSTRUMENT_RIGHTS_NOTE = (
    "The licence recorded for a table covers its response data only. It does "
    "not extend to the instrument: inclusion of item text implies no licence "
    "to reuse, reproduce or administer a scale, and copyright stays with the "
    "rights holders. Do not reproduce an instrument on the strength of the "
    "deposit licence; check the public notes for withdrawn or restricted text."
)

# Metadata columns filled in by human tagging. A table with none of them set
# is untagged, which is not the same as not matching a tag filter.
_TAG_COLUMNS = (
    "construct_type",
    "construct_name",
    "measurement_tool",
    "item_format",
    "sample",
    "language",
    "age_range",
)

AUTH_SETUP_MESSAGE = (
    "Usable Redivis credentials are unavailable or require renewed authorization. The Redivis SDK would open an "
    "interactive browser login, which cannot complete inside an MCP server. "
    "Authenticate once in a regular terminal with "
    "`python -c \"import irw; irw.list_tables()\"` (credentials are cached in "
    "~/.redivis), or set the REDIVIS_API_TOKEN environment variable for the "
    "MCP host, then retry."
)

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

    def fetch_table(
        self,
        table_name: str,
        *,
        wide: bool,
        dedup: bool,
        max_rows: Optional[int] = None,
        columns: Optional[List[str]] = None,
    ) -> Any: ...

    def filter_tables(self, **filters: Any) -> Any: ...

    def filter_names(self) -> List[str]: ...

    def filter_descriptions(self) -> Mapping[str, str]: ...

    def describe_filter(self, filter_name: str) -> Any: ...

    def itemtext(self, table_name: str) -> Any: ...

    def collections(self) -> pd.DataFrame: ...

    def citation(self, table_name: str) -> List[str]: ...

    def version_stamp(self) -> Optional[Tuple[int, str]]: ...


class PackageBackend:
    """Default backend that delegates to the public ``irw`` API."""

    @contextmanager
    def noninteractive(self):
        """Block SDK browser fallback, including after a failed token refresh.

        Called under the process-wide package lock because the SDK auth hook
        and stdout/warning capture are process-global state.
        """
        from redivis.common import auth

        original = auth.perform_oauth_login

        def refuse_login(*args, **kwargs):
            raise IRWMCPError("authentication_required", AUTH_SETUP_MESSAGE)

        auth.perform_oauth_login = refuse_login
        try:
            yield
        finally:
            auth.perform_oauth_login = original

    def ensure_ready(self) -> None:
        """Refuse to start a Redivis call that would block on a browser login.

        The Redivis SDK's fallback for missing credentials is an interactive
        device-authorization flow that prints a URL and polls for up to ten
        minutes. Inside a stdio MCP server that print is captured and the
        tool call simply hangs, so the absence of credentials has to be an
        error the client can read, not a wait.
        """
        if os.getenv("REDIVIS_API_TOKEN"):
            return
        if (Path.home() / ".redivis" / "python_credentials").is_file():
            return
        raise IRWMCPError("authentication_required", AUTH_SETUP_MESSAGE)

    def list_tables(self) -> pd.DataFrame:
        return irw.list_tables(source=SOURCE, include_metadata=True)

    def describe_table(self, table_name: str) -> Any:
        return irw.info(table_name, source=SOURCE, return_dict=True)

    def fetch_table(
        self,
        table_name: str,
        *,
        wide: bool,
        dedup: bool,
        max_rows: Optional[int] = None,
        columns: Optional[List[str]] = None,
    ) -> Any:
        return irw.fetch(
            table_name,
            source=SOURCE,
            wide=wide,
            dedup=dedup,
            max_rows=max_rows,
            columns=columns,
        )

    def filter_tables(self, **filters: Any) -> Any:
        return irw.filter(**filters)

    def filter_names(self) -> List[str]:
        return list(irw.get_filters())

    def filter_descriptions(self) -> Mapping[str, str]:
        # The descriptions are a module-level dict. describe_filter() also
        # returns them, but only after loading the metadata tables to compute
        # each filter's available values -- a network call and a slice of the
        # export quota, which the tool description must not cost at startup.
        from .operations.filter_info import FILTER_DESCRIPTIONS

        return dict(FILTER_DESCRIPTIONS)

    def describe_filter(self, filter_name: str) -> Any:
        return irw.describe_filter(filter_name)

    def itemtext(self, table_name: str) -> Any:
        return irw.itemtext(table_name)

    def collections(self) -> pd.DataFrame:
        return irw.collections()

    def citation(self, table_name: str) -> List[str]:
        return irw.save_bibtex(table_name)

    def version_stamp(self) -> Optional[Tuple[int, str]]:
        from .operations.version import current_version

        return current_version()


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
    """Turn whatever the package raised into a structured, safe error.

    Classification is the package's, not ours: ``_classify_error`` is the one
    place that knows a Redivis quota error arrives wearing an
    ``invalid_request`` code, that the machine-readable code lives in
    ``args[0]["error"]`` rather than the message, and that ``not_found`` is
    spelt with an underscore. A second list of substrings here would drift
    from it the first time Redivis changed a message.
    """
    if isinstance(error, IRWMCPError):
        return error
    if isinstance(error, InvalidFilterValue):
        return IRWMCPError("invalid_input", str(error))
    if isinstance(error, IRWMetadataUnavailable):
        return IRWMCPError(
            "upstream_unavailable",
            "IRW metadata could not be loaded. Check network access and try again.",
            retryable=True,
        )

    kind = _classify_error(error)
    # The classifier reads message text, and a bare TimeoutError() or
    # ConnectionResetError() carries none. The type says what the text
    # would have.
    if kind == "unknown" and (
        isinstance(error, (TimeoutError, ConnectionError))
        or re.search(r"timeout|connection", type(error).__name__, re.IGNORECASE)
    ):
        kind = "transient"
    if kind == "quota":
        return IRWMCPError(
            "quota_exceeded",
            "The Redivis export quota for this account is exhausted. Wait for "
            "the quota to reset before fetching more data.",
        )
    if kind == "auth":
        return IRWMCPError(
            "authentication_required",
            "Redivis authentication is required. Authenticate with the IRW "
            "package and retry.",
        )
    if kind == "not_found":
        return IRWMCPError("not_found", "The requested IRW resource was not found.")
    if kind == "transient":
        return IRWMCPError(
            "upstream_unavailable",
            "The IRW data service was temporarily unavailable. Retry the request.",
            retryable=True,
        )
    if kind == "invalid_request":
        detail = _sanitize_error(str(error))
        return IRWMCPError(
            "invalid_input",
            "Redivis rejected the request as invalid"
            + (f": {detail}" if detail else ".")
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


DESCRIBE_FILTER_MAX_VALUES = 300


def _filter_values(values: Any, state: "_ConversionState") -> Dict[str, Any]:
    """Normalise irw.describe_filter()['values'] into one shape.

    The package returns a different type per filter kind: a dict of summary
    statistics for numeric filters, a pandas Series of counts indexed by value
    for tag, licence, variable and collection filters, and a {True: n,
    False: n} dict for the boolean ones. An assistant needs the same three
    keys every time -- `available_values` is the list it may pass back to
    search_tables, `value_counts` says how common each is, and `summary` is
    the numeric range -- so the shape is fixed here rather than left to
    whatever pandas happened to produce.
    """
    if isinstance(values, pd.Series):
        # A missing index entry is not a value anyone can filter on, and a
        # repeated one is one value counted twice; neither should reach the
        # list an assistant copies from.
        merged: Dict[Any, int] = {}
        for key, count in values.dropna().items():
            if _is_missing(key):
                continue
            key = _jsonable(key, state, path="filter.values")
            merged[key] = merged.get(key, 0) + int(count)
        pairs = list(merged.items())
    elif isinstance(values, Mapping):
        keys = {str(k) for k in values}
        if keys and keys <= {"True", "False", "true", "false"}:
            pairs = [(bool(k in (True, "True", "true")), int(v)) for k, v in values.items()]
        else:
            return {
                "kind": "numeric",
                "available_values": None,
                "value_counts": None,
                "summary": _jsonable(dict(values), state, path="filter.values"),
                "truncated": False,
            }
    elif values is None:
        return {
            "kind": "unknown",
            "available_values": None,
            "value_counts": None,
            "summary": None,
            "truncated": False,
        }
    else:
        pairs = [(_jsonable(v, state, path="filter.values"), None) for v in _iter_values(values)]

    truncated = len(pairs) > DESCRIBE_FILTER_MAX_VALUES
    if truncated:
        state.add(
            f"{len(pairs)} distinct values; only the {DESCRIBE_FILTER_MAX_VALUES} "
            "most common are listed."
        )
        pairs = pairs[:DESCRIBE_FILTER_MAX_VALUES]
    kind = "boolean" if pairs and all(isinstance(k, bool) for k, _ in pairs) else "categorical"
    return {
        "kind": kind,
        "available_values": [k for k, _ in pairs],
        "value_counts": (
            {str(k): v for k, v in pairs} if all(v is not None for _, v in pairs) else None
        ),
        "summary": None,
        "truncated": truncated,
    }


def _name_set(value: Any) -> set:
    """Table names from whatever irw.filter() handed back, casefolded.

    filter() returns a pandas Series today. _iter_values would treat one as a
    single opaque value -- a Series is neither a Mapping nor a list -- and the
    set would come out holding one stringified frame, which matches nothing
    and reads as "no such tables".
    """
    if _is_missing(value):
        return set()
    if isinstance(value, pd.Series):
        items = value.tolist()
    elif isinstance(value, pd.DataFrame):
        column = "name" if "name" in value.columns else value.columns[0]
        items = value[column].tolist()
    elif isinstance(value, (list, tuple, set, frozenset, np.ndarray)):
        items = list(value)
    elif isinstance(value, str):
        items = [value]
    else:
        items = list(value) if hasattr(value, "__iter__") else [value]
    return {str(item).casefold() for item in items if not _is_missing(item)}


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


def _is_tagged(raw: Mapping[Any, Any]) -> bool:
    return any(not _is_missing(raw.get(column)) for column in _TAG_COLUMNS if column in raw)


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

    # Columnar, not a list of row objects: a row object repeats every column
    # name on every row, which for a 1,000-row page is the column names a
    # thousand times over -- about half the response and none of the content.
    # `columns` below names the fields, in order, for every row.
    rows = []
    for row_index, values in enumerate(
        view.itertuples(index=False, name=None), start=offset
    ):
        rows.append(
            [
                _jsonable(value, state, path=f"{row_key}[{row_index}].{name}")
                for name, value in zip(selected_names, values)
            ]
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


def _bound_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Bound complete records; never alter a value to make it fit."""
    page_key = next((key for key in ('rows', 'items', 'tables', 'collections')
                     if isinstance(payload.get(key), list) and 'offset' in payload), None)
    if page_key:
        payload['returned'] = len(payload[page_key])
        payload['next_offset'] = (
            payload['offset'] + payload['returned'] if payload.get('has_more') else None
        )

    def size():
        return len(json.dumps({"result": payload}, ensure_ascii=False, allow_nan=False).encode('utf-8'))

    if size() <= PAYLOAD_MAX_BYTES:
        return payload
    if page_key and payload[page_key]:
        records = payload[page_key]
        payload['has_more'] = True
        payload['truncated'] = True
        payload['warnings'].append('The payload size limit reduced this page; continue with next_offset.')
        low, high = 0, len(records) - 1
        while low < high:
            count = (low + high + 1) // 2
            payload[page_key] = records[:count]
            payload['returned'] = count
            payload['next_offset'] = payload['offset'] + count
            if size() <= PAYLOAD_MAX_BYTES:
                low = count
            else:
                high = count - 1
        if low:
            payload[page_key] = records[:low]
            payload['returned'] = low
            payload['next_offset'] = payload['offset'] + low
            return payload
    raise IRWMCPError(
        'response_too_large',
        'A complete record or metadata result exceeds the 256 KiB response limit. '
        'Select fewer columns or retrieve the resource directly with the Python package.',
    )


def _http_get_text(url: str) -> str:
    """Fetch a public text resource. `requests` is a dependency of redivis."""
    import requests

    import time

    started = time.monotonic()
    with requests.get(url, timeout=(5, 15), stream=True,
                      headers={"User-Agent": "irw-mcp"}) as response:
        response.raise_for_status()
        chunks = []
        size = 0
        for chunk in response.iter_content(chunk_size=65536):
            size += len(chunk)
            if size > PUBLIC_SOURCE_MAX_BYTES:
                raise ValueError("Public source exceeds the download limit")
            if time.monotonic() - started > 30:
                raise TimeoutError("Public source exceeded the download deadline")
            chunks.append(chunk)
        return b"".join(chunks).decode("utf-8-sig")


def _parse_issue_list(text: str) -> Dict[str, List[str]]:
    """Parse the embedded YAML as data, without executing the surrounding R."""
    import yaml

    match = re.search(r'issues\s*<-\s*yaml\.load\(r"---\((.*?)\)---"\)', text, re.DOTALL)
    if match is None:
        raise ValueError("The public issue page has no recognized YAML issue block")
    records = yaml.safe_load(match.group(1))
    if not isinstance(records, list):
        raise ValueError("The public issue block must be a list")
    issues: Dict[str, List[str]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each public issue must be an object")
        table, issue = record.get("table"), record.get("issue")
        if not isinstance(table, str) or not table.strip() or not isinstance(issue, str) or not issue.strip():
            raise ValueError("Each public issue requires nonempty table and issue strings")
        issues.setdefault(table.strip().casefold(), []).append(issue.strip())
    return issues


def _script_header(text: str) -> Tuple[str, bool]:
    """The leading comment block of a processing script, bounded.

    R and Python scripts open with ``#`` comments, Python sometimes with a
    docstring, Stata with ``*`` or ``//``. The block ends at the first line of
    code. A script with no header at all falls back to its opening lines.
    """
    lines = text.splitlines()
    header: List[str] = []
    in_docstring = False
    for line in lines:
        stripped = line.strip()
        if in_docstring:
            header.append(line)
            if stripped.endswith('"""') or stripped.endswith("'''"):
                in_docstring = False
            continue
        if stripped.startswith(('"""', "'''")):
            header.append(line)
            if not (len(stripped) > 3 and stripped.endswith(stripped[:3])):
                in_docstring = True
            continue
        if stripped == "" or stripped.startswith(("#", "*", "//")):
            header.append(line)
            continue
        break
    if not any(line.strip() for line in header):
        header = lines[:40]
    truncated = False
    if len(header) > PROCESSING_NOTES_MAX_LINES:
        header, truncated = header[:PROCESSING_NOTES_MAX_LINES], True
    joined = "\n".join(header).strip()
    if len(joined) > PROCESSING_NOTES_MAX_CHARS:
        joined, truncated = joined[:PROCESSING_NOTES_MAX_CHARS], True
    return joined, truncated


_SCRIPT_SUFFIX = re.compile(r"\.(py|r|do|ipynb|txt)$", re.IGNORECASE)


def _processing_header(path: str, text: str) -> Tuple[str, bool]:
    if not path.lower().endswith(".ipynb"):
        return _script_header(text)
    notebook = json.loads(text)
    blocks = []
    for cell in notebook.get("cells", []):
        source = cell.get("source", [])
        source = source if isinstance(source, str) else "".join(source)
        if cell.get("cell_type") == "markdown":
            blocks.append(source)
        elif cell.get("cell_type") == "code":
            # Only actual leading comments, never executable notebook code.
            comments = []
            for line in source.splitlines():
                if line.strip() and not line.lstrip().startswith("#"):
                    break
                comments.append(line)
            blocks.append("\n".join(comments))
            break
    lines = "\n\n".join(blocks).strip().splitlines()
    text = "\n".join(lines[:PROCESSING_NOTES_MAX_LINES])
    truncated = len(lines) > PROCESSING_NOTES_MAX_LINES or len(text) > PROCESSING_NOTES_MAX_CHARS
    return text[:PROCESSING_NOTES_MAX_CHARS], truncated


def _match_scripts(table_name: str, paths: List[str]) -> Tuple[List[str], str]:
    """Find the processing script(s) for a table among ``data/`` paths.

    Exact stem match first. Failing that, a prefix match in either direction
    catches multi-table scripts (``DART_Brysbaert_2020.R`` produces
    ``DART_Brysbaert_2020_1`` and friends) and tables named more fully than
    their script. Returns the paths and how they were matched.
    """
    wanted = table_name.casefold()
    stems: Dict[str, List[str]] = {}
    for path in paths:
        if not _SCRIPT_SUFFIX.search(path):
            continue
        stem = _SCRIPT_SUFFIX.sub("", path.rsplit("/", 1)[-1]).casefold()
        stems.setdefault(stem, []).append(path)
    if wanted in stems:
        paths = sorted(stems[wanted])
        return paths, "exact" if len(paths) == 1 else "ambiguous"
    candidates = [
        stem
        for stem in stems
        if len(stem) >= 8 and (
            any(wanted.startswith(stem + sep) or stem.startswith(wanted + sep)
                for sep in ("_", "-", "."))
        )
    ]
    if candidates:
        matched = sorted(path for stem in candidates for path in stems[stem])
        return matched, "prefix" if len(candidates) == 1 else "ambiguous"
    return [], "none"


class GitHubSource:
    """Public, quota-free reads from the IRW repositories on GitHub.

    Everything here is a raw file or a tree listing: no Redivis login and no
    export quota. Each resource is fetched once per process.
    """

    def __init__(self, fetch_text: Optional[Callable[[str], str]] = None) -> None:
        self._fetch = fetch_text or _http_get_text
        self._scripts: Optional[List[str]] = None
        self._scripts_truncated = False
        self._issues: Optional[Dict[str, List[str]]] = None
        self._overrides: Optional[Dict[str, List[Dict[str, str]]]] = None
        self._files: Dict[str, str] = {}
        self._commit: Optional[str] = None

    def commit(self) -> str:
        if self._commit is None:
            payload = json.loads(self._fetch(f"https://api.github.com/repos/{IRW_REPO}/commits/main"))
            sha = payload.get("sha", "")
            if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
                raise ValueError("GitHub did not return an immutable commit identifier")
            self._commit = sha
        return self._commit

    def source_url(self, path: str) -> str:
        return f"https://github.com/{IRW_REPO}/blob/{self.commit()}/{path}"

    def _raw_url(self, path: str) -> str:
        return f"https://raw.githubusercontent.com/{IRW_REPO}/{self.commit()}/{path}"

    def data_scripts(self) -> List[str]:
        if self._scripts is None:
            payload = json.loads(self._fetch(
                f"https://api.github.com/repos/{IRW_REPO}/git/trees/{self.commit()}?recursive=1"
            ))
            # GitHub silently caps a recursive tree and says so only in this
            # flag. A capped listing looks exactly like a repository with
            # fewer scripts in it, which would turn "no processing notes for
            # this table" into a false statement rather than an error.
            self._scripts_truncated = bool(payload.get("truncated"))
            self._scripts = sorted(
                entry["path"]
                for entry in payload.get("tree", [])
                if entry.get("type") == "blob"
                and str(entry.get("path", "")).startswith("data/")
            )
        return self._scripts

    @property
    def scripts_truncated(self) -> bool:
        return self._scripts_truncated

    def read_script(self, path: str) -> str:
        if path not in self._files:
            self._files[path] = self._fetch(self._raw_url(path))
        return self._files[path]

    def itemtext_issues(self) -> Dict[str, List[str]]:
        if self._issues is None:
            self._issues = _parse_issue_list(self._fetch(ITEMTEXT_ISSUES_QMD_URL))
        return self._issues

    def validator_overrides(self) -> Dict[str, List[Dict[str, str]]]:
        if self._overrides is None:
            import csv

            text = self._fetch(self._raw_url("processing_notes/validator_overrides.csv"))
            overrides: Dict[str, List[Dict[str, str]]] = {}
            for row in csv.DictReader(io.StringIO(text)):
                table = (row.get("table") or "").strip()
                if table:
                    overrides.setdefault(table.casefold(), []).append(dict(row))
            self._overrides = overrides
        return self._overrides


class IRWTools:
    """Synchronous tool implementations, separated from MCP registration."""

    def __init__(
        self,
        backend: Optional[IRWBackend] = None,
        source: Optional[GitHubSource] = None,
    ) -> None:
        self.backend = backend or PackageBackend()
        self.source = source or GitHubSource()
        self._capture_lock = _PACKAGE_CALL_LOCK
        self._irw_version: Optional[str] = None
        self._irw_released_at: Optional[str] = None
        self._irw_version_checked = False
        self._catalogue_cache: Optional[Dict[str, Dict[str, Any]]] = None
        self._filter_names: Optional[List[str]] = None
        self._filter_descriptions: Dict[str, Optional[str]] = {}

    def _catalogue(self) -> Dict[str, Dict[str, Any]]:
        """Quota-free per-table facts from the metadata table, by lowercase name.

        Used for pre-checks that must not cost a download: the size guard on
        fetch_table, the response-data licence on get_itemtext, and whether a
        table carries any tags at all.
        """
        if self._catalogue_cache is not None:
            return self._catalogue_cache
        frame, _ = self._call(self.backend.list_tables)
        catalogue: Dict[str, Dict[str, Any]] = {}
        if isinstance(frame, pd.DataFrame):
            for _, row in frame.iterrows():
                raw = row.to_dict()
                name = raw.get("name", raw.get("table"))
                if _is_missing(name):
                    continue
                key = str(name).casefold()
                if key in catalogue:
                    continue
                size = raw.get("n_responses")
                try:
                    number = float(size)
                    n_responses = (
                        int(number) if not isinstance(size, (bool, np.bool_))
                        and math.isfinite(number) and number >= 0
                        and number.is_integer() else None
                    )
                except (TypeError, ValueError, OverflowError):
                    n_responses = None
                licence = raw.get("Derived_License", raw.get("license"))
                variables = raw.get("variables")
                catalogue[key] = {
                    "name": str(name),
                    "n_responses": n_responses,
                    "variables": (
                        None
                        if _is_missing(variables)
                        else [
                            v
                            for v in re.split(r"[\s,;|]+", str(variables).strip())
                            if v
                        ]
                    ),
                    "license": None if _is_missing(licence) else str(licence),
                    "has_item_text": _as_bool(raw.get("has_item_text")),
                    "tagged": _is_tagged(raw),
                }
        self._catalogue_cache = catalogue
        return catalogue

    def _rights(self, table_name: str) -> Tuple[Dict[str, Any], List[str]]:
        """Rights that travel with item text: licence, the instrument rule, and
        the table's public notes from the item-text issues page."""
        notes_warnings: List[str] = []
        facts = self._catalogue().get(table_name.casefold()) or {}
        notes: List[str] = []
        try:
            notes = list(self.source.itemtext_issues().get(table_name.casefold(), []))
        except Exception as error:
            notes_warnings.append(
                "The public item-text notes could not be loaded "
                f"({type(error).__name__}); check {ITEMTEXT_ISSUES_PAGE} before "
                "using this text."
            )
        if notes:
            notes_warnings.append(
                f"This table has {len(notes)} public item-text note(s); read "
                "rights.public_notes before using the text."
            )
        rights = {
            "response_data_license": facts.get("license"),
            "response_data_license_field": "Derived License",
            "original_license": None,
            "original_license_note": ORIGINAL_LICENSE_NOTE,
            "instrument_rights": INSTRUMENT_RIGHTS_NOTE,
            "public_notes": notes,
            "public_notes_url": ITEMTEXT_ISSUES_PAGE,
        }
        return rights, notes_warnings

    def _call(
        self, callback: Callable[..., Any], *args: Any,
        requires_auth: bool = True, **kwargs: Any
    ) -> Tuple[Any, List[str]]:
        """Run package code without allowing stdout to corrupt MCP stdio."""
        callback_name = getattr(callback, "__name__", callback.__class__.__name__)
        ensure_ready = getattr(self.backend, "ensure_ready", None)
        with self._capture_lock:
            with redirect_stdout(io.StringIO()) as captured_stdout:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    try:
                        if requires_auth and ensure_ready is not None:
                            ensure_ready()
                        guard = getattr(self.backend, "noninteractive", None)
                        with guard() if requires_auth and guard else nullcontext():
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

    def _stamp(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Attach observed manifest provenance without claiming pinned reads.

        IRW is many independently versioned Redivis datasets; the manifest's
        ``irw_version`` is the only number that names the corpus as a whole.
        The manifest is fetched once per server process; when it cannot be
        loaded, results carry a warning rather than failing.
        """
        if not self._irw_version_checked:
            self._irw_version_checked = True
            stamp_fn = getattr(self.backend, "version_stamp", None)
            stamp = None
            if stamp_fn is not None:
                try:
                    stamp, _ = self._call(stamp_fn, requires_auth=False)
                except IRWMCPError:
                    stamp = None
            if stamp:
                number, released = stamp
                self._irw_version = str(number)
                self._irw_released_at = str(released)
        payload["irw_version"] = self._irw_version
        payload["irw_released_at"] = self._irw_released_at
        payload['retrieved_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
        payload['version_status'] = 'observed_manifest' if self._irw_version else 'unavailable'
        payload['data_pinned'] = False
        if self._irw_version is None:
            message = (
                "The IRW version manifest could not be loaded; this result is "
                "not associated with an observed IRW version."
            )
            warning_list = payload.setdefault("warnings", [])
            if message not in warning_list:
                warning_list.append(message)
        return _bound_payload(payload)

    def filter_names(self) -> List[str]:
        """The package's own filter list, cached per process.

        Read rather than restated: a hand-kept copy is a list that goes stale
        the first time `irw.filter()` gains an argument, and the failure mode
        is an assistant reporting that IRW cannot filter on something it can.
        """
        if self._filter_names is None:
            names, _ = self._call(self.backend.filter_names, requires_auth=False)
            self._filter_names = [str(name) for name in names]
        return self._filter_names

    def describe_filter(self, filter_name: str) -> Dict[str, Any]:
        """What one filter means and which values it actually takes."""
        filter_name = _validate_text(filter_name, "filter_name")
        known = self.filter_names()
        if filter_name not in known:
            raise IRWMCPError(
                "invalid_input",
                f"Unknown filter '{filter_name}'. IRW filters on: "
                f"{', '.join(known)}.",
            )
        details, package_warnings = self._call(
            self.backend.describe_filter, filter_name
        )
        state = _ConversionState()
        for message in package_warnings:
            state.add(message)
        # A None here is a filter the package can name but not enumerate --
        # today n_categories, has_item_text and collection on the shipped
        # package -- so fall back to its description rather than deny a
        # filter that irw.filter() accepts exists.
        if isinstance(details, Mapping):
            description = details.get("description")
            raw_values = details.get("values", details.get("available_values"))
        else:
            description, raw_values = details, None
        if _is_missing(description):
            description = self._filter_description(filter_name)
        normalised = _filter_values(raw_values, state)
        return self._stamp(
            {
                "source": SOURCE,
                "filter": filter_name,
                "description": None if _is_missing(description) else str(description),
                **normalised,
                "warnings": state.warnings,
            }
        )

    def _validate_filters(self, filters: Any) -> Dict[str, Any]:
        """Check filter names against the package before spending a call on them."""
        if filters is None:
            return {}
        if not isinstance(filters, Mapping):
            raise IRWMCPError(
                "invalid_input",
                "filters must be an object of filter name to value, for "
                'example {"construct_type": "Cognitive", "n_items": [10, 50]}.',
            )
        known = self.filter_names()
        cleaned: Dict[str, Any] = {}
        for key, value in filters.items():
            name = str(key)
            if name not in known:
                raise IRWMCPError(
                    "invalid_input",
                    f"Unknown filter '{name}'. IRW filters on: "
                    f"{', '.join(known)}. Call describe_filter for the values "
                    "one of them takes.",
                )
            if value is None:
                continue
            try:
                validate_filter_value(name, value)
            except InvalidFilterValue as error:
                raise IRWMCPError("invalid_input", str(error)) from None
            cleaned[name] = value
        return cleaned

    def search_tables(
        self,
        query: str = "",
        filters: Optional[Mapping[str, Any]] = None,
        limit: int = SEARCH_DEFAULT_LIMIT,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Free-text search over the catalogue, narrowed by `irw.filter()`.

        The filtering is the package's, not ours. An adapter that reimplements
        it drifts from it -- the first version of this server accepted five
        filters where the package has nineteen, applied them with its own
        matching rules, and dropped the coverage caveats the package already
        carries in FILTER_DESCRIPTIONS. Those caveats are the load-bearing
        part for an assistant: a filter that quietly restricts the search to
        the tagged subset turns "no match" into "no data", which is a
        confident wrong answer rather than a missing feature.
        """
        query = _validate_text(query, "query", allow_empty=True)
        if query and not re.search(r"\w", query, flags=re.UNICODE):
            raise IRWMCPError("invalid_input", "query must contain searchable letters or numbers.")
        selected = self._validate_filters(filters)
        limit = _validate_limit(limit, SEARCH_DEFAULT_LIMIT, SEARCH_MAX_LIMIT)
        offset = _validate_offset(offset)

        state = _ConversionState()
        allowed: Optional[set[str]] = None
        if selected:
            # irw.filter() defaults density to [0.5, 1] and warns when that
            # drops tables. A caller who asked for nothing about density did
            # not ask for sparse tables to disappear, so opt out unless they
            # named it -- and let the package's own warning through when they
            # did.
            call_filters = dict(selected)
            call_filters.setdefault("density", None)
            names, filter_warnings = self._call(
                self.backend.filter_tables, **call_filters
            )
            for message in filter_warnings:
                state.add(message)
            allowed = _name_set(names)
            if not allowed:
                return self._stamp(
                    {
                        "source": SOURCE,
                        "tables": [],
                        "total": 0,
                        "offset": offset,
                        "limit": limit,
                        "has_more": False,
                        "n_untagged_in_catalogue": None,
                        "filters_applied": selected,
                        "caveats": self._search_caveats(selected, None, None),
                        "warnings": state.warnings,
                    }
                )

        frame, package_warnings = self._call(self.backend.list_tables)
        if not isinstance(frame, pd.DataFrame):
            raise IRWMCPError(
                "serialization_error", "IRW catalogue was not returned as a table."
            )

        query_casefold = query.casefold()
        query_tokens = re.findall(r"\w+", query_casefold, flags=re.UNICODE)
        for message in package_warnings:
            state.add(message)
        matches: List[Tuple[int, str, Dict[str, Any]]] = []
        seen_names: set[str] = set()
        n_untagged = 0

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
            tagged = _is_tagged(raw)
            if not tagged:
                n_untagged += 1

            if allowed is not None and name_key not in allowed:
                continue

            # The card is lean but the search is not: the query still runs over
            # every metadata field, including the ones describe_table keeps.
            searchable = " ".join(
                _search_text(value) for value in raw.values()
            ).casefold()
            if query_tokens and not all(token in searchable for token in query_tokens):
                continue

            card = {
                field: record[field]
                for field in SEARCH_CARD_FIELDS
                if field in record
            }
            card["name"] = name_text
            card["tagged"] = tagged

            score = 0
            if query_casefold:
                if name_key == query_casefold:
                    score += 100000
                elif query_casefold in name_key:
                    score += 10000
                score += sum(100 if token in name_key else 10 for token in query_tokens)
            matches.append((score, name_key, card))

        matches.sort(key=lambda item: (-item[0], item[1]))
        total = len(matches)
        page = [card for _, _, card in matches[offset : offset + limit]]
        return self._stamp(
            {
                "source": SOURCE,
                "tables": page,
                "total": total,
                "offset": offset,
                "limit": limit,
                "has_more": offset + len(page) < total,
                "n_untagged_in_catalogue": n_untagged,
                "filters_applied": selected,
                "caveats": self._search_caveats(
                    selected, n_untagged, len(seen_names)
                ),
                "warnings": state.warnings,
            }
        )

    def _search_caveats(
        self,
        selected: Mapping[str, Any],
        n_untagged: Optional[int],
        n_total: Optional[int],
    ) -> List[str]:
        """Caveats for this search, with the package's own filter text attached.

        Each applied filter contributes the description the package already
        keeps for it, so a caveat added to FILTER_DESCRIPTIONS reaches the
        assistant without anyone editing this file.
        """
        caveats: List[str] = []
        if n_untagged is not None and n_total:
            caveats.append(
                f"Tags are human-added and incomplete: {n_untagged} of "
                f"{n_total} tables carry no tags at all (see `tagged` on each "
                "record). An untagged table is not a non-matching table, and "
                "any tag-based filter silently restricts you to the tagged "
                "subset."
            )
        else:
            caveats.append(
                "Tags are human-added and incomplete. An untagged table is "
                "not a non-matching table, and any tag-based filter silently "
                "restricts you to the tagged subset."
            )
        if selected:
            caveats.append(
                "No density filter was applied unless you asked for one; "
                "irw.filter() defaults to density=[0.5, 1], which would have "
                "dropped sparse tables silently."
            )
        for name in selected:
            description = self._filter_description(name)
            if description:
                caveats.append(f"{name}: {description}")
        caveats.append(
            "Each record is a summary card. Call describe_table for a table's "
            "full metadata, and get_processing_notes before recommending it."
        )
        return caveats

    def _filter_description(self, filter_name: str) -> Optional[str]:
        """One filter's description text, or None if it cannot be loaded.

        Deliberately not describe_filter(): that one computes each filter's
        available values from the metadata tables, so calling it here would
        put a Redivis download -- and the quota it spends -- in the path of
        building a tool description at server startup. The descriptions
        themselves are a plain dict in the package.
        """
        if not self._filter_descriptions:
            try:
                descriptions, _ = self._call(
                    self.backend.filter_descriptions, requires_auth=False
                )
            except IRWMCPError:
                descriptions = {}
            self._filter_descriptions = {
                str(key): (str(value).strip() or None)
                for key, value in (descriptions or {}).items()
            }
        return self._filter_descriptions.get(filter_name)

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
        return self._stamp(
            {
                "source": SOURCE,
                "table": table_name,
                "metadata": dict(metadata),
                "schema": schema,
                "warnings": state.warnings,
            }
        )

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

        # The page is bounded on the wire, not after the download: irw.fetch()
        # takes max_rows and columns and hands both to Redivis's read session,
        # so a page of a 107M-response table costs a page. Paging past the
        # window means asking for offset+limit rows and dropping the offset --
        # still bounded, and the only option Redivis's read API offers.
        #
        # dedup and wide are the exception. Each describes the whole table:
        # dedup can only drop the duplicates it can see, and long2resp
        # reshapes whatever rows it is handed. Capping either produces a
        # confident, wrong-shaped answer, so those keep the catalogue
        # pre-check instead.
        facts = self._catalogue().get(table_name.casefold())
        guard_warnings: List[str] = []

        # Column names now go to Redivis, which rejects an unknown one with a
        # wire error the caller cannot act on. The catalogue already carries
        # the variable list, so an obvious typo is caught here and named --
        # for nothing, and before the request. When the catalogue has no
        # variable list, the check is skipped rather than guessed at.
        known_columns = (facts or {}).get("variables")
        if columns and known_columns and not wide:
            known = {str(column).casefold() for column in known_columns if column}
            unknown = [
                column for column in columns if column.casefold() not in known
            ]
            if unknown:
                raise IRWMCPError(
                    "invalid_input",
                    f"Unknown column(s) for '{table_name}': "
                    f"{', '.join(unknown)}. This table has: "
                    f"{', '.join(str(c) for c in known_columns)}.",
                )

        needs_whole_table = wide or dedup
        if not needs_whole_table and offset + limit > FETCH_MAX_WINDOW:
            raise IRWMCPError(
                'request_too_large',
                f'offset + limit must not exceed {FETCH_MAX_WINDOW:,}; offset paging rereads preceding rows. '
                'Use the Python package for larger exports.',
            )
        max_rows: Optional[int] = None if needs_whole_table else offset + limit
        if needs_whole_table:
            reason = "wide=true" if wide else "dedup=true"
            if facts is None:
                raise IRWMCPError('table_size_unknown', f'Cannot verify table size for {reason}; use an ordinary bounded preview.')
            elif facts.get("n_responses") is None:
                raise IRWMCPError('table_size_unknown', f'No response count is available for {reason}; use an ordinary bounded preview.')
            elif facts["n_responses"] > FETCH_MAX_RESPONSES:
                raise IRWMCPError(
                    "table_too_large",
                    f"Table '{table_name}' has {facts['n_responses']:,} "
                    f"responses, above the {FETCH_MAX_RESPONSES:,} guard, and "
                    f"{reason} describes the whole table so it cannot be "
                    "bounded to a page. Retry with wide=false and dedup=false "
                    "for a bounded sample of the raw rows, or use "
                    "describe_table for its statistics.",
                )
            guard_warnings.append(
                f"{reason} is computed over the whole table, so this call "
                "downloaded all of it rather than one page."
            )

        frame, package_warnings = self._call(
            self.backend.fetch_table,
            table_name,
            wide=wide,
            dedup=dedup,
            max_rows=max_rows,
            columns=None if wide else columns,
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
            initial_warnings=guard_warnings + package_warnings,
        )
        if max_rows is not None:
            # `total_rows` from _page_dataframe counts the rows that arrived,
            # which under a wire-level cap is the cap. Reporting that as the
            # table's size would be a plain lie, and `has_more: false` at the
            # end of a full window would be a worse one.
            fetched = int(len(frame))
            payload["total_rows"] = None
            payload["has_more"] = fetched >= max_rows
            payload["truncated"] = payload["has_more"]
            estimate = (facts or {}).get("n_responses")
            payload["total_rows_estimate"] = estimate
            payload["total_rows_note"] = (
                "Only the requested window was downloaded, so the table's row "
                "count is not known from this call. total_rows_estimate is the "
                "catalogue's response count"
                + (
                    "; it counts all columns' responses, not the rows returned here."
                    if columns
                    else " for the whole table."
                )
            )
        payload.update(
            {"source": SOURCE, "table": table_name, "wide": wide, "dedup": dedup}
        )
        return self._stamp(payload)

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
        rights, rights_warnings = self._rights(table_name)
        if isinstance(value, str) or value is None:
            state = _ConversionState()
            for message in package_warnings + rights_warnings:
                state.add(message)
            state.add(ITEMTEXT_DISCLAIMER)
            facts = self._catalogue().get(table_name.casefold()) or {}
            if facts.get("has_item_text") is True:
                state.add(
                    "The IRW catalogue lists item text for this table but the "
                    "irw package could not fetch it. Treat that as a package or "
                    "shard fault, not as confirmation that no text exists; "
                    "report it at "
                    "https://github.com/itemresponsewarehouse/Python-pkg/issues."
                )
            return self._stamp(
                {
                    "source": SOURCE,
                    "table": table_name,
                    "available": False,
                    "availability_status": 'fetch_failed' if facts.get('has_item_text') is True else 'unavailable',
                    "rights": rights,
                    "items": [],
                    "columns": [],
                    "total_items": None if facts.get('has_item_text') is True else 0,
                    "offset": offset,
                    "limit": limit,
                    "returned": 0,
                    "has_more": False,
                    "truncated": False,
                    "disclaimer": ITEMTEXT_DISCLAIMER,
                    "warnings": state.warnings,
                }
            )

        payload = _page_dataframe(
            value,
            row_key="items",
            total_key="total_items",
            limit=limit,
            offset=offset,
            columns=None,
            initial_warnings=package_warnings + rights_warnings + [ITEMTEXT_DISCLAIMER],
        )
        payload.update(
            {
                "source": SOURCE,
                "table": table_name,
                "available": True,
                "availability_status": 'available',
                "rights": rights,
                "disclaimer": ITEMTEXT_DISCLAIMER,
            }
        )
        return self._stamp(payload)

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
        return self._stamp(
            {
                "source": SOURCE,
                "collections": page,
                "total": total,
                "offset": offset,
                "limit": limit,
                "has_more": offset + len(page) < total,
                "warnings": state.warnings,
            }
        )

    def get_citation(self, table_name: str) -> Dict[str, Any]:
        table_name = _validate_table_name(table_name)
        entries, package_warnings = self._call(self.backend.citation, table_name)
        state = _ConversionState()
        for message in package_warnings:
            state.add(message)
        if entries is None:
            entries = []
        elif isinstance(entries, str):
            entries = [entries]
        bibtex = [str(entry).strip() for entry in entries if str(entry).strip()]
        if not bibtex:
            state.add(
                f"No BibTeX entry could be resolved for '{table_name}'. Cite the "
                "IRW itself and check the table's bibliography metadata by hand."
            )
        return self._stamp(
            {
                "source": SOURCE,
                "table": table_name,
                "available": bool(bibtex),
                "bibtex": bibtex,
                "warnings": state.warnings,
            }
        )

    def get_processing_notes(self, table_name: str) -> Dict[str, Any]:
        table_name = _validate_table_name(table_name)
        state = _ConversionState()
        try:
            scripts = self.source.data_scripts()
        except Exception as error:
            raise IRWMCPError(
                "upstream_unavailable",
                "The IRW repository listing could not be loaded from GitHub "
                f"({type(error).__name__}). Retry later.",
                retryable=True,
            ) from None
        paths, match = _match_scripts(table_name, scripts)
        notes: List[Dict[str, Any]] = []
        if match == "ambiguous":
            state.add("Several processing scripts match this table family; no script was selected. Inspect candidate_paths.")
        for path in ([] if match == "ambiguous" else paths[:5]):
            try:
                text = self.source.read_script(path)
                header, truncated = _processing_header(path, text)
            except Exception as error:
                state.add(f"Could not read {path} ({type(error).__name__}).")
                continue
            notes.append(
                {
                    "path": path,
                    "url": self.source.source_url(path),
                    "header": header,
                    "truncated": truncated,
                }
            )
        if match == "none":
            state.add(
                f"No processing script named after '{table_name}' was found under "
                f"data/ in {IRW_REPO}. The table may have been processed under "
                "another name or outside the repository; nothing here says how "
                "its id, items or covariates were built."
            )
            if getattr(self.source, "scripts_truncated", False):
                state.add(
                    "GitHub truncated the repository listing, so the script "
                    "may exist and simply not be in the part that was "
                    "returned. Treat this as 'not looked up', not as 'not "
                    f"there': check {IRW_REPO_BLOB_URL}data/ directly."
                )
        elif match == "prefix":
            state.add(
                "Matched by name prefix, not exactly: the script may produce "
                "several tables. Confirm it mentions this one before relying on it."
            )
        overrides: List[Dict[str, str]] = []
        try:
            overrides = self.source.validator_overrides().get(table_name.casefold(), [])
        except Exception as error:
            state.add(
                f"Validator overrides could not be loaded ({type(error).__name__})."
            )
        return self._stamp(
            {
                "source": SOURCE,
                "table": table_name,
                "match": match,
                "candidate_paths": paths,
                "scripts": notes,
                "validator_overrides": overrides,
                "guides": {
                    "data_standard": IRW_REPO_BLOB_URL + "datastandard.md",
                    "processing_instructions": IRW_REPO_BLOB_URL
                    + "processing_notes/DataProcessingInstructions.md",
                },
                "warnings": state.warnings,
            }
        )


SERVER_INSTRUCTIONS = (
    "Read-only access to the Item Response Warehouse (IRW), a corpus of item "
    "response datasets in one long format (id, item, resp). Every response "
    "carries an observed irw_version when available, not a pinned data snapshot; the server's own "
    "version number is the irw Python package, not the data. Before "
    "recommending a table, call get_processing_notes: facts metadata cannot "
    "express (whether id links people across waves, what a cov_* column really "
    "means, which columns were excluded) live only there. Tags are incomplete, "
    "so an untagged table is not a non-match. Response direction is not "
    "harmonised across items, duplicate id-item rows can be real data, and item "
    "text carries no licence to reuse an instrument."
)


def _search_tables_doc(tools: "IRWTools") -> str:
    """Assemble search_tables' description from the package's filter list."""
    header = (
        "Search IRW tables by free text and by the irw package's own filters.\n"
        "\n"
        "`query` matches every metadata field, including ones the result card "
        "omits. `filters` is an object passed straight to irw.filter(), so the "
        "filter names and semantics are the package's -- for example "
        '{\"construct_type\": \"Cognitive\", \"n_items\": [10, 50]}. Numeric '
        "filters take a number for an exact match or [min, max] for a range "
        "(None for no bound); tag filters take a string or a list of strings, "
        "matched with OR.\n"
        "\n"
        "Results are a summary card each, paginated (default 20, maximum 100); "
        "call describe_table for a table's full metadata. Unlike irw.filter(), "
        "no default density filter is applied, so sparse tables are not "
        "silently dropped.\n"
        "\n"
        "Read `caveats` before trusting a filtered result. Tags are human-added "
        "and incomplete, so an untagged table is NOT a non-matching table and "
        "every tag-based filter restricts you to the tagged subset; each record "
        "carries `tagged` and the response carries `n_untagged_in_catalogue`.\n"
        "\n"
        "Matching is deterministic and case-insensitive. Two collection "
        "meanings that are easy to get wrong: `treat` means an experimental "
        "assignment is recorded, not merely that a treatment occurred, and "
        "`longitudinal` is derived by grepping the variable string, so "
        "cov_birthdate and cov_startdate match too -- confirm a real `wave` "
        "or `date` column before treating a table as a panel."
    )
    try:
        names = tools.filter_names()
    except Exception:  # a catalogue that will not load must not stop startup
        return header + (
            "\n\nCall describe_filter to list the available filters and the "
            "values each one takes."
        )
    lines = [header, "", "Available filters (describe_filter gives the values each takes):"]
    for name in names:
        description = tools._filter_description(name)
        lines.append(f"- {name}: {description}" if description else f"- {name}")
    return "\n".join(lines)


def create_server(
    backend: Optional[IRWBackend] = None,
    source: Optional[GitHubSource] = None,
) -> Any:
    """Create the MCP server; import the optional SDK only when requested."""
    try:
        from mcp.server import MCPServer
    except ImportError as error:
        raise RuntimeError(
            "The MCP extra is not installed. Use Python 3.10+ and run "
            "`python -m pip install 'irw[mcp]'`."
        ) from error

    tools = IRWTools(backend, source)
    server = MCPServer(
        name="IRW",
        version=str(getattr(irw, "__version__", "unknown")),
        description="Read-only access to Item Response Warehouse tables and metadata.",
        instructions=SERVER_INSTRUCTIONS,
    )

    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.types import ToolAnnotations
    from functools import wraps
    import inspect
    from typing import get_type_hints
    from .mcp_models import OUTPUT_MODELS
    from pydantic import ValidationError

    def typed_tool(**options: Any) -> Callable:
        """Preserve the result envelope while publishing concrete contracts."""
        def decorate(function: Callable) -> Callable:
            model = OUTPUT_MODELS[options["name"]]

            @wraps(function)
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                payload = function(*args, **kwargs)
                try:
                    normalized = model.model_validate({"result": payload}).model_dump(by_alias=True)["result"]
                    return {"result": _bound_payload(normalized)}
                except ValidationError:
                    raise ToolError(str(IRWMCPError(
                        "serialization_error", "IRW returned a result that violates the tool response schema."
                    ))) from None
                except IRWMCPError as error:
                    raise ToolError(str(error)) from None

            wrapped.__annotations__ = get_type_hints(function)
            wrapped.__annotations__["return"] = model
            signature = inspect.signature(function)
            wrapped.__signature__ = signature.replace(
                parameters=[parameter.replace(annotation=wrapped.__annotations__.get(name, parameter.annotation))
                            for name, parameter in signature.parameters.items()],
                return_annotation=model,
            )
            return server.tool(**options)(wrapped)
        return decorate

    def deliver(call: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        """Hand a structured error to the model instead of losing it.

        The SDK treats any exception other than its own ToolError as a crash:
        the client gets the bare text "Error executing tool <name>" and the
        code, message and retryable flag that IRWMCPError carries stay in the
        server log. Every refusal this server makes -- table_too_large, an
        unknown column, a missing credential -- was reaching the assistant
        as that one uninformative line. Raising ToolError with the JSON
        payload puts it in the is_error result the model reads.
        """
        try:
            return call()
        except IRWMCPError as error:
            raise ToolError(str(error)) from None

    read_only = ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
    )

    # The description is built from the package's own filter list rather than
    # written here, so a filter added to irw.filter() appears in this tool
    # without anyone remembering to edit this file. It has to be passed to the
    # decorator: the SDK reads the docstring at registration, so assigning
    # __doc__ afterwards registers an empty description and no test that only
    # calls the tool would notice.
    @typed_tool(
        name="search_tables",
        description=_search_tables_doc(tools),
        annotations=read_only,
        structured_output=True,
    )
    def search_tables(
        query: str = "",
        filters: Optional[Dict[str, Any]] = None,
        limit: int = SEARCH_DEFAULT_LIMIT,
        offset: int = 0,
    ) -> Dict[str, Any]:
        return deliver(lambda: tools.search_tables(query, filters, limit, offset))

    @typed_tool(name="describe_filter", annotations=read_only, structured_output=True)
    def describe_filter(filter_name: str) -> Dict[str, Any]:
        """Explain one search filter and list the values it actually takes.

        Call this before guessing a filter value: the tag vocabularies are
        closed sets with specific spellings ("Affective/mental health", not
        "mental health"), and a value that matches nothing returns an empty
        result that looks exactly like an absence of data.
        """
        return deliver(lambda: tools.describe_filter(filter_name))

    @typed_tool(name="describe_table", annotations=read_only, structured_output=True)
    def describe_table(table_name: str) -> Dict[str, Any]:
        """Return metadata and statistics for one IRW table without fetching rows."""
        return deliver(lambda: tools.describe_table(table_name))

    @typed_tool(name="fetch_table", annotations=read_only, structured_output=True)
    def fetch_table(
        table_name: str,
        limit: int = ROW_DEFAULT_LIMIT,
        offset: int = 0,
        columns: Optional[List[str]] = None,
        wide: bool = False,
        dedup: bool = False,
    ) -> Dict[str, Any]:
        """Fetch a bounded page of rows from one IRW table.

        The default page is 100 rows and the maximum is 1,000. The window is
        bounded on the wire to offset + limit, at most 10,000 rows. Later
        pages reread the prefix. `columns` selects output columns, including
        item columns after wide reshaping. Rows come back
        columnar: `columns` names the fields and each entry of `rows` is a
        list of values in that order.

        Because only the window is downloaded, `total_rows` is null and
        `has_more` says whether the window came back full;
        `total_rows_estimate` carries the catalogue's response count.
        A page is the table's first rows in storage order, not a random
        sample: use it to confirm the shape matches the metadata, not to
        estimate anything about the table. describe_table has the
        statistics.

        wide=true and dedup=true are the exception. Both describe the whole
        table, so they cannot be bounded to a page: the call downloads the
        table, warns that it did, and is refused above 1,000,000 responses
        (error table_too_large), or if size is unknown (table_size_unknown).
        Application JSON is capped at 256 KiB. Follow next_offset when a page
        is reduced; individual values are never shortened to fit.

        Response values keep the source's coding: higher resp is consistent
        within an item, but reverse-scored items are NOT recoded, so direction
        may vary across items. Duplicate id-item pairs can be real data
        (trials, waves, raters); they are kept unless dedup=true.
        """
        return deliver(lambda: tools.fetch_table(table_name, limit, offset, columns, wide, dedup))

    @typed_tool(name="get_itemtext", annotations=read_only, structured_output=True)
    def get_itemtext(
        table_name: str,
        limit: int = ROW_DEFAULT_LIMIT,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Return a bounded page of item-level text with its rights information.

        `rights` carries the response-data licence (IRW's Derived License,
        not the source deposit's Original License, which is not in the
        metadata this server reads and is reported as null), the
        instrument-rights rule, and the table's public notes from the
        item-text issues page (withdrawn wording, machine translations, known
        mismatches). The deposit licence is not an instrument licence: never
        reproduce a scale on its strength. Text is reconstructed with partial
        review; verify against the source.
        """
        return deliver(lambda: tools.get_itemtext(table_name, limit, offset))

    @typed_tool(name="list_collections", annotations=read_only, structured_output=True)
    def list_collections(
        limit: int = COLLECTION_DEFAULT_LIMIT,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List IRW's labelled collections and their metadata."""
        return deliver(lambda: tools.list_collections(limit, offset))

    @typed_tool(name="get_citation", annotations=read_only, structured_output=True)
    def get_citation(table_name: str) -> Dict[str, Any]:
        """Return BibTeX for the original data producers of one IRW table.

        Cite the original producers, not only the IRW, when using a table.
        """
        return deliver(lambda: tools.get_citation(table_name))

    @typed_tool(
        name="get_processing_notes", annotations=read_only, structured_output=True
    )
    def get_processing_notes(table_name: str) -> Dict[str, Any]:
        """Return the header notes of the script that built a table, from GitHub.

        No Redivis login or export quota. Read this before recommending a
        table: facts metadata cannot express live only here, such as whether
        `id` links people across waves or fell back to the row index, what a
        `cov_*` column really means, and which source columns were excluded.
        `match` says how the script was found: exact, prefix (a multi-table
        script), or none.
        """
        return deliver(lambda: tools.get_processing_notes(table_name))

    return server


def main() -> None:
    """Run the local stdio server for an MCP host."""
    # The redivis client draws tqdm progress bars during downloads. They go to
    # stderr, so they cannot corrupt the stdio protocol, but they flood the MCP
    # host's log with carriage-return spam on every fetch.
    os.environ.setdefault("TQDM_DISABLE", "1")
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
