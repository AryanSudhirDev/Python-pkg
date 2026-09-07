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
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
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
    "No Redivis credentials were found. The Redivis SDK would open an "
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

    def citation(self, table_name: str) -> List[str]: ...

    def version_stamp(self) -> Optional[Tuple[int, str]]: ...


class PackageBackend:
    """Default backend that delegates to the public ``irw`` API."""

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

    def fetch_table(self, table_name: str, *, wide: bool, dedup: bool) -> Any:
        return irw.fetch(table_name, source=SOURCE, wide=wide, dedup=dedup)

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
    if "quota" in message:
        return IRWMCPError(
            "quota_exceeded",
            "The Redivis export quota for this account is exhausted. Wait for "
            "the quota to reset before fetching more data.",
        )
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


def _http_get_text(url: str) -> str:
    """Fetch a public text resource. `requests` is a dependency of redivis."""
    import requests

    response = requests.get(url, timeout=30, headers={"User-Agent": "irw-mcp"})
    response.raise_for_status()
    return response.text


def _parse_issue_list(text: str) -> Dict[str, List[str]]:
    """Parse the per-table issue list embedded in itemtext_issues.qmd.

    The page is generated from a YAML list of ``- table: name`` / ``issue: |-``
    entries. That fixed shape is parsed directly rather than adding a YAML
    dependency to the package for one file.
    """
    issues: Dict[str, List[str]] = {}
    table: Optional[str] = None
    block: List[str] = []
    in_block = False

    def flush() -> None:
        if table and block:
            issues.setdefault(table.casefold(), []).append(
                " ".join(line.strip() for line in block).strip()
            )

    for line in text.splitlines():
        head = re.match(r"^- table:\s*(\S+)", line)
        if head:
            flush()
            table, block, in_block = head.group(1), [], False
            continue
        if table is None:
            continue
        if re.match(r"^\s{2}issue:\s*[|>]", line):
            in_block = True
            continue
        if in_block:
            if line.strip() == "" or line.startswith("    "):
                block.append(line)
            else:
                flush()
                table, block, in_block = None, [], False
    flush()
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


_SCRIPT_SUFFIX = re.compile(r"\.(py|r|do|ipynb|txt|rdata|xlsx)$", re.IGNORECASE)


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
        stem = _SCRIPT_SUFFIX.sub("", path.rsplit("/", 1)[-1]).casefold()
        stems.setdefault(stem, []).append(path)
    if wanted in stems:
        return sorted(stems[wanted]), "exact"
    candidates = [
        stem
        for stem in stems
        if len(stem) >= 8 and (wanted.startswith(stem) or stem.startswith(wanted))
    ]
    if candidates:
        best = max(candidates, key=len)
        return sorted(stems[best]), "prefix"
    return [], "none"


class GitHubSource:
    """Public, quota-free reads from the IRW repositories on GitHub.

    Everything here is a raw file or a tree listing: no Redivis login and no
    export quota. Each resource is fetched once per process.
    """

    def __init__(self, fetch_text: Optional[Callable[[str], str]] = None) -> None:
        self._fetch = fetch_text or _http_get_text
        self._scripts: Optional[List[str]] = None
        self._issues: Optional[Dict[str, List[str]]] = None
        self._overrides: Optional[Dict[str, List[Dict[str, str]]]] = None
        self._files: Dict[str, str] = {}

    def data_scripts(self) -> List[str]:
        if self._scripts is None:
            payload = json.loads(self._fetch(IRW_REPO_TREE_URL))
            self._scripts = sorted(
                entry["path"]
                for entry in payload.get("tree", [])
                if entry.get("type") == "blob"
                and str(entry.get("path", "")).startswith("data/")
            )
        return self._scripts

    def read_script(self, path: str) -> str:
        if path not in self._files:
            self._files[path] = self._fetch(IRW_REPO_RAW_URL + path)
        return self._files[path]

    def itemtext_issues(self) -> Dict[str, List[str]]:
        if self._issues is None:
            self._issues = _parse_issue_list(self._fetch(ITEMTEXT_ISSUES_QMD_URL))
        return self._issues

    def validator_overrides(self) -> Dict[str, List[Dict[str, str]]]:
        if self._overrides is None:
            import csv

            text = self._fetch(IRW_REPO_RAW_URL + "processing_notes/validator_overrides.csv")
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
        self._capture_lock = threading.Lock()
        self._irw_version: Optional[str] = None
        self._irw_released_at: Optional[str] = None
        self._irw_version_checked = False
        self._catalogue_cache: Optional[Dict[str, Dict[str, Any]]] = None

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
                    n_responses = None if _is_missing(size) else int(float(size))
                except (TypeError, ValueError):
                    n_responses = None
                licence = raw.get("Derived_License", raw.get("license"))
                catalogue[key] = {
                    "name": str(name),
                    "n_responses": n_responses,
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
            "instrument_rights": INSTRUMENT_RIGHTS_NOTE,
            "public_notes": notes,
            "public_notes_url": ITEMTEXT_ISSUES_PAGE,
        }
        return rights, notes_warnings

    def _call(
        self, callback: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Tuple[Any, List[str]]:
        """Run package code without allowing stdout to corrupt MCP stdio."""
        callback_name = getattr(callback, "__name__", callback.__class__.__name__)
        ensure_ready = getattr(self.backend, "ensure_ready", None)
        with self._capture_lock:
            with redirect_stdout(io.StringIO()) as captured_stdout:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    try:
                        if ensure_ready is not None:
                            ensure_ready()
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
        """Attach the IRW corpus version so any result can be pinned.

        IRW is many independently versioned Redivis datasets; the manifest's
        ``irw_version`` is the only number that names the corpus as a whole.
        The manifest is fetched once per server process; when it cannot be
        loaded, results ship unpinned with a warning rather than failing.
        """
        if not self._irw_version_checked:
            self._irw_version_checked = True
            stamp_fn = getattr(self.backend, "version_stamp", None)
            stamp = None
            if stamp_fn is not None:
                try:
                    stamp, _ = self._call(stamp_fn)
                except IRWMCPError:
                    stamp = None
            if stamp:
                number, released = stamp
                self._irw_version = str(number)
                self._irw_released_at = str(released)
        payload["irw_version"] = self._irw_version
        payload["irw_released_at"] = self._irw_released_at
        if self._irw_version is None:
            message = (
                "The IRW version manifest could not be loaded; this result is "
                "not pinned to an IRW version."
            )
            warning_list = payload.setdefault("warnings", [])
            if message not in warning_list:
                warning_list.append(message)
        return payload

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
            record["tagged"] = _is_tagged(raw)
            if not record["tagged"]:
                n_untagged += 1

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
        caveats = [
            f"Tags are human-added and incomplete: {n_untagged} of "
            f"{len(seen_names)} tables carry no tags at all (see `tagged` on "
            "each record). An untagged table is not a non-matching table."
        ]
        if collection is not None:
            caveats.append(
                "Filtering on a collection restricts results to tables whose "
                "tags or variables put them in it; untagged tables that "
                "measure the same thing are not returned."
            )
        if longitudinal is not None:
            caveats.append(
                "`longitudinal` is derived by grepping the variable string, so "
                "names like cov_birthdate and cov_startdate also match. Confirm "
                "an actual `wave` or `date` column in `variables`."
            )
        return self._stamp(
            {
                "source": SOURCE,
                "tables": page,
                "total": total,
                "offset": offset,
                "limit": limit,
                "has_more": offset + len(page) < total,
                "n_untagged_in_catalogue": n_untagged,
                "caveats": caveats,
                "warnings": state.warnings,
            }
        )

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

        # Size guard as a pre-check, not a truncation: irw.fetch() has no row
        # argument, so by the time rows could be counted the whole table has
        # already been exported against the account's quota. n_responses is in
        # the metadata table, which costs nothing to read.
        facts = self._catalogue().get(table_name.casefold())
        guard_warnings: List[str] = []
        if facts is None:
            guard_warnings.append(
                f"'{table_name}' is not in the IRW catalogue, so its size could "
                "not be checked before downloading."
            )
        elif facts.get("n_responses") is None:
            guard_warnings.append(
                f"The catalogue has no response count for '{table_name}', so its "
                "size could not be checked before downloading."
            )
        elif facts["n_responses"] > FETCH_MAX_RESPONSES:
            raise IRWMCPError(
                "table_too_large",
                f"Table '{table_name}' has {facts['n_responses']:,} responses, "
                f"above the {FETCH_MAX_RESPONSES:,} guard. Any fetch downloads "
                "the whole table against the account's 30-day Redivis export "
                "quota, so this server refuses it. Use describe_table for its "
                "statistics, or fetch it deliberately with the irw package "
                "outside this server.",
            )

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
            initial_warnings=guard_warnings + package_warnings,
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
                    "rights": rights,
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
        for path in paths[:5]:
            try:
                text = self.source.read_script(path)
            except Exception as error:
                state.add(f"Could not read {path} ({type(error).__name__}).")
                continue
            header, truncated = _script_header(text)
            notes.append(
                {
                    "path": path,
                    "url": IRW_REPO_BLOB_URL + path,
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
    "carries irw_version, the citable version of the data; the server's own "
    "version number is the irw Python package, not the data. Before "
    "recommending a table, call get_processing_notes: facts metadata cannot "
    "express (whether id links people across waves, what a cov_* column really "
    "means, which columns were excluded) live only there. Tags are incomplete, "
    "so an untagged table is not a non-match. Response direction is not "
    "harmonised across items, duplicate id-item rows can be real data, and item "
    "text carries no licence to reuse an instrument."
)


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
        """Search IRW table names and metadata; deterministic and case-insensitive.

        Results are paginated with a default limit of 20 and a maximum of 100.
        Filters: collection, variable, license, longitudinal, has_item_text.
        Read `caveats` before trusting a filtered result:
        - Tags are human-added and incomplete (thinner among recent tables). An
          untagged table is NOT a non-matching table; each record carries
          `tagged`, and collection or tag filters silently restrict you to the
          tagged subset.
        - `longitudinal` is derived by grepping the variable string, so
          cov_birthdate and cov_startdate match too; confirm a real `wave` or
          `date` column in `variables`.
        - `treat` (the rct collection) means an experimental assignment is
          recorded, not merely that a treatment occurred.
        - Unlike irw.filter(), no default density filter is applied, so sparse
          tables appear; check `density` before per-person conclusions.
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

        The default page is 100 rows and the maximum is 1,000; use offset for
        later pages and columns to shrink the response. Tables above 1,000,000
        responses are refused before any download (error table_too_large):
        irw.fetch() has no row argument, so a fetch exports the whole table
        against the account's 30-day Redivis export quota, and paging happens
        locally afterwards. Response values keep the source's coding: higher
        resp is consistent within an item, but reverse-scored items are NOT
        recoded, so direction may vary across items. Duplicate id-item pairs
        can be real data (trials, waves, raters); they are kept unless
        dedup=true.
        """
        return tools.fetch_table(table_name, limit, offset, columns, wide, dedup)

    @server.tool(name="get_itemtext", annotations=read_only, structured_output=True)
    def get_itemtext(
        table_name: str,
        limit: int = ROW_DEFAULT_LIMIT,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Return a bounded page of item-level text with its rights information.

        `rights` carries the response-data licence, the instrument-rights rule
        and the table's public notes from the item-text issues page (withdrawn
        wording, machine translations, known mismatches). The deposit licence
        is not an instrument licence: never reproduce a scale on its strength.
        Text is reconstructed with partial review; verify against the source.
        """
        return tools.get_itemtext(table_name, limit, offset)

    @server.tool(name="list_collections", annotations=read_only, structured_output=True)
    def list_collections(
        limit: int = COLLECTION_DEFAULT_LIMIT,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List IRW's labelled collections and their metadata."""
        return tools.list_collections(limit, offset)

    @server.tool(name="get_citation", annotations=read_only, structured_output=True)
    def get_citation(table_name: str) -> Dict[str, Any]:
        """Return BibTeX for the original data producers of one IRW table.

        Cite the original producers, not only the IRW, when using a table.
        """
        return tools.get_citation(table_name)

    @server.tool(
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
        return tools.get_processing_notes(table_name)

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
