"""Internal Redivis table fetching utilities."""

# Required for the `Exception | None` annotations below: PEP 604 unions are
# 3.10+, and the package supports 3.9 (see pyproject requires-python). Without
# this, `import irw` raises TypeError on 3.9.
from __future__ import annotations

import re
import time
from typing import Any, Callable, List, Optional, Tuple, TypeVar
import warnings

T = TypeVar("T")

_TRANSIENT_ERROR_MARKERS = (
    "timeout",
    "temporar",
    "connection",
    "server error",
    "502",
    "503",
    "expected to be able to read",
    "message body",
    "incomplete read",
    "connection reset",
    "broken pipe",
    "chunkedencodingerror",
    "read timed out",
    "remotedisconnected",
    "protocolerror",
)


def _get_table(ds: Any, name: str) -> Any:
    """Get a table handle and ensure properties are loaded.

    A missing table must surface as not-found here rather than being
    swallowed: `_search_datasets` relies on that signal to move on to the
    next shard. With it swallowed, `ds.table(name)` hands back a phantom
    handle and the *first* shard searched claims every name -- which is how
    item text for the 732 tables in `irw_text` came back "not available"
    once the 15-table `irw_text_2` shard was put ahead of it. Other failures
    (a transient property load) keep the lazy handle, as before; the fetch
    that follows will raise properly if the table really is unreachable.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*No reference id was provided for the table.*")
        tbl = ds.table(name)
        try:
            tbl.get()  # populate properties if needed
        except Exception as e:
            if _classify_error(e) == "not_found":
                raise
        return tbl


# Redivis caps exported bytes per rolling window and reports the cap as a 400
# invalid_request, so quota has to be recognised BEFORE invalid_request or a
# user who hit the 30-day export limit is told their table is malformed. Rate
# limiting and resource exhaustion are grouped here because the user response
# is the same: wait, or run the query server-side. Ported from
# Rpkg/R/redivis-errors.R:22-28.
_QUOTA_ERROR_PATTERN = re.compile(
    r"cannot export more than"
    r"|export(?:ed)? .{0,40}(?:within|in the past)"
    r"|quota"
    r"|rate.?limit"
    r"|too many requests"
    r"|resource_exhausted"
    r"|\b429\b",
    re.IGNORECASE,
)

# Ported from Rpkg/R/redivis-errors.R:49-57. Previously unclassified in Python,
# so an expired login surfaced as a raw exception.
_AUTH_ERROR_PATTERN = re.compile(
    r"unauthorized|unauthenticated|not authenticated|authentication"
    r"|permission_denied|access_denied|forbidden|invalid_grant|invalid_token"
    r"|login required|not authorized|must be logged in|\b401\b",
    re.IGNORECASE,
)

# R's arm is `not_found_error|not\s*found`, which does not match the bare
# `not_found` code Redivis returns -- \s does not match an underscore. The
# character class covers both spellings.
_NOT_FOUND_PATTERN = re.compile(r"not[_\s]*found", re.IGNORECASE)

# Internal dataset refs leak into Redivis error text; they are noise to a user
# who never typed them. Ported from Rpkg/R/redivis-errors.R:79, with the nom
# warehouse added -- R's arm predates irw_nominal being reachable from Python.
_DATASET_REF_PATTERN = re.compile(
    r"(?:item_response_warehouse(?:_\d+)?|irw_nominal|irw_simsyn|irw_competitions"
    r"|irw_text(?:_\d+)?|irw_meta):[a-z0-9]+",
    re.IGNORECASE,
)


def _error_text(err: Exception) -> str:
    """The searchable text of an error: its Redivis payload plus its message.

    Redivis raises with a dict as args[0]; the machine-readable code lives in
    its ``error`` key and the human explanation in ``error_description``. Both
    matter -- quota, for instance, arrives with code ``invalid_request`` and is
    identifiable only from the description.
    """
    parts = [str(err)]
    try:
        payload = err.args[0]
        if isinstance(payload, dict):
            parts.extend(
                str(payload.get(k, "")) for k in ("error", "error_description", "message")
            )
    except Exception:
        pass
    return " ".join(p for p in parts if p)


def _sanitize_error(msg: Optional[str]) -> str:
    """Strip internal Redivis dataset paths out of an error message."""
    if not msg:
        return ""
    msg = _DATASET_REF_PATTERN.sub("IRW", msg)
    return re.sub(r"\s+", " ", msg).strip()


def _classify_error(err: Exception) -> str:
    """Classify Redivis-style errors for control flow.

    Returns one of ``quota``, ``invalid_request``, ``not_found``, ``auth``,
    ``transient`` or ``unknown``. The order of the checks is load-bearing and
    matches Rpkg/R/redivis-errors.R:37-61: quota is tested first because it
    arrives wearing an ``invalid_request`` code.
    """
    text = _error_text(err)

    if _QUOTA_ERROR_PATTERN.search(text):
        return "quota"

    payload = None
    try:
        if isinstance(err.args[0], dict):
            payload = err.args[0]
    except Exception:
        pass
    if payload:
        code = str(payload.get("error", "")).lower()
        if "invalid_request" in code:
            return "invalid_request"
        if "not_found" in code:
            return "not_found"

    if "invalid_request" in text.lower():
        return "invalid_request"
    if _NOT_FOUND_PATTERN.search(text):
        return "not_found"
    if _AUTH_ERROR_PATTERN.search(text):
        return "auth"
    if any(marker in text.lower() for marker in _TRANSIENT_ERROR_MARKERS):
        return "transient"
    return "unknown"


# Errors that are terminal for the whole search rather than for one datasource:
# there is no point trying the next warehouse when the account is out of export
# quota, the login has expired, or this specific table is malformed.
_TERMINAL_ERROR_KINDS = ("quota", "auth", "invalid_request")


def _terminal_error_message(err: Exception, table_name: str) -> str:
    """User-facing text for a terminal error, naming the actual condition.

    Every one of these used to be reported as "invalid format", including
    export-quota exhaustion -- so a user who had hit the 30-day cap went
    looking for a problem in the warehouse that does not exist. Text mirrors
    Rpkg/R/redivis-errors.R:97-143 so the two clients say the same thing, with
    one deliberate divergence: R points at `irw_table_sets()` as the
    quota-free alternative and Python has no such function yet, so the Python
    text points at a server-side query instead.
    """
    kind = _classify_error(err)
    detail = _sanitize_error(_format_error(err))

    if kind == "quota":
        return (
            "\nRedivis export quota or rate limit reached, so the table could not be "
            "downloaded.\n"
            "This is an account-wide limit on exported bytes, not a problem with the "
            "table.\n"
            "It resets as the rolling window rolls over; to get results sooner, run a "
            "server-side\n"
            "query in the Redivis web interface, which does not count against the "
            "export quota.\n"
            f"Underlying error: {detail}"
        )
    if kind == "auth":
        return (
            "\nRedivis authentication failed. Sign in via the browser when prompted, "
            "or see the README troubleshooting section.\n"
            f"Underlying error: {detail}"
        )
    return f"\nTable '{table_name}' cannot be fetched due to an invalid format."


def _retry_transient(
    callback: Callable[[], T],
    *,
    max_attempts: int = 3,
    base_delay: float = 1.0,
) -> T:
    """Retry callback on transient Redivis/network read failures."""
    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return callback()
        except Exception as e:
            last_err = e
            if _classify_error(e) != "transient" or attempt == max_attempts - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))
    if last_err is not None:
        raise last_err
    raise RuntimeError("retry loop exited without result")


def _format_error(err: Exception | None) -> str:
    """Pretty-print Redivis JSON error payloads when available."""
    if err is None:
        return "not found"
    try:
        payload = err.args[0]
        if isinstance(payload, dict):
            body = {k: payload.get(k) for k in ("status", "error", "error_description") if k in payload}
            if body:
                return str(body)
        return str(payload)
    except Exception:
        return str(err)


def _search_datasets(
    datasets: List[Any],
    callback: Callable[[Any], T],
) -> Tuple[Optional[T], Optional[Exception], Optional[Exception]]:
    """
    Try callback(ds) for each Redivis dataset, in order, until one succeeds.

    Returns (result, last_other_error, terminal_error).
    On success, result is set and both errors are None.
    When every dataset reports not-found, all three return values are None.

    A terminal error stops the search: quota and auth apply to every
    datasource, and an invalid format is a property of the table itself. Pass
    it to _terminal_error_message() for the text to show the user -- the class
    matters, since quota arrives looking like an invalid request.
    """
    last_other: Exception | None = None
    for ds in datasets:
        try:
            return callback(ds), None, None
        except Exception as e:
            kind = _classify_error(e)
            if kind in _TERMINAL_ERROR_KINDS:
                return None, None, e
            if kind == "not_found":
                continue
            last_other = e
    return None, last_other, None
