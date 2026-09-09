"""Fetch operations for IRW data.

This module contains all functionality related to fetching data from IRW datasets.
"""

from typing import Iterable, Union, Dict, Optional, Any, List, Tuple
import warnings
import logging
import numpy as np
import pandas as pd

from ..utils.redivis.tables import (
    _get_table,
    _classify_error,
    _format_error,
    _search_datasets,
    _retry_transient,
    _terminal_error_message,
)

logger = logging.getLogger(__name__)


def fetch(
    datasets: List[Any],
    name: Union[str, Iterable[str], pd.Series],
    *,
    dedup: bool = False,
    max_rows: Optional[int] = None,
    columns: Optional[Iterable[str]] = None,
) -> Union[pd.DataFrame, Dict[str, Optional[pd.DataFrame]], None]:
    """
    Fetch one or more IRW tables.
    
    Parameters
    ----------
    datasets : List[Any]
        List of Redivis dataset objects to search through.
    name : str | Iterable[str] | pandas.Series
        Single table id → returns DataFrame or None.
        Multiple (list, Series, etc.) → returns dict[name -> DataFrame | None].
        Can also pass a pandas Series (e.g., from filter() method).
    dedup : bool, default False
        After fetch, keep one row per (id,item[,wave]).
    max_rows : int, optional
        Stop after this many rows. Pushed to Redivis, so only these rows are
        exported and billed against the account's quota.
    columns : iterable of str, optional
        Restrict to these columns. Also pushed to Redivis.
    
    Returns
    -------
    pd.DataFrame | dict[str, pd.DataFrame | None] | None
        Single table name returns DataFrame or None.
        Multiple table names return dict mapping names to DataFrame or None.
    
    Raises
    ------
    ValueError
        If name is empty.
    TypeError
        If name is not a string, iterable of strings, or pandas Series.
    
    Examples
    --------
    >>> import irw
    >>> from irw.operations.fetch import fetch
    >>> from irw.utils.redivis import _init_main_datasets
    >>> 
    >>> # Fetch a single table
    >>> datasets = _init_main_datasets()
    >>> df = fetch(datasets, "agn_kay_2025")
    >>> 
    >>> # Fetch multiple tables
    >>> tables = fetch(datasets, ["agn_kay_2025", "pks_probability"])
    >>> 
    >>> # Fetch from filter results (pandas Series)
    >>> filtered = irw.filter(construct_type="Affective/mental health")
    >>> tables = fetch(datasets, filtered)
    >>> 
    >>> # Fetch with deduplication
    >>> df_dedup = fetch(datasets, "agn_kay_2025", dedup=True)
    """
    # Handle pandas Series
    if isinstance(name, pd.Series):
        if name.empty:
            raise ValueError("name Series cannot be empty")
        name = name.tolist()
    
    # Check if empty (after Series conversion)
    if isinstance(name, str):
        if not name:
            raise ValueError("name cannot be empty")
        return _fetch_one_table(
            datasets, name, dedup=dedup, max_rows=max_rows, columns=columns
        )
    
    # Validate iterable
    if not name:
        raise ValueError("name cannot be empty")
    
    try:
        name_list = list(name)
        if not name_list:
            raise ValueError("name iterable cannot be empty")
    except TypeError:
        raise TypeError("name must be a string, iterable of strings, or pandas Series")

    out: Dict[str, Optional[pd.DataFrame]] = {}
    for nm in name_list:
        if not isinstance(nm, str):
            raise TypeError(f"All names must be strings, got {type(nm)}")
        key = str(nm)
        out[key] = _fetch_one_table(
            datasets, key, dedup=dedup, max_rows=max_rows, columns=columns
        )
    return out


def _validate_pushdown(
    max_rows: Optional[int],
    columns: Optional[Iterable[str]],
) -> Tuple[Optional[int], Optional[List[str]]]:
    """Check the pushdown arguments and normalise `columns` to a list.

    Both are handed straight to Redivis, which is the whole point: a row cap
    applied after the download has already cost the account's export quota,
    and IRW's response counts are skewed enough (a median table of ~8k
    responses against a p99 of ~44M) that the difference is invisible in
    testing and only shows up on the tables people most want to look at.
    """
    if max_rows is not None:
        # bool is an int subclass, and fetch(..., max_rows=True) is a mistake.
        if isinstance(max_rows, bool) or not isinstance(max_rows, int):
            raise TypeError(f"max_rows must be an int or None; got {type(max_rows)}")
        if max_rows < 1:
            raise ValueError(f"max_rows must be at least 1; got {max_rows}")

    column_list: Optional[List[str]] = None
    if columns is not None:
        if isinstance(columns, str):
            raise TypeError(
                "columns must be an iterable of column names, not a single "
                f"string; pass ['{columns}'] to select one column."
            )
        column_list = list(columns)
        if not column_list:
            raise ValueError("columns cannot be empty; pass None for all columns.")
        bad = [c for c in column_list if not isinstance(c, str) or not c.strip()]
        if bad:
            raise TypeError(f"All column names must be non-empty strings; got {bad!r}")

    return max_rows, column_list


def _fetch_one_table(
    datasets: List[Any],
    name: str,
    *,
    dedup: bool,
    max_rows: Optional[int] = None,
    columns: Optional[Iterable[str]] = None,
) -> Optional[pd.DataFrame]:
    """
    Internal: fetch a single table using the provided datasets.
    """
    if not isinstance(name, str) or name == "":
        # Not a Redivis condition at all -- this is a bad argument. It shared
        # the "invalid format" wording with the Redivis path, which made the
        # two indistinguishable in a log.
        logger.warning(
            f"Table name must be a non-empty string; got {name!r}. Nothing was fetched."
        )
        return None

    max_rows, column_list = _validate_pushdown(max_rows, columns)

    # A row cap and dedup describe different tables. Dedup drops duplicate
    # (id, item) rows, and it can only drop the ones it can see, so on a capped
    # fetch it dedups a prefix and reports success -- the caller has no way to
    # tell that from a dedup of the whole table. Say so rather than refuse:
    # a capped peek at a table with known duplicates is still a reasonable
    # thing to ask for.
    if max_rows is not None and dedup:
        warnings.warn(
            f"In dataset '{name}': dedup=True was applied to the first "
            f"{max_rows} rows only, not to the whole table. Duplicate "
            "(id, item) rows outside that window were not seen.",
            UserWarning,
            stacklevel=2,
        )

    def _load_table(ds: Any) -> pd.DataFrame:
        tbl = _get_table(ds, name)
        df = _retry_transient(
            lambda: tbl.to_pandas_dataframe(
                max_rows, variables=None if dedup else column_list
            )
        )

        # --- inline transforms needed only for fetch() ---

        # Coerce 'resp' to numeric (exact R-style warning text)
        if "resp" in df.columns:
            s = df["resp"]

            def missing_like(x) -> bool:
                if x is None:
                    return True
                if isinstance(x, float) and pd.isna(x):
                    return True
                if isinstance(x, str) and x.strip().upper() in {"", "NA", "NAN", "NULL"}:
                    return True
                return False

            if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
                coerced = pd.to_numeric(s.where(s.map(missing_like), s), errors="coerce")
                bad = (~s.map(missing_like)) & coerced.isna()
                if bad.any():
                    warnings.warn(
                        f"In dataset '{name}': 'resp' column contained non-numeric values that could not be coerced. Some NAs were introduced.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                df = df.copy()
                df["resp"] = coerced

        # Dedup without pandas GroupBy.apply warning
        if dedup:
            if "date" in df.columns:
                warnings.warn(
                    f"Deduplication skipped for dataset '{name}': 'date' column detected (timestamped responses).",
                    UserWarning, stacklevel=2,
                )
            elif {"id", "item"}.issubset(df.columns):
                keys = ["id", "item"] + (["wave"] if "wave" in df.columns else [])
                msg = (
                    f"Deduplicated dataset '{name}': one response randomly retained per (id, item, wave) group."
                    if "wave" in df.columns
                    else f"Deduplicated dataset '{name}': one response randomly retained per (id, item) pair."
                )
                n0 = len(df)
                rng = np.random.RandomState(42)
                groups = df.groupby(keys, dropna=False).indices  # dict[group_key -> ndarray of row indices]
                if groups:
                    chosen_idx = [rng.choice(ix) for ix in groups.values()]
                    df = df.iloc[sorted(chosen_idx)].reset_index(drop=True)
                    if len(df) < n0:
                        logger.info(msg)
                    else:
                        logger.info(f"Deduplication not needed for dataset '{name}': no duplicate responses found.")
                else:
                    logger.info(f"Deduplication not needed for dataset '{name}': no duplicate responses found.")
            else:
                warnings.warn(
                    f"Deduplication skipped for dataset '{name}': missing id or item column.",
                    UserWarning, stacklevel=2,
                )

        if dedup and column_list is not None:
            df = df.loc[:, column_list]

        return pd.DataFrame(df)

    result, last_other, terminal = _search_datasets(datasets, _load_table)
    if terminal is not None:
        # Names the actual condition -- quota exhaustion used to be reported
        # here as an invalid format. warn-and-return-None is kept deliberately:
        # R stops(), Python's fetch path has always returned None.
        message = _terminal_error_message(terminal, name)
        logger.warning(message)
        warnings.warn(message, UserWarning, stacklevel=2)
        return None
    if result is not None:
        return result

    if last_other is not None:
        logger.error(f"Error fetching dataset '{name}' : {_format_error(last_other)}")
        if _classify_error(last_other) == "transient":
            warnings.warn(
                f"Transient error fetching '{name}' from Redivis after retries. "
                "Try running the fetch again in a fresh session.",
                UserWarning,
                stacklevel=2,
            )
    else:
        msg = f"Table '{name}' does not exist in the IRW database."
        warnings.warn(msg, UserWarning, stacklevel=2)
        logger.warning(msg)
    return None
