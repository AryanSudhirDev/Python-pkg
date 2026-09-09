"""
Simplified IRW API - all functions at module level.

Usage:
    import irw
    
    # Database operations
    irw.list_tables()
    irw.filter(construct_type="Affective/mental health")
    irw.info()  # Database info
    irw.info("agn_kay_2025")  # Table info
    
    # Table operations
    irw.fetch("agn_kay_2025")
    irw.itemtext("agn_kay_2025")
    irw.save_bibtex("agn_kay_2025")
    irw.download("agn_kay_2025")
    irw.long2resp(df)  # df is a DataFrame from fetch()
"""

from __future__ import annotations
import warnings
import datetime
from typing import Optional, Union, Dict, List, Literal
import pandas as pd
from .utils.redivis import _init_main_datasets, _init_sim_dataset, _init_comp_dataset, _init_nom_dataset
from .utils.redivis.item_text import _list_itemtext_tables, _itemtext_disclaimer
from .utils.long2resp import long2resp as _long2resp
from .operations.fetch import fetch as _fetch
from .operations.list_tables import list_tables as _list_tables, list_tables_basic
from .operations.info import info_for
from .operations.filter import filter_tables
from .operations.filter_info import get_filters as _get_filters_func, describe_filter as _describe_filter
from .operations.version import version as _version
from .utils.redivis.table_metadata import (
    get_collections_table as _get_collections_table,
    get_collection_members_table as _get_collection_members_table,
)
from .utils.table_helpers import (
    _get_table_info_dict,
    _get_table_itemtext,
    _format_table_info
)


def _get_datasets(source: str = "main"):
    """Get datasets for the given source."""
    if source == "main":
        return _init_main_datasets()
    elif source == "sim":
        return [_init_sim_dataset()]
    elif source == "comp":
        return [_init_comp_dataset()]
    elif source == "nom":
        return [_init_nom_dataset()]
    else:
        raise ValueError(f"Unknown source '{source}'. Must be one of: 'main', 'sim', 'comp', 'nom'")


def list_tables(source: str = "main", include_metadata: bool = False) -> pd.DataFrame:
    """
    List available IRW tables.
    
    Parameters
    ----------
    source : str, default "main"
        Dataset source to use. Options: "main", "sim", "comp", "nom"
    include_metadata : bool, default False
        If True and source="main", return enriched IRW metadata.
        If False, return basic Redivis properties.
        
    Returns
    -------
    pandas.DataFrame
        Table listing with metadata.
    """
    datasets = _get_datasets(source)
    if source == "main" and include_metadata:
        return _list_tables(datasets)
    else:
        return list_tables_basic(datasets)


def filter(**kwargs) -> pd.Series:
    """
    Filter IRW tables based on metadata criteria.
    
    Only works for main IRW datasets.
    
    Parameters
    ----------
    **kwargs
        Filter parameters (n_responses, construct_type, etc.)
        
    Returns
    -------
    pandas.Series
        Sorted Series of table names that match filters.
    """
    datasets = _get_datasets("main")
    return filter_tables(datasets, **kwargs)


def info(table_name: Optional[str] = None, source: str = "main", return_dict: bool = False) -> Union[None, Dict]:
    """
    Get information about the IRW database or a specific table.
    
    Parameters
    ----------
    table_name : str, optional
        If provided, returns info for that table. If None, returns database info.
    source : str, default "main"
        Dataset source (only used for database info).
    return_dict : bool, default False
        If True and table_name provided, returns dictionary instead of formatted string.
        
    Returns
    -------
    None, dict, or str
        Database info prints and returns None.
        Table info prints and returns None (unless return_dict=True, then returns dict).
    """
    if table_name is None:
        # Database info
        datasets = _get_datasets(source)
        title = {
            "main": "IRW Database Information",
            "sim": "IRW Simulation Database Information",
            "comp": "IRW Competition Database Information",
            "nom": "IRW Nominal Response Database Information"
        }.get(source, "IRW Database Information")
        return info_for(datasets, title=title)
    else:
        # Table info
        if source != "main":
            raise ValueError(f"Table info is only available for main IRW datasets (source='main'). Current source is '{source}'.")
        
        result = _get_table_info_dict(table_name)
        
        if not result:
            message = f"No metadata available for table: {table_name}"
            print(message)
            return None if not return_dict else {}
        
        formatted_message = _format_table_info(table_name, result)
        print(formatted_message)
        
        if return_dict:
            return result
        else:
            return None  # Already printed, don't return string


def fetch(
    table_name: Union[str, List[str]], 
    source: str = "main", 
    *, 
    dedup: bool = False,
    wide: bool = False,
    max_rows: Optional[int] = None,
    columns: Optional[List[str]] = None,
) -> Union[pd.DataFrame, Dict[str, Optional[pd.DataFrame]]]:
    """
    Fetch one or more IRW tables.
    
    Parameters
    ----------
    table_name : str or list of str
        Single table name or list of table names.
    source : str, default "main"
        Dataset source to use. Options: "main", "sim", "comp", "nom".
    dedup : bool, default False
        Apply deduplication to responses.
    wide : bool, default False
        If True, automatically convert to wide-format response matrix using long2resp().
        Only works for single table fetch. For multiple tables, use long2resp() separately.
    max_rows : int, optional
        Return at most this many rows per table. The cap is pushed to Redivis,
        so only these rows are exported and charged against the account's
        export quota -- the reason to prefer it over slicing the result.
        These are the table's first ``max_rows`` rows in storage order, not a
        random sample, so they are not representative of the table.
    columns : list of str, optional
        Return only these output columns, in the requested order. Pushed to
        Redivis for ordinary reads; applied after deduplication or reshaping
        when those operations need additional input columns.
        
    Returns
    -------
    pandas.DataFrame or dict[str, pandas.DataFrame]
        Single table returns DataFrame, multiple return dict.
        If wide=True, returns wide-format response matrix instead of long format.
    """
    from .operations.fetch import _validate_pushdown

    max_rows, columns = _validate_pushdown(max_rows, columns)
    datasets = _get_datasets(source)
    if wide and max_rows is not None:
        # long2resp infers the response matrix from the rows it is given, so a
        # capped fetch yields a matrix of whoever happened to be in the first
        # max_rows rows -- usually a handful of ids with full response vectors
        # and nothing else. Useful as a shape check, misleading as data.
        warnings.warn(
            f"wide=True with max_rows={max_rows} reshapes only the first "
            f"{max_rows} rows, so the response matrix covers whichever ids "
            "appear in them rather than the table's respondents.",
            UserWarning,
            stacklevel=2,
        )
    result = _fetch(
        datasets, table_name, dedup=dedup, max_rows=max_rows,
        columns=None if wide else columns
    )
    
    # Handle single DataFrame result
    if isinstance(result, pd.DataFrame):
        # If wide=True, convert to response matrix
        if wide:
            result = _long2resp(result, wave=None, id_density_threshold=0.1, agg_method="mean")
            return result if columns is None else result.loc[:, columns]
        return result
    
    # Handle dict of DataFrames
    elif isinstance(result, dict):
        out = {}
        for k, v in result.items():
            if v is not None:
                # If wide=True, convert each table
                if wide:
                    reshaped = _long2resp(v, wave=None, id_density_threshold=0.1, agg_method="mean")
                    out[k] = reshaped if columns is None else reshaped.loc[:, columns]
                else:
                    out[k] = v
            else:
                out[k] = None
        return out
    
    else:
        return result


def itemtext(table_name: str) -> Union[pd.DataFrame, str]:
    """
    Get item-level text for a table.
    
    Parameters
    ----------
    table_name : str
        Name of the IRW table.
        
    Returns
    -------
    pandas.DataFrame or str
        DataFrame with item text if available, otherwise message string.

    Notes
    -----
    IRW item text is reconstructed from published articles, codebooks, and
    supplementary materials using a largely automated pipeline with partial
    human review, and is provided **for research purposes only**. We make no
    guarantee that the text is accurate, complete, or correctly aligned with
    the ``item`` identifiers in the corresponding IRW response data; the
    original source document is always authoritative. Users are responsible
    for verifying item text against that source before drawing substantive
    conclusions. Documented discrepancies are listed at
    https://itemresponsewarehouse.org/itemtext_issues.html

    Note also that the licenses recorded for IRW response data do not extend to
    the instruments themselves. Inclusion of item text here implies no license
    to reuse an instrument; copyright and related rights remain with the
    original rights holders, and users are responsible for obtaining any
    permissions their intended use requires.
    """
    _itemtext_disclaimer()
    return _get_table_itemtext(table_name)


def save_bibtex(
    table_names: Union[str, List[str]], 
    output_file: Optional[str] = None
) -> List[str]:
    """
    Get/save BibTeX entries for one or more IRW tables.
    
    Updates the BibTeX key to match the table name. Attempts to fetch BibTeX
    from the bibliography table first, then falls back to DOI-based lookup if needed.
    
    If output_file is provided, saves entries to file. Otherwise, returns entries.
    
    Parameters
    ----------
    table_names : str or list of str
        Single table name or list of table names for which BibTeX entries are generated.
    output_file : str, optional
        File path to save BibTeX entries. If None, returns entries instead.
        
    Returns
    -------
    list of str
        List of valid BibTeX entries.
        
    Examples
    --------
    >>> import irw
    >>> 
    >>> # Get BibTeX for a single table
    >>> bibtex = irw.save_bibtex("agn_kay_2025")
    >>> 
    >>> # Save BibTeX for multiple tables to file
    >>> irw.save_bibtex(["agn_kay_2025", "pks_probability"], "refs.bib")
    """
    import re
    import urllib.request
    from urllib.error import URLError
    
    # Normalize input to list
    if isinstance(table_names, str):
        table_names = [table_names]
    
    # Initialize lists
    valid_entries = []
    missing_tables = []
    missing_doi_tables = []
    
    # Fetch the full biblio table once (uses cache)
    from .utils.redivis.table_metadata import get_biblio_table
    from .utils.redivis.table_metadata import _get_existing_tables
    
    biblio_df = get_biblio_table()
    existing_tables = _get_existing_tables()
    
    # Process each table name
    for table_name in table_names:
        # Check if table exists in IRW
        if table_name.lower() not in existing_tables:
            missing_tables.append(table_name)
            continue
        
        # Get the corresponding row from biblio (case-insensitive)
        biblio_entry = biblio_df[biblio_df['table'].str.lower() == table_name.lower()]
        if biblio_entry.empty:
            missing_doi_tables.append(table_name)
            continue
        
        # Get BibTeX and DOI from the row
        bibtex = biblio_entry.iloc[0].get('BibTex') if 'BibTex' in biblio_entry.columns else None
        doi = biblio_entry.iloc[0].get('DOI__for_paper_') if 'DOI__for_paper_' in biblio_entry.columns else None
        
        # Clean BibTeX
        if bibtex is not None and pd.notna(bibtex):
            bibtex = str(bibtex).strip()
        else:
            bibtex = None
        
        # Step 1: Try manual BibTeX
        if bibtex and '@' in bibtex:
            # Valid BibTeX found
            pass
        else:
            # Step 2: Try DOI-based BibTeX
            bibtex = None
            if doi is not None and pd.notna(doi):
                doi_str = str(doi).strip()
                if doi_str:
                    try:
                        # Fetch BibTeX from DOI.org
                        url = f"https://doi.org/{doi_str}"
                        req = urllib.request.Request(
                            url,
                            headers={'Accept': 'application/x-bibtex'}
                        )
                        with urllib.request.urlopen(req, timeout=10) as response:
                            fetched_bibtex = response.read().decode('utf-8').strip()
                            if fetched_bibtex and '@' in fetched_bibtex:
                                bibtex = fetched_bibtex
                    except (URLError, Exception):
                        pass
        
        # Step 3: If all else fails
        if not bibtex or '@' not in bibtex:
            missing_doi_tables.append(table_name)
            continue
        
        # Step 4: Replace BibTeX key with table name
        # Pattern: @type{key, -> @type{table_name,
        bibtex = re.sub(
            r'@(\w+)\{[^,]+,',
            f'@\\1{{{table_name},',
            bibtex,
            count=1
        )
        
        valid_entries.append(bibtex)
    
    # Remove duplicates while preserving order
    if valid_entries:
        seen = set()
        unique_entries = []
        for entry in valid_entries:
            if entry not in seen:
                seen.add(entry)
                unique_entries.append(entry)
        
        if output_file:
            # Save to file
            with open(output_file, 'w', encoding='utf-8') as f:
                for entry in unique_entries:
                    f.write(entry + '\n\n')
        
        return unique_entries
    
    return valid_entries


def download(table_name: str, path: Optional[str] = None, overwrite: bool = False) -> str:
    """
    Download a table using Redivis's native download method.
    
    Parameters
    ----------
    table_name : str
        Name of the IRW table.
    path : str, optional
        Path to save file. If None, saves in current directory with table name.
    overwrite : bool, default False
        Whether to overwrite existing file.
        
    Returns
    -------
    str
        Path to saved file.
    """
    from .utils.redivis.tables import (
        _get_table,
        _format_error,
        _search_datasets,
        _terminal_error_message,
    )
    import os

    all_datasets = []
    for source in ["main", "sim", "comp", "nom"]:
        datasets = _get_datasets(source)
        all_datasets.extend(datasets if isinstance(datasets, list) else [datasets])

    def _download_from(ds):
        table_obj = _get_table(ds, table_name)
        if not hasattr(table_obj, "download"):
            raise ValueError(f"Table '{table_name}' does not support downloading.")
        table_obj.download(path=path, overwrite=overwrite)
        return path if path is not None else os.path.join(os.getcwd(), table_name)

    saved_path, last_other, terminal = _search_datasets(all_datasets, _download_from)
    if terminal is not None:
        # download() is the path most likely to hit the export quota, and it is
        # where "invalid format" was most misleading.
        raise ValueError(_terminal_error_message(terminal, table_name)) from terminal
    if saved_path is not None:
        return saved_path

    if last_other is not None:
        raise ValueError(
            f"Error downloading table '{table_name}': {_format_error(last_other)}"
        ) from last_other
    raise ValueError(f"Table '{table_name}' does not exist in the IRW database.")


def long2resp(
    df: pd.DataFrame,
    wave: Optional[int] = None,
    id_density_threshold: Optional[float] = 0.1,
    agg_method: Literal["mean", "mode", "median", "first"] = "mean"
) -> pd.DataFrame:
    """
    Convert IRW long-format data to wide-format response matrix.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Long-format DataFrame with columns: id, item, resp (and optionally wave).
        Typically obtained from irw.fetch().
    wave : int, optional
        Filter by wave. Defaults to most frequent wave if None.
    id_density_threshold : float, optional
        Minimum response density (0.0-1.0). None to disable. Default 0.1.
    agg_method : str, default "mean"
        How to handle multiple id-item pairs: "mean", "mode", "median", "first".
        
    Returns
    -------
    pandas.DataFrame
        Wide-format response matrix where rows are ids and columns are items.
        
    Examples
    --------
    >>> import irw
    >>> 
    >>> # Fetch data
    >>> df = irw.fetch("agn_kay_2025")
    >>> 
    >>> # Convert to wide format
    >>> resp_matrix = irw.long2resp(df)
    """
    return _long2resp(df, wave=wave, id_density_threshold=id_density_threshold, agg_method=agg_method)


def get_filters() -> List[str]:
    """
    Get list of available filter parameter names.
    
    Returns
    -------
    list[str]
        List of all available filter parameter names.
        
    Examples
    --------
    >>> import irw
    >>> filters = irw.get_filters()
    >>> print(filters)  # ['n_responses', 'n_participants', ...]
    """
    return _get_filters_func()


def describe_filter(filter_name: str) -> Optional[Dict]:
    """
    Describe a filter and show available values.
    
    Parameters
    ----------
    filter_name : str
        Name of the filter to describe.
        
    Returns
    -------
    dict or None
        Dictionary with 'description' and 'values', or None if not found.
    """
    return _describe_filter(_get_datasets("main"), filter_name)


def list_tables_with_itemtext() -> List[str]:
    """
    List tables that have item-level text available.
    
    Returns
    -------
    list of str
        Sorted list of table names with item text.

    Notes
    -----
    IRW item text is reconstructed from published articles, codebooks, and
    supplementary materials using a largely automated pipeline with partial
    human review, and is provided **for research purposes only**. We make no
    guarantee that the text is accurate, complete, or correctly aligned with
    the ``item`` identifiers in the corresponding IRW response data; the
    original source document is always authoritative. Users are responsible
    for verifying item text against that source before drawing substantive
    conclusions. Documented discrepancies are listed at
    https://itemresponsewarehouse.org/itemtext_issues.html

    Note also that the licenses recorded for IRW response data do not extend to
    the instruments themselves. Inclusion of item text here implies no license
    to reuse an instrument; copyright and related rights remain with the
    original rights holders, and users are responsible for obtaining any
    permissions their intended use requires.
    """
    _itemtext_disclaimer()
    available_tables = _list_itemtext_tables()
    return sorted(list(available_tables))



def collections(kind: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
    """
    List IRW collections: labelled groupings of tables.

    Collections cover study designs ("rct", "clustered", "q_matrix"), instrument
    families ("big_five", "promis") and constructs ("depression", "math"). A
    table can belong to several at once.

    The ``coverage`` column says how much of the warehouse each collection's
    rule actually searched, and it matters whenever you need *all* of
    something:

    - ``metadata-complete``  — searched every documented table.
    - ``tagged-subset-only`` — searched only the ~62% of tables that carry tag
      annotations, and coverage is far lower in the newer warehouses. Not
      exhaustive; do not describe these as "all".
    - ``curated-only``       — every member chosen by hand.

    Parameters
    ----------
    kind : str or list of str, optional
        Restrict to "design", "instrument" and/or "construct".

    Returns
    -------
    pandas.DataFrame
        One row per collection.

    Examples
    --------
    >>> import irw
    >>> irw.collections()
    >>> irw.collections(kind="instrument")
    """
    reg = _get_collections_table().copy()

    # n_tables in the published registry is the build-time count. Membership is
    # filtered to live tables on fetch, so recompute rather than hand back a
    # number that may overstate what can actually be retrieved.
    #
    # If membership cannot be fetched, keep the published counts rather than
    # failing: the registry is useful on its own, and this is the only part of
    # collections() that needs the large table.
    try:
        members = _get_collection_members_table()
    except Exception:
        members = pd.DataFrame(columns=["table", "collection"])
    if not members.empty:
        live = members.groupby("collection")["table"].nunique()
        reg["n_tables"] = reg["collection"].map(live).fillna(0).astype("Int64")

    if kind is not None:
        want = [kind] if isinstance(kind, str) else list(kind)
        valid = set(reg["kind"].dropna().unique())
        unknown = [k for k in want if k not in valid]
        if unknown:
            raise ValueError(
                f"Unknown kind(s): {unknown}. Valid: {sorted(valid)}."
            )
        reg = reg[reg["kind"].isin(want)]

    return reg.sort_values(["kind", "collection"]).reset_index(drop=True)


def collection(name: str, quiet: bool = False) -> List[str]:
    """
    Get the table names in one IRW collection.

    The result can be passed straight to :func:`fetch` or :func:`save_bibtex`.

    Prints the collection's definition and, when it does not cover the whole
    warehouse, says so — see the ``coverage`` note in :func:`collections`.

    Parameters
    ----------
    name : str
        A single collection name, e.g. "big_five".
    quiet : bool, default False
        Suppress the definition/coverage message.

    Returns
    -------
    list of str
        Sorted table names.

    Examples
    --------
    >>> import irw
    >>> tabs = irw.collection("depression")
    >>> dat = irw.fetch(tabs)
    >>> # Cross-collection: math assessments administered in booklets
    >>> sorted(set(irw.collection("math")) & set(irw.collection("multistage")))
    """
    if not isinstance(name, str):
        raise ValueError("`name` must be a single collection name. See irw.collections().")

    reg = _get_collections_table()
    known = list(reg["collection"])
    if name not in known:
        import difflib
        near = difflib.get_close_matches(name, known, n=3, cutoff=0.6)
        hint = (f" Did you mean: {near}?" if near
                else f" See irw.collections() for the {len(known)} available.")
        raise ValueError(f'No collection named "{name}".{hint}')

    members = _get_collection_members_table()
    tables = sorted(set(members.loc[members["collection"] == name, "table"]))

    if not quiet:
        row = reg[reg["collection"] == name].iloc[0]
        print(f"{name}: {len(tables)} table{'' if len(tables) == 1 else 's'}")
        if pd.notna(row.get("definition")):
            print(f"  {row['definition']}")
        if row.get("coverage") != "metadata-complete":
            print(f"  Coverage: {row.get('coverage')} -- this collection does not "
                  f"search the whole warehouse; see irw.collections().")

    return tables


def collection_members(
    tables: Optional[Union[str, List[str]]] = None,
    collection: Optional[Union[str, List[str]]] = None,
) -> pd.DataFrame:
    """
    IRW collection membership, one row per (table, collection) pair.

    Use this for the inverse question: what collections is a given table in?

    Parameters
    ----------
    tables : str or list of str, optional
        Restrict to these table names (case-insensitive).
    collection : str or list of str, optional
        Restrict to these collections.

    Returns
    -------
    pandas.DataFrame
        Columns: table, collection, basis.

    Examples
    --------
    >>> import irw
    >>> irw.collection_members(tables="frac20")
    """
    df = _get_collection_members_table().copy()

    if tables is not None:
        want = [tables] if isinstance(tables, str) else list(tables)
        want_lower = {t.lower() for t in want}
        df = df[df["table"].str.lower().isin(want_lower)]
    if collection is not None:
        want = [collection] if isinstance(collection, str) else list(collection)
        df = df[df["collection"].isin(want)]

    return df.sort_values(["table", "collection"]).reset_index(drop=True)


def version(date: Optional[Union[str, datetime.date, datetime.datetime]] = None) -> pd.DataFrame:
    """
    Which IRW version was live, and the Redivis version of every dataset in it.

    IRW is eleven Redivis datasets versioned independently, so no Redivis
    version describes the corpus. The IRW version number does: it increments
    whenever any dataset is published and names one exact combination of the
    eleven, so a paper can cite it and a reader can reconstruct the data.

    Parameters
    ----------
    date : str, date, or datetime, optional
        Report the version that was live then, e.g. ``"2026-08-01"``. Defaults
        to the newest version.

    Returns
    -------
    pandas.DataFrame
        Columns ``dataset``, ``version``, ``released_at``, ``approximate``;
        one row per dataset. The IRW version number is in
        ``.attrs["irw_version"]``.

    Notes
    -----
    Date lookups before 2026-07-21 are approximate and warn. Redivis overwrote
    its own release timestamps for the older warehouse shards during a platform
    migration, so 142 of the corpus\' 332 released versions cannot be dated
    exactly; the manifest brackets those rather than asserting them.

    Examples
    --------
    >>> irw.version()                     # doctest: +SKIP
    >>> irw.version("2026-08-01")         # doctest: +SKIP
    """
    return _version(date)
