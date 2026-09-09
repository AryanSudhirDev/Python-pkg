"""Filter IRW tables based on metadata criteria."""

from typing import List, Optional, Union
import re
import math
from numbers import Real
import pandas as pd
import numpy as np
from ..operations.list_tables import list_tables, IRWMetadataUnavailable


NUMERIC_FILTERS = frozenset({
    'n_responses', 'n_categories', 'n_participants', 'n_items',
    'responses_per_participant', 'responses_per_item', 'density',
})
BOOLEAN_FILTERS = frozenset({'longitudinal', 'has_item_text'})


class InvalidFilterValue(ValueError):
    """A filter value cannot express the documented predicate."""


def validate_filter_value(name, value):
    if value is None:
        return
    if name in NUMERIC_FILTERS:
        values = value if isinstance(value, list) else [value]
        if len(values) not in (1, 2):
            raise InvalidFilterValue(f'{name} requires a number or a one-/two-element list.')
        for endpoint in values:
            if endpoint is None and len(values) == 2:
                continue
            if isinstance(endpoint, (bool, np.bool_)) or not isinstance(endpoint, Real) or not math.isfinite(endpoint):
                raise InvalidFilterValue(f'{name} requires finite numbers; null is allowed only as a range endpoint.')
        if len(values) == 2 and all(v is not None for v in values) and values[0] > values[1]:
            raise InvalidFilterValue(f'{name} range minimum must not exceed its maximum.')
    elif name in BOOLEAN_FILTERS:
        if not isinstance(value, (bool, np.bool_)):
            raise InvalidFilterValue(f'{name} requires a boolean.')
    else:
        values = value if isinstance(value, list) else [value]
        if not values or any(not isinstance(v, str) or not v.strip() for v in values):
            raise InvalidFilterValue(f'{name} requires a non-empty string or list of non-empty strings.')


def _require_column(df: pd.DataFrame, column: str) -> None:
    """Refuse to silently skip a filter whose metadata column is missing.

    Every filter below is a no-op when its column is absent, so a metadata
    outage would quietly widen the result to the entire catalogue instead of
    narrowing it. Fail instead.
    """
    if column not in df.columns:
        raise IRWMetadataUnavailable(
            f"Cannot filter on '{column}': that column is missing from the IRW "
            f"metadata, so the filter would be silently ignored and every table "
            f"returned. This usually means the metadata tables could not be loaded."
        )


def _apply_numeric_filter(
    df: pd.DataFrame, 
    column: str, 
    value: Optional[Union[float, int, List[Optional[Union[float, int]]]]]
) -> pd.DataFrame:
    """Apply numeric filter (exact value or range).
    
    Supports:
    - Single value: exact match
    - List of length 1: exact match
    - List of length 2: range [min, max], where None means infinity
    """
    if value is None:
        return df
    validate_filter_value(column, value)
    _require_column(df, column)
    
    if isinstance(value, Real):
        # Exact match
        mask = df[column] == value
    elif isinstance(value, list) and len(value) == 1:
        # Single value in list
        mask = df[column] == value[0]
    elif isinstance(value, list) and len(value) == 2:
        # Range [min, max], where None means infinity
        min_val, max_val = value[0], value[1]
        if min_val is None:
            min_val = -np.inf
        if max_val is None or max_val == float('inf') or max_val == np.inf:
            max_val = np.inf
        mask = (df[column] >= min_val) & (df[column] <= max_val)
    else:
        return df
    
    return df[mask].copy()


def _apply_tag_filter(
    df: pd.DataFrame,
    column: str,
    values: Optional[Union[str, List[str]]]
) -> pd.DataFrame:
    """Apply tag-based filter (exact match, can be multiple values).
    
    Handles comma-separated values in tag columns (e.g., "value1, value2").
    Matches if any value in the comma-separated list matches any filter value.
    """
    if values is None:
        return df
    _require_column(df, column)
    
    if isinstance(values, str):
        values = [values]
    
    # Build mask checking each row
    mask = pd.Series(False, index=df.index)
    
    for idx in df.index:
        row_value = df.loc[idx, column]

        # List/tuple must be tested BEFORE pd.isna(): on a list, pd.isna()
        # returns an elementwise array and `if` on it raises "truth value of an
        # array is ambiguous". That made the isinstance branch below
        # unreachable for list-valued columns such as `collections`.
        if isinstance(row_value, (list, tuple)):
            tag_list = [str(v).strip() for v in row_value if v is not None and str(v).strip()]
            mask.loc[idx] = any(tag in values for tag in tag_list)
            continue

        if pd.isna(row_value):
            continue

        mask.loc[idx] = any(tag in values for tag in _split_tags(row_value))

    return df[mask].copy()


def _split_tags(cell) -> List[str]:
    """Split a multi-select tag cell into its atoms.

    Tag columns such as ``sample`` and ``construct type`` are multi-select,
    stored as one comma-joined string. ``03_tags.R`` in the pipeline repo
    normalizes them on export (issue #1720): stray quotes are removed, atoms
    are sorted into canonical order, and no tag value contains a comma -- the
    one that did, ``"Internet-based (Mturkers, etc)"``, was renamed to
    ``"Internet-based"``. That is what makes a plain split correct here.

    Before that normalization this split was wrong: the comma inside that value
    broke it into ``"Internet-based (Mturkers"`` and ``"etc)"``, so
    ``filter(sample="Internet-based (Mturkers, etc)")`` silently matched
    nothing across ~450 tables. R carried a 34-line quote-aware parser to cope;
    Python carried none. Both now do a plain split, deliberately.

    Do not reintroduce quote handling here -- fix the vocabulary instead, so
    that no tag value ever contains the delimiter.
    """
    if isinstance(cell, (list, tuple)):
        return [str(v).strip() for v in cell if v is not None and str(v).strip()]
    return [v.strip() for v in str(cell).split(',') if v.strip()]


def _apply_variable_filter(
    df: pd.DataFrame,
    variables: Optional[Union[str, List[str]]]
) -> pd.DataFrame:
    """Apply variable presence filter (exact names or prefix matching).
    
    ALL variables in the filter list must be present (AND logic).
    Variables containing '_' are treated as prefix matches.
    """
    if variables is None:
        return df
    _require_column(df, 'variables')
    
    if isinstance(variables, str):
        variables = [variables]
    
    # Build mask - ALL variables must match (AND logic)
    mask = pd.Series(True, index=df.index)
    
    # Check each row's variables column
    for idx in df.index:
        row_vars = df.loc[idx, 'variables']
        
        # Handle different data types
        if pd.isna(row_vars):
            mask.loc[idx] = False
            continue
        
        if isinstance(row_vars, (list, tuple)):
            # If it's a list/tuple, check directly
            var_list = [str(v).lower() for v in row_vars]
        elif isinstance(row_vars, str):
            # R uses pipe-separated: "var1|var2|var3"
            var_list = [v.strip().lower() for v in re.split(r'\|\s*', str(row_vars)) if v.strip()]
        else:
            # Convert to string and try pipe-separated
            var_str = str(row_vars).lower()
            var_list = [v.strip() for v in re.split(r'\|\s*', var_str) if v.strip()]
        
        # Check if ALL filter variables match
        all_match = True
        for var in variables:
            var_lower = var.lower()
            
            # Check if variable contains '_' (R logic: any underscore means prefix match)
            if '_' in var_lower:
                # Prefix matching - remove trailing underscore if present
                var_prefix = var_lower.rstrip('_')
                matches = any(v.startswith(var_prefix) for v in var_list)
            else:
                # Exact match
                matches = var_lower in var_list
            
            if not matches:
                all_match = False
                break
        
        mask.loc[idx] = all_match
    
    return df[mask].copy()


def _apply_longitudinal_filter(
    df: pd.DataFrame,
    longitudinal: Optional[bool]
) -> pd.DataFrame:
    """Apply longitudinal filter."""
    if longitudinal is None:
        return df
    _require_column(df, 'longitudinal')
    
    mask = df['longitudinal'] == longitudinal
    return df[mask].copy()


def filter_tables(
    datasets: List,
    n_responses: Optional[Union[float, int, List[Optional[Union[float, int]]]]] = None,
    n_categories: Optional[Union[float, int, List[Optional[Union[float, int]]]]] = None,
    n_participants: Optional[Union[float, int, List[Optional[Union[float, int]]]]] = None,
    n_items: Optional[Union[float, int, List[Optional[Union[float, int]]]]] = None,
    responses_per_participant: Optional[Union[float, int, List[Optional[Union[float, int]]]]] = None,
    responses_per_item: Optional[Union[float, int, List[Optional[Union[float, int]]]]] = None,
    density: Optional[Union[float, int, List[Optional[Union[float, int]]]]] = [0.5, 1],
    var: Optional[Union[str, List[str]]] = None,
    age_range: Optional[Union[str, List[str]]] = None,
    child_age: Optional[Union[str, List[str]]] = None,
    construct_type: Optional[Union[str, List[str]]] = None,
    construct_name: Optional[Union[str, List[str]]] = None,
    sample: Optional[Union[str, List[str]]] = None,
    measurement_tool: Optional[Union[str, List[str]]] = None,
    item_format: Optional[Union[str, List[str]]] = None,
    language: Optional[Union[str, List[str]]] = None,
    longitudinal: Optional[bool] = None,
    has_item_text: Optional[bool] = None,
    license: Optional[Union[str, List[str]]] = None,
    collection: Optional[Union[str, List[str]]] = None,
) -> pd.Series:
    """
    Filter IRW tables based on metadata criteria.
    
    Returns the names of datasets in the Item Response Warehouse (IRW) that match
    user-specified metadata, tag values, variable presence, and license criteria.
    
    This function only works for the main IRW database (source="main").
    
    Parameters
    ----------
    n_responses : float, int, or list of length 1 or 2, optional
        Filter by total number of responses.
        - Single value: exact match (e.g., n_responses=1000)
        - List of length 2: range [min, max] (e.g., n_responses=[1000, None] for >= 1000)
    
    n_categories : float, int, or list of length 1 or 2, optional
        Filter by number of unique response categories.
        - List of length 2: range [min, max], use None for infinity (e.g., [3, None] for >= 3)
    
    n_participants : float, int, or list of length 1 or 2, optional
        Filter by number of unique participants.
        - List of length 2: range [min, max], use None for infinity (e.g., [500, None] for >= 500)
    
    n_items : float, int, or list of length 1 or 2, optional
        Filter by number of unique items.
        - List of length 2: range [min, max], use None for infinity (e.g., [10, 50] for between 10-50)
    
    responses_per_participant : float, int, or list of length 1 or 2, optional
        Filter by average responses per participant.
        - List of length 2: range [min, max], use None for infinity
    
    responses_per_item : float, int, or list of length 1 or 2, optional
        Filter by average responses per item.
        - List of length 2: range [min, max], use None for infinity
    
    density : float, int, list of length 1 or 2, or None, optional
        Filter by matrix density (proportion of cells with valid responses).
        A density of 1 means every person responded to every item (100% of cells have valid responses).
        Lower density indicates that some individuals did not respond to all items.
        Default is [0.5, 1] to exclude sparse matrices.
        - List of length 2: range [min, max], use None for infinity (e.g., density=[0.5, None] for >= 0.5)
        - Set to None to disable density filtering
    
    var : str or list of str, optional
        Filter datasets by presence of variables.
        - Exact names: "rt", "wave"
        - Prefix matching: "cov_" (matches any variable starting with "cov_")
    
    age_range : str or list of str, optional
        Filter by participant age group (e.g., "Adult (18+)").
    
    child_age : str or list of str, optional
        Filter by child age subgroup.
    
    construct_type : str or list of str, optional
        Filter by high-level construct category (e.g., "Affective/mental health").
        Can provide multiple values as a list for OR logic (e.g., ["Affective/mental health", "Cognitive"]).
    
    construct_name : str or list of str, optional
        Filter by specific construct (e.g., "Big Five").
    
    sample : str or list of str, optional
        Filter by sample type or recruitment method (e.g., "Educational", "Clinical").
    
    measurement_tool : str or list of str, optional
        Filter by instrument type (e.g., "Survey/questionnaire").
    
    item_format : str or list of str, optional
        Filter by item format (e.g., "Likert Scale/selected response").
    
    language : str or list of str, optional
        Filter by language used (e.g., "eng").
    
    longitudinal : bool or None, optional
        Filter longitudinal datasets.
        - True: include only datasets flagged as longitudinal
        - False: exclude datasets flagged as longitudinal
        - None: no filter (default)
    has_item_text : bool or None, optional
        Filter by whether reconstructed item text is available.
        - True: include only datasets with item text
        - False: exclude datasets with item text
        - None: no filter (default)
    
    license : str or list of str, optional
        Filter datasets by license (e.g., "CC BY 4.0").
        Can provide multiple values as a list for OR logic.

    collection : str or list of str, optional
        Filter datasets by collection membership (e.g., "rct", "big_five",
        "depression"). OR within the argument: ["rct", "response_time"] returns
        the union, not the intersection -- for that, intersect two
        irw.collection() results. Raises ValueError on an unknown name.
        See irw.collections() for what exists and how complete each one is.

    Returns
    -------
    pandas.Series
        A sorted Series of dataset names (table names) that match all specified filters.
        Returns empty Series if no matches are found.
    
    Examples
    --------
    >>> import irw
    >>> 
    >>> # Numeric filters
    >>> filtered = irw.filter(n_responses=[1000, None], n_items=[10, 50])  # >= 1000 responses, 10-50 items
    >>> 
    >>> # Variable presence
    >>> filtered = irw.filter(var="rt")
    >>> filtered = irw.filter(var=["wave", "cov_"])
    >>> 
    >>> # Tag metadata filtering (single value)
    >>> filtered = irw.filter(construct_type="Affective/mental health", sample="Educational")
    >>> 
    >>> # Tag metadata filtering (multiple values with OR logic)
    >>> filtered = irw.filter(construct_type=["Affective/mental health", "Cognitive"])
    >>> filtered = irw.filter(sample=["Educational", "Clinical"])
    >>> 
    >>> # License filtering
    >>> filtered = irw.filter(license="CC BY 4.0")
    >>> 
    >>> # Filter by response category complexity
    >>> filtered = irw.filter(n_categories=2)  # binary
    >>> filtered = irw.filter(n_categories=[3, 5])  # small multi-category
    """
    # Validate before loading data, including when the catalogue is empty.
    for filter_name, filter_value in locals().copy().items():
        if filter_name != 'datasets':
            validate_filter_value(filter_name, filter_value)

    # Get all tables with metadata
    df = list_tables(datasets)
    
    if df.empty or 'name' not in df.columns:
        return pd.Series([], dtype=str, name='name')
    
    # Apply numeric filters
    df = _apply_numeric_filter(df, 'n_responses', n_responses)
    df = _apply_numeric_filter(df, 'n_categories', n_categories)
    df = _apply_numeric_filter(df, 'n_participants', n_participants)
    df = _apply_numeric_filter(df, 'n_items', n_items)
    df = _apply_numeric_filter(df, 'responses_per_participant', responses_per_participant)
    df = _apply_numeric_filter(df, 'responses_per_item', responses_per_item)
    
    # Apply density filter (with default handling)
    # Default is [0.5, 1] to exclude sparse matrices
    if density is not None:
        df_before_density = df.copy()
        df = _apply_numeric_filter(df, 'density', density)
        
        # Show message if default density removed datasets
        # Check if density equals default [0.5, 1]
        is_default_density = (
            isinstance(density, list) and 
            len(density) == 2 and 
            density[0] == 0.5 and 
            density[1] == 1
        )
        
        num_removed = len(df_before_density) - len(df)
        if is_default_density and num_removed > 0:
            import warnings
            warnings.warn(
                f"Note: Default density filter (0.5-1) removed {num_removed} dataset(s). "
                f"Set density=None to disable.",
                UserWarning,
                stacklevel=2
            )
    
    # Apply variable presence filter
    df = _apply_variable_filter(df, var)

    # has_item_text is a plain boolean column on list_tables, the same shape
    # as longitudinal. It was filterable by slicing list_tables and not
    # through filter(), and a gap like that is what makes a caller
    # reimplement the filtering instead of calling it.
    if has_item_text is not None:
        _require_column(df, 'has_item_text')
        df = df[df['has_item_text'] == has_item_text].copy()
    
    # Apply tag filters (using cleaned-up column names from list_tables)
    if age_range is not None:
        df = _apply_tag_filter(df, 'age_range', age_range)
    if child_age is not None:
        df = _apply_tag_filter(df, 'child_age', child_age)
    if construct_type is not None:
        df = _apply_tag_filter(df, 'construct_type', construct_type)
    if construct_name is not None:
        df = _apply_tag_filter(df, 'construct_name', construct_name)
    if collection is not None:
        # Singular arg name (matches R's `collection=`), plural column name
        # (matches the list-valued content). _apply_tag_filter already handles
        # list row values with OR semantics, so no new primitive is needed.
        _require_column(df, 'collections')
        _known = {c for v in df['collections'] for c in (v if isinstance(v, list) else [])}
        _want = [collection] if isinstance(collection, str) else list(collection)
        _unknown = [c for c in _want if c not in _known]
        if _unknown:
            raise ValueError(
                f"Unknown collection(s): {_unknown}. "
                f"See irw.collections() for the {len(_known)} available."
            )
        df = _apply_tag_filter(df, 'collections', collection)
    if sample is not None:
        df = _apply_tag_filter(df, 'sample', sample)
    if measurement_tool is not None:
        df = _apply_tag_filter(df, 'measurement_tool', measurement_tool)
    if item_format is not None:
        df = _apply_tag_filter(df, 'item_format', item_format)
    if language is not None:
        df = _apply_tag_filter(df, 'language', language)
    
    # Apply longitudinal filter
    df = _apply_longitudinal_filter(df, longitudinal)
    
    # Apply license filter
    df = _apply_tag_filter(df, 'license', license)
    
    # Return sorted Series of table names
    if df.empty:
        return pd.Series([], dtype=str, name='name')
    
    result = df['name'].sort_values().reset_index(drop=True)
    return result
