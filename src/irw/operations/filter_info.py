"""Functions to explore available filter options."""

from typing import List, Dict, Any, Optional
import pandas as pd
import re
from ..utils.redivis.table_metadata import (
    get_metadata_table,
    get_tags_table,
    get_biblio_table,
    get_collections_table,
    get_collection_members_table,
)
from ..utils.redivis.item_text import _list_itemtext_tables
from ..operations.list_tables import _build_base_table_list
from ..operations.filter import _split_tags


# Filter descriptions with usage instructions
FILTER_DESCRIPTIONS = {
    'n_responses': 'Total number of responses in the dataset. Use a single number for exact match, or a list [min, max] for a range (use None for no upper limit, e.g., [1000, None] for ≥1000).',
    'n_categories': 'Number of unique response categories. Use a single number for exact match, or a list [min, max] for a range (use None for no upper limit).',
    'n_participants': 'Number of unique participants (id) in the dataset. Use a single number for exact match, or a list [min, max] for a range (use None for no upper limit).',
    'n_items': 'Number of unique items. Use a single number for exact match, or a list [min, max] for a range (use None for no upper limit).',
    'responses_per_participant': 'Average number of responses per participant. Use a single number for exact match, or a list [min, max] for a range (use None for no upper limit).',
    'responses_per_item': 'Average number of responses per item. Use a single number for exact match, or a list [min, max] for a range (use None for no upper limit).',
    'density': 'Matrix density (proportion of cells with valid responses). A density of 1 means every person responded to every item (100% of cells have valid responses). Lower density indicates that some individuals did not respond to all items. Use a single number for exact match, or a list [min, max] for a range (use None for no upper limit). Default is [0.5, 1] to exclude sparse matrices.',
    'var': 'Filter by presence of specific variables in the dataset. Use exact variable names (e.g., "rt", "wave") or prefix matching (e.g., "cov_" matches any variable starting with "cov_"). Provide a single string or a list of strings. All specified variables must be present (AND logic).',
    'age_range': 'Participant age group (e.g., "Adult (18+)"). Use a single string or a list of strings for OR logic (e.g., ["Adult (18+)", "Child"]).',
    'child_age': 'Child age subgroup (for child-focused studies). Use a single string or a list of strings for OR logic.',
    'construct_type': 'High-level construct category (e.g., "Affective/mental health"). Use a single string or a list of strings for OR logic (e.g., ["Affective/mental health", "Cognitive"]).',
    'construct_name': 'Specific construct name (e.g., "Big Five"). Use a single string or a list of strings for OR logic.',
    'sample': 'Sample type or recruitment method (e.g., "Educational", "Clinical"). Use a single string or a list of strings for OR logic.',
    'measurement_tool': 'Instrument type (e.g., "Survey/questionnaire"). Use a single string or a list of strings for OR logic.',
    'item_format': 'Item format (e.g., "Likert Scale/selected response"). Use a single string or a list of strings for OR logic.',
    'language': 'Primary language used (e.g., "eng"). Use a single string or a list of strings for OR logic.',
    'longitudinal': 'Whether the dataset is longitudinal (i.e., has wave or date variables). Use True to include only longitudinal datasets, False to exclude longitudinal datasets, or None for no filter.',
    'has_item_text': 'Whether reconstructed item text is available for the dataset. Use True for tables with item text, False for those without, or None for no filter. Item text is available for a minority of tables and is reconstructed from source documents with partial review, so its absence is not a judgement about the response data.',
    'license': 'Dataset license type (e.g., "CC BY 4.0"). Use a single string or a list of strings for OR logic.',
    'collection': 'Collection membership (e.g., "rct", "big_five", "depression"). Collections are labelled groupings of IRW tables -- study designs, instrument families and constructs. A table can be in several at once. Use a single string or a list of strings for OR logic (the union, not the intersection). See irw.collections() for what exists, and note the coverage column: collections derived from tags searched only ~62% of tables, so they are not exhaustive.'
}


def get_filters() -> List[str]:
    """
    Get list of available filter parameter names.
    
    Returns
    -------
    list of str
        List of all available filter parameter names.
        
    Examples
    --------
    >>> import irw
    >>> 
    >>> # Get list of available filters
    >>> filters = irw.get_filters()
    >>> print(filters)  # ['n_responses', 'n_participants', ...]
    """
    return list(FILTER_DESCRIPTIONS.keys())


def describe_filter(datasets: List, filter_name: str) -> Optional[Dict[str, Any]]:
    """
    Describe a specific filter and show its available values.
    
    Returns a dictionary with two keys:
    - 'description': Explanation of what the filter is and how to use it
    - 'values': Available values for the filter:
              - Numeric filters: dict with summary statistics (min, max, mean, median, etc.)
              - Categorical filters: pandas.Series with value_counts (sorted by count)
              - 'var': pandas.Series with variable frequency counts (sorted by count)
              - 'longitudinal': dict with counts for True/False
    
    **Performance Note:** This function only loads the necessary metadata (stats, tags, or
    bibliography) based on the filter type. For example, requesting 'n_responses' only loads
    the stats table, not tags or bibliography. All loaded data is cached for subsequent calls.
    
    Parameters
    ----------
    datasets : List
        Redivis dataset objects (typically from IRW._datasets).
    filter_name : str
        Name of the filter to describe (e.g., 'n_responses', 'construct_type', 'var').
        
    Returns
    -------
    dict or None
        Dictionary with keys 'description' and 'values', or None if filter not found.
        
    Examples
    --------
    >>> import irw
    >>> 
    >>> # Describe numeric filter
    >>> info = irw.describe_filter('n_responses')
    >>> print(info['description'])  # What the filter is and how to use it
    >>> print(info['values'])  # Summary statistics
    >>> 
    >>> # Describe categorical filter
    >>> info = irw.describe_filter('construct_type')
    >>> print(info['description'])  # What the filter is and how to use it
    >>> print(info['values'])  # Available values with counts
    >>> 
    >>> # Describe variable filter
    >>> info = irw.describe_filter('var')
    >>> print(info['description'])  # How to use variable filtering
    >>> print(info['values'].head(10))  # Top 10 most common variables
    """
    # Check if filter name is valid
    if filter_name not in FILTER_DESCRIPTIONS:
        return None
    
    description = FILTER_DESCRIPTIONS[filter_name]
    result = {'description': description}
    
    # Get base table list (needed for merging)
    base = _build_base_table_list(datasets)
    if base.empty:
        return None
    
    # Numeric filters - only need stats/metadata table
    numeric_filters = [
        'n_responses',
        # n_categories lives in the metadata table beside the other counts.
        # It used to have its own branch below that looked for it in the tags
        # table, which has never carried it, so describe_filter('n_categories')
        # returned None for a filter that filter() accepts -- the same "the
        # list is a lie" failure the drift test in tests/test_filter_info.py
        # guards against, one layer down.
        'n_categories',
        'n_participants',
        'n_items',
        'responses_per_participant',
        'responses_per_item',
        'density'
    ]
    
    if filter_name in numeric_filters:
        metadata_df = get_metadata_table()
        if metadata_df.empty:
            return None
        
        # Merge with base to get all tables
        metadata_df['name_lower'] = metadata_df['table'].str.lower()
        base['name_lower'] = base['name'].str.lower()
        merged = base.merge(metadata_df, left_on='name_lower', right_on='name_lower', how='left')
        
        if filter_name not in merged.columns:
            return None
        series = merged[filter_name].dropna()
        if series.empty:
            return None
        result['values'] = {
            'min': float(series.min()),
            'max': float(series.max()),
            'mean': float(series.mean()),
            'median': float(series.median()),
            'std': float(series.std()),
            'count': len(series),
            'non_null_count': len(series),
            'null_count': len(merged) - len(series)
        }
        return result
    
    # Categorical/tag filters - only need tags table
    categorical_filters = [
        'construct_type',
        'construct_name',
        'age_range',
        'child_age',
        'sample',
        'item_format',
        'measurement_tool',
        'language'
    ]
    
    if filter_name in categorical_filters:
        tags_df = get_tags_table()
        if tags_df.empty:
            return None
        
        # Handle column name mapping
        col_name = filter_name
        if filter_name == 'child_age':
            col_name = 'child_age__for_child_focused_studies_'
        elif filter_name == 'language':
            col_name = 'primary_language_s_'
        
        # Merge with base
        tags_df['name_lower'] = tags_df['table'].str.lower()
        base['name_lower'] = base['name'].str.lower()
        merged = base.merge(tags_df, left_on='name_lower', right_on='name_lower', how='left')
        
        if col_name not in merged.columns:
            return None
        # Count tag ATOMS, not whole cells. `sample` and `construct type` are
        # multi-select and stored as one comma-joined string, so counting
        # cells reported the ~37 combinations as if each were a value --
        # while filter() matches on atoms via _split_tags(). The function that
        # tells you the valid inputs disagreed with the one that consumes
        # them: a user copying "General/non-specific, Internet-based" out of
        # describe_filter() and into filter() got zero matches.
        #
        # Applied to every categorical column rather than a multi-select
        # allowlist: splitting a comma-free cell is a no-op, so single-valued
        # columns such as `item format` and `age range` are unaffected. R does
        # the same -- irw_tag_options() and the filter path share one
        # .irw_split_tags().
        atoms = merged[col_name].dropna().apply(_split_tags).explode().dropna()
        value_counts = atoms.value_counts()
        if value_counts.empty:
            return None
        # Sort by count (descending), then by value (ascending)
        result_df = value_counts.reset_index()
        result_df.columns = ['value', 'count']
        result_df = result_df.sort_values(['count', 'value'], ascending=[False, True])
        result['values'] = pd.Series(result_df['count'].values, index=result_df['value'].values, name=filter_name)
        return result
    
    # License filter - only need bibliography table
    if filter_name == 'license':
        biblio_df = get_biblio_table()
        if biblio_df.empty:
            return None
        
        # Merge with base
        biblio_df['name_lower'] = biblio_df['table'].str.lower()
        base['name_lower'] = base['name'].str.lower()
        merged = base.merge(biblio_df, left_on='name_lower', right_on='name_lower', how='left')
        
        if 'Derived_License' not in merged.columns:
            return None
        value_counts = merged['Derived_License'].value_counts(dropna=True)
        if value_counts.empty:
            return None
        # Sort by count (descending), then by value (ascending)
        result_df = value_counts.reset_index()
        result_df.columns = ['value', 'count']
        result_df = result_df.sort_values(['count', 'value'], ascending=[False, True])
        result['values'] = pd.Series(result_df['count'].values, index=result_df['value'].values, name='license')
        return result
    
    # Special filters
    if filter_name == 'var':
        # Variables column is in metadata table, not tags table
        metadata_df = get_metadata_table()
        if metadata_df.empty:
            return None
        
        # Merge with base
        metadata_df['name_lower'] = metadata_df['table'].str.lower()
        base['name_lower'] = base['name'].str.lower()
        merged = base.merge(metadata_df, left_on='name_lower', right_on='name_lower', how='left')
        
        if 'variables' not in merged.columns:
            return None
        
        # Extract variables and count frequency across datasets
        var_counts = {}
        for vars_str in merged['variables'].dropna():
            if pd.isna(vars_str):
                continue
            if isinstance(vars_str, str):
                # R uses pipe-separated: "var1|var2|var3"
                vars_list = [v.strip() for v in re.split(r'\|\s*', str(vars_str)) if v.strip()]
                for var in vars_list:
                    var_counts[var] = var_counts.get(var, 0) + 1
        
        if not var_counts:
            result['values'] = pd.Series(dtype=int)
            return result
        
        # Convert to Series and sort by count (descending), then by variable name (ascending) for ties
        var_series = pd.Series(var_counts)
        # Sort by count descending, then by index (variable name) ascending for ties
        var_series = var_series.sort_index().sort_values(ascending=False, kind='mergesort')
        
        result['values'] = var_series
        return result
    
    if filter_name == 'longitudinal':
        # Need metadata table for longitudinal flag
        metadata_df = get_metadata_table()
        if metadata_df.empty:
            return None
        
        metadata_df['name_lower'] = metadata_df['table'].str.lower()
        base['name_lower'] = base['name'].str.lower()
        merged = base.merge(metadata_df, left_on='name_lower', right_on='name_lower', how='left')
        
        if 'longitudinal' not in merged.columns:
            return None
        long_counts = merged['longitudinal'].value_counts(dropna=True)
        result['values'] = {
            True: int(long_counts.get(True, 0)),
            False: int(long_counts.get(False, 0))
        }
        return result

    if filter_name == 'has_item_text':
        # Computed, not stored: a table has item text when its name is in one
        # of the irw_text shards. Same source list_tables() uses for the
        # column filter() reads, so these counts are the ones filter() sees.
        try:
            with_text = _list_itemtext_tables()
        except Exception:
            return None
        names = base['name'].dropna().astype(str).str.lower()
        n_true = int(names.isin(with_text).sum())
        result['values'] = {True: n_true, False: int(len(names) - n_true)}
        return result

    if filter_name == 'collection':
        # The registry names every collection, including ones whose rule
        # matched nothing that is live today; the members table gives the live
        # count. Reported the same way collections() does, so a name copied
        # from here into filter(collection=...) is one filter() knows.
        registry = get_collections_table()
        if registry.empty or 'collection' not in registry.columns:
            return None
        try:
            members = get_collection_members_table()
        except Exception:
            members = pd.DataFrame(columns=['table', 'collection'])
        live: Dict[str, int] = {}
        if not members.empty and {'table', 'collection'} <= set(members.columns):
            live = members.groupby('collection')['table'].nunique().to_dict()
        rows = [
            (str(name), int(live.get(name, 0)))
            for name in registry['collection'].dropna().drop_duplicates()
        ]
        rows.sort(key=lambda row: (-row[1], row[0]))
        result['values'] = pd.Series(
            [count for _, count in rows],
            index=[name for name, _ in rows],
            name='collection',
        )
        return result

    return None
