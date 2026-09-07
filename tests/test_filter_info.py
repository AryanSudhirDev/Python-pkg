"""Offline coverage for describe_filter()'s categorical branch.

describe_filter() is how a Python user discovers what they can filter on, so
its output has to agree with what filter() actually matches. filter() matches
tag ATOMS (via _split_tags); describe_filter() used to count whole cells, so
for the multi-select columns it reported combinations that filter() would
never match. See issue #8.

No network: both metadata lookups are patched.
"""

import pandas as pd
import pytest

from irw.operations import filter_info


@pytest.fixture
def patched(monkeypatch):
    """describe_filter() over a fixed tags table, with no Redivis behind it."""
    base = pd.DataFrame({"name": ["t1", "t2", "t3", "t4"]})

    tags = pd.DataFrame({
        "table": ["t1", "t2", "t3", "t4"],
        # multi-select: comma-joined atoms
        "sample": [
            "Educational, Internet-based",
            "Educational, Internet-based",
            "Clinical",
            None,
        ],
        # single-valued: a plain split is a no-op
        "age_range": ["Adult", "Adult", "Child", None],
    })

    monkeypatch.setattr(filter_info, "_build_base_table_list", lambda ds: base.copy())
    monkeypatch.setattr(filter_info, "get_tags_table", lambda: tags.copy())
    return None


def test_multi_select_values_are_reported_as_atoms_not_combinations(patched):
    """Regression: `sample` reported "Educational, Internet-based" as a value.

    That string is not a value of anything -- it is two values in one cell --
    and passing it back to filter(sample=...) matches zero tables.
    """
    result = filter_info.describe_filter([], "sample")
    values = result["values"]

    assert set(values.index) == {"Educational", "Internet-based", "Clinical"}
    assert values["Educational"] == 2
    assert values["Internet-based"] == 2
    assert values["Clinical"] == 1
    assert "Educational, Internet-based" not in values.index


def test_single_valued_columns_are_unchanged_by_the_split(patched):
    """Splitting a comma-free cell is a no-op, so no multi-select allowlist."""
    result = filter_info.describe_filter([], "age_range")
    values = result["values"]

    assert set(values.index) == {"Adult", "Child"}
    assert values["Adult"] == 2
    assert values["Child"] == 1


def test_missing_tags_are_not_counted_as_a_value(patched):
    """t4 has no tags; "nan" must not appear as a tag atom."""
    values = filter_info.describe_filter([], "sample")["values"]
    assert "nan" not in values.index
    assert values.sum() == 5  # 2 + 2 + 1, from three tagged tables


def test_every_filter_description_names_a_real_filter_argument():
    """The drift guard: FILTER_DESCRIPTIONS is the list callers are shown, and
    filter_tables' signature is the list that works. A name in one and not the
    other is either a filter nobody can find or one that raises TypeError."""
    import inspect

    from irw.operations.filter import filter_tables
    from irw.operations.filter_info import FILTER_DESCRIPTIONS

    parameters = set(inspect.signature(filter_tables).parameters) - {"datasets"}
    assert set(FILTER_DESCRIPTIONS) == parameters


def test_has_item_text_is_filterable():
    import pandas as pd
    from unittest.mock import patch

    from irw.operations.filter import filter_tables

    frame = pd.DataFrame(
        {
            "name": ["with_text", "without_text"],
            "has_item_text": [True, False],
            "density": [1.0, 1.0],
        }
    )
    with patch("irw.operations.filter.list_tables", return_value=frame):
        assert filter_tables([], has_item_text=True).tolist() == ["with_text"]
        assert filter_tables([], has_item_text=False).tolist() == ["without_text"]
        assert sorted(filter_tables([]).tolist()) == ["with_text", "without_text"]
