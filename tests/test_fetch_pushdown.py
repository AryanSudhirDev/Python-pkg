"""Tests for the row and column pushdown on fetch (no Redivis network calls).

The point of these arguments is *where* the limit is applied, so the assertion
that matters is on what reached `to_pandas_dataframe` -- a test that only
checked the returned row count would pass just as well against a slice taken
after the whole table was downloaded, which is the bug they exist to prevent.
"""

import warnings

import pandas as pd
import pytest

from irw.operations.fetch import _validate_pushdown, fetch


class _Table:
    """Records the arguments the wire call was made with."""

    def __init__(self, frame):
        self._frame = frame
        self.calls = []

    def get(self):
        return None

    def to_pandas_dataframe(self, max_results=None, *, variables=None, **kwargs):
        self.calls.append({"max_results": max_results, "variables": variables})
        frame = self._frame
        if variables is not None:
            frame = frame.loc[:, list(variables)]
        if max_results is not None:
            frame = frame.iloc[:max_results]
        return frame.copy()


class _Dataset:
    def __init__(self, table):
        self._table = table

    def table(self, name):
        return self._table


def _frame(n=10):
    return pd.DataFrame(
        {
            "id": [f"p{i // 2}" for i in range(n)],
            "item": [f"q{i % 2}" for i in range(n)],
            "resp": list(range(n)),
            "cov_age": [30 + i for i in range(n)],
        }
    )


def test_max_rows_reaches_redivis_rather_than_slicing_afterwards():
    table = _Table(_frame())
    out = fetch([_Dataset(table)], "t", max_rows=3)
    assert table.calls == [{"max_results": 3, "variables": None}]
    assert len(out) == 3


def test_columns_reach_redivis():
    table = _Table(_frame())
    out = fetch([_Dataset(table)], "t", columns=["id", "item"])
    assert table.calls == [{"max_results": None, "variables": ["id", "item"]}]
    assert list(out.columns) == ["id", "item"]


def test_defaults_are_unchanged_so_an_uncapped_fetch_still_asks_for_everything():
    table = _Table(_frame())
    fetch([_Dataset(table)], "t")
    assert table.calls == [{"max_results": None, "variables": None}]


def test_pushdown_applies_to_every_table_in_a_multi_table_fetch():
    tables = [_Table(_frame()), _Table(_frame())]
    fetch([_Dataset(tables[0])], "a", max_rows=2)
    fetch([_Dataset(tables[1])], "b", max_rows=2)
    assert all(t.calls[0]["max_results"] == 2 for t in tables)


def test_dedup_on_a_capped_fetch_warns_that_it_only_saw_the_window():
    table = _Table(_frame())
    with pytest.warns(UserWarning, match="first 4 rows only"):
        fetch([_Dataset(table)], "t", max_rows=4, dedup=True)


def test_dedup_without_a_cap_does_not_warn():
    table = _Table(_frame())
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        fetch([_Dataset(table)], "t", dedup=True)


@pytest.mark.parametrize(
    "value, error",
    [
        (0, ValueError),
        (-1, ValueError),
        (True, TypeError),
        (1.5, TypeError),
        ("10", TypeError),
    ],
)
def test_max_rows_rejects_values_that_would_silently_mean_something_else(value, error):
    with pytest.raises(error):
        _validate_pushdown(value, None)


def test_columns_rejects_a_bare_string():
    """`columns="id"` would otherwise reach Redivis as ['i', 'd']."""
    with pytest.raises(TypeError, match=r"\['id'\]"):
        _validate_pushdown(None, "id")


def test_columns_rejects_an_empty_selection():
    with pytest.raises(ValueError):
        _validate_pushdown(None, [])


def test_columns_is_normalised_to_a_list_so_a_generator_survives():
    _, columns = _validate_pushdown(None, (c for c in ["id", "resp"]))
    assert columns == ["id", "resp"]


def test_wide_on_a_capped_fetch_warns_that_the_matrix_is_not_the_respondents():
    """api.fetch's own guard: long2resp reshapes whatever rows it is handed."""
    from unittest.mock import patch

    import irw

    table = _Table(_frame())
    with patch("irw.api._get_datasets", return_value=[_Dataset(table)]):
        with pytest.warns(UserWarning, match="first 4 rows"):
            irw.fetch("t", max_rows=4, wide=True)


def test_api_fetch_passes_pushdown_through_to_the_wire():
    from unittest.mock import patch

    import irw

    table = _Table(_frame())
    with patch("irw.api._get_datasets", return_value=[_Dataset(table)]):
        irw.fetch("t", max_rows=5, columns=["id", "item", "resp"])
    assert table.calls == [
        {"max_results": 5, "variables": ["id", "item", "resp"]}
    ]
