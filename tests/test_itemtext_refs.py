"""Tests for config-driven item text shard discovery (no Redivis network calls).

Redivis caps a dataset at 1000 tables, so item text is a shard list the same way
response data is. `ITEMTEXT_REFS` has one entry today; these tests drive the
two-shard path so the cutover is a config edit rather than a code change.
"""

from unittest.mock import MagicMock, patch

import pytest

from irw.config import ITEMTEXT_REFS
from irw.utils.redivis.cache import metadata_cache
from irw.utils.redivis import item_text as it


@pytest.fixture(autouse=True)
def clear_cache():
    metadata_cache.clear()
    yield
    metadata_cache.clear()


class FakeTable:
    """`name` is a reserved constructor kwarg on MagicMock, so use a real object."""

    def __init__(self, name):
        self.name = name


def fake_shard(ds_id, bases):
    ds = MagicMock()
    ds._id = ds_id
    ds.list_tables.return_value = [FakeTable(f"{b}__items") for b in bases]
    return ds


def test_itemtext_refs_is_a_list_oldest_to_newest():
    assert isinstance(ITEMTEXT_REFS, tuple)
    assert ITEMTEXT_REFS[0] == ("datapages", "irw_text:07b6")


def test_cache_key_changes_when_a_shard_is_added(monkeypatch):
    before = it._itemtext_datasets_cache_key()
    monkeypatch.setattr(
        "irw.utils.redivis.item_text.ITEMTEXT_REFS",
        (("datapages", "irw_text:07b6"), ("datapages", "irw_text_2:zzzz")),
    )
    after = it._itemtext_datasets_cache_key()
    assert before != after
    assert "irw_text_2:zzzz" in after


@patch("irw.utils.redivis.item_text._init_datasets_from_refs")
def test_shards_are_searched_newest_first(mock_init, monkeypatch):
    monkeypatch.setattr(
        "irw.utils.redivis.item_text.ITEMTEXT_REFS",
        (("u", "irw_text:1"), ("u", "irw_text_2:2")),
    )
    old, new = fake_shard("irw_text:1", []), fake_shard("irw_text_2:2", [])
    mock_init.return_value = [old, new]          # config order, oldest first
    assert it._get_itemtext_datasets() == [new, old]


@patch("irw.utils.redivis.item_text._init_datasets_from_refs")
def test_listing_is_the_union_across_shards(mock_init, monkeypatch):
    monkeypatch.setattr(
        "irw.utils.redivis.item_text.ITEMTEXT_REFS",
        (("u", "irw_text:1"), ("u", "irw_text_2:2")),
    )
    mock_init.return_value = [
        fake_shard("irw_text:1", ["alpha", "shared"]),
        fake_shard("irw_text_2:2", ["shared", "Beta"]),
    ]
    assert it._list_itemtext_tables() == {"alpha", "shared", "beta"}


@patch("irw.utils.redivis.item_text._init_datasets_from_refs")
def test_unavailable_shard_is_skipped_not_fatal(mock_init, monkeypatch):
    # A freshly created shard has no released version, so a read-only token
    # cannot open it. skip_unavailable is what keeps item text working anyway.
    monkeypatch.setattr(
        "irw.utils.redivis.item_text.ITEMTEXT_REFS",
        (("u", "irw_text:1"), ("u", "irw_text_2:2")),
    )
    mock_init.return_value = [fake_shard("irw_text:1", ["alpha"])]
    assert it._list_itemtext_tables() == {"alpha"}
    _args, kwargs = mock_init.call_args
    assert kwargs.get("skip_unavailable") is True


@patch("irw.utils.redivis.item_text._init_datasets_from_refs")
def test_fetch_routes_to_the_shard_that_has_the_table(mock_init, monkeypatch):
    """The whole point: never ask a remembered "the" dataset.

    `_search_datasets` skips a not-found and moves on, so a table living only in
    the older shard is still reachable once a newer shard exists.
    """
    monkeypatch.setattr(
        "irw.utils.redivis.item_text.ITEMTEXT_REFS",
        (("u", "irw_text:1"), ("u", "irw_text_2:2")),
    )
    old, new = fake_shard("irw_text:1", ["only_old"]), fake_shard("irw_text_2:2", [])
    mock_init.return_value = [old, new]

    def get_table(ds, name):
        if ds is new:
            raise Exception("Not found: datapages.irw_text_2:2:v1_0.only_old__items")
        return f"table<{name}@{ds._id}>"

    with patch("irw.utils.redivis.tables._get_table", side_effect=get_table):
        assert it._get_itemtext_table("only_old") == "table<only_old__items@irw_text:1>"


@patch("irw.utils.redivis.item_text._init_datasets_from_refs")
def test_fetch_prefers_the_newest_shard_holding_the_table(mock_init, monkeypatch):
    monkeypatch.setattr(
        "irw.utils.redivis.item_text.ITEMTEXT_REFS",
        (("u", "irw_text:1"), ("u", "irw_text_2:2")),
    )
    old, new = fake_shard("irw_text:1", ["shared"]), fake_shard("irw_text_2:2", ["shared"])
    mock_init.return_value = [old, new]
    with patch("irw.utils.redivis.tables._get_table",
               side_effect=lambda ds, name: f"table<{name}@{ds._id}>"):
        assert it._get_itemtext_table("shared") == "table<shared__items@irw_text_2:2>"


@patch("irw.utils.redivis.item_text._init_datasets_from_refs")
def test_mixed_case_name_resolves_to_the_stored_lowercase_table(mock_init, monkeypatch):
    """Item text is lower-cased on upload; response tables often are not.

    `HEARD_Roch_2022_K6` is stored as `heard_roch_2022_k6__items`. Redivis'
    lookup happens to be case-insensitive, so this pins the resolution we
    perform ourselves rather than the server behaviour we would otherwise be
    depending on.
    """
    monkeypatch.setattr("irw.utils.redivis.item_text.ITEMTEXT_REFS",
                        (("u", "irw_text:1"),))
    mock_init.return_value = [fake_shard("irw_text:1", ["heard_roch_2022_k6"])]

    assert it._itemtext_name_index()["heard_roch_2022_k6"] == "heard_roch_2022_k6"
    asked = []
    with patch("irw.utils.redivis.tables._get_table",
               side_effect=lambda ds, name: asked.append(name) or "tbl"):
        assert it._get_itemtext_table("HEARD_Roch_2022_K6") == "tbl"
    assert asked == ["heard_roch_2022_k6__items"], asked


@patch("irw.utils.redivis.item_text._init_datasets_from_refs")
def test_name_index_prefers_the_newest_shard(mock_init, monkeypatch):
    monkeypatch.setattr("irw.utils.redivis.item_text.ITEMTEXT_REFS",
                        (("u", "irw_text:1"), ("u", "irw_text_2:2")))
    mock_init.return_value = [fake_shard("irw_text:1", ["Shared"]),
                              fake_shard("irw_text_2:2", ["shared"])]
    # Reversed to newest-first inside _get_itemtext_datasets, so shard 2 wins.
    assert it._itemtext_name_index()["shared"] == "shared"


class _Handle:
    def __init__(self, name, missing):
        self.name = name
        self._missing = missing

    def get(self):
        if self._missing:
            raise Exception(f"Not found: datapages.irw_text_2:2:v1_1.{self.name}")


class _Shard:
    """A dataset whose `table()` hands back a handle that only fails on get()."""

    def __init__(self, ds_id, present):
        self._id = ds_id
        self._present = set(present)

    def list_tables(self):
        return [FakeTable(n) for n in self._present]

    def table(self, name):
        return _Handle(name, name not in self._present)


def test_real_get_table_surfaces_not_found_instead_of_a_phantom_handle():
    """Regression: `_get_table` swallowed the not-found from `tbl.get()`, so the
    newest shard claimed every name and 732 tables lost their item text."""
    from irw.utils.redivis.tables import _get_table

    new = _Shard("irw_text_2:2", [])
    old = _Shard("irw_text:1", ["only_old__items"])
    with pytest.raises(Exception, match="Not found"):
        _get_table(new, "only_old__items")
    assert _get_table(old, "only_old__items").name == "only_old__items"


@patch("irw.utils.redivis.item_text._init_datasets_from_refs")
def test_lookup_falls_through_a_newer_shard_using_the_real_get_table(mock_init, monkeypatch):
    """Same scenario, end to end, without mocking `_get_table` away -- the mock in
    the routing test above is exactly what hid the bug."""
    monkeypatch.setattr(
        "irw.utils.redivis.item_text.ITEMTEXT_REFS",
        (("u", "irw_text:1"), ("u", "irw_text_2:2")),
    )
    old = _Shard("irw_text:1", ["only_old__items"])
    new = _Shard("irw_text_2:2", ["newer__items"])
    mock_init.return_value = [old, new]
    handle = it._get_itemtext_table("only_old")
    assert handle.name == "only_old__items"
    assert it._get_itemtext_table("newer").name == "newer__items"
    assert it._get_itemtext_table("absent") is None
