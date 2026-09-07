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


@pytest.fixture
def every_loader_patched(monkeypatch):
    """Every metadata loader describe_filter() can reach, with no Redivis
    behind any of them, so the whole filter list can be walked offline."""
    base = pd.DataFrame({"name": ["t1", "t2", "t3"]})
    metadata = pd.DataFrame(
        {
            "table": ["t1", "t2", "t3"],
            "n_responses": [10, 20, 30],
            "n_categories": [2, 5, None],
            "n_participants": [1, 2, 3],
            "n_items": [10, 10, 10],
            "responses_per_participant": [10.0, 10.0, 10.0],
            "responses_per_item": [1.0, 2.0, 3.0],
            "density": [1.0, 0.5, 0.2],
            "variables": ["id| item| resp", "id| item| resp| wave", "id| item| resp"],
            "longitudinal": [False, True, False],
        }
    )
    tags = pd.DataFrame(
        {
            "table": ["t1", "t2", "t3"],
            "age_range": ["Adult", "Child", None],
            "child_age__for_child_focused_studies_": [None, "5-10", None],
            "sample": ["Educational", "Clinical", None],
            "construct_type": ["Cognitive", "Affective/mental health", None],
            "measurement_tool": ["Survey", "Survey", None],
            "item_format": ["Likert", "Likert", None],
            "primary_language_s_": ["eng", "eng", None],
            "construct_name": ["Big Five", "PHQ-9", None],
        }
    )
    biblio = pd.DataFrame(
        {"table": ["t1", "t2", "t3"], "Derived_License": ["CC BY 4.0", "CC0", None]}
    )
    registry = pd.DataFrame(
        {"collection": ["depression", "rct", "empty_rule"], "kind": ["construct", "design", "construct"]}
    )
    members = pd.DataFrame(
        {"table": ["t2", "t1", "t2"], "collection": ["depression", "rct", "rct"]}
    )
    monkeypatch.setattr(filter_info, "_build_base_table_list", lambda ds: base.copy())
    monkeypatch.setattr(filter_info, "get_metadata_table", lambda: metadata.copy())
    monkeypatch.setattr(filter_info, "get_tags_table", lambda: tags.copy())
    monkeypatch.setattr(filter_info, "get_biblio_table", lambda: biblio.copy())
    monkeypatch.setattr(filter_info, "get_collections_table", lambda: registry.copy())
    monkeypatch.setattr(filter_info, "get_collection_members_table", lambda: members.copy())
    monkeypatch.setattr(filter_info, "_list_itemtext_tables", lambda: {"t1"})
    return None


def test_every_advertised_filter_can_be_described(every_loader_patched):
    """describe_filter() is how a caller learns what a filter takes. On the
    shipped package three of the twenty names in get_filters() -- n_categories,
    has_item_text and collection -- came back None, so the tool that tells you
    the valid inputs denied that filters filter() accepts exist."""
    missing = [
        name for name in filter_info.get_filters()
        if filter_info.describe_filter([], name) is None
    ]
    assert missing == []


def test_n_categories_is_read_from_the_metadata_table(every_loader_patched):
    """It was looked up in the tags table, which has never carried it."""
    values = filter_info.describe_filter([], "n_categories")["values"]
    assert values["min"] == 2.0
    assert values["max"] == 5.0
    assert values["count"] == 2
    assert values["null_count"] == 1


def test_has_item_text_counts_come_from_the_text_shards(every_loader_patched):
    values = filter_info.describe_filter([], "has_item_text")["values"]
    assert values == {True: 1, False: 2}


def test_collection_lists_the_registry_with_live_counts(every_loader_patched):
    """Every registry entry appears, including one with no live members, so
    a caller sees the vocabulary filter(collection=...) accepts and not just
    the collections that happen to be populated."""
    values = filter_info.describe_filter([], "collection")["values"]
    assert list(values.index) == ["rct", "depression", "empty_rule"]
    assert values["rct"] == 2
    assert values["depression"] == 1
    assert values["empty_rule"] == 0


def test_collection_survives_a_malformed_registry_or_members_table(monkeypatch):
    """A duplicated or blank registry row is one collection or none, and a
    members table without the expected columns means zero live counts, not
    a KeyError from the function that lists what can be filtered on."""
    base = pd.DataFrame({"name": ["t1", "t2"]})
    registry = pd.DataFrame({"collection": ["rct", None, "rct", "math"]})
    monkeypatch.setattr(filter_info, "_build_base_table_list", lambda ds: base.copy())
    monkeypatch.setattr(filter_info, "get_collections_table", lambda: registry.copy())
    monkeypatch.setattr(
        filter_info, "get_collection_members_table", lambda: pd.DataFrame({"foo": [1]})
    )
    values = filter_info.describe_filter([], "collection")["values"]
    assert list(values.index) == ["math", "rct"]
    assert values.tolist() == [0, 0]
