"""Regressions exercising package transforms behind the actual MCP client."""

import asyncio
from unittest.mock import patch

import pandas as pd
import pytest

import irw
from irw.mcp import IRWTools, create_server
from irw.operations.filter import InvalidFilterValue, _apply_numeric_filter, filter_tables
from test_fetch_pushdown import _Dataset, _Table
from test_mcp import FakeBackend, FakeSource


def wave_frame():
    return pd.DataFrame({
        'id': [101, 101, 202, 202], 'item': ['q1', 'q1', 'q1', 'q1'],
        'resp': [1, 2, 3, 4], 'wave': [1, 2, 1, 2],
    })


@pytest.mark.parametrize('columns', [['id', 'item', 'resp'], ['resp']])
def test_output_projection_preserves_waves_during_dedup(columns):
    table = _Table(wave_frame())
    with patch('irw.api._get_datasets', return_value=[_Dataset(table)]):
        result = irw.fetch('t', dedup=True, columns=columns)
    assert len(result) == 4
    assert list(result.columns) == columns
    assert table.calls[0]['variables'] is None


def test_projection_preserves_timestamp_skip():
    frame = wave_frame().rename(columns={'wave': 'date'})
    with patch('irw.api._get_datasets', return_value=[_Dataset(_Table(frame))]):
        with pytest.warns(UserWarning, match='timestamped'):
            result = irw.fetch('t', dedup=True, columns=['resp'])
    assert len(result) == 4


def test_missing_dedup_keys_are_reported():
    frame = pd.DataFrame({'resp': [1, 1]})
    with patch('irw.api._get_datasets', return_value=[_Dataset(_Table(frame))]):
        with pytest.warns(UserWarning, match='missing id or item'):
            assert len(irw.fetch('t', dedup=True)) == 2


def test_wide_selection_is_output_selection():
    frame = wave_frame().drop(columns='wave')
    frame['item'] = ['q1', 'q2', 'q1', 'q2']
    with patch('irw.api._get_datasets', return_value=[_Dataset(_Table(frame))]):
        result = irw.fetch('t', wide=True, columns=['id', 'q2'])
    assert result.to_dict('list') == {'id': [101, 202], 'q2': [2.0, 4.0]}


def test_real_transforms_through_mcp():
    pytest.importorskip('mcp')
    from mcp import Client

    class Backend(FakeBackend):
        def fetch_table(self, table_name, **kwargs):
            frame = wave_frame().drop(columns='wave')
            frame['item'] = ['q1', 'q2', 'q1', 'q2']
            with patch('irw.api._get_datasets', return_value=[_Dataset(_Table(frame))]):
                return irw.fetch(table_name, **kwargs)

    async def check():
        async with Client(create_server(Backend(), FakeSource())) as client:
            result = await client.call_tool('fetch_table', {
                'table_name': 'alpha_depression', 'wide': True, 'columns': ['id', 'q2'],
            })
            assert not result.is_error
            payload = result.structured_content['result']
            assert payload['rows'] == [[101, 2.0], [202, 4.0]]
            assert [c['name'] for c in payload['columns']] == ['id', 'q2']
    asyncio.run(check())


@pytest.mark.parametrize('value', ['50', [10, 20, 30], {}, True, [None], [50, 5], float('nan'), float('inf')])
def test_numeric_filter_never_silently_ignores_invalid_values(value):
    frame = pd.DataFrame({'n_items': [5, 50, 500]})
    with pytest.raises(InvalidFilterValue):
        _apply_numeric_filter(frame, 'n_items', value)
    with patch('irw.operations.filter.list_tables') as load:
        with pytest.raises(InvalidFilterValue):
            filter_tables([], n_items=value)
        load.assert_not_called()


@pytest.mark.parametrize('kwargs', [{'longitudinal': 'true'}, {'sample': []}, {'construct_type': 1}])
def test_filter_types_are_checked_before_io(kwargs):
    with patch('irw.operations.filter.list_tables') as load:
        with pytest.raises(InvalidFilterValue):
            filter_tables([], **kwargs)
        load.assert_not_called()


def test_punctuation_query_is_not_catalogue_browse():
    from irw.mcp import IRWMCPError
    with pytest.raises(IRWMCPError, match='searchable'):
        IRWTools(FakeBackend(), FakeSource()).search_tables('!!!')
