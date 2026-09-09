"""Processing sources must be textual and unambiguous."""

import json
import pytest

from irw.mcp import _match_scripts, _processing_header
from irw.mcp import _parse_issue_list
from test_mcp import FakeSource
from unittest.mock import patch
from irw.mcp import _http_get_text


class StreamingResponse:
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield from self.chunks


def test_public_download_stops_at_byte_limit_and_closes_response():
    response = StreamingResponse([b"abc", b"def", b"unread"])
    with patch("requests.get", return_value=response) as get:
        with patch("irw.mcp.PUBLIC_SOURCE_MAX_BYTES", 5):
            with pytest.raises(ValueError, match="download limit"):
                _http_get_text("https://example.test/source")
    assert response.closed
    assert get.call_args.kwargs["stream"] is True
    assert get.call_args.kwargs["timeout"] == (5, 15)


def test_public_download_has_total_deadline():
    response = StreamingResponse([b"abc"])
    with patch("requests.get", return_value=response):
        with patch("time.monotonic", side_effect=[0, 31]):
            with pytest.raises(TimeoutError, match="deadline"):
                _http_get_text("https://example.test/source")
    assert response.closed


def test_public_download_decodes_split_unicode_and_bom():
    response = StreamingResponse([b"\xef\xbb\xbf\xc3", b"\xa9"])
    with patch("requests.get", return_value=response):
        assert _http_get_text("https://example.test/source") == "\u00e9"


def test_yaml_issue_styles_and_quoted_names():
    text = '''issues <- yaml.load(r"---(
- table: "Study_2026"
  issue: "Quoted: note"
- table: Study_2026
  issue: >-
    Folded
    paragraph.
- table: other
  issue: |-
    First paragraph.

    Second paragraph.
)---")'''
    result = _parse_issue_list(text)
    assert result["study_2026"] == ["Quoted: note", "Folded paragraph."]
    assert result["other"] == ["First paragraph.\n\nSecond paragraph."]


@pytest.mark.parametrize("text", ["No issue block", 'issues <- yaml.load(r"---(null)---")',
    'issues <- yaml.load(r"---([{table: a, issue: 42}])---")'])
def test_unrecognized_or_invalid_issue_data_is_not_empty_success(text):
    with pytest.raises(ValueError):
        _parse_issue_list(text)


def test_processing_sources_use_one_immutable_revision():
    source = FakeSource()
    source.data_scripts()
    source.read_script("data/alpha_depression.py")
    source.validator_overrides()
    assert sum(url.endswith("commits/main") for url in source.calls) == 1
    for url in source.calls:
        if not url.endswith("commits/main"):
            assert "a" * 40 in url


def test_binary_files_cannot_be_processing_scripts():
    assert _match_scripts("study_2026", ["data/study_2026.xlsx", "data/study_2026.RData"]) == ([], "none")


def test_family_matching_requires_separator():
    assert _match_scripts("study_2026", ["data/study_20260.py"]) == ([], "none")
    assert _match_scripts("study_2026_phq", ["data/study_2026.py"]) == (["data/study_2026.py"], "prefix")


def test_multiple_family_matches_are_not_arbitrarily_selected():
    paths = ["data/study_2026_a.py", "data/study_2026_b.py"]
    assert _match_scripts("study_2026", paths) == (paths, "ambiguous")


def test_multiple_exact_scripts_are_also_ambiguous():
    paths = ["data/study_2026.R", "data/study_2026.py"]
    assert _match_scripts("study_2026", paths) == (paths, "ambiguous")


def test_notebook_notes_exclude_code_and_outputs():
    notebook = {"cells": [
        {"cell_type": "markdown", "source": ["# Decisions\n", "IDs use row numbers."]},
        {"cell_type": "code", "source": ["# Retain missing values\n", "print('not documentation')"],
         "outputs": [{"text": "not documentation either"}]},
    ]}
    header, truncated = _processing_header("data/study.ipynb", json.dumps(notebook))
    assert "IDs use row numbers" in header
    assert "Retain missing values" in header
    assert "not documentation" not in header
    assert not truncated
