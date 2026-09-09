"""Wire contracts loaded only with the optional MCP server."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model


class Result(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source: str
    warnings: List[str]
    irw_version: Optional[str]
    irw_released_at: Optional[str]
    retrieved_at: str
    version_status: Literal["observed_manifest", "unavailable"]
    data_pinned: Literal[False]


class Page(Result):
    offset: int
    limit: int
    returned: int
    next_offset: Optional[int]
    has_more: bool


class Search(Page):
    tables: List[Dict[str, Any]]
    total: int
    n_untagged_in_catalogue: Optional[int]
    filters_applied: Dict[str, Any]
    caveats: List[str]
    truncated: Optional[bool] = None


class Filter(Result):
    filter: str
    description: Optional[str]
    kind: Literal["numeric", "categorical", "boolean", "unknown"]
    available_values: Optional[List[Any]]
    value_counts: Optional[Dict[str, int]]
    summary: Optional[Dict[str, Any]]
    truncated: bool


class Description(Result):
    table: str
    metadata: Dict[str, Any]
    table_schema: Any = Field(alias="schema")


class Column(BaseModel):
    name: str
    dtype: str


class Fetch(Page):
    table: str
    columns: List[Column]
    rows: List[List[Any]]
    total_rows: Optional[int]
    total_rows_estimate: Optional[int] = None
    total_rows_note: Optional[str] = None
    truncated: bool
    wide: bool
    dedup: bool


class Itemtext(Page):
    table: str
    available: bool
    availability_status: Literal["available", "unavailable", "fetch_failed"]
    rights: Dict[str, Any]
    items: List[List[Any]]
    columns: List[Column]
    total_items: Optional[int]
    truncated: bool
    disclaimer: str


class Collections(Page):
    collections: List[Dict[str, Any]]
    total: int
    truncated: Optional[bool] = None


class Citation(Result):
    table: str
    available: bool
    bibtex: List[str]


class Script(BaseModel):
    path: str
    url: str
    header: str
    truncated: bool


class Notes(Result):
    table: str
    match: Literal["exact", "prefix", "ambiguous", "none"]
    candidate_paths: List[str]
    scripts: List[Script]
    validator_overrides: List[Dict[str, str]]
    guides: Dict[str, str]


OUTPUT_MODELS = {
    name: create_model(name + "Output", result=(model, ...))
    for name, model in {
        "search_tables": Search, "describe_filter": Filter,
        "describe_table": Description, "fetch_table": Fetch,
        "get_itemtext": Itemtext, "list_collections": Collections,
        "get_citation": Citation, "get_processing_notes": Notes,
    }.items()
}
