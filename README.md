# `irw`: A Python Package for the Item Response Warehouse

This repository hosts the Python package `irw`, which provides programmatic access to the [Item Response Warehouse (IRW)](https://itemresponsewarehouse.org/), an open repository of harmonized item response data hosted on Redivis.

Project map: [`ARCHITECTURE.md`](https://github.com/ben-domingue/irw/blob/main/ARCHITECTURE.md) in `ben-domingue/irw` — which repo owns
what, where the data lives, and which document is authoritative when two disagree.

## Installation

**Recommended: Use a virtual environment** (prevents conflicts with other packages):

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
python -m pip install --upgrade pip
python -m pip install irw
```

To upgrade an existing install:

```bash
python -m pip install --upgrade irw
```

### Development install

To run against unreleased code on `main`:

```bash
python -m pip install "git+https://github.com/itemresponsewarehouse/Python-pkg.git"
```

Note that this install does not upgrade cleanly: pip resolves the version from
the clone, sees it already installed and skips, even under `--upgrade`. Use it
when you want a specific commit, not as a way to stay current.

### Requirements

- Python 3.9 or higher
- pip 

If you encounter any installation issues, please [open an issue](https://github.com/itemresponsewarehouse/Python-pkg/issues).

## IMPORTANT: Redivis Authentication

The IRW tables are hosted on [Redivis](https://redivis.com), a data management platform. To access these datasets, you'll need to:

1. Have a Redivis account (create one at <https://redivis.com/?createAccount> if you don't have one).

2. Authenticate using the Redivis Python Client:
   1. When you first use a function in `irw` that connects to Redivis (e.g. `list_tables()`), a browser window will open, prompting you to sign in to your Redivis account.
   2. After signing in, click **Allow** to grant access for the Redivis Python Client.
   3. Once authentication is successful, close the browser window. You will see the message "Authentication was successful" in console.

**Note:** You only need to authenticate once per session. For detailed instructions, refer to the [Redivis Python Client documentation](https://apidocs.redivis.com/client-libraries/redivis-python).

## Usage Examples

See the `examples/` directory:
- `example.py` - Complete workflow example
- `available_methods.md` - Reference guide for all available methods

Example workflow:
```python
import irw

# Get database information
irw.info()

# View available tables
tables = irw.list_tables()
tables_with_metadata = irw.list_tables(include_metadata=True)

# Get table info
irw.info("agn_kay_2025")  # Table metadata

# Fetch a table
df = irw.fetch("agn_kay_2025")
# Convert to response matrix
resp_matrix = irw.long2resp(df)

# Explore available filters
filters = irw.get_filters()  # Returns list of filter names
irw.describe_filter('construct_type')  # Get values for a specific filter

# Filter and fetch tables
filtered = irw.filter(n_responses=[1000, None], construct_type="Affective/mental health")
dfs = irw.fetch(filtered)

# Get BibTeX citation
irw.save_bibtex("agn_kay_2025")  # Returns BibTeX entry
# Download table
irw.download("agn_kay_2025", path="data.csv")

# Browse collections: labelled groupings of tables
irw.collections()                     # all collections, with coverage and table counts
tabs = irw.collection("depression")   # the table names in one collection
irw.collection_members(tables="frac20")  # which collections is this table in?

# Which version of IRW is this? (cite this number)
irw.version()                         # newest IRW version and its dataset pins
irw.version("2026-08-01")             # what was live on that date
```

## MCP server

IRW can run as a local, read-only Model Context Protocol server for an
MCP-capable research assistant (issue ben-domingue/irw#1713). Eight tools:

| Tool | What it does | Costs Redivis quota? |
|---|---|---|
| `search_tables` | free-text search plus `irw.filter()`'s own filters, passed straight through; a summary card per hit, each saying whether it is `tagged` | no |
| `describe_filter` | what one filter means and which values it takes | no |
| `describe_table` | statistics, tags, bibliography for one table | no |
| `get_processing_notes` | the header of the script that built the table, from the IRW GitHub repository: whether `id` links across waves, what a `cov_*` means, what was excluded | no (no login either) |
| `fetch_table` | a bounded page of rows, bounded on the wire | a page |
| `get_itemtext` | a bounded page of item text with a `rights` object: response-data licence, the instrument-rights rule, and the table's public notes | small |
| `list_collections` | the labelled collections | no |
| `get_citation` | BibTeX for the original data producers | no |

`search_tables` does not define its own filter vocabulary. It takes a `filters`
object, checks the names against `irw.get_filters()` and hands it to
`irw.filter()`, so the server cannot drift from the package and cannot offer a
smaller filter set than it has. `describe_filter` exposes
`irw.describe_filter()` so an assistant can look up a tag vocabulary rather
than guess at a spelling. One deliberate difference from `irw.filter()`: no
default `density` filter is applied, because its `[0.5, 1]` default silently
removes sparse tables from a search nobody asked to be about density.

Every response carries an observed `irw_version` and `irw_released_at` when
the public manifest is available, plus `retrieved_at`. This does not pin reads
from independently versioned Redivis tables: `data_pinned` is explicitly false.
Processing-script links do use an immutable Git commit. The server's own
version is the `irw` package version, not the data.

The MCP server requires Python 3.10 or newer because the current MCP SDK does.
It does not make OpenAI calls and does not require an OpenAI key; the host
application is responsible for the model. Redivis authentication is still
handled by the `irw` package.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install ".[mcp]"
.venv/bin/python -c "import sys; from pathlib import Path; print(Path(sys.executable).with_name('irw-mcp'))"
```

These commands install the local checkout. For an index installation, use
`"irw[mcp]>=0.1.4"` only after that release is published; 0.1.3 does not include
this server. On Windows, use `.venv\Scripts\python.exe` instead.

Authenticate with Redivis **before** first use. The Redivis SDK's interactive
browser login cannot complete inside an MCP server, so the server refuses to
start a call without credentials (error code `authentication_required`) rather
than hanging. Either run one call in a regular terminal --
`python -c "import irw; irw.list_tables()"` -- which caches credentials in
`~/.redivis`, or set `REDIVIS_API_TOKEN` in the MCP host's environment.

Configure an MCP host to start the installed launcher by its absolute path.
Add the `irw` entry to any existing `mcpServers` object; do not replace other
servers. Tokens set in a terminal are not necessarily inherited by a desktop
application: configure the host's environment without committing secrets.

```json
{
  "mcpServers": {
    "irw": {
      "command": "/absolute/path/to/.venv/bin/irw-mcp",
      "args": []
    }
  }
}
```

`fetch_table` and `get_itemtext` return bounded pages (default 100 rows,
`offset` for the next page, `has_more` and `truncated` fields; maximum 1,000
response rows and 500 item-text rows). These are storage-order previews, not
random samples. Follow `next_offset`; a full page does not prove another row
exists. Ordinary response-data windows are bounded on the wire:
`fetch_table` passes `max_rows` and `columns` to `irw.fetch()`, which hands
both to Redivis's read session. Each page reads `offset + limit` rows, including
the prefix discarded for later pages; this sum is capped at 10,000. Item text
may require fetching an entire text shard before paging. Rows come back
columnar -- `columns` names the fields and each entry of `rows` is a list of
values in that order -- which is about half the response size of repeating
every column name on every row.

Because only the window is downloaded, `total_rows` is `null` and `has_more`
reports whether the window came back full; `total_rows_estimate` carries the
catalogue's response count. Claiming the window size as the table's size would
be the more convenient answer and the wrong one.

`wide=true` and `dedup=true` are the exception. Both are computed over the
whole table -- dedup can only drop the duplicates it can see, and the reshape
uses whatever rows it is given -- so they cannot be bounded to a page. Those
calls download the table, say so in `warnings`, and are refused above
1,000,000 responses (error `table_too_large`) or when size is unknown
(`table_size_unknown`), following the size threshold the agents
briefing (`llms.txt`) gives researchers.

Responses have a 256 KiB application-JSON limit. Pages shrink only by complete
records and include `next_offset`; indivisible oversized records return an
error, not shortened wording or citations. `columns` selects output columns,
including item columns after `wide=true` reshaping. Numeric filter values must
be finite numbers or one/two-element numeric lists; malformed filters fail.

Processing notes are best-effort header extraction, not a complete execution
trace or proof of suitability. Ambiguous script families are reported without
choosing a script. Item text distinguishes `available`, `unavailable`, and
`fetch_failed`; an expected text retrieval failure is not an empty dataset.

The tool descriptions carry the traps the briefing documents: tags are
incomplete, so an untagged table is not a non-match; `longitudinal` is a grep
of the variable string; response direction is not recoded across items;
duplicate id-item rows can be real data; and the deposit licence is not an
instrument licence. The per-filter caveats come from the package's own
`FILTER_DESCRIPTIONS`, so a caveat added there reaches an assistant without
anyone editing the server. Item text may be reconstructed or incomplete;
verify it against the original source, and read `rights.public_notes`
(withdrawn wording, machine translations, known mismatches) before using it.

`rights.response_data_license` is IRW's **Derived License** for the response
data. IRW records an `Original License` for the source deposit separately, and
it is not carried in the metadata the package reads, so `rights` reports it as
`null` with a note rather than passing the derived licence off as an answer
about the source.

The process uses stdio, so it is intended to be launched by a local MCP host.
It is not a hosted HTTP endpoint and cannot be called directly by a static
GitHub Pages browser widget.

## Development

### Setting up Development Environment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/itemresponsewarehouse/Python-pkg.git
   cd Python-pkg
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**:
   ```bash
   pip install -e .
   ```

## Feedback and Contributions

If you encounter issues or have suggestions for improving `irw`, please submit them on the [GitHub Issues page](https://github.com/itemresponsewarehouse/Python-pkg/issues). Contributions are welcome!
