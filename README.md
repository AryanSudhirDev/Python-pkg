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
MCP-capable research assistant (issue ben-domingue/irw#1713). Seven tools:

| Tool | What it does | Costs Redivis quota? |
|---|---|---|
| `search_tables` | catalogue search with collection / variable / licence / longitudinal / item-text filters; every record says whether it is `tagged` | no |
| `describe_table` | statistics, tags, bibliography for one table | no |
| `get_processing_notes` | the header of the script that built the table, from the IRW GitHub repository: whether `id` links across waves, what a `cov_*` means, what was excluded | no (no login either) |
| `fetch_table` | a bounded page of rows; refuses tables above 1,000,000 responses **before** downloading | yes, the whole table |
| `get_itemtext` | a bounded page of item text with a `rights` object: response-data licence, the instrument-rights rule, and the table's public notes | small |
| `list_collections` | the labelled collections | no |
| `get_citation` | BibTeX for the original data producers | no |

Every response carries `irw_version` and `irw_released_at`, the citable
version of the corpus, so anything an assistant produces can be pinned. The
server's own version is the `irw` package version, not the data.

The MCP server requires Python 3.10 or newer because the current MCP SDK does.
It does not make OpenAI calls and does not require an OpenAI key; the host
application is responsible for the model. Redivis authentication is still
handled by the `irw` package.

```bash
python -m pip install "irw[mcp]"
```

Authenticate with Redivis **before** first use. The Redivis SDK's interactive
browser login cannot complete inside an MCP server, so the server refuses to
start a call without credentials (error code `authentication_required`) rather
than hanging. Either run one call in a regular terminal --
`python -c "import irw; irw.list_tables()"` -- which caches credentials in
`~/.redivis`, or set `REDIVIS_API_TOKEN` in the MCP host's environment.

Configure an MCP host to start this local process:

```json
{
  "command": "irw-mcp",
  "args": []
}
```

`fetch_table` and `get_itemtext` return bounded pages (default 100 rows,
`offset` for the next page, `has_more` and `truncated` fields; maximum 1,000
response rows and 500 item-text rows). Paging happens locally: `irw.fetch()`
has no row argument, so a fetch downloads the whole table against the
account's 30-day Redivis export quota. That is why the size guard is a
pre-check on the catalogue's `n_responses` (error `table_too_large`) rather
than a truncation after the download -- the same 1,000,000-response rule the
agents briefing (`llms.txt`) gives researchers.

The tool descriptions carry the traps the briefing documents: tags are
incomplete, so an untagged table is not a non-match; `longitudinal` is a grep
of the variable string; response direction is not recoded across items;
duplicate id-item rows can be real data; and the deposit licence is not an
instrument licence. Item text may be reconstructed or incomplete; verify it
against the original source, and read `rights.public_notes` (withdrawn
wording, machine translations, known mismatches) before using it.

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
