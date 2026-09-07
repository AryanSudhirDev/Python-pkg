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
