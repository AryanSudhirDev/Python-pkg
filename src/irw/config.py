"""Configuration constants for IRW package.

To add another main IRW Redivis warehouse, append a (user, dataset_ref) tuple to
MAIN_REFS below. All package operations (list_tables, fetch, filter, download,
info, etc.) discover tables across every entry in MAIN_REFS automatically — no
other code changes are required. Item text works the same way via ITEMTEXT_REFS.
Both are lists because Redivis caps a dataset at 1000 tables.

See docs/DEVELOPERS.md for the full contributor checklist and test commands.
"""

from typing import Tuple, ClassVar

# Main IRW response-data warehouses on Redivis.
# Declared oldest-to-newest; fetch searches them newest-first, so a table present
# in more than one warehouse resolves to its most recent copy.
MAIN_REFS: ClassVar[Tuple[Tuple[str, str], ...]] = (
    ("datapages", "item_response_warehouse:as2e"),
    ("datapages", "item_response_warehouse_2:epbx"),
    ("datapages", "item_response_warehouse_3:5xaj"),
    ("datapages", "item_response_warehouse_4:980f"),
    ("datapages", "item_response_warehouse_5:3ykx"),
    ("datapages", "item_response_warehouse_6:fpe6"),
)
SIM_REF: ClassVar[Tuple[str, str]] = ("datapages", "irw_simsyn:0btg")
COMP_REF: ClassVar[Tuple[str, str]] = ("datapages", "irw_competitions:cmd7")

# The nominal-response warehouse. Unlike sim and comp, this source is tagged --
# Rpkg's .irw_tag_sources is c("core", "nom") -- but it has no collections, so
# collections() must keep erroring for it rather than returning an empty frame.
NOM_REF: ClassVar[Tuple[str, str]] = ("datapages", "irw_nominal:614n")

# Main IRW metadata dataset references (only for main IRW)
META_REF: ClassVar[Tuple[str, str]] = ("datapages", "irw_meta:bdxt")

# IRW item text shards on Redivis.
# Redivis caps a dataset at 1000 tables, so item text shards the way response
# data does. Declared oldest-to-newest; lookups search newest-first, so a table
# present in more than one shard resolves to its most recent copy.
# See Rpkg/inst/developer/warehouses.md for the checklist to add one.
ITEMTEXT_REFS: ClassVar[Tuple[Tuple[str, str], ...]] = (
    ("datapages", "irw_text:07b6"),
    ("datapages", "irw_text_2:ae47"),
)

# Main IRW metadata table references.
#
# Address these by BARE NAME, never by the qualified `name:refid` form. Redivis
# mints a fresh reference id for every table on every release, so a hardcoded id
# stops resolving inside the next version: v19.3 had metadata:h5gs, v21.0 has
# metadata:hb7q. The package always asks for the latest version tag, so a stale
# id yields "Not found" and every metadata-dependent call degrades -- info()
# raises, and list_tables(include_metadata=True) and filter() silently fall back
# to names only. The redivis client warns you to prefer the qualified reference;
# for these tables, do not. (The R package addresses tables by name and was
# never affected.)
META_TABLES: ClassVar[dict[str, str]] = {
    "metadata": "metadata",
    "tags": "tags",
    "biblio": "biblio",
    # Collections (issue #1633). `collections` is the registry, one row per
    # collection; `collection_members` is long, one row per (table, collection).
    "collections": "collections",
    "collection_members": "collection_members",
}

# Package metadata
PACKAGE_NAME: str = "irw"
# The one place the package version is written in Python. `irw.__version__`
# re-exports this, and tests/test_version_string.py asserts it matches
# pyproject.toml. Bump it whenever a change needs to reach existing installs:
# the briefing's `pip install git+...` line resolves the version, sees it
# already installed and SKIPS -- even with --upgrade -- so an unchanged version
# string means users silently keep the old code.
VERSION: str = "0.1.4"
DESCRIPTION: str = "A Python package for the Item Response Warehouse (IRW)"

__all__ = [
    "MAIN_REFS",
    "SIM_REF", 
    "COMP_REF",
    "NOM_REF",
    "META_REF",
    "ITEMTEXT_REFS",
    "META_TABLES",
    "PACKAGE_NAME",
    "VERSION",
    "DESCRIPTION",
]
