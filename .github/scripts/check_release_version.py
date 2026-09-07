"""Assert that a release tag, pyproject.toml and irw.config.VERSION agree.

This exists because of a failure in the sibling R package. Rpkg's fix for the
rotated Redivis reference ids landed on `main` on 2026-09-04, two days after the
v1.1.2 tag, and DESCRIPTION was never bumped. `pak` and `remotes` decide whether
to upgrade by comparing version numbers, so every existing install reported
itself current and kept the broken code for three days (Rpkg#153). pip behaves
the same way: `pip install git+...` resolves the version, sees it installed and
skips even under `--upgrade`.

A fix under a stale version number is indistinguishable from no fix at all. This
script makes that state unrepresentable at release time rather than relying on
anyone remembering.

Usage:
    python .github/scripts/check_release_version.py            # literals only
    python .github/scripts/check_release_version.py v0.1.3     # literals + tag

Both files are parsed as text on purpose. `tomllib` is 3.11+, the package
supports 3.9, and nothing here should require the package to be installed.
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"
CONFIG = ROOT / "src" / "irw" / "config.py"


def pyproject_version() -> str:
    # Split on [project] first: [build-system] has no version key today, but a
    # bare search would happily match one added later.
    text = PYPROJECT.read_text()
    project = text.split("[project]", 1)[1]
    match = re.search(r'^version\s*=\s*"([^"]+)"', project, re.MULTILINE)
    if not match:
        sys.exit(f"no version found under [project] in {PYPROJECT}")
    return match.group(1)


def config_version() -> str:
    match = re.search(
        r'^VERSION:\s*str\s*=\s*"([^"]+)"', CONFIG.read_text(), re.MULTILINE
    )
    if not match:
        sys.exit(f"no VERSION found in {CONFIG}")
    return match.group(1)


def main() -> None:
    pyproject = pyproject_version()
    config = config_version()

    if pyproject != config:
        sys.exit(
            f"version mismatch: pyproject.toml says {pyproject}, "
            f"irw.config.VERSION says {config}. pip installs by the pyproject "
            f"version, so the number users see would not be the number pip "
            f"resolves."
        )

    if not re.fullmatch(r"\d+\.\d+\.\d+", pyproject):
        sys.exit(f"{pyproject} is not a plain X.Y.Z release number")

    if len(sys.argv) > 1 and sys.argv[1]:
        tag = sys.argv[1]
        if not tag.startswith("v"):
            sys.exit(f"tag {tag!r} does not start with 'v'")
        if tag[1:] != pyproject:
            sys.exit(
                f"tag {tag} does not match the package version {pyproject}. "
                f"Bump both version literals and merge that before tagging; do "
                f"not move the tag onto an unbumped commit."
            )
        print(f"ok: tag {tag} == pyproject == irw.config.VERSION == {pyproject}")
        return

    print(f"ok: pyproject == irw.config.VERSION == {pyproject} (no tag to check)")


if __name__ == "__main__":
    main()
