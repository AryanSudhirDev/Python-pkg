"""Run isolated core/MCP compatibility checks and retain a JSON audit.

Usage: python tests/run_compatibility_matrix.py /tmp/irw-compatibility
Requires uv. No credentials or live service calls are used.
"""

import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = []
    cases = [(version, "core") for version in ("3.9", "3.10", "3.11", "3.12", "3.13")]
    cases += [(version, sdk) for version in ("3.10", "3.13") for sdk in ("2.1.0", "latest")]
    env = dict(os.environ)
    env.pop("RUN_REDIVIS_TESTS", None)
    env.pop("REDIVIS_API_TOKEN", None)
    for version, sdk in cases:
        name = version + "-" + sdk
        directory = output / name
        python = directory / "bin" / "python"
        commands = [
            ["uv", "venv", "--python", version, str(directory)],
            ["uv", "pip", "install", "--python", str(python),
             str(root) + ("[mcp]" if sdk != "core" else ""), "pytest", "PyYAML>=6,<7"],
        ]
        if sdk not in ("core", "latest"):
            commands.append(["uv", "pip", "install", "--python", str(python), "mcp==" + sdk])
        commands.append([str(python), "-m", "pytest", "tests/", "-q"])
        case = {"python": version, "sdk": sdk, "steps": []}
        for command in commands:
            result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=300)
            case["steps"].append({"command": command, "returncode": result.returncode,
                                  "output": result.stdout + result.stderr})
            if result.returncode:
                break
        case["passed"] = all(step["returncode"] == 0 for step in case["steps"])
        results.append(case)
        (output / "results.json").write_text(json.dumps(results, indent=2))
        print(name + ": " + ("PASS" if case["passed"] else "FAIL"), flush=True)
    return 0 if all(case["passed"] for case in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
