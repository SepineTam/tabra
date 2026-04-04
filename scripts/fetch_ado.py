"""
Fetch Stata ado source files to .verify/ado_plus/

Usage: uv run scripts/fetch_ado.py <command>

Workflow:
1. Write a temp do-file (findfile + cp)
2. Execute via stata-mcp tool do
3. Verify the file was copied to .verify/ado_plus/<command>.ado
"""

import subprocess
import sys
from pathlib import Path

# scripts/fetch_ado.py -> project root
ROOT = Path(__file__).resolve().parents[1]
VERIFY_ADO = ROOT / ".verify" / "ado_plus"
DO_TMP = ROOT / "tmp" / "_fetch_ado.do"


def fetch_ado(command: str) -> None:
    VERIFY_ADO.mkdir(parents=True, exist_ok=True)
    dest = VERIFY_ADO / f"{command}.ado"

    # Write do-file: findfile to get path, cp to copy
    DO_TMP.parent.mkdir(parents=True, exist_ok=True)
    DO_TMP.write_text(
        f'findfile {command}.ado\n'
        f'cp "`r(fn)\'" "{dest}"\n'
    )

    # Execute
    result = subprocess.run(
        ["stata-mcp", "tool", "do", str(DO_TMP)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Execution failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    # Verify
    if dest.exists():
        print(str(dest.relative_to(ROOT)))
    else:
        print(f"Ado file not found. Command may be built-in: {command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run scripts/fetch_ado.py <command>", file=sys.stderr)
        sys.exit(1)
    fetch_ado(sys.argv[1])
