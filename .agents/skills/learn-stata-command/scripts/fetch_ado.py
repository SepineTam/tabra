"""从 Stata 拉取 ado 源码文件到 .verify/ado_plus/

用法: uv run .claude/skills/learn_stata_command/scripts/fetch_ado.py <command>

流程:
1. 写临时 do-file (findfile + cp)
2. 通过 stata-mcp tool do 执行
3. 确认文件已复制到 .verify/ado_plus/<command>.ado
"""

import subprocess
import sys
from pathlib import Path

# .claude/skills/learn_stata_command/scripts/fetch_ado.py -> 项目根目录
ROOT = Path(__file__).resolve().parents[5]
VERIFY_ADO = ROOT / ".verify" / "ado_plus"
DO_TMP = ROOT / "tmp" / "_fetch_ado.do"


def fetch_ado(command: str) -> None:
    VERIFY_ADO.mkdir(parents=True, exist_ok=True)
    dest = VERIFY_ADO / f"{command}.ado"

    # 写 do-file: findfile 拿路径，cp 复制到目标
    DO_TMP.parent.mkdir(parents=True, exist_ok=True)
    DO_TMP.write_text(
        f'findfile {command}.ado\n'
        f'cp "`r(fn)\'" "{dest}"\n'
    )

    # 执行
    result = subprocess.run(
        ["stata-mcp", "tool", "do", str(DO_TMP)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"执行失败:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    # 确认
    if dest.exists():
        print(str(dest.relative_to(ROOT)))
    else:
        print(f"ado 文件未找到，命令可能是 built-in: {command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: uv run .claude/skills/learn_stata_command/scripts/fetch_ado.py <command>", file=sys.stderr)
        sys.exit(1)
    fetch_ado(sys.argv[1])
