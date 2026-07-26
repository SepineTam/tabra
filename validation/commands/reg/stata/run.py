#!/usr/bin/env python3
"""Run Stata reg cases and normalize the extracted TSV as strict JSON."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REG_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REG_DIRECTORY))

from runner_common import (  # noqa: E402
    make_document,
    normalize_raw_tsv,
    verify_dataset,
    write_normalized_result,
)


def resolve_stata_command() -> tuple[str, list[str]]:
    configured_command = os.environ.get("STATA_CMD")
    if configured_command:
        command = shlex.split(configured_command)
        if command and (Path(command[0]).is_file() or shutil.which(command[0])):
            return "direct", command
        raise RuntimeError(f"STATA_CMD is not executable: {configured_command}")

    configured_mcp = os.environ.get("STATA_MCP_CMD")
    if configured_mcp:
        command = shlex.split(configured_mcp)
        if command and (Path(command[0]).is_file() or shutil.which(command[0])):
            return "mcp", command
        raise RuntimeError(f"STATA_MCP_CMD is not executable: {configured_mcp}")

    stata_mcp = shutil.which("stata-mcp")
    if stata_mcp:
        return "mcp", [stata_mcp]
    uvx = Path("/opt/homebrew/bin/uvx")
    if uvx.is_file():
        return "mcp", [str(uvx), "stata-mcp"]

    for candidate in ("stata-mp", "stata-se", "stata", "/usr/local/bin/stata-mp"):
        executable = shutil.which(candidate)
        if executable is None and Path(candidate).is_file():
            executable = str(Path(candidate).resolve())
        if executable:
            return "direct", [executable]
    raise RuntimeError(
        "Stata is required but was not found. Set STATA_CMD to the Stata executable."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()

    dataset_sha256, n_rows = verify_dataset(arguments.data)
    stata_backend, stata_command = resolve_stata_command()
    do_file = Path(__file__).with_name("run.do").resolve()
    with tempfile.TemporaryDirectory(prefix="tabra-reg-stata-") as temporary_directory:
        raw_path = Path(temporary_directory) / "stata-results.tsv"
        if stata_backend == "mcp":
            driver_path = Path(temporary_directory) / "driver.do"
            driver_source = do_file.read_text(encoding="utf-8")
            driver_source = driver_source.replace(
                "args data_path output_path",
                f'local data_path "{arguments.data.resolve()}"\n'
                f'local output_path "{raw_path}"',
                1,
            )
            driver_path.write_text(driver_source, encoding="utf-8")
            invocation = [
                *stata_command,
                "tool",
                "do",
                str(driver_path),
                "--is-read-log",
                "true",
                "--enable-smcl",
                "false",
                "--timeout",
                "120",
            ]
        else:
            invocation = [
                *stata_command,
                "-b",
                "do",
                str(do_file),
                str(arguments.data.resolve()),
                str(raw_path),
            ]
        completed_process = subprocess.run(
            invocation,
            cwd=temporary_directory,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed_process.returncode != 0 or not raw_path.is_file():
            diagnostic = (completed_process.stderr or completed_process.stdout).strip()
            raise RuntimeError(
                f"Stata reg validation failed with code {completed_process.returncode}: "
                f"{diagnostic[-2000:]}"
            )
        engine_version, cases = normalize_raw_tsv(raw_path)

    document = make_document(
        engine_name="stata",
        engine_version=engine_version,
        dataset_path=arguments.data,
        dataset_sha256=dataset_sha256,
        n_rows=n_rows,
        cases=cases,
    )
    write_normalized_result(arguments.output, document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
