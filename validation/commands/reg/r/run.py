#!/usr/bin/env python3
"""Run R lm cases and normalize the extracted TSV as strict JSON."""

from __future__ import annotations

import argparse
import os
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


def resolve_rscript() -> str:
    configured_executable = os.environ.get("RSCRIPT_CMD")
    candidates = [configured_executable] if configured_executable else [
        "Rscript",
        "/usr/local/bin/Rscript",
        "/opt/homebrew/bin/Rscript",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        executable = shutil.which(candidate)
        if executable is None and Path(candidate).is_file():
            executable = str(Path(candidate).resolve())
        if executable:
            return executable
    raise RuntimeError(
        "R is required but Rscript was not found. Install the locked R version "
        "or set RSCRIPT_CMD to an executable."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()

    dataset_sha256, n_rows = verify_dataset(arguments.data)
    rscript = resolve_rscript()
    r_file = Path(__file__).with_name("run.R").resolve()
    with tempfile.TemporaryDirectory(prefix="tabra-reg-r-") as temporary_directory:
        raw_path = Path(temporary_directory) / "r-results.tsv"
        completed_process = subprocess.run(
            [rscript, str(r_file), str(arguments.data.resolve()), str(raw_path)],
            cwd=temporary_directory,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed_process.returncode != 0 or not raw_path.is_file():
            diagnostic = (completed_process.stderr or completed_process.stdout).strip()
            raise RuntimeError(
                f"R reg validation failed with code {completed_process.returncode}: "
                f"{diagnostic[-2000:]}"
            )
        engine_version, cases = normalize_raw_tsv(raw_path)

    document = make_document(
        engine_name="r",
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
