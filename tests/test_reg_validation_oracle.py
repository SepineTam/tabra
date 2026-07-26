from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REG_DIRECTORY = REPOSITORY_ROOT / "validation" / "commands" / "reg"
ORACLE_PATH = REG_DIRECTORY / "oracles" / "code_audit_oracle.py"


def _load_oracle_module():
    module_specification = importlib.util.spec_from_file_location(
        "reg_code_audit_oracle", ORACLE_PATH
    )
    assert module_specification is not None
    assert module_specification.loader is not None
    oracle_module = importlib.util.module_from_spec(module_specification)
    module_specification.loader.exec_module(oracle_module)
    return oracle_module


def test_frozen_hashes_match_current_reg_inputs():
    oracle_module = _load_oracle_module()
    frozen_paths = {
        "manifest": REG_DIRECTORY / "manifest.yaml",
        "cases": REG_DIRECTORY / "cases.yaml",
        "data": REG_DIRECTORY / "data" / "reg_fixture.csv",
    }

    actual_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in frozen_paths.items()
    }

    assert actual_hashes == oracle_module.FROZEN_HASHES


def test_reg_oracle_formal_entry_passes(tmp_path):
    actual_path = tmp_path / "python-results.json"
    runner = subprocess.run(
        [
            sys.executable,
            str(REG_DIRECTORY / "python" / "run.py"),
            "--data",
            str(REG_DIRECTORY / "data" / "reg_fixture.csv"),
            "--output",
            str(actual_path),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert runner.returncode == 0, runner.stderr

    oracle = subprocess.run(
        [
            sys.executable,
            str(ORACLE_PATH),
            "--manifest",
            str(REG_DIRECTORY / "manifest.yaml"),
            "--cases",
            str(REG_DIRECTORY / "cases.yaml"),
            "--data",
            str(REG_DIRECTORY / "data" / "reg_fixture.csv"),
            "--actual",
            str(actual_path),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert oracle.returncode == 0, oracle.stderr
    oracle_result = json.loads(oracle.stdout)
    assert oracle_result["oracle_status"] == "PASS"
    assert all(oracle_result["hash_checks"].values())
    assert all(
        comparison["passed"]
        for comparison in oracle_result["comparisons"].values()
    )
