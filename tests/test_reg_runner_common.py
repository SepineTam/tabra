from __future__ import annotations

import csv
import hashlib
import importlib.util
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER_COMMON_PATH = (
    REPOSITORY_ROOT / "validation" / "commands" / "reg" / "runner_common.py"
)


def _load_runner_common():
    module_specification = importlib.util.spec_from_file_location(
        "reg_runner_common", RUNNER_COMMON_PATH
    )
    assert module_specification is not None
    assert module_specification.loader is not None
    module = importlib.util.module_from_spec(module_specification)
    module_specification.loader.exec_module(module)
    return module


def _write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(rows[0]),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def test_canonical_row_hash_uses_raw_order_and_terminal_lf():
    runner_common = _load_runner_common()
    row_ids = [str(index) for index in range(1, 121)]

    assert runner_common.hash_included_row_ids(row_ids) == (
        "11ebba9a3453b6af0b448a00ad5c27aa9f5508a1cfdfacfe130c6752545dcf76"
    )
    assert runner_common.hash_included_row_ids(list(reversed(row_ids))) != (
        runner_common.hash_included_row_ids(row_ids)
    )
    assert hashlib.sha256("\n".join(row_ids).encode("utf-8")).hexdigest() != (
        runner_common.hash_included_row_ids(row_ids)
    )


@pytest.mark.parametrize("row_ids", [["1", "1"], ["1", ""], ["1", None]])
def test_canonical_row_hash_rejects_invalid_row_ids(row_ids):
    runner_common = _load_runner_common()

    with pytest.raises(RuntimeError, match="row_id"):
        runner_common.hash_included_row_ids(row_ids)


def test_external_sidecar_maps_actual_rows_and_omitted_terms(tmp_path):
    runner_common = _load_runner_common()
    dataset_path = tmp_path / "fixture.csv"
    dataset_path.write_text(
        "row_id,y,x1,x2\nraw-01,1,2,3\nraw-02,2,3,4\nraw-03,3,4,5\n",
        encoding="utf-8",
    )
    raw_path = tmp_path / "results.tsv"
    base = {
        "engine_version": "test 1.0",
        "n_obs": 2,
        "df_model": 2,
        "df_resid": 1,
        "estimate": 1,
        "std_error": 1,
        "statistic": 1,
        "p_value": 0.5,
        "r_squared": 0.5,
        "r_squared_adj": 0.5,
        "f_stat": 1,
        "f_p_value": 0.5,
        "ss_residual": 1,
        "ss_model": 1,
        "ss_total": 2,
        "root_mse": 1,
    }
    rows = [
        {**base, "case_id": "basic", "term": "x1", "omitted": 0},
        {**base, "case_id": "basic", "term": "x2", "omitted": 1},
        {**base, "case_id": "basic", "term": "_cons", "omitted": 0},
        {
            **base,
            "case_id": "no_constant",
            "term": "x1",
            "omitted": 0,
            "n_obs": 1,
        },
        {
            **base,
            "case_id": "no_constant",
            "term": "x2",
            "omitted": 0,
            "n_obs": 1,
        },
    ]
    _write_tsv(raw_path, rows)
    sample_path = tmp_path / "samples.tsv"
    _write_tsv(
        sample_path,
        [
            {"case_id": "basic", "original_row_number": 1},
            {"case_id": "basic", "original_row_number": 2},
            {"case_id": "no_constant", "original_row_number": 3},
        ],
    )

    _, cases = runner_common.normalize_raw_tsv(
        raw_path, sample_path, dataset_path
    )

    assert [case["case_id"] for case in cases] == ["basic", "no_constant"]
    assert cases[0]["included_row_sha256"] == runner_common.hash_included_row_ids(
        ["raw-01", "raw-02"]
    )
    assert cases[0]["omitted_parameters"] == ["x2"]
    assert cases[1]["included_row_sha256"] == runner_common.hash_included_row_ids(
        ["raw-03"]
    )
