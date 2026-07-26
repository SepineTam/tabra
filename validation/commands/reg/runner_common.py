"""Shared normalization helpers for the reg cross-engine runners."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


REG_DIRECTORY = Path(__file__).resolve().parent
CASES_PATH = REG_DIRECTORY / "cases.yaml"


def load_case_specifications() -> list[dict[str, Any]]:
    """Load JSON-compatible YAML without requiring a YAML dependency."""
    document = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    return document["cases"]


def verify_dataset(dataset_path: Path) -> tuple[str, int]:
    """Verify the fixture SHA-256 and return its digest and row count."""
    dataset_path = dataset_path.resolve()
    sidecar_path = dataset_path.with_suffix(dataset_path.suffix + ".sha256")
    if not sidecar_path.is_file():
        raise RuntimeError(f"Missing SHA-256 sidecar: {sidecar_path}")

    expected_digest = sidecar_path.read_text(encoding="ascii").split()[0]
    actual_digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if actual_digest != expected_digest:
        raise RuntimeError(
            f"Dataset SHA-256 mismatch: expected {expected_digest}, got {actual_digest}"
        )

    with dataset_path.open("r", encoding="utf-8", newline="") as dataset_file:
        n_rows = sum(1 for _ in csv.DictReader(dataset_file))
    if n_rows < 1:
        raise RuntimeError("The validation dataset is empty")
    return actual_digest, n_rows


def write_normalized_result(output_path: Path, document: dict[str, Any]) -> None:
    """Write strict JSON and reject NaN or Infinity values."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def make_document(
    *,
    engine_name: str,
    engine_version: str,
    dataset_path: Path,
    dataset_sha256: str,
    n_rows: int,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = {
        "case_ids": [case["case_id"] for case in cases],
        "nobs": [case["n_obs"] for case in cases],
        "df_model": [case["df_model"] for case in cases],
        "df_resid": [case["df_resid"] for case in cases],
        "parameter_names": [
            [parameter["name"] for parameter in case["parameters"]]
            for case in cases
        ],
        "coefficients": [
            [parameter["estimate"] for parameter in case["parameters"]]
            for case in cases
        ],
        "standard_errors": [
            [parameter["std_error"] for parameter in case["parameters"]]
            for case in cases
        ],
        "test_statistics": [
            [parameter["statistic"] for parameter in case["parameters"]]
            for case in cases
        ],
        "p_values": [
            [parameter["p_value"] for parameter in case["parameters"]]
            for case in cases
        ],
    }
    for statistic_name in (
        "r_squared",
        "r_squared_adj",
        "f_stat",
        "f_p_value",
        "ss_residual",
        "ss_model",
        "ss_total",
        "root_mse",
    ):
        metrics[statistic_name] = [
            case["statistics"][statistic_name] for case in cases
        ]

    case_status = [case["status"] for case in cases]
    return {
        "schema_version": 1,
        "command": "reg",
        "engine": engine_name,
        "status": "completed" if all(
            status == "completed" for status in case_status
        ) else "failed",
        "converged": all(case["converged"] for case in cases),
        "skipped": any(status == "skipped" for status in case_status),
        "sample": {
            "nobs": n_rows,
            "data_sha256": dataset_sha256,
        },
        "metrics": metrics,
        "metadata": {
            "engine_version": engine_version,
            "dataset_path": str(dataset_path.resolve()),
            "case_status": case_status,
        },
    }


def normalize_raw_tsv(raw_path: Path) -> tuple[str, list[dict[str, Any]]]:
    """Normalize the engine-neutral TSV emitted by Stata and R."""
    with raw_path.open("r", encoding="utf-8", newline="") as raw_file:
        rows = list(csv.DictReader(raw_file, delimiter="\t"))
    if not rows:
        raise RuntimeError(f"External engine produced no result rows: {raw_path}")

    expected_case_ids = [case["case_id"] for case in load_case_specifications()]
    case_groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        case_groups.setdefault(row["case_id"], []).append(row)
    if list(case_groups) != expected_case_ids:
        raise RuntimeError(
            f"External engine returned cases {list(case_groups)}, expected {expected_case_ids}"
        )

    engine_versions = {row["engine_version"] for row in rows}
    if len(engine_versions) != 1:
        raise RuntimeError("External engine returned inconsistent version strings")

    cases: list[dict[str, Any]] = []
    for case_id in expected_case_ids:
        case_rows = case_groups[case_id]
        first_row = case_rows[0]
        parameters = [
            {
                "name": row["term"],
                "estimate": float(row["estimate"]),
                "std_error": float(row["std_error"]),
                "statistic": float(row["statistic"]),
                "p_value": float(row["p_value"]),
            }
            for row in case_rows
        ]
        cases.append(
            {
                "case_id": case_id,
                "status": "completed",
                "converged": True,
                "n_obs": int(float(first_row["n_obs"])),
                "df_model": int(float(first_row["df_model"])),
                "df_resid": int(float(first_row["df_resid"])),
                "parameters": parameters,
                "statistics": {
                    "r_squared": float(first_row["r_squared"]),
                    "r_squared_adj": float(first_row["r_squared_adj"]),
                    "f_stat": float(first_row["f_stat"]),
                    "f_p_value": float(first_row["f_p_value"]),
                    "ss_residual": float(first_row["ss_residual"]),
                    "ss_model": float(first_row["ss_model"]),
                    "ss_total": float(first_row["ss_total"]),
                    "root_mse": float(first_row["root_mse"]),
                },
            }
        )
    return engine_versions.pop(), cases
