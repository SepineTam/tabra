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


def hash_included_row_ids(row_ids: list[str]) -> str:
    if any(not isinstance(row_id, str) or row_id == "" for row_id in row_ids):
        raise RuntimeError("Every row_id must be a non-empty string")
    if len(row_ids) != len(set(row_ids)):
        raise RuntimeError("row_id values must be unique")
    preimage = "".join(f"{row_id}\n" for row_id in row_ids).encode("utf-8")
    return hashlib.sha256(preimage).hexdigest()


def load_raw_row_ids(dataset_path: Path) -> list[str]:
    with dataset_path.open("r", encoding="utf-8", newline="") as dataset_file:
        rows = list(csv.DictReader(dataset_file))
    if not rows or "row_id" not in rows[0]:
        raise RuntimeError("The validation dataset must contain row_id")
    row_ids = [row["row_id"] for row in rows]
    hash_included_row_ids(row_ids)
    return row_ids


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
    case_specifications = load_case_specifications()
    expected_case_ids = [case["case_id"] for case in case_specifications]
    if [case["case_id"] for case in cases] != expected_case_ids:
        raise RuntimeError("Normalized cases do not match the frozen case order")
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
        "schema_version": 2,
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
            "cases_sha256": hashlib.sha256(CASES_PATH.read_bytes()).hexdigest(),
            "case_execution": [
                {
                    "case_id": case["case_id"],
                    "seed": case_specification["seed"],
                    "included_row_sha256": case["included_row_sha256"],
                    "omitted_parameters": case["omitted_parameters"],
                }
                for case, case_specification in zip(
                    cases, case_specifications, strict=True
                )
            ],
        },
    }


def normalize_raw_tsv(
    raw_path: Path,
    sample_path: Path,
    dataset_path: Path,
) -> tuple[str, list[dict[str, Any]]]:
    """Normalize the engine-neutral TSV emitted by Stata and R."""
    with raw_path.open("r", encoding="utf-8", newline="") as raw_file:
        rows = list(csv.DictReader(raw_file, delimiter="\t"))
    if not rows:
        raise RuntimeError(f"External engine produced no result rows: {raw_path}")

    case_specifications = load_case_specifications()
    expected_case_ids = [case["case_id"] for case in case_specifications]
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

    with sample_path.open("r", encoding="utf-8", newline="") as sample_file:
        sample_rows = list(csv.DictReader(sample_file, delimiter="\t"))
    if not sample_rows:
        raise RuntimeError(f"External engine produced no sample rows: {sample_path}")
    sample_groups: dict[str, list[int]] = {}
    for row in sample_rows:
        case_id = row["case_id"]
        try:
            original_row_number = int(row["original_row_number"])
        except (TypeError, ValueError) as exc:
            raise RuntimeError("External sample row number must be an integer") from exc
        sample_groups.setdefault(case_id, []).append(original_row_number)
    if list(sample_groups) != expected_case_ids:
        raise RuntimeError(
            f"External sample returned cases {list(sample_groups)}, "
            f"expected {expected_case_ids}"
        )
    raw_row_ids = load_raw_row_ids(dataset_path)

    cases: list[dict[str, Any]] = []
    for case_specification in case_specifications:
        case_id = case_specification["case_id"]
        case_rows = case_groups[case_id]
        first_row = case_rows[0]
        expected_parameters = [*case_specification["independent"]]
        if case_specification["include_constant"]:
            expected_parameters.append("_cons")
        actual_parameters = [row["term"] for row in case_rows]
        if actual_parameters != expected_parameters:
            raise RuntimeError(
                f"External engine returned parameters {actual_parameters} for "
                f"{case_id}, expected {expected_parameters}"
            )
        omitted_flags: list[bool] = []
        for row in case_rows:
            if row.get("omitted") not in {"0", "1"}:
                raise RuntimeError("External omitted flag must be 0 or 1")
            omitted_flags.append(row["omitted"] == "1")
        omitted_parameters = [
            parameter
            for parameter, omitted in zip(
                expected_parameters, omitted_flags, strict=True
            )
            if omitted
        ]
        sample_row_numbers = sample_groups[case_id]
        if (
            len(sample_row_numbers) != len(set(sample_row_numbers))
            or sample_row_numbers != sorted(sample_row_numbers)
            or any(
                row_number < 1 or row_number > len(raw_row_ids)
                for row_number in sample_row_numbers
            )
        ):
            raise RuntimeError(
                f"External sample rows for {case_id} must be unique, valid, "
                "and in original data order"
            )
        n_obs = int(float(first_row["n_obs"]))
        if len(sample_row_numbers) != n_obs:
            raise RuntimeError(
                f"External sample row count for {case_id} does not match n_obs"
            )
        included_row_ids = [
            raw_row_ids[row_number - 1] for row_number in sample_row_numbers
        ]
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
                "included_row_sha256": hash_included_row_ids(included_row_ids),
                "omitted_parameters": omitted_parameters,
                "status": "completed",
                "converged": True,
                "n_obs": n_obs,
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
