#!/usr/bin/env python3
"""Independent algebra oracle for the frozen ``reg`` validation cases.

This module deliberately does not import Tabra. Expected values are derived from
the frozen data with NumPy SVD/lstsq and SciPy distribution survival functions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy import stats


FROZEN_HASHES = {
    "manifest": "533e96bde7c0fd15dcc532f127789ab300c21f378c7b344ac99be6a4cafa26d7",
    "cases": "d6dc76648bcd0433795b60c2f65d46cc89167565384c23e24d864525d42ea7e7",
    "data": "53f6fea92a7978cbce09212c224eb6e29e9c581ab920e0285ef541c4b09d37c4",
}


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON-compatible YAML or JSON document."""
    return json.loads(path.read_text(encoding="utf-8"))


def read_numeric_csv(path: Path) -> dict[str, np.ndarray]:
    """Read the frozen numeric CSV without using Tabra or pandas."""
    with path.open("r", encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        raise RuntimeError("The oracle dataset is empty")
    columns = rows[0].keys()
    return {
        column: np.asarray([float(row[column]) for row in rows], dtype=np.float64)
        for column in columns
    }


def derive_case(
    data: dict[str, np.ndarray], case_specification: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, float]]:
    """Derive all frozen OLS metrics using independent linear algebra."""
    response = data[case_specification["dependent"]]
    design = np.column_stack(
        [data[name] for name in case_specification["independent"]]
    )
    parameter_names = list(case_specification["independent"])
    if case_specification["include_constant"]:
        design = np.column_stack([design, np.ones(response.size)])
        parameter_names.append("_cons")

    n_obs, n_parameters = design.shape
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        design, response, rcond=None
    )
    if rank != n_parameters:
        raise RuntimeError(
            f"Oracle design is rank deficient: rank={rank}, k={n_parameters}"
        )

    fitted = design @ coefficients
    residuals = response - fitted
    ss_residual = float(residuals @ residuals)
    if case_specification["include_constant"]:
        centered_response = response - np.mean(response)
        ss_total = float(centered_response @ centered_response)
        df_model = n_parameters - 1
        adjusted_numerator_df = n_obs - 1
    else:
        ss_total = float(response @ response)
        df_model = n_parameters
        adjusted_numerator_df = n_obs
    df_resid = n_obs - n_parameters
    if df_resid <= 0:
        raise RuntimeError(f"Oracle residual degrees of freedom are {df_resid}")

    ss_model = ss_total - ss_residual
    residual_variance = ss_residual / df_resid
    _, _, right_singular_vectors = np.linalg.svd(design, full_matrices=False)
    xtx_inverse = (
        right_singular_vectors.T
        @ np.diag(np.square(1.0 / singular_values))
        @ right_singular_vectors
    )
    covariance = residual_variance * xtx_inverse
    standard_errors = np.sqrt(np.diag(covariance))
    test_statistics = coefficients / standard_errors
    p_values = 2.0 * stats.t.sf(np.abs(test_statistics), df_resid)

    r_squared = 1.0 - ss_residual / ss_total
    r_squared_adj = (
        1.0
        - (1.0 - r_squared) * adjusted_numerator_df / df_resid
    )
    f_stat = (ss_model / df_model) / (ss_residual / df_resid)
    f_p_value = float(stats.f.sf(f_stat, df_model, df_resid))

    metrics = {
        "nobs": n_obs,
        "df_model": df_model,
        "df_resid": df_resid,
        "parameter_names": parameter_names,
        "coefficients": coefficients.tolist(),
        "standard_errors": standard_errors.tolist(),
        "test_statistics": test_statistics.tolist(),
        "p_values": p_values.tolist(),
        "r_squared": r_squared,
        "r_squared_adj": r_squared_adj,
        "f_stat": f_stat,
        "f_p_value": f_p_value,
        "ss_residual": ss_residual,
        "ss_model": ss_model,
        "ss_total": ss_total,
        "root_mse": float(np.sqrt(residual_variance)),
    }
    invariants = {
        "rank": int(rank),
        "n_parameters": n_parameters,
        "condition_number_x": float(singular_values[0] / singular_values[-1]),
        "normal_equation_max_abs": float(np.max(np.abs(design.T @ residuals))),
        "ss_decomposition_abs_error": abs(
            ss_total - ss_model - ss_residual
        ),
        "residual_sum_abs": float(abs(np.sum(residuals)))
        if case_specification["include_constant"]
        else 0.0,
    }
    return metrics, invariants


def compare_values(
    actual: Any, expected: Any, tolerance: dict[str, Any]
) -> dict[str, Any]:
    """Compare one frozen metric and return structured numeric evidence."""
    if tolerance.get("exact"):
        passed = actual == expected
        return {"passed": passed, "actual": actual, "expected": expected}

    def nested_structure(value: Any) -> Any:
        if isinstance(value, list):
            return [nested_structure(item) for item in value]
        return None

    def flatten_numeric(value: Any) -> list[float]:
        if isinstance(value, list):
            return [number for item in value for number in flatten_numeric(item)]
        return [float(value)]

    same_shape = nested_structure(actual) == nested_structure(expected)
    actual_array = np.asarray(flatten_numeric(actual), dtype=np.float64)
    expected_array = np.asarray(flatten_numeric(expected), dtype=np.float64)
    finite = bool(
        np.all(np.isfinite(actual_array)) and np.all(np.isfinite(expected_array))
    )
    absolute_error = np.abs(actual_array - expected_array) if same_shape else np.array([np.inf])
    denominator = np.maximum(np.abs(expected_array), np.finfo(float).tiny)
    relative_error = absolute_error / denominator
    passed = bool(
        same_shape
        and finite
        and np.allclose(
            actual_array,
            expected_array,
            atol=float(tolerance.get("atol", 0.0)),
            rtol=float(tolerance.get("rtol", 0.0)),
        )
    )
    return {
        "passed": passed,
        "max_abs_error": float(np.max(absolute_error)),
        "max_rel_error": float(np.max(relative_error)),
        "actual": actual,
        "expected": expected,
    }


def main() -> int:
    """Run the independent oracle against a normalized Python result."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--actual", required=True, type=Path)
    arguments = parser.parse_args()

    paths = {
        "manifest": arguments.manifest,
        "cases": arguments.cases,
        "data": arguments.data,
    }
    actual_hashes = {name: file_sha256(path) for name, path in paths.items()}
    hash_checks = {
        name: actual_hashes[name] == expected_hash
        for name, expected_hash in FROZEN_HASHES.items()
    }
    if not all(hash_checks.values()):
        raise RuntimeError(
            f"Frozen input hash mismatch: expected={FROZEN_HASHES}, actual={actual_hashes}"
        )

    manifest = read_json(arguments.manifest)
    cases_document = read_json(arguments.cases)
    actual_document = read_json(arguments.actual)
    data = read_numeric_csv(arguments.data)
    expected_by_metric: dict[str, list[Any]] = {
        metric_name: [] for metric_name in manifest["metrics"]
    }
    case_invariants: dict[str, dict[str, float]] = {}
    for case_specification in cases_document["cases"]:
        expected_case, invariants = derive_case(data, case_specification)
        case_id = case_specification["case_id"]
        case_invariants[case_id] = invariants
        expected_by_metric["case_ids"].append(case_id)
        for metric_name in expected_by_metric:
            if metric_name != "case_ids":
                expected_by_metric[metric_name].append(expected_case[metric_name])

    comparisons: dict[str, dict[str, Any]] = {}
    for metric_name, metric_specification in manifest["metrics"].items():
        comparisons[metric_name] = compare_values(
            actual_document["metrics"][metric_name],
            expected_by_metric[metric_name],
            metric_specification["tolerance"],
        )
        if metric_specification.get("safety", {}).get("reject_zero_mismatch"):
            actual_array = np.asarray(
                [
                    number
                    for case_values in actual_document["metrics"][metric_name]
                    for number in (
                        case_values if isinstance(case_values, list) else [case_values]
                    )
                ],
                dtype=np.float64,
            )
            expected_array = np.asarray(
                [
                    number
                    for case_values in expected_by_metric[metric_name]
                    for number in (
                        case_values if isinstance(case_values, list) else [case_values]
                    )
                ],
                dtype=np.float64,
            )
            zero_mismatch = bool(np.any((actual_array == 0.0) != (expected_array == 0.0)))
            comparisons[metric_name]["zero_mismatch"] = zero_mismatch
            comparisons[metric_name]["passed"] = bool(
                comparisons[metric_name]["passed"] and not zero_mismatch
            )

    status_checks = {
        "document_completed": actual_document.get("status") == "completed",
        "document_converged": actual_document.get("converged") is True,
        "document_not_skipped": actual_document.get("skipped") is False,
        "sample_nobs": actual_document.get("sample", {}).get("nobs")
        == len(data["row_id"]),
        "sample_data_sha256": actual_document.get("sample", {}).get("data_sha256")
        == FROZEN_HASHES["data"],
    }
    passed = bool(
        all(hash_checks.values())
        and all(status_checks.values())
        and all(comparison["passed"] for comparison in comparisons.values())
    )
    output = {
        "oracle_status": "PASS" if passed else "FAIL",
        "oracle_is_independent": True,
        "imports_tabra": False,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "frozen_hashes": actual_hashes,
        "hash_checks": hash_checks,
        "status_checks": status_checks,
        "case_invariants": case_invariants,
        "comparisons": comparisons,
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
