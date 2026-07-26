"""Three-engine comparison using tolerances frozen before execution."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from itertools import combinations
from numbers import Real
from typing import Any

from .models import (
    CommandManifest,
    EngineRunStatus,
    NormalizedResult,
    ToleranceMode,
    ToleranceSpec,
    thaw_json,
)

_VERSION_NUMBER = re.compile(r"(?<!\d)(\d+(?:\.\d+)*)")


def _violation(
    code: str,
    message: str,
    *,
    engine: str | None = None,
    metric: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {"code": code, "message": message}
    if engine is not None:
        value["engine"] = engine
    if metric is not None:
        value["metric"] = metric
    return value


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _unsafe_value_reason(value: Any, path: str = "value") -> str | None:
    if value is None:
        return f"{path} is null"
    if isinstance(value, bool):
        return None
    if isinstance(value, Real):
        if not math.isfinite(float(value)):
            return f"{path} is NaN or infinite"
        return None
    if isinstance(value, Mapping):
        for key, item in value.items():
            reason = _unsafe_value_reason(item, f"{path}.{key}")
            if reason is not None:
                return reason
        return None
    if _is_sequence(value):
        for index, item in enumerate(value):
            reason = _unsafe_value_reason(item, f"{path}[{index}]")
            if reason is not None:
                return reason
    return None


def _numeric_comparison(
    left: Any,
    right: Any,
    tolerance: ToleranceSpec,
    path: str,
    reject_zero_mismatch: bool = False,
) -> tuple[bool, list[str], float]:
    if isinstance(left, bool) or isinstance(right, bool):
        return False, [f"{path} contains a boolean, not a numeric value"], 0.0
    if isinstance(left, Real) and isinstance(right, Real):
        left_number = float(left)
        right_number = float(right)
        difference = abs(left_number - right_number)
        if reject_zero_mismatch and ((left_number == 0.0) != (right_number == 0.0)):
            return (
                False,
                [f"{path} loses a positive value to numerical zero"],
                difference,
            )
        passed = math.isclose(
            left_number,
            right_number,
            abs_tol=tolerance.atol,
            rel_tol=tolerance.rtol,
        )
        issues = [] if passed else [f"{path} differs by {difference:.17g}"]
        return passed, issues, difference

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return False, [f"{path} has different mapping keys"], 0.0
        passed = True
        issues: list[str] = []
        maximum_difference = 0.0
        for key in sorted(left):
            child_passed, child_issues, child_difference = _numeric_comparison(
                left[key],
                right[key],
                tolerance,
                f"{path}.{key}",
                reject_zero_mismatch,
            )
            passed = passed and child_passed
            issues.extend(child_issues)
            maximum_difference = max(maximum_difference, child_difference)
        return passed, issues, maximum_difference

    if _is_sequence(left) and _is_sequence(right):
        if len(left) != len(right):
            return False, [f"{path} has different sequence lengths"], 0.0
        passed = True
        issues = []
        maximum_difference = 0.0
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            child_passed, child_issues, child_difference = _numeric_comparison(
                left_item,
                right_item,
                tolerance,
                f"{path}[{index}]",
                reject_zero_mismatch,
            )
            passed = passed and child_passed
            issues.extend(child_issues)
            maximum_difference = max(maximum_difference, child_difference)
        return passed, issues, maximum_difference

    return False, [f"{path} has incompatible or non-numeric values"], 0.0


def _tolerance_json(tolerance: ToleranceSpec) -> dict[str, Any]:
    if tolerance.mode is ToleranceMode.EXACT:
        return {"exact": True}
    return {"atol": tolerance.atol, "rtol": tolerance.rtol}


def _version_tuple(value: str) -> tuple[int, ...] | None:
    match = _VERSION_NUMBER.search(value)
    if match is None:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def _version_satisfies_requirement(
    actual_version: str,
    requirement: Mapping[str, Any],
) -> bool:
    actual = _version_tuple(actual_version)
    if actual is None:
        return False
    if "locked_version" in requirement:
        expected = _version_tuple(str(requirement["locked_version"]))
        return actual == expected
    minimum = _version_tuple(str(requirement["minimum_version"]))
    if minimum is None:
        return False
    width = max(len(actual), len(minimum))
    return actual + (0,) * (width - len(actual)) >= minimum + (0,) * (
        width - len(minimum)
    )


def compare_results(
    manifest: CommandManifest,
    results: Sequence[NormalizedResult],
) -> dict[str, Any]:
    """Compare Python, Stata, and R results without inferring tolerances.

    The returned report is deterministic. It can therefore be recomputed by
    the release gate to detect stale or edited comparison artifacts.
    """

    violations: list[dict[str, Any]] = []
    results_by_engine: dict[str, NormalizedResult] = {}
    for result in results:
        if result.engine in results_by_engine:
            violations.append(
                _violation(
                    "duplicate_engine",
                    f"More than one normalized result was supplied for {result.engine}",
                    engine=result.engine,
                )
            )
            continue
        results_by_engine[result.engine] = result

    for engine in manifest.engines:
        if engine not in results_by_engine:
            violations.append(
                _violation(
                    "missing_engine",
                    f"No normalized result was supplied for {engine}",
                    engine=engine,
                )
            )

    for engine, result in sorted(results_by_engine.items()):
        if engine not in manifest.engines:
            violations.append(
                _violation(
                    "unexpected_engine",
                    f"Engine {engine} is not declared by the manifest",
                    engine=engine,
                )
            )
        if result.command != manifest.command:
            violations.append(
                _violation(
                    "command_mismatch",
                    f"Expected command {manifest.command}, got {result.command}",
                    engine=engine,
                )
            )
        if result.status is not EngineRunStatus.COMPLETED:
            violations.append(
                _violation(
                    "run_not_completed",
                    f"Engine run status is {result.status.value}",
                    engine=engine,
                )
            )
        if result.skipped:
            violations.append(
                _violation(
                    "run_skipped",
                    "Skipped engine runs cannot pass scientific validation",
                    engine=engine,
                )
            )
        if not result.converged:
            violations.append(
                _violation(
                    "not_converged",
                    "Non-converged estimates cannot pass scientific validation",
                    engine=engine,
                )
            )
        if result.nobs <= 0:
            violations.append(
                _violation(
                    "empty_sample",
                    "The estimation sample must contain at least one observation",
                    engine=engine,
                )
            )
        if result.data_sha256 != manifest.data_sha256:
            violations.append(
                _violation(
                    "dataset_hash_mismatch",
                    "The engine result does not use the dataset frozen by the manifest",
                    engine=engine,
                )
            )
        case_ids = result.metrics.get("case_ids")
        if case_ids != manifest.supported_cases:
            violations.append(
                _violation(
                    "required_case_mismatch",
                    "The engine result does not contain the frozen required case set",
                    engine=engine,
                    metric="case_ids",
                )
            )
        case_status = result.metadata["case_status"]
        if (
            len(case_status) != len(manifest.supported_cases)
            or any(status != "completed" for status in case_status)
        ):
            violations.append(
                _violation(
                    "case_status_mismatch",
                    "Every frozen case must have one completed status",
                    engine=engine,
                )
            )
        engine_version = str(result.metadata["engine_version"])
        requirement = manifest.environment_requirements[engine]
        if not _version_satisfies_requirement(engine_version, requirement):
            violations.append(
                _violation(
                    "engine_version_mismatch",
                    f"Engine version {engine_version!r} violates the frozen requirement",
                    engine=engine,
                )
            )

    complete_results = [
        results_by_engine[engine]
        for engine in manifest.engines
        if engine in results_by_engine
    ]
    if complete_results:
        expected_nobs = complete_results[0].nobs
        expected_data_sha256 = complete_results[0].data_sha256
        for result in complete_results[1:]:
            if result.nobs != expected_nobs:
                violations.append(
                    _violation(
                        "sample_size_mismatch",
                        f"Expected nobs={expected_nobs}, got {result.nobs}",
                        engine=result.engine,
                    )
                )
            if result.data_sha256 != expected_data_sha256:
                violations.append(
                    _violation(
                        "sample_hash_mismatch",
                        "All engines must use the exact same input data",
                        engine=result.engine,
                    )
                )

    cases_hash_values = {
        engine: results_by_engine[engine].cases_sha256
        for engine in manifest.engines
        if engine in results_by_engine
    }
    cases_hash_passed = (
        len(cases_hash_values) == len(manifest.engines)
        and all(
            digest == manifest.cases_sha256
            for digest in cases_hash_values.values()
        )
    )
    for engine, digest in cases_hash_values.items():
        if digest != manifest.cases_sha256:
            violations.append(
                _violation(
                    "case_specification_hash_mismatch",
                    "The engine result does not use the cases frozen by the manifest",
                    engine=engine,
                )
            )

    execution_order_passed: dict[str, bool] = {}
    for engine in manifest.engines:
        result = results_by_engine.get(engine)
        if result is None:
            execution_order_passed[engine] = False
            continue
        execution_case_ids = tuple(
            str(case["case_id"]) for case in result.case_execution
        )
        passed = execution_case_ids == manifest.supported_cases
        execution_order_passed[engine] = passed
        if not passed:
            violations.append(
                _violation(
                    "case_execution_mismatch",
                    "Case execution must exactly match the frozen case order",
                    engine=engine,
                )
            )

    case_evidence_items: list[dict[str, Any]] = []
    for case_index, case_id in enumerate(manifest.supported_cases):
        execution_values: dict[str, Any] = {}
        seed_values: dict[str, Any] = {}
        included_row_values: dict[str, Any] = {}
        omitted_parameter_values: dict[str, Any] = {}
        for engine in manifest.engines:
            result = results_by_engine.get(engine)
            if result is None or case_index >= len(result.case_execution):
                continue
            case_execution = result.case_execution[case_index]
            execution_values[engine] = case_execution["case_id"]
            seed_values[engine] = case_execution["seed"]
            included_row_values[engine] = case_execution["included_row_sha256"]
            omitted_parameter_values[engine] = thaw_json(
                case_execution["omitted_parameters"]
            )

        execution_passed = (
            len(execution_values) == len(manifest.engines)
            and all(value == case_id for value in execution_values.values())
        )
        seed_passed = (
            len(seed_values) == len(manifest.engines)
            and all(value == manifest.seed for value in seed_values.values())
        )
        included_row_passed = (
            len(included_row_values) == len(manifest.engines)
            and len(set(included_row_values.values())) == 1
        )
        omitted_parameters_passed = (
            len(omitted_parameter_values) == len(manifest.engines)
            and len(
                {
                    tuple(parameters)
                    for parameters in omitted_parameter_values.values()
                }
            )
            == 1
        )
        if not seed_passed:
            violations.append(
                _violation(
                    "case_seed_mismatch",
                    f"Case {case_id} must use the seed frozen by the manifest",
                )
            )
        if not included_row_passed:
            violations.append(
                _violation(
                    "included_row_hash_mismatch",
                    f"Engines used different estimation rows for case {case_id}",
                )
            )
        if not omitted_parameters_passed:
            violations.append(
                _violation(
                    "omitted_parameters_mismatch",
                    f"Engines omitted different parameters for case {case_id}",
                )
            )
        case_passed = (
            execution_passed
            and seed_passed
            and included_row_passed
            and omitted_parameters_passed
        )
        case_evidence_items.append(
            {
                "case_id": case_id,
                "case_execution": {
                    "values": execution_values,
                    "passed": execution_passed,
                },
                "seed": {
                    "expected": manifest.seed,
                    "values": seed_values,
                    "passed": seed_passed,
                },
                "included_row_sha256": {
                    "values": included_row_values,
                    "passed": included_row_passed,
                },
                "omitted_parameters": {
                    "values": omitted_parameter_values,
                    "passed": omitted_parameters_passed,
                },
                "passed": case_passed,
            }
        )

    provenance_blocking_codes = {
        "dataset_hash_mismatch",
        "sample_hash_mismatch",
        "sample_size_mismatch",
        "required_case_mismatch",
        "case_specification_hash_mismatch",
        "case_execution_mismatch",
        "case_seed_mismatch",
        "included_row_hash_mismatch",
        "omitted_parameters_mismatch",
    }
    metric_comparison_performed = (
        cases_hash_passed
        and all(execution_order_passed.values())
        and all(item["passed"] for item in case_evidence_items)
        and not any(
            violation["code"] in provenance_blocking_codes
            for violation in violations
        )
    )

    metric_reports: list[dict[str, Any]] = []
    for metric in manifest.metrics if metric_comparison_performed else ():
        metric_values: dict[str, Any] = {}
        for engine in manifest.engines:
            result = results_by_engine.get(engine)
            if result is None:
                continue
            if metric.name not in result.metrics:
                violations.append(
                    _violation(
                        "missing_metric",
                        f"Required metric {metric.name} is missing",
                        engine=engine,
                        metric=metric.name,
                    )
                )
                continue
            metric_value = result.metrics[metric.name]
            unsafe_reason = _unsafe_value_reason(metric_value)
            if unsafe_reason is not None:
                violations.append(
                    _violation(
                        "unsafe_metric_value",
                        unsafe_reason,
                        engine=engine,
                        metric=metric.name,
                    )
                )
                continue
            metric_values[engine] = metric_value

        pair_reports: list[dict[str, Any]] = []
        for left_engine, right_engine in combinations(manifest.engines, 2):
            if left_engine not in metric_values or right_engine not in metric_values:
                continue
            left = metric_values[left_engine]
            right = metric_values[right_engine]
            if metric.tolerance.mode is ToleranceMode.EXACT:
                passed = left == right
                issues = [] if passed else ["Values are not exactly equal"]
                maximum_difference: float | None = None
            else:
                passed, issues, maximum_difference = _numeric_comparison(
                    left,
                    right,
                    metric.tolerance,
                    metric.name,
                    metric.reject_zero_mismatch,
                )
            pair_report: dict[str, Any] = {
                "left_engine": left_engine,
                "right_engine": right_engine,
                "passed": passed,
                "issues": issues,
            }
            if maximum_difference is not None:
                pair_report["max_abs_difference"] = maximum_difference
            pair_reports.append(pair_report)
            if not passed:
                violations.append(
                    _violation(
                        "metric_mismatch",
                        f"{left_engine} and {right_engine} differ for {metric.name}",
                        metric=metric.name,
                    )
                )

        metric_reports.append(
            {
                "name": metric.name,
                "tolerance": _tolerance_json(metric.tolerance),
                "safety": {
                    "reject_zero_mismatch": metric.reject_zero_mismatch,
                },
                "values": {
                    engine: thaw_json(value)
                    for engine, value in sorted(metric_values.items())
                },
                "pairs": pair_reports,
                "passed": (
                    len(metric_values) == len(manifest.engines)
                    and len(pair_reports) == 3
                    and all(pair["passed"] for pair in pair_reports)
                ),
            }
        )

    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "three_engine_comparison",
        "command": manifest.command,
        "manifest_sha256": manifest.sha256,
        "result_sha256": {
            engine: result.sha256
            for engine, result in sorted(results_by_engine.items())
        },
        "engines": list(manifest.engines),
        "case_evidence": {
            "cases_sha256": {
                "expected": manifest.cases_sha256,
                "values": cases_hash_values,
                "passed": cases_hash_passed,
            },
            "cases": case_evidence_items,
        },
        "metric_comparison_performed": metric_comparison_performed,
        "metrics": metric_reports,
        "violations": violations,
        "passed": (
            metric_comparison_performed
            and not violations
            and all(metric["passed"] for metric in metric_reports)
        ),
    }
    return report
