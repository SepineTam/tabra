"""Tests for the strict scientific validation infrastructure."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from tabra.validation import (
    SchemaError,
    StateTransitionError,
    ValidationStatus,
    compare_results,
    evaluate_gate,
    load_manifest,
    load_normalized_result,
    load_registry,
    sha256_json,
    transition_status,
)
from tabra.validation.cli import main

DATA_SHA256 = "a" * 64
ENGINE_VERSIONS = {
    "python": "3.11.11",
    "stata": "19.5",
    "r": "R version 4.4.1 (2024-06-14)",
}


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return path


def _manifest_value() -> dict[str, object]:
    return {
        "schema_version": 1,
        "command": "reg",
        "engines": ["python", "stata", "r"],
        "tolerances_frozen": True,
        "environment_requirements": {
            "python": {"minimum_version": "3.11"},
            "stata": {"minimum_version": "16"},
            "r": {"locked_version": "4.4.1", "lockfile": "r/renv.lock"},
        },
        "supported_cases": ["basic", "no_constant"],
        "data": {"sha256": DATA_SHA256},
        "metrics": {
            "case_ids": {"tolerance": {"exact": True}},
            "nobs": {"tolerance": {"exact": True}},
            "coefficients": {"tolerance": {"atol": 1e-8, "rtol": 1e-6}},
        },
    }


def _result_value(
    engine: str,
    *,
    status: str = "completed",
    converged: bool = True,
    skipped: bool = False,
    nobs: int = 100,
    data_sha256: str = DATA_SHA256,
    metrics: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "command": "reg",
        "engine": engine,
        "status": status,
        "converged": converged,
        "skipped": skipped,
        "sample": {"nobs": nobs, "data_sha256": data_sha256},
        "metrics": metrics
        if metrics is not None
        else {
            "case_ids": ["basic", "no_constant"],
            "nobs": 100,
            "coefficients": [1.0, 2.0],
        },
        "metadata": {
            "engine_version": ENGINE_VERSIONS[engine],
            "dataset_path": "/frozen/reg_fixture.csv",
            "case_status": ["completed", "completed"],
        },
    }


def _load_complete_case(tmp_path: Path):
    manifest_path = _write_json(tmp_path / "manifest.yaml", _manifest_value())
    result_paths = [
        _write_json(
            tmp_path / f"{engine}.json",
            _result_value(
                engine,
                metrics={
                    "case_ids": ["basic", "no_constant"],
                    "nobs": 100,
                    "coefficients": [1.0 + index * 1e-9, 2.0],
                },
            ),
        )
        for index, engine in enumerate(("python", "stata", "r"))
    ]
    return (
        load_manifest(manifest_path),
        [load_normalized_result(path) for path in result_paths],
        result_paths,
    )


def _audit_value(
    kind: str,
    manifest_sha256: str,
    commit: str,
    run_id: str = "test-run",
) -> dict[str, object]:
    audit = {
        "schema_version": 1,
        "command": "reg",
        "run_id": run_id,
        f"{kind}_audit_status": "PASS",
        "frozen_inputs": {
            "manifest_sha256": manifest_sha256,
            "commit": commit,
        },
        "findings": [],
    }
    if kind == "code":
        audit["audited_commit"] = commit
    return audit


def _patch_git_environment(monkeypatch, commit: str, *, dirty: bool = False):
    monkeypatch.setattr(
        "tabra.validation.gate.collect_environment",
        lambda workdir=None: {
            "python": {},
            "platform": {},
            "distributions": {},
            "git": {"commit": commit, "tracked_files_dirty": dirty},
        },
    )


def test_module_entry_point_exposes_cli_help():
    completed_process = subprocess.run(
        [sys.executable, "-m", "tabra.validation", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 0
    assert "Run strict, auditable three-engine scientific validation" in (
        completed_process.stdout
    )


def test_validation_status_allows_only_forward_auditable_transitions():
    assert (
        transition_status("discovered", "spec_frozen")
        is ValidationStatus.SPEC_FROZEN
    )
    assert (
        transition_status("comparison_passed", "gated") is ValidationStatus.GATED
    )

    with pytest.raises(StateTransitionError, match="Illegal validation transition"):
        transition_status("gated", "preflight_passed")
    with pytest.raises(StateTransitionError):
        transition_status("failed", "spec_frozen")


def test_manifest_requires_frozen_explicit_tolerances(tmp_path):
    value = _manifest_value()
    value["tolerances_frozen"] = False
    manifest_path = _write_json(tmp_path / "manifest.yaml", value)

    with pytest.raises(SchemaError, match="tolerances_frozen"):
        load_manifest(manifest_path)

    value = _manifest_value()
    value["metrics"] = {"coefficient": {"tolerance": {}}}
    _write_json(manifest_path, value)
    with pytest.raises(SchemaError, match="exact"):
        load_manifest(manifest_path)


def test_manifest_is_immutable_and_requires_all_three_engines(tmp_path):
    manifest_path = _write_json(tmp_path / "manifest.yaml", _manifest_value())
    manifest = load_manifest(manifest_path)

    with pytest.raises(FrozenInstanceError):
        manifest.command = "logit"  # type: ignore[misc]

    value = _manifest_value()
    value["engines"] = ["python", "stata"]
    _write_json(manifest_path, value)
    with pytest.raises(SchemaError, match="exactly python, stata, and r"):
        load_manifest(manifest_path)


def test_manifest_retains_frozen_data_cases_and_engine_requirements(tmp_path):
    manifest_path = _write_json(tmp_path / "manifest.yaml", _manifest_value())

    manifest = load_manifest(manifest_path)

    assert manifest.data_sha256 == DATA_SHA256
    assert manifest.supported_cases == ("basic", "no_constant")
    assert manifest.environment_requirements["r"]["locked_version"] == "4.4.1"


def test_registry_resolves_manifests_and_blocks_path_escape(tmp_path):
    manifest_path = _write_json(
        tmp_path / "commands" / "reg" / "manifest.yaml", _manifest_value()
    )
    registry_path = _write_json(
        tmp_path / "registry.yaml",
        {
            "schema_version": 1,
            "commands": {
                "reg": {"manifest": "commands/reg/manifest.yaml", "enabled": True}
            },
        },
    )
    registry = load_registry(registry_path)

    assert registry.entries[0].manifest_path == str(manifest_path.resolve())

    outside_manifest = _write_json(tmp_path.parent / "outside.yaml", _manifest_value())
    _write_json(
        registry_path,
        {
            "schema_version": 1,
            "commands": {
                "reg": {"manifest": f"../{outside_manifest.name}", "enabled": True}
            },
        },
    )
    with pytest.raises(SchemaError, match="escapes"):
        load_registry(registry_path)


def test_normalized_result_rejects_duplicate_keys_and_nonfinite_json(tmp_path):
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(
        '{"schema_version": 1, "schema_version": 1}', encoding="utf-8"
    )
    with pytest.raises(SchemaError, match="Duplicate JSON key"):
        load_normalized_result(duplicate_path)

    nan_path = tmp_path / "nan.json"
    value = _result_value("python")
    raw = json.dumps(value).replace("[1.0, 2.0]", "[NaN, 2.0]")
    nan_path.write_text(raw, encoding="utf-8")
    with pytest.raises(SchemaError, match="Non-finite JSON number"):
        load_normalized_result(nan_path)


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {"dataset_path": "/data.csv", "case_status": ["completed"]},
        {"engine_version": "3.11.11", "case_status": ["completed"]},
        {"engine_version": "3.11.11", "dataset_path": "/data.csv"},
    ],
)
def test_normalized_result_requires_complete_metadata(tmp_path, metadata):
    value = _result_value("python")
    if metadata is None:
        value.pop("metadata")
    else:
        value["metadata"] = metadata
    result_path = _write_json(tmp_path / "python.json", value)

    with pytest.raises(SchemaError, match="metadata"):
        load_normalized_result(result_path)


def test_three_engine_comparison_passes_exact_and_numeric_metrics(tmp_path):
    manifest, results, _ = _load_complete_case(tmp_path)

    report = compare_results(manifest, results)

    assert report["passed"] is True
    assert report["violations"] == []
    assert all(metric["passed"] for metric in report["metrics"])
    assert len(report["metrics"][1]["pairs"]) == 3


def test_comparison_rejects_shared_wrong_dataset(tmp_path):
    manifest_path = _write_json(tmp_path / "manifest.yaml", _manifest_value())
    result_paths = [
        _write_json(
            tmp_path / f"{engine}.json",
            _result_value(engine, data_sha256="b" * 64),
        )
        for engine in ("python", "stata", "r")
    ]

    report = compare_results(
        load_manifest(manifest_path),
        [load_normalized_result(path) for path in result_paths],
    )

    assert report["passed"] is False
    assert "dataset_hash_mismatch" in {
        violation["code"] for violation in report["violations"]
    }


def test_comparison_rejects_shared_case_omission(tmp_path):
    manifest_path = _write_json(tmp_path / "manifest.yaml", _manifest_value())
    result_paths = []
    for engine in ("python", "stata", "r"):
        value = _result_value(
            engine,
            metrics={
                "case_ids": ["basic"],
                "nobs": 100,
                "coefficients": [1.0, 2.0],
            },
        )
        value["metadata"]["case_status"] = ["completed"]
        result_paths.append(_write_json(tmp_path / f"{engine}.json", value))

    report = compare_results(
        load_manifest(manifest_path),
        [load_normalized_result(path) for path in result_paths],
    )

    assert report["passed"] is False
    assert "required_case_mismatch" in {
        violation["code"] for violation in report["violations"]
    }


def test_comparison_rejects_unapproved_engine_version(tmp_path):
    manifest_path = _write_json(tmp_path / "manifest.yaml", _manifest_value())
    result_paths = []
    for engine in ("python", "stata", "r"):
        value = _result_value(engine)
        if engine == "r":
            value["metadata"]["engine_version"] = "R version 4.4.2"
        result_paths.append(_write_json(tmp_path / f"{engine}.json", value))

    report = compare_results(
        load_manifest(manifest_path),
        [load_normalized_result(path) for path in result_paths],
    )

    assert report["passed"] is False
    assert "engine_version_mismatch" in {
        violation["code"] for violation in report["violations"]
    }


@pytest.mark.parametrize(
    ("engine", "overrides", "expected_code"),
    [
        ("r", {"status": "failed"}, "run_not_completed"),
        ("r", {"status": "skipped", "skipped": True}, "run_skipped"),
        ("stata", {"converged": False}, "not_converged"),
        ("python", {"nobs": 99}, "sample_size_mismatch"),
        ("r", {"data_sha256": "b" * 64}, "sample_hash_mismatch"),
        (
            "stata",
            {"metrics": {"case_ids": ["basic", "no_constant"], "nobs": 100}},
            "missing_metric",
        ),
        (
            "python",
            {
                "metrics": {
                    "case_ids": ["basic", "no_constant"],
                    "nobs": 100,
                    "coefficients": [None, 2.0],
                }
            },
            "unsafe_metric_value",
        ),
    ],
)
def test_scientific_safety_failures_cannot_pass(
    tmp_path, engine, overrides, expected_code
):
    manifest_path = _write_json(tmp_path / "manifest.yaml", _manifest_value())
    result_paths = []
    for current_engine in ("python", "stata", "r"):
        current_overrides = overrides if current_engine == engine else {}
        result_paths.append(
            _write_json(
                tmp_path / f"{current_engine}.json",
                _result_value(current_engine, **current_overrides),
            )
        )

    report = compare_results(
        load_manifest(manifest_path),
        [load_normalized_result(path) for path in result_paths],
    )

    assert report["passed"] is False
    assert expected_code in {violation["code"] for violation in report["violations"]}


def test_out_of_tolerance_metric_and_missing_engine_fail(tmp_path):
    manifest, results, _ = _load_complete_case(tmp_path)
    changed_path = _write_json(
        tmp_path / "r-changed.json",
        _result_value(
            "r",
            metrics={
                "case_ids": ["basic", "no_constant"],
                "nobs": 100,
                "coefficients": [1.1, 2.0],
            },
        ),
    )

    mismatch_report = compare_results(
        manifest,
        [results[0], results[1], load_normalized_result(changed_path)],
    )
    missing_report = compare_results(manifest, results[:2])

    assert mismatch_report["passed"] is False
    assert any(
        violation["code"] == "metric_mismatch"
        for violation in mismatch_report["violations"]
    )
    assert missing_report["passed"] is False
    assert any(
        violation["code"] == "missing_engine"
        for violation in missing_report["violations"]
    )


def test_probability_safety_rejects_zero_underflow_hidden_by_tolerance(tmp_path):
    value = _manifest_value()
    value["metrics"] = {
        "p_values": {
            "tolerance": {"atol": 1e-10, "rtol": 1e-8},
            "safety": {"reject_zero_mismatch": True},
        }
    }
    manifest_path = _write_json(tmp_path / "manifest.yaml", value)
    result_paths = []
    for engine, p_value in (("python", 0.0), ("stata", 1e-90), ("r", 1e-90)):
        result_paths.append(
            _write_json(
                tmp_path / f"{engine}.json",
                _result_value(
                    engine,
                    metrics={
                        "case_ids": ["basic", "no_constant"],
                        "p_values": [p_value],
                    },
                ),
            )
        )

    report = compare_results(
        load_manifest(manifest_path),
        [load_normalized_result(path) for path in result_paths],
    )

    assert report["passed"] is False
    assert any(
        "numerical zero" in issue
        for issue in report["metrics"][0]["pairs"][0]["issues"]
    )


def test_gate_recomputes_evidence_and_generates_hashed_attestation(
    tmp_path, monkeypatch
):
    manifest, results, _ = _load_complete_case(tmp_path)
    comparison = compare_results(manifest, results)
    commit = "c" * 40
    _patch_git_environment(monkeypatch, commit)

    attestation = evaluate_gate(
        manifest,
        results,
        comparison,
        run_id="test-run",
        code_audit=_audit_value("code", manifest.sha256, commit),
        theory_audit=_audit_value("theory", manifest.sha256, commit),
        workdir=tmp_path,
    )

    assert attestation["passed"] is True, attestation
    assert attestation["kind"] == "validation_attestation"
    assert attestation["status"] == "passed"
    assert attestation["manifest"]["sha256"] == manifest.sha256
    digest = attestation.pop("attestation_sha256")
    assert digest == sha256_json(attestation)


def test_gate_rejects_edited_or_stale_comparison(tmp_path, monkeypatch):
    manifest, results, _ = _load_complete_case(tmp_path)
    comparison = compare_results(manifest, results)
    comparison["passed"] = False
    commit = "c" * 40
    _patch_git_environment(monkeypatch, commit)

    decision = evaluate_gate(
        manifest,
        results,
        comparison,
        run_id="test-run",
        code_audit=_audit_value("code", manifest.sha256, commit),
        theory_audit=_audit_value("theory", manifest.sha256, commit),
        workdir=tmp_path,
    )

    assert decision["passed"] is False
    assert decision["status"] == "failed"
    assert decision["violations"][0]["code"] == "comparison_report_mismatch"


def test_gate_requires_both_independent_audits(tmp_path):
    manifest, results, _ = _load_complete_case(tmp_path)
    comparison = compare_results(manifest, results)

    decision = evaluate_gate(manifest, results, comparison, run_id="audit-run")

    assert decision["passed"] is False
    assert {item["code"] for item in decision["violations"]} == {
        "missing_code_audit",
        "missing_theory_audit",
    }


def test_gate_rejects_an_audit_from_another_run(tmp_path, monkeypatch):
    manifest, results, _ = _load_complete_case(tmp_path)
    comparison = compare_results(manifest, results)
    commit = "c" * 40
    _patch_git_environment(monkeypatch, commit)

    decision = evaluate_gate(
        manifest,
        results,
        comparison,
        run_id="current-run",
        code_audit=_audit_value("code", manifest.sha256, commit, "old-run"),
        theory_audit=_audit_value(
            "theory", manifest.sha256, commit, "current-run"
        ),
    )

    assert decision["passed"] is False
    assert any(
        item["code"] == "code_audit_run_mismatch"
        for item in decision["violations"]
    )


@pytest.mark.parametrize(
    ("stale_field", "expected_code"),
    [
        ("manifest", "code_audit_manifest_mismatch"),
        ("commit", "code_audit_commit_mismatch"),
    ],
)
def test_gate_rejects_stale_audit_identity(
    tmp_path, monkeypatch, stale_field, expected_code
):
    manifest, results, _ = _load_complete_case(tmp_path)
    comparison = compare_results(manifest, results)
    commit = "c" * 40
    _patch_git_environment(monkeypatch, commit)
    code_manifest = "d" * 64 if stale_field == "manifest" else manifest.sha256
    code_commit = "d" * 40 if stale_field == "commit" else commit

    decision = evaluate_gate(
        manifest,
        results,
        comparison,
        run_id="test-run",
        code_audit=_audit_value("code", code_manifest, code_commit),
        theory_audit=_audit_value("theory", manifest.sha256, commit),
    )

    assert decision["passed"] is False
    assert expected_code in {
        violation["code"] for violation in decision["violations"]
    }


def test_gate_rejects_missing_git_identity(tmp_path, monkeypatch):
    manifest, results, _ = _load_complete_case(tmp_path)
    comparison = compare_results(manifest, results)
    commit = "c" * 40
    monkeypatch.setattr(
        "tabra.validation.gate.collect_environment",
        lambda workdir=None: {
            "git": {"commit": None, "tracked_files_dirty": None}
        },
    )

    decision = evaluate_gate(
        manifest,
        results,
        comparison,
        run_id="test-run",
        code_audit=_audit_value("code", manifest.sha256, commit),
        theory_audit=_audit_value("theory", manifest.sha256, commit),
        workdir=tmp_path,
    )

    assert decision["passed"] is False
    assert decision["violations"][0]["code"] == "git_identity_unavailable"


def test_gate_rejects_conflicting_audit_commit_fields(tmp_path, monkeypatch):
    manifest, results, _ = _load_complete_case(tmp_path)
    comparison = compare_results(manifest, results)
    commit = "c" * 40
    _patch_git_environment(monkeypatch, commit)
    code_audit = _audit_value("code", manifest.sha256, commit)
    code_audit["frozen_inputs"]["commit"] = "d" * 40

    decision = evaluate_gate(
        manifest,
        results,
        comparison,
        run_id="test-run",
        code_audit=code_audit,
        theory_audit=_audit_value("theory", manifest.sha256, commit),
    )

    assert decision["passed"] is False
    assert "code_audit_commit_mismatch" in {
        violation["code"] for violation in decision["violations"]
    }


def test_gate_rejects_unknown_git_cleanliness(tmp_path, monkeypatch):
    manifest, results, _ = _load_complete_case(tmp_path)
    comparison = compare_results(manifest, results)
    commit = "c" * 40
    monkeypatch.setattr(
        "tabra.validation.gate.collect_environment",
        lambda workdir=None: {
            "git": {"commit": commit, "tracked_files_dirty": None}
        },
    )

    decision = evaluate_gate(
        manifest,
        results,
        comparison,
        run_id="test-run",
        code_audit=_audit_value("code", manifest.sha256, commit),
        theory_audit=_audit_value("theory", manifest.sha256, commit),
        workdir=tmp_path,
    )

    assert decision["passed"] is False
    assert decision["violations"][0]["code"] == "git_cleanliness_unavailable"


def test_gate_cli_never_overwrites_existing_evidence(tmp_path, capsys):
    manifest, results, result_paths = _load_complete_case(tmp_path)
    comparison_path = _write_json(
        tmp_path / "comparison.json", compare_results(manifest, results)
    )
    code_audit_path = _write_json(
        tmp_path / "code-audit.json",
        _audit_value("code", manifest.sha256, "c" * 40),
    )
    theory_audit_path = _write_json(
        tmp_path / "theory-audit.json",
        _audit_value("theory", manifest.sha256, "c" * 40),
    )
    output_path = _write_json(tmp_path / "attestation.json", {"keep": True})
    argv = [
        "gate",
        "--manifest",
        str(tmp_path / "manifest.yaml"),
        "--comparison",
        str(comparison_path),
        "--run-id",
        "test-run",
        "--code-audit",
        str(code_audit_path),
        "--theory-audit",
        str(theory_audit_path),
        "--output",
        str(output_path),
    ]
    for result_path in result_paths:
        argv.extend(["--result", str(result_path)])

    assert main(argv) == 2
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"keep": True}
    assert "Refusing to overwrite immutable gate output" in capsys.readouterr().out


def test_cli_preflight_compare_and_gate(tmp_path, capsys, monkeypatch):
    manifest, _, result_paths = _load_complete_case(tmp_path)
    manifest_path = Path(manifest.source_path)
    comparison_path = tmp_path / "comparison.json"
    attestation_path = tmp_path / "attestation.json"
    commit = "c" * 40
    _patch_git_environment(monkeypatch, commit)
    code_audit_path = _write_json(
        tmp_path / "code-audit.json",
        _audit_value("code", manifest.sha256, commit),
    )
    theory_audit_path = _write_json(
        tmp_path / "theory-audit.json",
        _audit_value("theory", manifest.sha256, commit),
    )

    assert (
        main(
            [
                "preflight",
                "--manifest",
                str(manifest_path),
                "--schema-only",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["passed"] is True

    compare_args = ["compare", "--manifest", str(manifest_path)]
    for path in result_paths:
        compare_args.extend(["--result", str(path)])
    compare_args.extend(["--output", str(comparison_path)])
    assert main(compare_args) == 0

    gate_args = [
        "gate",
        "--manifest",
        str(manifest_path),
        "--comparison",
        str(comparison_path),
        "--run-id",
        "test-run",
        "--code-audit",
        str(code_audit_path),
        "--theory-audit",
        str(theory_audit_path),
        "--workdir",
        str(tmp_path),
    ]
    for path in result_paths:
        gate_args.extend(["--result", str(path)])
    gate_args.extend(["--output", str(attestation_path)])
    assert main(gate_args) == 0
    assert json.loads(attestation_path.read_text(encoding="utf-8"))["passed"] is True


def test_cli_returns_nonzero_for_failed_scientific_comparison(tmp_path):
    manifest, _, result_paths = _load_complete_case(tmp_path)
    failed_result = _write_json(
        tmp_path / "r-failed.json",
        _result_value("r", status="skipped", skipped=True),
    )
    args = ["compare", "--manifest", manifest.source_path]
    for path in [result_paths[0], result_paths[1], failed_result]:
        args.extend(["--result", str(path)])

    assert main(args) == 1


def test_reg_formula_map_separates_mathematical_and_compatibility_sources():
    formula_map_path = (
        Path(__file__).resolve().parents[1]
        / "validation"
        / "commands"
        / "reg"
        / "formula_map.csv"
    )
    with formula_map_path.open(encoding="utf-8", newline="") as formula_map_file:
        rows = list(csv.DictReader(formula_map_file))

    required_columns = {
        "mathematical_reference",
        "mathematical_location",
        "compatibility_reference",
        "compatibility_location",
    }
    assert required_columns <= set(rows[0])

    software_specific_formula_ids = {
        "REG-CASE-001",
        "REG-SAMPLE-001",
        "REG-PARAM-001",
        "REG-NOCONSTANT-001",
    }
    for row in rows:
        if row["status"] != "implemented":
            continue
        assert row["compatibility_reference"]
        assert row["compatibility_location"]
        if row["formula_id"] in software_specific_formula_ids:
            assert row["mathematical_reference"] == "not_applicable"
            assert row["mathematical_location"] == "not_applicable"
        else:
            assert row["mathematical_reference"] not in {
                "",
                "not_applicable",
            }
            assert row["mathematical_location"] not in {
                "",
                "not_applicable",
            }
            assert "StataCorp" not in row["mathematical_reference"]
