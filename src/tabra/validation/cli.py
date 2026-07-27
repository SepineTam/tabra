"""Command-line interface for scientific validation artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .compare import check_comparison_integrity, compare_results
from .errors import ValidationError
from .gate import evaluate_gate
from .io import (
    load_json_document,
    load_manifest,
    load_normalized_result,
    load_registry,
    write_json,
)
from .models import ValidationStatus


def _resolve_executable(configured: str | None, candidates: list[str]) -> str | None:
    if configured:
        parts = shlex.split(configured)
        if not parts:
            return None
        executable = parts[0]
        resolved = shutil.which(executable)
        if resolved:
            return resolved
        path = Path(executable)
        return str(path.resolve()) if path.is_file() else None
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
        path = Path(candidate)
        if path.is_file():
            return str(path.resolve())
    return None


def _engine_preflight() -> dict[str, dict[str, Any]]:
    stata = _resolve_executable(
        os.environ.get("STATA_CMD"),
        [
            "stata-mcp",
            "stata-mp",
            "stata-se",
            "stata",
            "/usr/local/bin/stata-mp",
        ],
    )
    if stata is None and Path("/opt/homebrew/bin/uvx").is_file():
        stata = "/opt/homebrew/bin/uvx stata-mcp"
    rscript = _resolve_executable(
        os.environ.get("RSCRIPT_CMD"),
        ["Rscript", "/opt/homebrew/bin/Rscript", "/usr/local/bin/Rscript"],
    )
    return {
        "python": {"available": True, "executable": sys.executable},
        "stata": {"available": stata is not None, "executable": stata},
        "r": {"available": rscript is not None, "executable": rscript},
    }


def _emit(value: dict[str, Any], output: str | None) -> None:
    if output is not None:
        write_json(output, value)
        return
    print(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )


def _preflight(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if args.manifest is not None:
        manifest = load_manifest(args.manifest)
        engine_environment = _engine_preflight()
        missing_engines = [
            engine
            for engine in manifest.engines
            if not engine_environment[engine]["available"]
        ]
        passed = args.schema_only or not missing_engines
        report = {
            "schema_version": 1,
            "kind": "preflight",
            "status": (
                ValidationStatus.PREFLIGHT_PASSED.value
                if passed
                else ValidationStatus.BLOCKED_ENV.value
            ),
            "passed": passed,
            "command": manifest.command,
            "manifest_sha256": manifest.sha256,
            "engines": list(manifest.engines),
            "required_metrics": [metric.name for metric in manifest.metrics],
            "tolerances_frozen": manifest.tolerances_frozen,
            "engine_environment": engine_environment,
            "missing_engines": missing_engines,
            "schema_only": args.schema_only,
        }
        return report, 0 if passed else 1

    registry = load_registry(args.registry)
    report = {
        "schema_version": 1,
        "kind": "preflight",
        "status": ValidationStatus.PREFLIGHT_PASSED.value,
        "passed": True,
        "registry_sha256": registry.sha256,
        "commands": [
            {
                "command": entry.command,
                "manifest": entry.manifest_path,
                "enabled": entry.enabled,
            }
            for entry in registry.entries
        ],
    }
    return report, 0


def _compare(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if args.output is not None and Path(args.output).exists():
        raise ValidationError(
            "Refusing to overwrite immutable comparison output: "
            f"{args.output}; create a new run ID"
        )
    manifest = load_manifest(args.manifest)
    results = [load_normalized_result(path) for path in args.result]
    report = compare_results(manifest, results)
    return report, 0 if report["passed"] else 1


def _check_comparison(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    manifest = load_manifest(args.manifest)
    results = [load_normalized_result(path) for path in args.result]
    comparison = load_json_document(args.comparison)
    report, _ = check_comparison_integrity(manifest, results, comparison)
    return report, 0 if report["passed"] else 1


def _gate(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if args.output is not None and Path(args.output).exists():
        raise ValidationError(
            f"Refusing to overwrite immutable gate output: {args.output}"
        )
    manifest = load_manifest(args.manifest)
    results = [load_normalized_result(path) for path in args.result]
    comparison = load_json_document(args.comparison)
    code_audit = load_json_document(args.code_audit)
    theory_audit = load_json_document(args.theory_audit)
    report = evaluate_gate(
        manifest,
        results,
        comparison,
        run_id=args.run_id,
        code_audit=code_audit,
        theory_audit=theory_audit,
        workdir=args.workdir,
        allow_dirty=args.allow_dirty,
    )
    return report, 0 if report["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    """Build the public validation CLI parser."""

    parser = argparse.ArgumentParser(
        prog="python -m tabra.validation",
        description="Run strict, auditable three-engine scientific validation.",
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    preflight_parser = subparsers.add_parser(
        "preflight", help="Validate a frozen manifest or registry before execution."
    )
    preflight_source = preflight_parser.add_mutually_exclusive_group(required=True)
    preflight_source.add_argument("--manifest", help="Path to a command manifest.")
    preflight_source.add_argument("--registry", help="Path to the command registry.")
    preflight_parser.add_argument("--output", help="Optional JSON output path.")
    preflight_parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Validate structure without requiring local engine executables.",
    )
    preflight_parser.set_defaults(handler=_preflight)

    compare_parser = subparsers.add_parser(
        "compare", help="Compare Python, Stata, and R normalized results."
    )
    compare_parser.add_argument("--manifest", required=True)
    compare_parser.add_argument(
        "--result",
        required=True,
        action="append",
        help="Normalized result JSON; provide once per engine.",
    )
    compare_parser.add_argument("--output", help="Optional comparison JSON path.")
    compare_parser.set_defaults(handler=_compare)

    check_comparison_parser = subparsers.add_parser(
        "check-comparison",
        help="Verify a stored comparison against the canonical recomputed object.",
    )
    check_comparison_parser.add_argument("--manifest", required=True)
    check_comparison_parser.add_argument("--comparison", required=True)
    check_comparison_parser.add_argument(
        "--result",
        required=True,
        action="append",
        help="Normalized result JSON; provide once per engine.",
    )
    check_comparison_parser.add_argument(
        "--output", help="Optional comparison integrity JSON path."
    )
    check_comparison_parser.set_defaults(handler=_check_comparison)

    gate_parser = subparsers.add_parser(
        "gate", help="Recompute comparison evidence and issue an attestation."
    )
    gate_parser.add_argument("--manifest", required=True)
    gate_parser.add_argument("--comparison", required=True)
    gate_parser.add_argument("--run-id", required=True)
    gate_parser.add_argument("--code-audit", required=True)
    gate_parser.add_argument("--theory-audit", required=True)
    gate_parser.add_argument(
        "--result",
        required=True,
        action="append",
        help="Normalized result JSON; provide once per engine.",
    )
    gate_parser.add_argument(
        "--workdir",
        default=str(Path.cwd()),
        help="Repository used for Git environment metadata.",
    )
    gate_parser.add_argument("--output", help="Attestation or failure JSON path.")
    gate_parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help=(
            "Authorize a non-release development attestation from a dirty tracked "
            "worktree; the authorization is recorded."
        ),
    )
    gate_parser.set_defaults(handler=_gate)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the validation CLI and return a process status code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report, exit_code = args.handler(args)
    except (ValidationError, OSError) as exc:
        report = {
            "schema_version": 1,
            "kind": "validation_error",
            "status": ValidationStatus.FAILED.value,
            "passed": False,
            "error": str(exc),
        }
        output = getattr(args, "output", None)
        if output is not None and Path(output).exists():
            # Immutable evidence must never be replaced, including by an error report.
            output = None
        _emit(report, output)
        return 2
    _emit(report, args.output)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
