"""Independent scientific release gate and attestation generation."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .compare import compare_results
from .io import sha256_json
from .models import CommandManifest, NormalizedResult, ValidationStatus


def _audit_violations(
    audit: dict[str, Any] | None,
    *,
    kind: str,
    command: str,
    run_id: str,
) -> list[dict[str, str]]:
    if audit is None:
        return [
            {
                "code": f"missing_{kind}_audit",
                "message": f"A passing {kind} audit is required before gating",
            }
        ]
    violations: list[dict[str, str]] = []
    if audit.get("kind") != f"{kind}_audit":
        violations.append(
            {
                "code": f"invalid_{kind}_audit_kind",
                "message": f"Expected kind={kind}_audit",
            }
        )
    if audit.get("command") != command:
        violations.append(
            {
                "code": f"{kind}_audit_command_mismatch",
                "message": f"The {kind} audit is for a different command",
            }
        )
    if audit.get("run_id") != run_id:
        violations.append(
            {
                "code": f"{kind}_audit_run_mismatch",
                "message": f"The {kind} audit is not from run {run_id}",
            }
        )
    if audit.get("passed") is not True or audit.get("status") != "passed":
        violations.append(
            {
                "code": f"{kind}_audit_not_passed",
                "message": f"The {kind} audit did not pass",
            }
        )
    if audit.get("skipped") is True:
        violations.append(
            {
                "code": f"{kind}_audit_skipped",
                "message": f"A skipped {kind} audit cannot pass the gate",
            }
        )
    return violations


def _git_metadata(workdir: str | Path | None = None) -> dict[str, Any]:
    cwd = str(Path(workdir).resolve()) if workdir is not None else None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty_output = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        return {"commit": commit, "tracked_files_dirty": bool(dirty_output.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "tracked_files_dirty": None}


def collect_environment(workdir: str | Path | None = None) -> dict[str, Any]:
    """Collect a non-secret environment fingerprint for reproducibility."""

    distributions: dict[str, str | None] = {}
    for distribution in ("tabra", "numpy", "scipy", "pandas", "statsmodels"):
        try:
            distributions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            distributions[distribution] = None
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "distributions": distributions,
        "git": _git_metadata(workdir),
    }


def evaluate_gate(
    manifest: CommandManifest,
    results: Sequence[NormalizedResult],
    comparison: dict[str, Any],
    *,
    run_id: str = "unspecified",
    code_audit: dict[str, Any] | None = None,
    theory_audit: dict[str, Any] | None = None,
    workdir: str | Path | None = None,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Recompute comparison evidence and issue an attestation only on success."""

    audit_violations = [
        *_audit_violations(
            code_audit,
            kind="code",
            command=manifest.command,
            run_id=run_id,
        ),
        *_audit_violations(
            theory_audit,
            kind="theory",
            command=manifest.command,
            run_id=run_id,
        ),
    ]
    if audit_violations:
        return {
            "schema_version": 1,
            "kind": "gate_decision",
            "command": manifest.command,
            "run_id": run_id,
            "status": ValidationStatus.FAILED.value,
            "passed": False,
            "violations": audit_violations,
            "manifest_sha256": manifest.sha256,
        }

    recomputed = compare_results(manifest, results)
    if comparison != recomputed:
        return {
            "schema_version": 1,
            "kind": "gate_decision",
            "command": manifest.command,
            "run_id": run_id,
            "status": ValidationStatus.FAILED.value,
            "passed": False,
            "violations": [
                {
                    "code": "comparison_report_mismatch",
                    "message": (
                        "The stored comparison is stale or edited; regenerate it "
                        "from the current manifest and normalized results"
                    ),
                }
            ],
            "manifest_sha256": manifest.sha256,
            "supplied_comparison_sha256": sha256_json(comparison),
            "recomputed_comparison_sha256": sha256_json(recomputed),
        }

    if not recomputed["passed"]:
        return {
            "schema_version": 1,
            "kind": "gate_decision",
            "command": manifest.command,
            "run_id": run_id,
            "status": ValidationStatus.FAILED.value,
            "passed": False,
            "violations": recomputed["violations"],
            "manifest_sha256": manifest.sha256,
            "comparison_sha256": sha256_json(recomputed),
        }

    environment = collect_environment(workdir)
    dirty = environment["git"]["tracked_files_dirty"]
    if dirty is True and not allow_dirty:
        return {
            "schema_version": 1,
            "kind": "gate_decision",
            "command": manifest.command,
            "run_id": run_id,
            "status": ValidationStatus.FAILED.value,
            "passed": False,
            "violations": [
                {
                    "code": "dirty_worktree",
                    "message": (
                        "A release attestation requires a clean tracked worktree "
                        "or an explicitly authorized reproducible patch identity"
                    ),
                }
            ],
            "manifest_sha256": manifest.sha256,
            "comparison_sha256": sha256_json(recomputed),
            "environment": environment,
        }

    attestation: dict[str, Any] = {
        "schema_version": 1,
        "kind": "validation_attestation",
        "command": manifest.command,
        "run_id": run_id,
        "status": "passed",
        "passed": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": {
            "path": manifest.source_path,
            "sha256": manifest.sha256,
            "tolerances_frozen": manifest.tolerances_frozen,
        },
        "comparison_sha256": sha256_json(recomputed),
        "audits": {
            "code": {"sha256": sha256_json(code_audit)},
            "theory": {"sha256": sha256_json(theory_audit)},
        },
        "normalized_results": {
            result.engine: {"path": result.source_path, "sha256": result.sha256}
            for result in sorted(results, key=lambda item: item.engine)
        },
        "environment": environment,
        "dirty_worktree_authorized": allow_dirty,
    }
    attestation["attestation_sha256"] = sha256_json(attestation)
    return attestation
