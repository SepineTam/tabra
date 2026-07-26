"""Strict JSON-compatible YAML loaders for validation artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from .errors import SchemaError
from .models import (
    CommandManifest,
    EngineRunStatus,
    MetricSpec,
    NormalizedResult,
    RegistryEntry,
    ToleranceMode,
    ToleranceSpec,
    ValidationRegistry,
    freeze_json,
)

SCHEMA_VERSION = 1
REQUIRED_ENGINES = frozenset({"python", "stata", "r"})
_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA256 digest for a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON value deterministically for integrity hashing."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    """Return the SHA256 digest of a canonical JSON value."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    """Write a readable JSON artifact without allowing NaN or Infinity."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SchemaError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise SchemaError(f"Non-finite JSON number is not allowed: {value}")


def load_json_document(path: str | Path) -> dict[str, Any]:
    """Load strict JSON, also accepted when stored with a YAML extension."""

    source = Path(path)
    try:
        raw = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise SchemaError(f"Cannot read {source}: {exc}") from exc
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except json.JSONDecodeError as exc:
        raise SchemaError(
            f"{source} must contain JSON-compatible YAML: {exc.msg}"
        ) from exc
    if not isinstance(value, dict):
        raise SchemaError(f"{source} must contain a JSON object")
    return value


def _require_keys(value: dict[str, Any], keys: set[str], context: str) -> None:
    missing = sorted(keys - value.keys())
    if missing:
        raise SchemaError(f"{context} is missing required keys: {', '.join(missing)}")


def _validate_schema_version(value: Any, context: str) -> int:
    if type(value) is not int or value != SCHEMA_VERSION:
        raise SchemaError(
            f"{context}.schema_version must be the integer {SCHEMA_VERSION}"
        )
    return value


def _validate_identifier(value: Any, context: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise SchemaError(f"{context} must be a non-empty command-style identifier")
    return value


def _validate_sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise SchemaError(f"{context} must be a lowercase SHA256 digest")
    return value


def _load_tolerance(value: Any, context: str) -> ToleranceSpec:
    if not isinstance(value, dict):
        raise SchemaError(f"{context} must be an object")
    if value == {"exact": True}:
        return ToleranceSpec(mode=ToleranceMode.EXACT)

    allowed = {"atol", "rtol"}
    if not value or not set(value).issubset(allowed):
        raise SchemaError(
            f"{context} must be {{\"exact\": true}} or contain only atol/rtol"
        )
    atol = value.get("atol", 0.0)
    rtol = value.get("rtol", 0.0)
    for name, tolerance in (("atol", atol), ("rtol", rtol)):
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
            raise SchemaError(f"{context}.{name} must be numeric")
        if not math.isfinite(tolerance) or tolerance < 0:
            raise SchemaError(f"{context}.{name} must be finite and non-negative")
    return ToleranceSpec(
        mode=ToleranceMode.NUMERIC,
        atol=float(atol),
        rtol=float(rtol),
    )


def load_manifest(path: str | Path) -> CommandManifest:
    """Load and validate a frozen three-engine command manifest."""

    source = Path(path).resolve()
    value = load_json_document(source)
    _require_keys(
        value,
        {"schema_version", "command", "engines", "tolerances_frozen", "metrics"},
        "manifest",
    )
    schema_version = _validate_schema_version(value["schema_version"], "manifest")
    command = _validate_identifier(value["command"], "manifest.command")

    engines_value = value["engines"]
    if not isinstance(engines_value, list) or any(
        not isinstance(engine, str) for engine in engines_value
    ):
        raise SchemaError("manifest.engines must be a list of engine names")
    if len(engines_value) != len(set(engines_value)):
        raise SchemaError("manifest.engines must not contain duplicates")
    if set(engines_value) != REQUIRED_ENGINES:
        raise SchemaError("manifest.engines must contain exactly python, stata, and r")

    if value["tolerances_frozen"] is not True:
        raise SchemaError("manifest.tolerances_frozen must be true before any run")

    metrics_value = value["metrics"]
    if not isinstance(metrics_value, dict) or not metrics_value:
        raise SchemaError("manifest.metrics must be a non-empty object")
    metrics: list[MetricSpec] = []
    for metric_name, metric_value in metrics_value.items():
        _validate_identifier(metric_name, f"manifest.metrics key {metric_name!r}")
        if not isinstance(metric_value, dict):
            raise SchemaError(f"manifest.metrics.{metric_name} must be an object")
        _require_keys(metric_value, {"tolerance"}, f"manifest.metrics.{metric_name}")
        safety = metric_value.get("safety", {})
        if not isinstance(safety, dict) or not set(safety).issubset(
            {"reject_zero_mismatch"}
        ):
            raise SchemaError(
                f"manifest.metrics.{metric_name}.safety must contain only "
                "reject_zero_mismatch"
            )
        reject_zero_mismatch = safety.get("reject_zero_mismatch", False)
        if type(reject_zero_mismatch) is not bool:
            raise SchemaError(
                f"manifest.metrics.{metric_name}.safety.reject_zero_mismatch "
                "must be boolean"
            )
        metrics.append(
            MetricSpec(
                name=metric_name,
                tolerance=_load_tolerance(
                    metric_value["tolerance"],
                    f"manifest.metrics.{metric_name}.tolerance",
                ),
                reject_zero_mismatch=reject_zero_mismatch,
            )
        )

    return CommandManifest(
        schema_version=schema_version,
        command=command,
        engines=tuple(engines_value),
        metrics=tuple(metrics),
        tolerances_frozen=True,
        source_path=str(source),
        sha256=sha256_file(source),
    )


def load_registry(path: str | Path) -> ValidationRegistry:
    """Load a command registry and resolve manifests safely relative to it."""

    source = Path(path).resolve()
    value = load_json_document(source)
    _require_keys(value, {"schema_version", "commands"}, "registry")
    schema_version = _validate_schema_version(value["schema_version"], "registry")
    commands = value["commands"]
    if not isinstance(commands, dict) or not commands:
        raise SchemaError("registry.commands must be a non-empty object")

    entries: list[RegistryEntry] = []
    for command, entry_value in commands.items():
        _validate_identifier(command, f"registry command {command!r}")
        if not isinstance(entry_value, dict):
            raise SchemaError(f"registry.commands.{command} must be an object")
        _require_keys(entry_value, {"manifest"}, f"registry.commands.{command}")
        manifest_value = entry_value["manifest"]
        if not isinstance(manifest_value, str) or not manifest_value:
            raise SchemaError(f"registry.commands.{command}.manifest must be a path")
        enabled = entry_value.get("enabled", True)
        if type(enabled) is not bool:
            raise SchemaError(f"registry.commands.{command}.enabled must be boolean")

        relative_path = Path(manifest_value)
        if relative_path.is_absolute():
            raise SchemaError("registry manifest paths must be relative")
        manifest_path = (source.parent / relative_path).resolve()
        try:
            manifest_path.relative_to(source.parent)
        except ValueError as exc:
            raise SchemaError(
                f"registry.commands.{command}.manifest escapes the registry directory"
            ) from exc
        manifest = load_manifest(manifest_path)
        if manifest.command != command:
            raise SchemaError(
                f"Registry command {command} points to manifest for {manifest.command}"
            )
        entries.append(
            RegistryEntry(
                command=command,
                manifest_path=str(manifest_path),
                enabled=enabled,
            )
        )

    return ValidationRegistry(
        schema_version=schema_version,
        entries=tuple(entries),
        source_path=str(source),
        sha256=sha256_file(source),
    )


def load_normalized_result(path: str | Path) -> NormalizedResult:
    """Load and validate one engine's normalized scientific result."""

    source = Path(path).resolve()
    value = load_json_document(source)
    _require_keys(
        value,
        {
            "schema_version",
            "command",
            "engine",
            "status",
            "converged",
            "skipped",
            "sample",
            "metrics",
        },
        "normalized result",
    )
    schema_version = _validate_schema_version(value["schema_version"], "result")
    command = _validate_identifier(value["command"], "result.command")
    engine = value["engine"]
    if engine not in REQUIRED_ENGINES:
        raise SchemaError("result.engine must be python, stata, or r")
    try:
        status = EngineRunStatus(value["status"])
    except (TypeError, ValueError) as exc:
        raise SchemaError("result.status must be completed, failed, or skipped") from exc
    if type(value["converged"]) is not bool:
        raise SchemaError("result.converged must be boolean")
    if type(value["skipped"]) is not bool:
        raise SchemaError("result.skipped must be boolean")

    sample = value["sample"]
    if not isinstance(sample, dict):
        raise SchemaError("result.sample must be an object")
    _require_keys(sample, {"nobs", "data_sha256"}, "result.sample")
    nobs = sample["nobs"]
    if type(nobs) is not int or nobs < 0:
        raise SchemaError("result.sample.nobs must be a non-negative integer")
    data_sha256 = _validate_sha256(
        sample["data_sha256"], "result.sample.data_sha256"
    )

    metrics = value["metrics"]
    if not isinstance(metrics, dict):
        raise SchemaError("result.metrics must be an object")
    metadata = value.get("metadata", {})
    if not isinstance(metadata, dict):
        raise SchemaError("result.metadata must be an object")

    return NormalizedResult(
        schema_version=schema_version,
        command=command,
        engine=engine,
        status=status,
        converged=value["converged"],
        skipped=value["skipped"],
        nobs=nobs,
        data_sha256=data_sha256,
        metrics=freeze_json(metrics),
        metadata=freeze_json(metadata),
        source_path=str(source),
        sha256=sha256_file(source),
    )
