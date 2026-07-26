"""Immutable models used by the scientific validation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping

from .errors import StateTransitionError


class ValidationStatus(StrEnum):
    """Lifecycle states for one command validation run."""

    DISCOVERED = "discovered"
    SPEC_FROZEN = "spec_frozen"
    PREFLIGHT_PASSED = "preflight_passed"
    COMPARISON_PASSED = "comparison_passed"
    GATED = "gated"
    BLOCKED_ENV = "blocked_env"
    FAILED = "failed"


_LEGAL_TRANSITIONS: dict[ValidationStatus, frozenset[ValidationStatus]] = {
    ValidationStatus.DISCOVERED: frozenset(
        {
            ValidationStatus.SPEC_FROZEN,
            ValidationStatus.BLOCKED_ENV,
            ValidationStatus.FAILED,
        }
    ),
    ValidationStatus.SPEC_FROZEN: frozenset(
        {
            ValidationStatus.PREFLIGHT_PASSED,
            ValidationStatus.BLOCKED_ENV,
            ValidationStatus.FAILED,
        }
    ),
    ValidationStatus.PREFLIGHT_PASSED: frozenset(
        {ValidationStatus.COMPARISON_PASSED, ValidationStatus.FAILED}
    ),
    ValidationStatus.COMPARISON_PASSED: frozenset(
        {ValidationStatus.GATED, ValidationStatus.FAILED}
    ),
    ValidationStatus.GATED: frozenset(),
    ValidationStatus.BLOCKED_ENV: frozenset(),
    ValidationStatus.FAILED: frozenset(),
}


def transition_status(
    current: ValidationStatus | str,
    target: ValidationStatus | str,
) -> ValidationStatus:
    """Validate and return a workflow transition.

    Terminal states cannot be reopened. A new validation run must be created
    after either success or failure, preserving the audit history.
    """

    current_status = ValidationStatus(current)
    target_status = ValidationStatus(target)
    if target_status not in _LEGAL_TRANSITIONS[current_status]:
        raise StateTransitionError(
            f"Illegal validation transition: {current_status.value} -> "
            f"{target_status.value}"
        )
    return target_status


class EngineRunStatus(StrEnum):
    """Allowed execution states in a normalized engine result."""

    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ToleranceMode(StrEnum):
    """Comparison strategies frozen in the command manifest."""

    EXACT = "exact"
    NUMERIC = "numeric"


@dataclass(frozen=True, slots=True)
class ToleranceSpec:
    """An immutable exact or numeric tolerance specification."""

    mode: ToleranceMode
    atol: float = 0.0
    rtol: float = 0.0


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """One required output metric and its frozen tolerance."""

    name: str
    tolerance: ToleranceSpec
    reject_zero_mismatch: bool = False


@dataclass(frozen=True, slots=True)
class CommandManifest:
    """Validated, immutable command comparison specification."""

    schema_version: int
    command: str
    engines: tuple[str, ...]
    metrics: tuple[MetricSpec, ...]
    tolerances_frozen: bool
    source_path: str
    sha256: str

    def metric(self, name: str) -> MetricSpec:
        """Return a metric by name."""

        for metric in self.metrics:
            if metric.name == name:
                return metric
        raise KeyError(name)


@dataclass(frozen=True, slots=True)
class RegistryEntry:
    """One command registered for autonomous validation."""

    command: str
    manifest_path: str
    enabled: bool


@dataclass(frozen=True, slots=True)
class ValidationRegistry:
    """Validated registry of command manifests."""

    schema_version: int
    entries: tuple[RegistryEntry, ...]
    source_path: str
    sha256: str


def freeze_json(value: Any) -> Any:
    """Recursively make parsed JSON values immutable."""

    if isinstance(value, dict):
        return MappingProxyType({key: freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(freeze_json(item) for item in value)
    return value


def thaw_json(value: Any) -> Any:
    """Convert immutable JSON values back to serializable containers."""

    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class NormalizedResult:
    """Engine-independent result required for scientific comparison."""

    schema_version: int
    command: str
    engine: str
    status: EngineRunStatus
    converged: bool
    skipped: bool
    nobs: int
    data_sha256: str
    metrics: Mapping[str, Any]
    metadata: Mapping[str, Any]
    source_path: str
    sha256: str
