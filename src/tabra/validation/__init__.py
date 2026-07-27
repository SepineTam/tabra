"""Scientific validation infrastructure for Tabra commands."""

from .compare import check_comparison_integrity, compare_results
from .errors import SchemaError, StateTransitionError, ValidationError
from .gate import collect_environment, evaluate_gate
from .io import (
    load_manifest,
    load_normalized_result,
    load_registry,
    sha256_file,
    sha256_json,
    write_json,
)
from .models import (
    CommandManifest,
    EngineRunStatus,
    MetricSpec,
    NormalizedResult,
    ToleranceMode,
    ToleranceSpec,
    ValidationRegistry,
    ValidationStatus,
    transition_status,
)

__all__ = [
    "CommandManifest",
    "EngineRunStatus",
    "MetricSpec",
    "NormalizedResult",
    "SchemaError",
    "StateTransitionError",
    "ToleranceMode",
    "ToleranceSpec",
    "ValidationError",
    "ValidationRegistry",
    "ValidationStatus",
    "check_comparison_integrity",
    "collect_environment",
    "compare_results",
    "evaluate_gate",
    "load_manifest",
    "load_normalized_result",
    "load_registry",
    "sha256_file",
    "sha256_json",
    "transition_status",
    "write_json",
]
