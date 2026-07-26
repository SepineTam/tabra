"""Exceptions raised by the scientific validation infrastructure."""


class ValidationError(ValueError):
    """Base class for validation configuration and result errors."""


class SchemaError(ValidationError):
    """Raised when a registry, manifest, or result violates its schema."""


class StateTransitionError(ValidationError):
    """Raised when a validation workflow attempts an illegal transition."""
