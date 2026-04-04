"""Variable resolution utilities."""

import re


def resolve_var(var: str | list[str], var_list: list[str]) -> list[str]:
    """Resolve variable specification against available variable names.

    Tries exact match first. If every token is an exact column name, returns
    them directly. Otherwise, treats the raw input as a regex pattern and
    matches against all available names.

    Args:
        var: Variable specification — a single name, space-separated names,
            a regex pattern, or a list of names.
        var_list: Full list of available variable names (e.g. df.columns).

    Returns:
        list[str]: Resolved variable names.

    Example:
        >>> resolve_var("price mpg", ["price", "mpg", "weight"])
        ['price', 'mpg']
        >>> resolve_var("^r", ["price", "rep78", "gear_ratio"])
        ['rep78']
    """
    if isinstance(var, list):
        tokens = var
    else:
        tokens = var.split()

    # All tokens are exact names → return as-is
    if all(t in var_list for t in tokens):
        return tokens

    # Fallback: treat raw input as regex
    raw = " ".join(tokens) if isinstance(var, list) else var
    return [c for c in var_list if re.search(raw, c)]
