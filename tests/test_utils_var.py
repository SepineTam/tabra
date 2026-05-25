import re

import pytest

from tabra.utils.var import resolve_var


def test_resolve_var_exact_single_name():
    cols = ["price", "mpg", "weight"]
    assert resolve_var("mpg", cols) == ["mpg"]


def test_resolve_var_exact_multiple_names_with_spaces():
    cols = ["price", "mpg", "weight"]
    assert resolve_var("price weight", cols) == ["price", "weight"]


def test_resolve_var_list_exact_names_keeps_input_order():
    cols = ["price", "mpg", "weight"]
    assert resolve_var(["weight", "price"], cols) == ["weight", "price"]


def test_resolve_var_exact_tokens_preserve_duplicates():
    cols = ["price", "mpg", "weight"]
    assert resolve_var("price price", cols) == ["price", "price"]


def test_resolve_var_regex_from_string_matches_multiple_columns():
    cols = ["price", "rep78", "gear_ratio", "repair_count"]
    assert resolve_var(r"^rep", cols) == ["rep78", "repair_count"]


def test_resolve_var_regex_from_list_uses_joined_pattern_with_spaces():
    cols = ["gross_income", "net_income", "income_tax", "ratio"]
    with pytest.raises(KeyError, match=r"Variable 'income _tax' not found"):
        resolve_var([r"income", r"_tax"], cols)


def test_resolve_var_mixed_exact_and_non_exact_tokens_fallback_to_regex():
    cols = ["x1", "x2", "x1_lag", "y"]
    # Because not all tokens are exact names, input falls back to regex search.
    # The combined raw string becomes "x1 missing", which does not match any column.
    with pytest.raises(KeyError, match=r"Variable 'x1 missing' not found"):
        resolve_var("x1 missing", cols)


def test_resolve_var_regex_substring_match_behaviour():
    cols = ["price", "sprinter", "impression", "mpg"]
    # re.search performs substring matching based on the provided pattern.
    assert resolve_var("pri", cols) == ["price", "sprinter"]


def test_resolve_var_regex_result_respects_var_list_order():
    cols = ["rep2", "x", "rep1", "rep3"]
    assert resolve_var(r"^rep", cols) == ["rep2", "rep1", "rep3"]


def test_resolve_var_empty_string_returns_empty_exact_token_list():
    cols = ["price", "mpg", "weight"]
    # Empty string splits into an empty token list; all([]) is True.
    assert resolve_var("", cols) == []


def test_resolve_var_empty_list_returns_empty_list():
    cols = ["price", "mpg", "weight"]
    assert resolve_var([], cols) == []


def test_resolve_var_raises_key_error_when_no_match():
    cols = ["price", "mpg", "weight"]
    with pytest.raises(KeyError, match=r"Variable 'horsepower' not found"):
        resolve_var("horsepower", cols)


def test_resolve_var_propagates_invalid_regex_error_for_string_input():
    cols = ["price", "mpg", "weight"]
    with pytest.raises(re.error):
        resolve_var("[", cols)


def test_resolve_var_propagates_invalid_regex_error_for_list_input():
    cols = ["price", "mpg", "weight"]
    with pytest.raises(re.error):
        resolve_var(["(", "price"], cols)
