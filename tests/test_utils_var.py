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


def test_resolve_var_regex_from_string_matches_multiple_columns():
    cols = ["price", "rep78", "gear_ratio", "repair_count"]
    assert resolve_var(r"^rep", cols) == ["rep78", "repair_count"]


def test_resolve_var_regex_from_list_uses_joined_pattern_with_spaces():
    cols = ["gross_income", "net_income", "income_tax", "ratio"]
    with pytest.raises(KeyError, match=r"Variable 'income _tax' not found"):
        resolve_var([r"income", r"_tax"], cols)


def test_resolve_var_regex_substring_match_behaviour():
    cols = ["price", "sprinter", "impression", "mpg"]
    # re.search performs substring matching based on the provided pattern.
    assert resolve_var("pri", cols) == ["price", "sprinter"]


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
