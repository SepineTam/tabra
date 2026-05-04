"""Tests for DataOps.count()."""
import pandas as pd
import numpy as np
import pytest

from tabra import load_data


@pytest.fixture
def tab():
    df = pd.read_stata(".local/data/auto.dta", convert_categoricals=False)
    return load_data(df, is_display_result=False)


class TestCount:
    def test_count_all(self, tab):
        """count() returns total number of observations."""
        result = tab.data.count()
        assert result == 74

    def test_count_with_condition(self, tab):
        """count('price > 6000') returns filtered count."""
        result = tab.data.count("price > 6000")
        assert result == 23

    def test_count_equality(self, tab):
        """count('foreign == 1') counts foreign cars."""
        result = tab.data.count("foreign == 1")
        assert result == 22

    def test_count_with_missing(self, tab):
        """count() includes rows with missing values."""
        assert tab.data.count() == 74

    def test_count_condition_missing(self, tab):
        """count with condition involving missing values."""
        result = tab.data.count("rep78 > 3")
        assert isinstance(result, int)

    def test_count_returns_int(self, tab):
        """count always returns int, not float or numpy type."""
        result = tab.data.count()
        assert type(result) is int

    def test_count_zero_results(self, tab):
        """count condition matching nothing returns 0."""
        result = tab.data.count("price > 999999")
        assert result == 0

    def test_count_compound_condition(self, tab):
        """count with compound condition (and/or)."""
        result = tab.data.count("price > 5000 & foreign == 1")
        assert isinstance(result, int)
        assert result > 0
