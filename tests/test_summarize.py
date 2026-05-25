#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : test_summarize.py

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sum_data():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "price": np.random.randn(n) * 1000 + 5000,
        "mpg": np.random.randn(n) * 5 + 20,
        "weight": np.random.randn(n) * 500 + 3000,
        "cat_col": ["a", "b"] * 50,  # Non-numeric column should be excluded
    })
    return df


# ============ Basic mode ============

def test_summarize_returns_result_object(sum_data):
    """sum should return a SummarizeResult-like object."""
    from tabra import load_data

    tab = load_data(sum_data, is_display_result=False)
    result = tab.data.sum()
    assert result is not None
    assert hasattr(result, "summary")


def test_summarize_default_all_numeric(sum_data):
    """When var_list is None, only numeric columns should be included."""
    from tabra import load_data

    tab = load_data(sum_data, is_display_result=False)
    result = tab.data.sum()
    assert "price" in result.var_names
    assert "mpg" in result.var_names
    assert "weight" in result.var_names
    assert "cat_col" not in result.var_names


def test_summarize_specified_vars(sum_data):
    """When var_list is specified, only those variables should be included."""
    from tabra import load_data

    tab = load_data(sum_data, is_display_result=False)
    result = tab.data.sum(var_list=["price", "mpg"])
    assert result.var_names == ["price", "mpg"]


def test_summarize_obs_count(sum_data):
    """Obs should equal the number of rows."""
    from tabra import load_data

    tab = load_data(sum_data, is_display_result=False)
    result = tab.data.sum(var_list=["price"])
    assert result.obs["price"] == 100


def test_summarize_mean_matches_numpy(sum_data):
    """Mean should match NumPy results."""
    from tabra import load_data

    tab = load_data(sum_data, is_display_result=False)
    result = tab.data.sum(var_list=["price", "mpg"])
    assert np.isclose(result.mean["price"], sum_data["price"].mean(), atol=1e-10)
    assert np.isclose(result.mean["mpg"], sum_data["mpg"].mean(), atol=1e-10)


def test_summarize_std_matches_numpy(sum_data):
    """Std. dev should match NumPy with ddof=1."""
    from tabra import load_data

    tab = load_data(sum_data, is_display_result=False)
    result = tab.data.sum(var_list=["price"])
    assert np.isclose(result.std["price"], sum_data["price"].std(ddof=1), atol=1e-10)


def test_summarize_min_max(sum_data):
    """Min and max should match actual data values."""
    from tabra import load_data

    tab = load_data(sum_data, is_display_result=False)
    result = tab.data.sum(var_list=["price"])
    assert np.isclose(result.min_val["price"], sum_data["price"].min(), atol=1e-10)
    assert np.isclose(result.max_val["price"], sum_data["price"].max(), atol=1e-10)


def test_summarize_missing_values():
    """For columns with missing values, Obs should exclude NaN."""
    from tabra import load_data

    np.random.seed(42)
    df = pd.DataFrame({"x": [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10]})
    tab = load_data(df, is_display_result=False)
    result = tab.data.sum()
    assert result.obs["x"] == 8
    assert np.isclose(result.mean["x"], np.nanmean(df["x"]), atol=1e-10)


# ============ Detail mode ============

def test_summarize_detail_has_percentiles(sum_data):
    """detail=True should include percentiles."""
    from tabra import load_data

    tab = load_data(sum_data, is_display_result=False)
    result = tab.data.sum(var_list=["price"], detail=True)
    assert "1%" in result.percentiles["price"]
    assert "50%" in result.percentiles["price"]
    assert "99%" in result.percentiles["price"]


def test_summarize_detail_median(sum_data):
    """The detail 50% percentile should match NumPy median."""
    from tabra import load_data

    tab = load_data(sum_data, is_display_result=False)
    result = tab.data.sum(var_list=["price"], detail=True)
    assert np.isclose(
        result.percentiles["price"]["50%"],
        np.median(sum_data["price"]),
        atol=1e-10,
    )


def test_summarize_detail_skewness(sum_data):
    """Detail skewness should match SciPy."""
    from tabra import load_data
    from scipy.stats import skew

    tab = load_data(sum_data, is_display_result=False)
    result = tab.data.sum(var_list=["price"], detail=True)
    assert np.isclose(
        result.skewness["price"],
        skew(sum_data["price"], bias=False),
        atol=1e-8,
    )


def test_summarize_detail_kurtosis(sum_data):
    """Detail kurtosis should match SciPy."""
    from tabra import load_data
    from scipy.stats import kurtosis

    tab = load_data(sum_data, is_display_result=False)
    result = tab.data.sum(var_list=["price"], detail=True)
    assert np.isclose(
        result.kurtosis["price"],
        kurtosis(sum_data["price"], bias=False),
        atol=1e-8,
    )


# ============ Summary display ============

def test_summarize_summary_contains_headers(sum_data):
    """Summary output should contain core statistic headers."""
    from tabra import load_data

    tab = load_data(sum_data, is_display_result=False)
    result = tab.data.sum(var_list=["price", "mpg"])
    s = result.summary()
    assert "Obs" in s
    assert "Mean" in s
    assert "Std. dev" in s
    assert "Min" in s
    assert "Max" in s


def test_summarize_summary_contains_var_names(sum_data):
    """Summary output should contain variable names."""
    from tabra import load_data

    tab = load_data(sum_data, is_display_result=False)
    result = tab.data.sum(var_list=["price", "mpg"])
    s = result.summary()
    assert "price" in s
    assert "mpg" in s
