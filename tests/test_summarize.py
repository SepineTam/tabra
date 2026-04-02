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
        "cat_col": ["a", "b"] * 50,  # non-numeric, should be excluded
    })
    return df


# ============ Basic mode ============

def test_summarize_returns_result_object(sum_data):
    """summarize 应返回 SummarizeResult 对象"""
    from tabra import load_data

    tab = load_data(sum_data, is_display_result=False)
    result = tab.summarize()
    assert result is not None
    assert hasattr(result, "summary")


def test_summarize_default_all_numeric(sum_data):
    """var_list=None 应只包含数值列"""
    from tabra import load_data

    tab = load_data(sum_data, is_display_result=False)
    result = tab.summarize()
    assert "price" in result.var_names
    assert "mpg" in result.var_names
    assert "weight" in result.var_names
    assert "cat_col" not in result.var_names


def test_summarize_specified_vars(sum_data):
    """指定 var_list 应只包含指定列"""
    from tabra import load_data

    tab = load_data(sum_data, is_display_result=False)
    result = tab.summarize(var_list=["price", "mpg"])
    assert result.var_names == ["price", "mpg"]


def test_summarize_obs_count(sum_data):
    """Obs 应等于行数"""
    from tabra import load_data

    tab = load_data(sum_data, is_display_result=False)
    result = tab.summarize(var_list=["price"])
    assert result.obs["price"] == 100


def test_summarize_mean_matches_numpy(sum_data):
    """Mean 应与 numpy 一致"""
    from tabra import load_data

    tab = load_data(sum_data, is_display_result=False)
    result = tab.summarize(var_list=["price", "mpg"])
    assert np.isclose(result.mean["price"], sum_data["price"].mean(), atol=1e-10)
    assert np.isclose(result.mean["mpg"], sum_data["mpg"].mean(), atol=1e-10)


def test_summarize_std_matches_numpy(sum_data):
    """Std.Dev 应与 numpy ddof=1 一致"""
    from tabra import load_data

    tab = load_data(sum_data, is_display_result=False)
    result = tab.summarize(var_list=["price"])
    assert np.isclose(result.std["price"], sum_data["price"].std(ddof=1), atol=1e-10)


def test_summarize_min_max(sum_data):
    """Min/Max 应与实际一致"""
    from tabra import load_data

    tab = load_data(sum_data, is_display_result=False)
    result = tab.summarize(var_list=["price"])
    assert np.isclose(result.min_val["price"], sum_data["price"].min(), atol=1e-10)
    assert np.isclose(result.max_val["price"], sum_data["price"].max(), atol=1e-10)


def test_summarize_missing_values():
    """含缺失值的列，Obs 应排除 NaN"""
    from tabra import load_data

    np.random.seed(42)
    df = pd.DataFrame({"x": [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10]})
    tab = load_data(df, is_display_result=False)
    result = tab.summarize()
    assert result.obs["x"] == 8
    assert np.isclose(result.mean["x"], np.nanmean(df["x"]), atol=1e-10)


# ============ Detail mode ============

def test_summarize_detail_has_percentiles(sum_data):
    """detail=True 应包含 percentiles"""
    from tabra import load_data

    tab = load_data(sum_data, is_display_result=False)
    result = tab.summarize(var_list=["price"], detail=True)
    assert "1%" in result.percentiles["price"]
    assert "50%" in result.percentiles["price"]
    assert "99%" in result.percentiles["price"]


def test_summarize_detail_median(sum_data):
    """detail 50% 应与 numpy median 一致"""
    from tabra import load_data

    tab = load_data(sum_data, is_display_result=False)
    result = tab.summarize(var_list=["price"], detail=True)
    assert np.isclose(
        result.percentiles["price"]["50%"],
        np.median(sum_data["price"]),
        atol=1e-10,
    )


def test_summarize_detail_skewness(sum_data):
    """detail skewness 应与 scipy 一致"""
    from tabra import load_data
    from scipy.stats import skew

    tab = load_data(sum_data, is_display_result=False)
    result = tab.summarize(var_list=["price"], detail=True)
    assert np.isclose(
        result.skewness["price"],
        skew(sum_data["price"], bias=False),
        atol=1e-8,
    )


def test_summarize_detail_kurtosis(sum_data):
    """detail kurtosis 应与 scipy 一致"""
    from tabra import load_data
    from scipy.stats import kurtosis

    tab = load_data(sum_data, is_display_result=False)
    result = tab.summarize(var_list=["price"], detail=True)
    assert np.isclose(
        result.kurtosis["price"],
        kurtosis(sum_data["price"], bias=False),
        atol=1e-8,
    )


# ============ Summary display ============

def test_summarize_summary_contains_headers(sum_data):
    """summary 应包含 Obs/Mean/Std.Dev/Min/Max 表头"""
    from tabra import load_data

    tab = load_data(sum_data, is_display_result=False)
    result = tab.summarize(var_list=["price", "mpg"])
    s = result.summary()
    assert "Obs" in s
    assert "Mean" in s
    assert "Std. dev" in s
    assert "Min" in s
    assert "Max" in s


def test_summarize_summary_contains_var_names(sum_data):
    """summary 应包含变量名"""
    from tabra import load_data

    tab = load_data(sum_data, is_display_result=False)
    result = tab.summarize(var_list=["price", "mpg"])
    s = result.summary()
    assert "price" in s
    assert "mpg" in s
