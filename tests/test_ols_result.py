#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : test_ols_result.py

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ols_result():
    """创建一个 OLSResult 实例用于测试"""
    from tabra.models.estimate.ols import OLS

    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    noise = np.random.randn(n) * 0.5
    y = 2 + 3 * x1 + 1.5 * x2 + noise

    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    model = OLS()
    return model.fit(df, y="y", x=["x1", "x2"], is_con=True)


def test_summary_contains_r_squared(ols_result):
    """测试 summary 包含 R-squared"""
    summary = ols_result.summary()
    assert "R-squared" in summary


def test_summary_contains_number_of_obs(ols_result):
    """测试 summary 包含 Number of obs"""
    summary = ols_result.summary()
    assert "Number of obs" in summary
    assert "100" in summary


def test_summary_contains_constant(ols_result):
    """测试 summary 包含 _cons"""
    summary = ols_result.summary()
    assert "_cons" in summary


def test_summary_contains_variable_names(ols_result):
    """测试 summary 包含变量名 x1 和 x2"""
    summary = ols_result.summary()
    assert "x1" in summary
    assert "x2" in summary


def test_repr_equals_summary(ols_result):
    """测试 __repr__ 等于 summary()"""
    assert repr(ols_result) == ols_result.summary()


def test_save_creates_file(tmp_path, ols_result):
    """测试 save 方法创建文件"""
    file_path = tmp_path / "ols_result.txt"
    ols_result.save(str(file_path))

    assert file_path.exists()
    content = file_path.read_text()
    assert "R-squared" in content
    assert "_cons" in content


class TestSetStyle:
    def test_default_style_is_stata(self, ols_result):
        assert ols_result._style == "stata"

    def test_set_style_changes_style(self, ols_result):
        ols_result.set_style("python")
        assert ols_result._style == "python"


class TestSetDisplay:
    def test_set_display_prints_summary(self, ols_result, capsys):
        ols_result.set_display(True)
        captured = capsys.readouterr()
        assert "R-squared" in captured.out

    def test_set_display_false_does_not_print(self, ols_result, capsys):
        ols_result.set_display(False)
        captured = capsys.readouterr()
        assert captured.out == ""


class TestStyleDispatch:
    def test_unknown_style_falls_back_to_stata(self, ols_result):
        ols_result.set_style("nonexistent")
        summary = ols_result.summary()
        assert "R-squared" in summary
        assert "_cons" in summary
