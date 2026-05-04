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


class TestAnovaAlignment:
    @pytest.fixture
    def large_result(self):
        """Data with large SS values (like auto.dta) to test column overflow."""
        from tabra import load_data
        np.random.seed(42)
        n = 74
        weight = np.random.randn(n) * 1000 + 3000
        mpg = np.random.randn(n) * 10 + 20
        price = 1000 + 2 * weight - 50 * mpg + np.random.randn(n) * 2000
        df = pd.DataFrame({"price": price, "weight": weight, "mpg": mpg})
        data = load_data(df, is_display_result=False)
        return data.est.reg("price", ["weight", "mpg"])

    def test_right_side_equals_aligned(self, large_result):
        summary = large_result.summary()
        eq_cols = []
        for line in summary.split("\n"):
            if "=" in line:
                eq_cols.append(line.index("="))
        assert len(set(eq_cols)) == 1, (
            f"= signs not aligned at columns: {eq_cols}"
        )

    def test_pipe_aligned_in_anova(self, large_result):
        summary = large_result.summary()
        pipe_cols = []
        for line in summary.split("\n"):
            s = line.lstrip()
            if any(s.startswith(w) for w in ("Source", "Model", "Residual", "Total")):
                pipe_cols.append(line.index("|"))
        assert len(set(pipe_cols)) == 1, (
            f"| not aligned at columns: {pipe_cols}"
        )


class TestAnovaNoLeadingSpaces:
    def test_source_line_no_extra_leading_spaces(self, ols_result):
        summary = ols_result.summary()
        for line in summary.split("\n"):
            if "Source" in line:
                leading = len(line) - len(line.lstrip())
                assert leading <= 8, (
                    f"Source line has excessive leading spaces ({leading}): {repr(line)}"
                )
                break
        else:
            pytest.fail("No Source line found")

    def test_model_line_no_extra_leading_spaces(self, ols_result):
        summary = ols_result.summary()
        for line in summary.split("\n"):
            stripped = line.lstrip()
            if stripped.startswith("Model"):
                leading = len(line) - len(line.lstrip())
                assert leading <= 8, (
                    f"Model line has excessive leading spaces ({leading}): {repr(line)}"
                )
                break
        else:
            pytest.fail("No Model line found")


class TestYNameInSummary:
    def test_y_name_in_coef_header(self, ols_result):
        summary = ols_result.summary()
        lines = summary.split("\n")
        for line in lines:
            if "Coef." in line:
                assert "y" in line
                break
        else:
            pytest.fail("No coefficient header line found")

    def test_y_name_passed_through_api(self):
        from tabra import load_data
        np.random.seed(42)
        n = 50
        w = np.random.randn(n)
        m = np.random.randn(n)
        p = 3 + 2 * w - 1.5 * m + np.random.randn(n) * 0.3
        df = pd.DataFrame({"price": p, "weight": w, "mpg": m})
        data = load_data(df, is_display_result=False)
        result = data.est.reg("price", ["weight", "mpg"])
        summary = result.summary()
        lines = summary.split("\n")
        for line in lines:
            if "Coef." in line:
                assert "price" in line
                break
        else:
            pytest.fail("No coefficient header line found")


class TestConsOrdering:
    def test_cons_is_last_in_var_names(self, ols_result):
        assert ols_result.var_names[-1] == "_cons"

    def test_cons_is_last_line_in_summary(self, ols_result):
        summary = ols_result.summary()
        lines = [l for l in summary.split("\n") if l.strip() and not l.startswith("-")]
        coef_lines = []
        in_coef = False
        for line in lines:
            if "Coef." in line:
                in_coef = True
                continue
            if in_coef and "Source" in line:
                break
            if in_coef:
                coef_lines.append(line)
        last_coef = coef_lines[-1].strip()
        assert last_coef.startswith("_cons")


class TestStyleDispatch:
    def test_unknown_style_falls_back_to_stata(self, ols_result):
        ols_result.set_style("nonexistent")
        summary = ols_result.summary()
        assert "R-squared" in summary
        assert "_cons" in summary
