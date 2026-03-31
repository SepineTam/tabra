#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : test_stats.py

import pytest
from scipy import stats as sp_stats


class TestTCdf:
    """测试 t 分布 CDF 函数"""

    def test_positive_value(self):
        """测试正值 t=1.96, df=100"""
        from tabra.ops.stats import t_cdf

        result = t_cdf(1.96, 100)
        expected = sp_stats.t.cdf(1.96, 100)
        assert abs(result - expected) < 1e-10

    def test_negative_value(self):
        """测试负值 t=-1.96, df=100"""
        from tabra.ops.stats import t_cdf

        result = t_cdf(-1.96, 100)
        expected = sp_stats.t.cdf(-1.96, 100)
        assert abs(result - expected) < 1e-10

    def test_zero_value(self):
        """测试零值 t=0.0, df=30，应返回 0.5"""
        from tabra.ops.stats import t_cdf

        result = t_cdf(0.0, 30)
        assert abs(result - 0.5) < 1e-10

    def test_extreme_value(self):
        """测试极端值 t=100, df=5，应接近 1.0"""
        from tabra.ops.stats import t_cdf

        result = t_cdf(100, 5)
        assert result > 0.99999


class TestFCdf:
    """测试 F 分布 CDF 函数"""

    def test_basic_value(self):
        """测试基本值 f=3.0, df1=2, df2=50，与 scipy 对比"""
        from tabra.ops.stats import f_cdf

        result = f_cdf(3.0, 2, 50)
        expected = sp_stats.f.cdf(3.0, 2, 50)
        assert abs(result - expected) < 1e-10

    def test_zero_value(self):
        """测试零值 f=0.0"""
        from tabra.ops.stats import f_cdf

        result = f_cdf(0.0, 5, 10)
        expected = sp_stats.f.cdf(0.0, 5, 10)
        assert abs(result - expected) < 1e-10


class TestTPval:
    """测试 t 分布 p-value 函数"""

    def test_basic_value(self):
        """测试基本值 t=1.96, df=1000，约等于 0.05"""
        from tabra.ops.stats import t_pval

        result = t_pval(1.96, 1000)
        expected = 2 * (1 - sp_stats.t.cdf(abs(1.96), 1000))
        assert abs(result - expected) < 1e-10
        assert abs(result - 0.05) < 0.005  # 约等于 0.05

    def test_zero_value(self):
        """测试零值 t=0，应返回 1.0"""
        from tabra.ops.stats import t_pval

        result = t_pval(0.0, 30)
        assert abs(result - 1.0) < 1e-10

    def test_extreme_value(self):
        """测试极端值 t=100, df=5，应接近 0.0"""
        from tabra.ops.stats import t_pval

        result = t_pval(100, 5)
        assert result < 0.0001

    def test_negative_value(self):
        """测试负值，应与正值相同（双尾）"""
        from tabra.ops.stats import t_pval

        result_pos = t_pval(1.96, 100)
        result_neg = t_pval(-1.96, 100)
        assert abs(result_pos - result_neg) < 1e-10


class TestFPval:
    """测试 F 分布 p-value 函数"""

    def test_basic_value(self):
        """测试基本值，与 scipy 对比"""
        from tabra.ops.stats import f_pval

        result = f_pval(3.0, 2, 50)
        expected = 1 - sp_stats.f.cdf(3.0, 2, 50)
        assert abs(result - expected) < 1e-10

    def test_zero_value(self):
        """测试零值 f=0，应返回 1.0"""
        from tabra.ops.stats import f_pval

        result = f_pval(0.0, 5, 10)
        assert abs(result - 1.0) < 1e-10

    def test_large_value(self):
        """测试大值"""
        from tabra.ops.stats import f_pval

        result = f_pval(100.0, 3, 30)
        expected = 1 - sp_stats.f.cdf(100.0, 3, 30)
        assert abs(result - expected) < 1e-10
