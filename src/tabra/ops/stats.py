#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : stats.py

from scipy import stats as sp_stats


def t_cdf(t, df):
    """Compute the CDF of Student's t-distribution.

    Args:
        t: t-statistic value
        df: degrees of freedom

    Returns:
        CDF value at t
    """
    return float(sp_stats.t.cdf(t, df))


def f_cdf(f, df1, df2):
    """Compute the CDF of F-distribution.

    Args:
        f: F-statistic value
        df1: numerator degrees of freedom
        df2: denominator degrees of freedom

    Returns:
        CDF value at f
    """
    return float(sp_stats.f.cdf(f, df1, df2))


def t_pval(t, df):
    """Compute two-tailed p-value for t-statistic.

    Args:
        t: t-statistic value
        df: degrees of freedom

    Returns:
        Two-tailed p-value
    """
    return float(2 * (1 - sp_stats.t.cdf(abs(t), df)))


def f_pval(f, df1, df2):
    """Compute p-value for F-statistic (right-tailed).

    Args:
        f: F-statistic value
        df1: numerator degrees of freedom
        df2: denominator degrees of freedom

    Returns:
        Right-tailed p-value
    """
    return float(1 - sp_stats.f.cdf(f, df1, df2))
