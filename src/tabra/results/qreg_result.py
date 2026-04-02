#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : qreg_result.py

import numpy as np
from scipy import stats as sp_stats
from tabra.results.base import BaseResult


class QRegResult(BaseResult):
    def __init__(self, coef, std_err, t_stat, p_value, vce,
                 n_obs, k_vars, df_model, df_resid, var_names,
                 y_name, quantile, q_v, pseudo_r2,
                 sum_adev, sum_rdev, f_r, sparsity, bwidth,
                 resid, fitted):
        super().__init__()
        self._coef = coef
        self._std_err = std_err
        self._t_stat = t_stat
        self._p_value = p_value
        self._vce = vce
        self._n_obs = n_obs
        self._k_vars = k_vars
        self._df_model = df_model
        self._df_resid = df_resid
        self._var_names = var_names
        self._y_name = y_name
        self._quantile = quantile
        self._q_v = q_v
        self._pseudo_r2 = pseudo_r2
        self._sum_adev = sum_adev
        self._sum_rdev = sum_rdev
        self._f_r = f_r
        self._sparsity = sparsity
        self._bwidth = bwidth
        self._resid = resid
        self._fitted = fitted

    @property
    def coef(self):
        return self._coef

    @property
    def std_err(self):
        return self._std_err

    @property
    def t_stat(self):
        return self._t_stat

    @property
    def p_value(self):
        return self._p_value

    @property
    def vce(self):
        return self._vce

    @property
    def n_obs(self):
        return self._n_obs

    @property
    def k_vars(self):
        return self._k_vars

    @property
    def df_model(self):
        return self._df_model

    @property
    def df_resid(self):
        return self._df_resid

    @property
    def var_names(self):
        return self._var_names

    @property
    def quantile(self):
        return self._quantile

    @property
    def pseudo_r2(self):
        return self._pseudo_r2

    @property
    def sum_adev(self):
        return self._sum_adev

    @property
    def sum_rdev(self):
        return self._sum_rdev

    @property
    def sparsity(self):
        return self._sparsity

    @property
    def bwidth(self):
        return self._bwidth

    @property
    def resid(self):
        return self._resid

    @property
    def fitted(self):
        return self._fitted

    def _summary_style_stata(self):
        lines = []

        # Header
        if self._quantile == 0.5:
            title = "Median regression"
        else:
            title = f"{self._quantile} Quantile regression"

        lines.append(
            f"{title:<40s}"
            f"{'Number of obs':>14s} = {self._n_obs:>8d}"
        )

        lines.append(
            f"  Raw sum of deviations {self._sum_rdev:>9.1f} (about {self._q_v:.0f})"
        )
        lines.append(
            f"  Min sum of deviations {self._sum_adev:>9.1f}"
            f"{'':>{22 - len(f'{self._sum_adev:.1f}')}}"
            f"{'Pseudo R2':>14s} = {self._pseudo_r2:>8.4f}"
        )

        lines.append("")

        # Coefficient table
        lines.append(f"{'-' * 76}")
        lines.append(
            f"{self._y_name:>12s} | {'Coef.':>10s} {'Std. Err.':>10s} "
            f"{'t':>8s} {'P>|t|':>6s} {'[95% Conf. Interval]':>22s}"
        )
        lines.append(f"{'-' * 76}")

        t_crit = sp_stats.t.ppf(0.975, self._df_resid)
        for i, name in enumerate(self._var_names):
            c = self._coef[i]
            se = self._std_err[i]
            t = self._t_stat[i]
            p = self._p_value[i]
            lo = c - t_crit * se
            hi = c + t_crit * se
            lines.append(
                f"{name:>12s} | {c:>10.3f} {se:>10.3f} {t:>8.2f} "
                f"{p:>6.3f} {lo:>11.3f} {hi:>11.3f}"
            )

        lines.append(f"{'-' * 76}")
        return "\n".join(lines)

    def summary(self):
        dispatch = {
            "stata": self._summary_style_stata,
        }
        method = dispatch.get(self._style, self._summary_style_stata)
        return method()

    def save(self, path="qreg_result.txt"):
        with open(path, "w") as f:
            f.write(self.summary())

    def __repr__(self):
        return self.summary()
