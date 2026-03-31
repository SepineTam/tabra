#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : ols_result.py

import numpy as np
from tabra.results.base import BaseResult


class OLSResult(BaseResult):
    def __init__(self, coef, std_err, t_stat, p_value,
                 r_squared, r_squared_adj, f_stat, f_pval,
                 resid, fitted, n_obs, k_vars, var_names,
                 SSR, SSE, SST, df_model, df_resid, mse, root_mse):
        super().__init__()
        self._coef = coef
        self._std_err = std_err
        self._t_stat = t_stat
        self._p_value = p_value
        self._r_squared = r_squared
        self._r_squared_adj = r_squared_adj
        self._f_stat = f_stat
        self._f_pval = f_pval
        self._resid = resid
        self._fitted = fitted
        self._n_obs = n_obs
        self._k_vars = k_vars
        self._var_names = var_names
        self._SSR = SSR
        self._SSE = SSE
        self._SST = SST
        self._df_model = df_model
        self._df_resid = df_resid
        self._mse = mse
        self._root_mse = root_mse

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
    def r_squared(self):
        return self._r_squared

    @property
    def r_squared_adj(self):
        return self._r_squared_adj

    @property
    def f_stat(self):
        return self._f_stat

    @property
    def f_pval(self):
        return self._f_pval

    @property
    def resid(self):
        return self._resid

    @property
    def fitted(self):
        return self._fitted

    @property
    def n_obs(self):
        return self._n_obs

    @property
    def k_vars(self):
        return self._k_vars

    @property
    def var_names(self):
        return self._var_names

    @property
    def SSR(self):
        return self._SSR

    @property
    def SSE(self):
        return self._SSE

    @property
    def SST(self):
        return self._SST

    def _summary_style_stata(self):
        from scipy import stats

        lines = []
        ms_model = self._SSE / self._df_model if self._df_model > 0 else 0.0
        ms_resid = self._mse
        ms_total = self._SST / (self._n_obs - 1)

        lines.append(f"{'':>10s} Source | {'SS':>12s} {'df':>5s} {'MS':>12s}        Number of obs = {self._n_obs:>8d}")
        lines.append(f"{'':>10s}-------+{'-' * 40}    F({self._df_model}, {self._df_resid})  = {self._f_stat:>8.2f}")
        lines.append(f"{'':>10s}  Model | {self._SSE:>12.4f} {self._df_model:>5d} {ms_model:>12.6f}        Prob > F       = {self._f_pval:>7.4f}")
        lines.append(f"{'':>10s}Residual| {self._SSR:>12.4f} {self._df_resid:>5d} {ms_resid:>12.6f}        R-squared      = {self._r_squared:>7.4f}")
        lines.append(f"{'':>10s}-------+{'-' * 40}    Adj R-squared  = {self._r_squared_adj:>7.4f}")
        lines.append(f"{'':>10s}  Total | {self._SST:>12.4f} {self._n_obs - 1:>5d} {ms_total:>12.6f}        Root MSE       = {self._root_mse:>7.4f}")
        lines.append("")
        lines.append(f"{'-' * 76}")
        lines.append(f"{'':>12s} | {'Coef.':>10s} {'Std. Err.':>10s} {'t':>8s} {'P>|t|':>6s} {'[95% Conf. Interval]':>22s}")
        lines.append(f"{'-' * 76}")

        t_crit = stats.t.ppf(0.975, self._df_resid)
        for i, name in enumerate(self._var_names):
            c = self._coef[i]
            se = self._std_err[i]
            t = self._t_stat[i]
            p = self._p_value[i]
            lo = c - t_crit * se
            hi = c + t_crit * se
            lines.append(f"{name:>12s} | {c:>10.3f} {se:>10.3f} {t:>8.2f} {p:>6.3f} {lo:>11.3f} {hi:>11.3f}")

        lines.append(f"{'-' * 76}")
        return "\n".join(lines)

    def summary(self):
        dispatch = {
            "stata": self._summary_style_stata,
        }
        method = dispatch.get(self._style, self._summary_style_stata)
        return method()

    def save(self, path="ols_result.txt"):
        with open(path, "w") as f:
            f.write(self.summary())

    def __repr__(self):
        return self.summary()
