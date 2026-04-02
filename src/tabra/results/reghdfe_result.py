#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : reghdfe_result.py

import numpy as np
from tabra.results.base import BaseResult


class RegHDFEResult(BaseResult):

    def __init__(self, coef, std_err, t_stat, p_value,
                 r_squared, r_squared_adj, f_stat, f_pval,
                 resid, fitted, n_obs, k_vars, var_names,
                 SSR, SSE, SST, df_model, df_resid, mse, root_mse,
                 y_name="",
                 r2_within=0.0, r2_a_within=0.0,
                 df_a=0, n_hdfe=0, absorbed_fe=None):
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
        self._y_name = y_name
        self._r2_within = r2_within
        self._r2_a_within = r2_a_within
        self._df_a = df_a
        self._n_hdfe = n_hdfe
        self._absorbed_fe = absorbed_fe or []

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

    @property
    def df_model(self):
        return self._df_model

    @property
    def df_resid(self):
        return self._df_resid

    @property
    def mse(self):
        return self._mse

    @property
    def root_mse(self):
        return self._root_mse

    @property
    def r2_within(self):
        return self._r2_within

    @property
    def r2_a_within(self):
        return self._r2_a_within

    @property
    def df_a(self):
        return self._df_a

    @property
    def n_hdfe(self):
        return self._n_hdfe

    @property
    def absorbed_fe(self):
        return self._absorbed_fe

    def _summary_style_stata(self):
        from scipy import stats

        lines = []
        LW = 76

        def right(label, value, fmt="8.4f"):
            return f"{label:<{LW - 18}s}{value:>{fmt}}"

        def right_str(label, value):
            return f"{label:<{LW - 18}s}{value:>18s}"

        # Header
        lines.append(
            f"{'HDFE Linear regression':<{LW - 18}s}Number of obs   ={self._n_obs:>11d}"
        )
        lines.append(
            f"{'Absorbing ' + str(self._n_hdfe) + ' HDFE group' + ('s' if self._n_hdfe > 1 else ''):<{LW - 18}s}"
            + f"F({self._df_model:>3d}, {self._df_resid:>5d})={self._f_stat:>11.2f}"
        )
        lines.append(
            f"{'':<{LW - 18}s}Prob > F        ={self._f_pval:>11.4f}"
        )
        lines.append(right("R-squared", self._r_squared))
        lines.append(right("Adj R-squared", self._r_squared_adj))
        lines.append(right("Within R-sq.", self._r2_within))
        lines.append(right("Root MSE", self._root_mse))
        lines.append("")

        # Coefficient table
        lines.append("-" * LW)
        lines.append(
            f"{self._y_name:>12s} | {'Coef.':>10s} {'Std. Err.':>10s} {'t':>8s} {'P>|t|':>6s} {'[95% Conf. Interval]':>22s}"
        )
        lines.append("-" * LW)

        t_crit = stats.t.ppf(0.975, self._df_resid)
        for i, name in enumerate(self._var_names):
            c = self._coef[i]
            se = self._std_err[i]
            t = self._t_stat[i]
            p = self._p_value[i]
            lo = c - t_crit * se
            hi = c + t_crit * se
            lines.append(
                f"{name:>12s} | {c:>10.3f} {se:>10.3f} {t:>8.2f} {p:>6.3f} {lo:>11.3f} {hi:>11.3f}"
            )
        lines.append("-" * LW)

        # Absorbed FE footnote
        if self._absorbed_fe:
            lines.append("Absorbed degrees of freedom:")
            lines.append("-" * 53 + "+")
            lines.append(
                f" {'Absorbed FE':>12s} | {'Categories':>10s}  {'- Redundant':>12s}  = {'Num. Coefs':>10s} "
            )
            lines.append("-" * 53 + "+")
            for fe in self._absorbed_fe:
                nested_mark = " *" if fe.get("nested", False) else ""
                lines.append(
                    f" {fe['name']:>12s} | {fe['categories']:>10d}  {fe['redundant']:>12d}  {fe['num_coefs']:>10d} {nested_mark:>3s}"
                )
            lines.append("-" * 53 + "+")

        return "\n".join(lines)

    def summary(self):
        dispatch = {
            "stata": self._summary_style_stata,
        }
        method = dispatch.get(self._style, self._summary_style_stata)
        return method()

    def save(self, path="reghdfe_result.txt"):
        with open(path, "w") as f:
            f.write(self.summary())

    def __repr__(self):
        return self.summary()
