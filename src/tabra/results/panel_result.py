#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : panel_result.py

import numpy as np
from tabra.results.base import BaseResult


class PanelResult(BaseResult):
    def __init__(self, model_type, coef, std_err, t_stat, p_value,
                 r_squared, r_squared_adj, f_stat, f_pval,
                 resid, fitted, n_obs, k_vars, var_names,
                 SSR, SSE, SST, df_model, df_resid, mse, root_mse,
                 sigma_u=None, sigma_e=None, rho=None,
                 y_name="", **kwargs):
        super().__init__()
        self._model_type = model_type
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
        self._sigma_u = sigma_u
        self._sigma_e = sigma_e
        self._rho = rho
        self._y_name = y_name
        self._theta = kwargs.get("theta", None)
        self._n_groups = kwargs.get("n_groups", None)

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
    def sigma_u(self):
        return self._sigma_u

    @property
    def sigma_e(self):
        return self._sigma_e

    @property
    def rho(self):
        return self._rho

    @property
    def theta(self):
        return self._theta

    @property
    def n_groups(self):
        return self._n_groups

    def _summary_style_stata(self):
        from scipy import stats

        lines = []
        model_titles = {
            "fe": "Fixed-effects (within) regression",
            "re": "Random-effects GLS regression",
            "be": "Between regression (regression on group means)",
            "mle": "Random-effects ML regression",
            "pa": "GEE population-averaged model",
        }
        title = model_titles.get(self._model_type, "Panel regression")

        LW = 76
        RLW = 18

        def right(label, value, fmt="8.4f"):
            return f"{label:<{LW - RLW}s}{value:>{fmt}}"

        def right_str(label, value):
            return f"{label:<{LW - RLW}s}{value:>{RLW}s}"

        # Header
        lines.append(f"{title:<{LW - RLW}s}Number of obs     ={self._n_obs:>10d}")
        if self._n_groups is not None:
            lines.append(
                right_str("Group variable", "")
            )
            lines.append(
                f"{'':<{LW - RLW}s}Number of groups  ={self._n_groups:>10d}"
            )
        lines.append(right("R-squared", self._r_squared))
        if self._model_type in ("fe", "re"):
            lines.append(right("Adj R-squared", self._r_squared_adj))

        # F or chi2 test
        if self._model_type in ("fe", "be"):
            f_label = f"F({self._df_model}, {self._df_resid})"
            lines.append(right(f_label, self._f_stat, "8.2f"))
            lines.append(right("Prob > F", self._f_pval))
        elif self._model_type in ("re", "mle", "pa"):
            from scipy.stats import chi2
            wald_chi2 = float(self._coef @ np.linalg.inv(
                np.diag(self._std_err ** 2)) @ self._coef) if len(self._coef) > 0 else 0.0
            chi2_pval = 1 - chi2.cdf(wald_chi2, self._df_model)
            chi2_label = f"{'Wald' if self._model_type == 're' else 'LR' if self._model_type == 'mle' else 'Wald'} chi2({self._df_model})"
            lines.append(right(chi2_label, wald_chi2, "8.2f"))
            lines.append(right("Prob > chi2", chi2_pval))

        lines.append("")

        # Coefficient table
        lines.append("-" * LW)
        stat_label = "z" if self._model_type in ("re", "mle", "pa") else "t"
        lines.append(
            f"{self._y_name:>12s} | {'Coef.':>10s} {'Std. Err.':>10s} {stat_label:>8s} {'P>|' + stat_label + '|':>6s} {'[95% Conf. Interval]':>22s}"
        )
        lines.append("-" * LW)

        if stat_label == "z":
            t_crit = stats.norm.ppf(0.975)
            df_for_p = None
        else:
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

        # sigma_u, sigma_e, rho
        if self._sigma_u is not None and self._sigma_e is not None:
            lines.append(f"{'sigma_u':>12s} | {self._sigma_u:>10.4f}")
            lines.append(f"{'sigma_e':>12s} | {self._sigma_e:>10.4f}")
        if self._rho is not None:
            lines.append(
                f"{'rho':>12s} | {self._rho:>10.6f}   (fraction of variance due to u_i)"
            )

        return "\n".join(lines)

    def summary(self):
        dispatch = {
            "stata": self._summary_style_stata,
        }
        method = dispatch.get(self._style, self._summary_style_stata)
        return method()

    def save(self, path="panel_result.txt"):
        with open(path, "w") as f:
            f.write(self.summary())

    def __repr__(self):
        return self.summary()
