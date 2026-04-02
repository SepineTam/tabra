#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : tobit_result.py

import numpy as np
from scipy import stats as sp_stats
from tabra.results.base import BaseResult


class TobitResult(BaseResult):
    def __init__(self, coef, std_err, t_stat, p_value,
                 sigma, var_e, se_sigma,
                 ll, ll_0, pseudo_r2, chi2, chi2_pval,
                 n_obs, n_unc, n_lc, n_rc,
                 k_vars, df_m, var_names,
                 y_name="", converged=True,
                 ll_limit=None, ul_limit=None):
        super().__init__()
        self._coef = coef
        self._std_err = std_err
        self._t_stat = t_stat
        self._p_value = p_value
        self._sigma = sigma
        self._var_e = var_e
        self._se_sigma = se_sigma
        self._ll = ll
        self._ll_0 = ll_0
        self._pseudo_r2 = pseudo_r2
        self._chi2 = chi2
        self._chi2_pval = chi2_pval
        self._n_obs = n_obs
        self._n_unc = n_unc
        self._n_lc = n_lc
        self._n_rc = n_rc
        self._k_vars = k_vars
        self._df_m = df_m
        self._var_names = var_names
        self._y_name = y_name
        self._converged = converged
        self._ll_limit = ll_limit
        self._ul_limit = ul_limit

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
    def sigma(self):
        return self._sigma

    @property
    def var_e(self):
        return self._var_e

    @property
    def ll(self):
        return self._ll

    @property
    def ll_0(self):
        return self._ll_0

    @property
    def pseudo_r2(self):
        return self._pseudo_r2

    @property
    def chi2(self):
        return self._chi2

    @property
    def n_obs(self):
        return self._n_obs

    @property
    def n_unc(self):
        return self._n_unc

    @property
    def n_lc(self):
        return self._n_lc

    @property
    def n_rc(self):
        return self._n_rc

    @property
    def k_vars(self):
        return self._k_vars

    @property
    def var_names(self):
        return self._var_names

    @property
    def converged(self):
        return self._converged

    def _summary_style_stata(self):
        lines = []

        # Limits display
        ll_str = str(self._ll_limit) if self._ll_limit is not None else "-inf"
        ul_str = str(self._ul_limit) if self._ul_limit is not None else "+inf"

        # Header block
        lines.append(
            f"{'Tobit regression':<40s}"
            f"{'Number of obs':>14s} = {self._n_obs:>8d}"
        )
        lines.append(
            f"{'':40s}"
            f"{'Uncensored':>14s} = {self._n_unc:>8d}"
        )
        lines.append(
            f"{'Limits: Lower = ' + ll_str:<40s}"
            f"{'Left-censored':>14s} = {self._n_lc:>8d}"
        )
        lines.append(
            f"{'        Upper = ' + ul_str:<40s}"
            f"{'Right-censored':>14s} = {self._n_rc:>8d}"
        )
        chi2_label = f"LR chi2({self._df_m})"
        lines.append(
            f"{'':40s}"
            f"{chi2_label:>14s} = {self._chi2:>8.2f}"
        )
        lines.append(
            f"{'':40s}"
            f"{'Prob > chi2':>14s} = {self._chi2_pval:>8.4f}"
        )
        lines.append(
            f"{'Log likelihood = ' + f'{self._ll:.5f}':<40s}"
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

        t_crit = sp_stats.t.ppf(0.975, df=self._n_obs - self._k_vars)
        for i, name in enumerate(self._var_names):
            c = self._coef[i]
            se = self._std_err[i]
            t = self._t_stat[i]
            p = self._p_value[i]
            lo = c - t_crit * se
            hi = c + t_crit * se
            lines.append(
                f"{name:>12s} | {c:>10.4f} {se:>10.4f} {t:>8.2f} "
                f"{p:>6.3f} {lo:>11.4f} {hi:>11.4f}"
            )

        lines.append(f"{'-' * 76}")

        # sigma^2 row
        se_var_e = 2 * self._sigma * self._se_sigma
        lo_var_e = self._var_e - t_crit * se_var_e
        hi_var_e = self._var_e + t_crit * se_var_e
        var_label = f"var(e.{self._y_name})"
        lines.append(
            f"{var_label:>12s} | {self._var_e:>10.5f} {se_var_e:>10.6f} "
            f"{'':>8s} {'':>6s} {lo_var_e:>11.5f} {hi_var_e:>11.5f}"
        )
        lines.append(f"{'-' * 76}")

        return "\n".join(lines)

    def summary(self):
        dispatch = {
            "stata": self._summary_style_stata,
        }
        method = dispatch.get(self._style, self._summary_style_stata)
        return method()

    def save(self, path="tobit_result.txt"):
        with open(path, "w") as f:
            f.write(self.summary())

    def __repr__(self):
        return self.summary()
