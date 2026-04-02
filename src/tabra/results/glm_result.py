#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : glm_result.py

import numpy as np
from scipy import stats as sp_stats
from tabra.results.base import BaseResult


class GLMResult(BaseResult):
    def __init__(self, coef, std_err, z_stat, p_value,
                 ll, ll_0, pseudo_r2, chi2, chi2_pval,
                 n_obs, k_vars, df_m, var_names,
                 y_name="", model_name="GLM",
                 converged=True, n_iter=0,
                 family="gaussian", link="identity",
                 deviance=0.0, null_deviance=0.0,
                 V=None):
        super().__init__()
        self._coef = coef
        self._std_err = std_err
        self._z_stat = z_stat
        self._p_value = p_value
        self._ll = ll
        self._ll_0 = ll_0
        self._pseudo_r2 = pseudo_r2
        self._chi2 = chi2
        self._chi2_pval = chi2_pval
        self._n_obs = n_obs
        self._k_vars = k_vars
        self._df_m = df_m
        self._var_names = var_names
        self._y_name = y_name
        self._model_name = model_name
        self._converged = converged
        self._n_iter = n_iter
        self._family = family
        self._link = link
        self._deviance = deviance
        self._null_deviance = null_deviance
        self._V = V

    @property
    def coef(self):
        return self._coef

    @property
    def std_err(self):
        return self._std_err

    @property
    def z_stat(self):
        return self._z_stat

    @property
    def p_value(self):
        return self._p_value

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
    def chi2_pval(self):
        return self._chi2_pval

    @property
    def n_obs(self):
        return self._n_obs

    @property
    def k_vars(self):
        return self._k_vars

    @property
    def df_m(self):
        return self._df_m

    @property
    def var_names(self):
        return self._var_names

    @property
    def y_name(self):
        return self._y_name

    @property
    def model_name(self):
        return self._model_name

    @property
    def converged(self):
        return self._converged

    @property
    def family(self):
        return self._family

    @property
    def link(self):
        return self._link

    @property
    def deviance(self):
        return self._deviance

    @property
    def null_deviance(self):
        return self._null_deviance

    @property
    def V(self):
        return self._V

    def _summary_style_stata(self):
        lines = []

        # Header
        lines.append(
            f"{self._model_name:<40s}"
            f"{'Number of obs':>14s} = {self._n_obs:>8d}"
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
            f"{'Log likelihood = ' + f'{self._ll:.6f}':<40s}"
            f"{'Pseudo R2':>14s} = {self._pseudo_r2:>8.4f}"
        )

        lines.append(
            f"{'Family: ' + self._family + ', Link: ' + self._link:<40s}"
            f"{'Deviance':>14s} = {self._deviance:>8.4f}"
        )

        if not self._converged:
            lines.append("WARNING: Convergence not achieved")

        lines.append("")

        # Coefficient table
        lines.append(f"{'-' * 76}")
        lines.append(
            f"{self._y_name:>12s} | {'Coef.':>10s} {'Std. Err.':>10s} "
            f"{'z':>8s} {'P>|z|':>6s} {'[95% Conf. Interval]':>22s}"
        )
        lines.append(f"{'-' * 76}")

        z_crit = sp_stats.norm.ppf(0.975)
        for i, name in enumerate(self._var_names):
            c = self._coef[i]
            se = self._std_err[i]
            z = self._z_stat[i]
            p = self._p_value[i]
            lo = c - z_crit * se
            hi = c + z_crit * se
            lines.append(
                f"{name:>12s} | {c:>10.4f} {se:>10.4f} {z:>8.2f} "
                f"{p:>6.3f} {lo:>11.4f} {hi:>11.4f}"
            )

        lines.append(f"{'-' * 76}")
        return "\n".join(lines)

    def summary(self):
        dispatch = {
            "stata": self._summary_style_stata,
        }
        method = dispatch.get(self._style, self._summary_style_stata)
        return method()

    def save(self, path="glm_result.txt"):
        with open(path, "w") as f:
            f.write(self.summary())

    def __repr__(self):
        return self.summary()
