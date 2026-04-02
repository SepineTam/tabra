#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : mlogit_result.py

import numpy as np
from scipy import stats as sp_stats


class MLogitResult:
    def __init__(self, coef, std_err, z_stat, p_value,
                 ll, ll_0, pseudo_r2, chi2, chi2_pval,
                 n_obs, k_vars, k_cat, df_m,
                 var_names, y_name, categories, base_outcome,
                 converged, model_name, V):
        # coef: dict {category: array_of_coefs}
        self._coef = coef
        self._std_err = std_err
        self._z_stat = z_stat
        self._p_value = p_value
        self._ll = float(ll)
        self._ll_0 = float(ll_0)
        self._pseudo_r2 = float(pseudo_r2)
        self._chi2 = float(chi2)
        self._chi2_pval = float(chi2_pval)
        self._n_obs = int(n_obs)
        self._k_vars = int(k_vars)
        self._k_cat = int(k_cat)
        self._df_m = int(df_m)
        self._var_names = list(var_names)
        self._y_name = y_name
        self._categories = list(categories)
        self._base_outcome = base_outcome
        self._converged = bool(converged)
        self._model_name = model_name
        self._V = V

        self._style = "stata"
        self._display = False

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
    def k_cat(self):
        return self._k_cat

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
    def categories(self):
        return self._categories

    @property
    def base_outcome(self):
        return self._base_outcome

    @property
    def converged(self):
        return self._converged

    @property
    def model_name(self):
        return self._model_name

    @property
    def V(self):
        return self._V

    def set_style(self, style: str):
        self._style = style

    def set_display(self, is_display: bool = True):
        self._display = is_display
        if is_display:
            print(self.summary())

    def summary(self):
        dispatch = {
            "stata": self._summary_style_stata,
        }
        method = dispatch.get(self._style, self._summary_style_stata)
        return method()

    def _summary_style_stata(self):
        lines = []

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

        lines.append("")

        non_base = [c for c in self._categories if c != self._base_outcome]
        for cat in non_base:
            lines.append(f"--- Outcome {cat} (base = {self._base_outcome}) ---")
            lines.append(f"{'-' * 76}")
            lines.append(
                f"{self._y_name:>12s} | {'Coef.':>10s} {'Std. Err.':>10s} "
                f"{'z':>8s} {'P>|z|':>6s} {'[95% Conf. Interval]':>22s}"
            )
            lines.append(f"{'-' * 76}")

            z_crit = sp_stats.norm.ppf(0.975)
            for i, name in enumerate(self._var_names):
                c = self._coef[cat][i]
                se = self._std_err[cat][i]
                z = self._z_stat[cat][i]
                p = self._p_value[cat][i]
                lo = c - z_crit * se
                hi = c + z_crit * se
                lines.append(
                    f"{name:>12s} | {c:>10.4f} {se:>10.4f} {z:>8.2f} "
                    f"{p:>6.3f} {lo:>11.4f} {hi:>11.4f}"
                )
            lines.append(f"{'-' * 76}")
            lines.append("")

        return "\n".join(lines)

    def save(self, path="mlogit_result.txt"):
        with open(path, "w") as f:
            f.write(self.summary())

    def __repr__(self):
        return self.summary()
