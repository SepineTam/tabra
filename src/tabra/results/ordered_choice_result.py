#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : ordered_choice_result.py

import numpy as np


class OrderedChoiceResult:
    def __init__(self, coef, std_err, z_stat, p_value,
                 cutpoints, cutpoint_se, cutpoint_z, cutpoint_p,
                 ll, ll_0, pseudo_r2, chi2, chi2_pval,
                 n_obs, k_vars, k_cat, df_m,
                 var_names, y_name, categories,
                 converged, model_name, V):
        self.coef = np.asarray(coef)
        self.std_err = np.asarray(std_err)
        self.z_stat = np.asarray(z_stat)
        self.p_value = np.asarray(p_value)
        self.cutpoints = np.asarray(cutpoints)
        self.cutpoint_se = np.asarray(cutpoint_se)
        self.cutpoint_z = np.asarray(cutpoint_z)
        self.cutpoint_p = np.asarray(cutpoint_p)
        self.ll = float(ll)
        self.ll_0 = float(ll_0)
        self.pseudo_r2 = float(pseudo_r2)
        self.chi2 = float(chi2)
        self.chi2_pval = float(chi2_pval)
        self.n_obs = int(n_obs)
        self.k_vars = int(k_vars)
        self.k_cat = int(k_cat)
        self.df_m = int(df_m)
        self.var_names = list(var_names)
        self.y_name = y_name
        self.categories = np.asarray(categories)
        self.converged = bool(converged)
        self.model_name = model_name
        self.V = V

        self._style = "stata"
        self._display = False

    def set_style(self, style: str):
        self._style = style

    def set_display(self, is_display: bool):
        self._display = is_display
        if is_display:
            print(self._summary())

    def summary(self):
        return self._summary()

    def _summary(self):
        lines = []
        lines.append(f"{self.model_name}")
        lines.append(f"Number of obs = {self.n_obs:>8}")
        lines.append(f"LR chi2({self.df_m})    = {self.chi2:>8.2f}")
        lines.append(f"Prob > chi2   = {self.chi2_pval:>8.4f}")
        lines.append(f"Pseudo R2     = {self.pseudo_r2:>8.4f}")
        lines.append(f"Log likelihood= {self.ll:>8.3f}")
        lines.append("")

        header = f"{'':>30} | {'Coef.':>10} {'Std. Err.':>10} {'z':>8} {'P>|z|':>8} {'[95% CI]':>20}"
        lines.append(header)
        lines.append("-" * 90)

        from scipy import stats as sp_stats
        for i, name in enumerate(self.var_names):
            z = self.z_stat[i]
            p = self.p_value[i]
            ci_lo = self.coef[i] - 1.96 * self.std_err[i]
            ci_hi = self.coef[i] + 1.96 * self.std_err[i]
            lines.append(
                f"{name:>30} | {self.coef[i]:>10.4f} {self.std_err[i]:>10.4f} "
                f"{z:>8.2f} {p:>8.4f} {ci_lo:>9.4f} {ci_hi:>9.4f}"
            )

        lines.append("-" * 90)
        for j in range(len(self.cutpoints)):
            name = f"cut{j + 1}"
            lines.append(
                f"{name:>30} | {self.cutpoints[j]:>10.4f} {self.cutpoint_se[j]:>10.4f}"
            )

        return "\n".join(lines)

    def __repr__(self):
        return self._summary()
