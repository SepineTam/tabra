#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : binary_choice_result.py

import numpy as np
from scipy import stats as sp_stats
from tabra.results.base import BaseResult


class BinaryChoiceResult(BaseResult):
    def __init__(self, coef, std_err, z_stat, p_value,
                 ll, ll_0, pseudo_r2, chi2, chi2_pval,
                 n_obs, k_vars, df_m, var_names,
                 y_name="", model_name="Binary Choice",
                 converged=True, n_iter=0,
                 vce_type="OIM",
                 y_true=None, y_pred=None):
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
        self._vce_type = vce_type
        self._y_true = y_true
        self._y_pred = y_pred

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
    def var_names(self):
        return self._var_names

    def _summary_style_stata(self):
        lines = []

        # Header block
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

        ll_label = "Log likelihood" if self._vce_type == "OIM" else "Log pseudolikelihood"
        lines.append(
            f"{ll_label + ' = ' + f'{self._ll:.6f}':<40s}"
            f"{'Pseudo R2':>14s} = {self._pseudo_r2:>8.4f}"
        )

        lines.append("")

        # Coefficient table
        lines.append(f"{'-' * 76}")
        se_header = "Std. Err." if self._vce_type == "OIM" else "std. err."
        lines.append(
            f"{self._y_name:>12s} | {'Coef.':>10s} {se_header:>10s} "
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

    def save(self, path="binary_choice_result.txt"):
        with open(path, "w") as f:
            f.write(self.summary())

    def confusion_matrix(self):
        from tabra.results.confusion_matrix_result import ConfusionMatrixResult
        if self._y_true is None:
            raise ValueError("No prediction data stored.")
        labels = sorted(set(self._y_true.astype(int)) | set(self._y_pred.astype(int)))
        k = len(labels)
        label_to_idx = {l: i for i, l in enumerate(labels)}
        matrix = np.zeros((k, k), dtype=int)
        for t, p in zip(self._y_true, self._y_pred):
            matrix[label_to_idx[int(t)], label_to_idx[int(p)]] += 1
        accuracy = float(np.sum(self._y_true == self._y_pred) / len(self._y_true))
        return ConfusionMatrixResult(
            matrix=matrix,
            labels=[str(l) for l in labels],
            accuracy=accuracy,
            n_obs=len(self._y_true),
        )

    def __repr__(self):
        return self.summary()
