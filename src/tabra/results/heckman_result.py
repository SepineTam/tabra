#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : heckman_result.py

import numpy as np
from scipy import stats as sp_stats
from tabra.results.base import BaseResult


class HeckmanResult(BaseResult):
    def __init__(self, outcome_coef, outcome_se, outcome_z, outcome_p,
                 select_coef, select_se, select_z, select_p,
                 athrho, athrho_se, lnsigma, lnsigma_se,
                 rho, rho_se, sigma, sigma_se,
                 lambda_, lambda_se,
                 ll, n_obs, n_selected, n_nonselected,
                 chi2, chi2_pval, df_m,
                 outcome_var_names, select_var_names,
                 y_name="", converged=True, method="mle",
                 lr_chi2=0.0, lr_pval=1.0, V=None):
        super().__init__()
        self._outcome_coef = outcome_coef
        self._outcome_se = outcome_se
        self._outcome_z = outcome_z
        self._outcome_p = outcome_p
        self._select_coef = select_coef
        self._select_se = select_se
        self._select_z = select_z
        self._select_p = select_p
        self._athrho = athrho
        self._athrho_se = athrho_se
        self._lnsigma = lnsigma
        self._lnsigma_se = lnsigma_se
        self._rho = rho
        self._rho_se = rho_se
        self._sigma = sigma
        self._sigma_se = sigma_se
        self._lambda = lambda_
        self._lambda_se = lambda_se
        self._ll = ll
        self._n_obs = n_obs
        self._n_selected = n_selected
        self._n_nonselected = n_nonselected
        self._chi2 = chi2
        self._chi2_pval = chi2_pval
        self._df_m = df_m
        self._outcome_var_names = outcome_var_names
        self._select_var_names = select_var_names
        self._y_name = y_name
        self._converged = converged
        self._method = method
        self._lr_chi2 = lr_chi2
        self._lr_pval = lr_pval
        self._V = V

    @property
    def outcome_coef(self):
        return self._outcome_coef

    @property
    def outcome_se(self):
        return self._outcome_se

    @property
    def select_coef(self):
        return self._select_coef

    @property
    def select_se(self):
        return self._select_se

    @property
    def rho(self):
        return self._rho

    @property
    def sigma(self):
        return self._sigma

    @property
    def lambda_(self):
        return self._lambda

    @property
    def ll(self):
        return self._ll

    @property
    def n_obs(self):
        return self._n_obs

    @property
    def n_selected(self):
        return self._n_selected

    @property
    def n_nonselected(self):
        return self._n_nonselected

    @property
    def chi2(self):
        return self._chi2

    @property
    def converged(self):
        return self._converged

    @property
    def outcome_var_names(self):
        return self._outcome_var_names

    @property
    def select_var_names(self):
        return self._select_var_names

    def _summary_style_stata(self):
        lines = []
        z_crit = sp_stats.norm.ppf(0.975)

        if self._method == "twostep":
            lines.append(
                f"{'Heckman selection model -- two-step estimates':<40s}"
                f"{'Number of obs':>14s} = {self._n_obs:>8d}"
            )
        else:
            lines.append(
                f"{'Heckman selection model':<40s}"
                f"{'Number of obs':>14s} = {self._n_obs:>8d}"
            )
        lines.append(
            f"{'(regression model with sample selection)':<40s}"
            f"{'Selected':>14s} = {self._n_selected:>8d}"
        )
        lines.append(
            f"{'':40s}"
            f"{'Nonselected':>14s} = {self._n_nonselected:>8d}"
        )

        chi2_label = f"Wald chi2({self._df_m})"
        lines.append(
            f"{'':40s}"
            f"{chi2_label:>14s} = {self._chi2:>8.2f}"
        )

        if self._method == "mle":
            lines.append(
                f"{'Log likelihood = ' + f'{self._ll:.3f}':<40s}"
                f"{'Prob > chi2':>14s} = {self._chi2_pval:>8.4f}"
            )
        else:
            lines.append(
                f"{'':40s}"
                f"{'Prob > chi2':>14s} = {self._chi2_pval:>8.4f}"
            )

        lines.append("")
        lines.append(f"{'-' * 76}")
        lines.append(
            f"{self._y_name:>12s} | {'Coef.':>10s} {'Std. Err.':>10s} "
            f"{'z':>8s} {'P>|z|':>6s} {'[95% Conf. Interval]':>22s}"
        )
        lines.append(f"{'-' * 76}")

        # Outcome equation
        lines.append(f"{'-' * 12}-+{'-' * 63}")
        eq_label = self._y_name
        lines.append(f"{eq_label:>12s} |")

        for i, name in enumerate(self._outcome_var_names):
            c = self._outcome_coef[i]
            se = self._outcome_se[i]
            z = self._outcome_z[i]
            p = self._outcome_p[i]
            lo = c - z_crit * se
            hi = c + z_crit * se
            lines.append(
                f"{name:>12s} | {c:>10.4f} {se:>10.4f} {z:>8.2f} "
                f"{p:>6.3f} {lo:>11.4f} {hi:>11.4f}"
            )

        # Selection equation
        lines.append(f"{'-' * 12}-+{'-' * 63}")
        lines.append(f"{'select':>12s} |")

        for i, name in enumerate(self._select_var_names):
            c = self._select_coef[i]
            se = self._select_se[i]
            z = self._select_z[i]
            p = self._select_p[i]
            lo = c - z_crit * se
            hi = c + z_crit * se
            lines.append(
                f"{name:>12s} | {c:>10.4f} {se:>10.4f} {z:>8.2f} "
                f"{p:>6.3f} {lo:>11.4f} {hi:>11.4f}"
            )

        # Ancillary parameters (MLE only)
        if self._method == "mle":
            lines.append(f"{'-' * 12}-+{'-' * 63}")
            c = self._athrho
            se = self._athrho_se
            z = c / se if se > 0 else 0
            p = 2 * (1 - sp_stats.norm.cdf(abs(z)))
            lo = c - z_crit * se
            hi = c + z_crit * se
            lines.append(
                f"{'/athrho':>12s} | {c:>10.4f} {se:>10.4f} {z:>8.2f} "
                f"{p:>6.3f} {lo:>11.4f} {hi:>11.4f}"
            )

            c = self._lnsigma
            se = self._lnsigma_se
            z = c / se if se > 0 else 0
            p = 2 * (1 - sp_stats.norm.cdf(abs(z)))
            lo = c - z_crit * se
            hi = c + z_crit * se
            lines.append(
                f"{'/lnsigma':>12s} | {c:>10.4f} {se:>10.4f} {z:>8.2f} "
                f"{p:>6.3f} {lo:>11.4f} {hi:>11.4f}"
            )

        lines.append(f"{'-' * 76}")

        # Derived parameters
        lines.append(f"{'rho':>12s} | {self._rho:>10.4f} {self._rho_se:>10.4f}")
        lines.append(f"{'sigma':>12s} | {self._sigma:>10.4f} {self._sigma_se:>10.4f}")
        lines.append(f"{'lambda':>12s} | {self._lambda:>10.4f} {self._lambda_se:>10.4f}")

        # LR test (MLE only)
        if self._method == "mle" and self._lr_chi2 > 0:
            lines.append(
                f"LR test of indep. eqns. (rho = 0): "
                f"chi2(1) = {self._lr_chi2:.2f}        "
                f"Prob > chi2 = {self._lr_pval:.4f}"
            )

        return "\n".join(lines)

    def summary(self):
        dispatch = {
            "stata": self._summary_style_stata,
        }
        method = dispatch.get(self._style, self._summary_style_stata)
        return method()

    def save(self, path="heckman_result.txt"):
        with open(path, "w") as f:
            f.write(self.summary())

    def __repr__(self):
        return self.summary()
