#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : ivtobit_result.py

import numpy as np
from tabra.results.base import BaseResult


class IVTobitResult(BaseResult):
    def __init__(self, coef, std_err, z_stat, p_value,
                 n_obs, n_lc, n_rc, n_unc,
                 ll, chi2, chi2_pval,
                 sigma, sigma_se,
                 endog_test_stat, endog_test_pval,
                 var_names, y_name,
                 method, converged, vce_type,
                 ll_limit, ul_limit):
        super().__init__()
        self._coef = coef
        self._std_err = std_err
        self._z_stat = z_stat
        self._p_value = p_value
        self._n_obs = n_obs
        self._n_lc = n_lc
        self._n_rc = n_rc
        self._n_unc = n_unc
        self._ll = ll
        self._chi2 = chi2
        self._chi2_pval = chi2_pval
        self._sigma = sigma
        self._sigma_se = sigma_se
        self._endog_test_stat = endog_test_stat
        self._endog_test_pval = endog_test_pval
        self._var_names = var_names
        self._y_name = y_name
        self._method = method
        self._converged = converged
        self._vce_type = vce_type
        self._ll_limit = ll_limit
        self._ul_limit = ul_limit

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
    def n_obs(self):
        return self._n_obs

    @property
    def n_lc(self):
        return self._n_lc

    @property
    def n_rc(self):
        return self._n_rc

    @property
    def n_unc(self):
        return self._n_unc

    @property
    def ll(self):
        return self._ll

    @property
    def chi2(self):
        return self._chi2

    @property
    def chi2_pval(self):
        return self._chi2_pval

    @property
    def sigma(self):
        return self._sigma

    @property
    def sigma_se(self):
        return self._sigma_se

    @property
    def endog_test_stat(self):
        return self._endog_test_stat

    @property
    def endog_test_pval(self):
        return self._endog_test_pval

    @property
    def var_names(self):
        return self._var_names

    @property
    def y_name(self):
        return self._y_name

    @property
    def method(self):
        return self._method

    @property
    def converged(self):
        return self._converged

    @property
    def vce_type(self):
        return self._vce_type

    def summary(self):
        lines = []
        lines.append(f"IV tobit ({self._method})")
        lines.append(f"Dependent variable: {self._y_name}")
        lines.append(f"Number of obs = {self._n_obs}")
        if self._n_lc > 0:
            lines.append(f"Left-censored = {self._n_lc}")
        if self._n_rc > 0:
            lines.append(f"Right-censored = {self._n_rc}")
        lines.append(f"Uncensored = {self._n_unc}")
        if self._chi2 is not None:
            lines.append(f"Wald chi2 = {self._chi2:.2f}")
            lines.append(f"Prob > chi2 = {self._chi2_pval:.4f}")
        if self._ll is not None:
            lines.append(f"Log likelihood = {self._ll:.4f}")
        lines.append(f"sigma = {self._sigma:.4f}")
        lines.append("")

        header = f"{'':>15s} {'Coef':>12s} {'Std. Err.':>12s} {'z':>10s} {'P>|z|':>10s}"
        lines.append(header)
        lines.append("-" * 65)

        for i, name in enumerate(self._var_names):
            lines.append(
                f"{name:>15s} {self._coef[i]:12.4f} {self._std_err[i]:12.4f} "
                f"{self._z_stat[i]:10.2f} {self._p_value[i]:10.4f}"
            )

        lines.append("")
        if self._endog_test_stat is not None:
            lines.append(
                f"Endogeneity test: chi2 = {self._endog_test_stat:.4f}, "
                f"p = {self._endog_test_pval:.4f}"
            )
        if not self._converged:
            lines.append("WARNING: did not converge")

        return "\n".join(lines)

    def save(self, path):
        with open(path, "w") as f:
            f.write(self.summary())
