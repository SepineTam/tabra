#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : ivprobit_result.py

import numpy as np
from tabra.results.base import BaseResult


class IVProbitResult(BaseResult):
    def __init__(self, coef, std_err, z_stat, p_value,
                 n_obs, ll, chi2, chi2_pval,
                 rho, rho_se,
                 endog_test_stat, endog_test_pval,
                 var_names, y_name,
                 method, converged, vce_type):
        super().__init__()
        self._coef = coef
        self._std_err = std_err
        self._z_stat = z_stat
        self._p_value = p_value
        self._n_obs = n_obs
        self._ll = ll
        self._chi2 = chi2
        self._chi2_pval = chi2_pval
        self._rho = rho
        self._rho_se = rho_se
        self._endog_test_stat = endog_test_stat
        self._endog_test_pval = endog_test_pval
        self._var_names = var_names
        self._y_name = y_name
        self._method = method
        self._converged = converged
        self._vce_type = vce_type

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
    def ll(self):
        return self._ll

    @property
    def chi2(self):
        return self._chi2

    @property
    def chi2_pval(self):
        return self._chi2_pval

    @property
    def rho(self):
        return self._rho

    @property
    def rho_se(self):
        return self._rho_se

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
        lines.append(f"IV probit ({self._method})")
        lines.append(f"Dependent variable: {self._y_name}")
        lines.append(f"Number of obs = {self._n_obs}")
        if self._chi2 is not None:
            lines.append(f"Wald chi2 = {self._chi2:.2f}")
            lines.append(f"Prob > chi2 = {self._chi2_pval:.4f}")
        if self._ll is not None:
            lines.append(f"Log likelihood = {self._ll:.4f}")
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
        if self._rho is not None:
            lines.append(f"rho = {self._rho:.4f}, se = {self._rho_se:.4f}")
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
