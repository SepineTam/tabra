#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : iv_result.py

import numpy as np
from tabra.results.base import BaseResult


class IVResult(BaseResult):
    def __init__(self, coef, std_err, z_stat, p_value,
                 r_squared, r_squared_adj, root_mse,
                 n_obs, k_vars, df_m, first_stage_f,
                 j_stat, j_pval,
                 endog_test_stat, endog_test_pval,
                 idstat, idpval,
                 widstat,
                 var_names, y_name,
                 estimator, vce_type,
                 endog_names, exog_names, inst_names,
                 df_r=None, df_a=None,
                 F=None, N_hdfe=None,
                 kappa=None, var_beta=None):
        super().__init__()
        self._coef = coef
        self._std_err = std_err
        self._z_stat = z_stat
        self._p_value = p_value
        self._r_squared = r_squared
        self._r_squared_adj = r_squared_adj
        self._root_mse = root_mse
        self._n_obs = n_obs
        self._k_vars = k_vars
        self._df_m = df_m
        self._df_r = df_r
        self._df_a = df_a
        self._F = F
        self._N_hdfe = N_hdfe
        self._first_stage_f = first_stage_f
        self._j_stat = j_stat
        self._j_pval = j_pval
        self._endog_test_stat = endog_test_stat
        self._endog_test_pval = endog_test_pval
        self._idstat = idstat
        self._idp = idpval
        self._widstat = widstat
        self._var_names = var_names
        self._y_name = y_name
        self._estimator = estimator
        self._vce_type = vce_type
        self._endog_names = endog_names
        self._exog_names = exog_names
        self._inst_names = inst_names
        self._kappa = kappa
        self._var_beta = var_beta

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
    def r_squared(self):
        return self._r_squared

    @property
    def r_squared_adj(self):
        return self._r_squared_adj

    @property
    def root_mse(self):
        return self._root_mse

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
    def df_r(self):
        return self._df_r

    @property
    def df_a(self):
        return self._df_a

    @property
    def F(self):
        return self._F

    @property
    def N_hdfe(self):
        return self._N_hdfe

    @property
    def first_stage_f(self):
        return self._first_stage_f

    @property
    def j_stat(self):
        return self._j_stat

    @property
    def j_pval(self):
        return self._j_pval

    @property
    def endog_test_stat(self):
        return self._endog_test_stat

    @property
    def endog_test_pval(self):
        return self._endog_test_pval

    @property
    def idstat(self):
        return self._idstat

    @property
    def idp(self):
        return self._idp

    @property
    def widstat(self):
        return self._widstat

    @property
    def var_names(self):
        return self._var_names

    @property
    def y_name(self):
        return self._y_name

    @property
    def estimator(self):
        return self._estimator

    @property
    def vce_type(self):
        return self._vce_type

    @property
    def endog_names(self):
        return self._endog_names

    @property
    def exog_names(self):
        return self._exog_names

    @property
    def inst_names(self):
        return self._inst_names

    @property
    def kappa(self):
        return self._kappa

    @property
    def var_beta(self):
        return self._var_beta

    def summary(self):
        lines = []
        lines.append(f"IV regression ({self._estimator.upper()})")
        lines.append(f"Dependent variable: {self._y_name}")
        lines.append(f"Number of obs = {self._n_obs}")
        if self._df_m > 0:
            lines.append(f"Wald chi2({self._df_m}) = {self._wald_chi2():.2f}")
            lines.append(f"Prob > chi2 = {self._wald_pval():.4f}")
        lines.append(f"Root MSE = {self._root_mse:.4f}")
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
        lines.append(f"Endogenous: {', '.join(self._endog_names)}")
        lines.append(f"Exogenous:  {', '.join(self._exog_names)}")
        lines.append(f"Instruments: {', '.join(self._inst_names)}")
        lines.append("")

        if self._idstat is not None:
            lines.append(f"Anderson LM = {self._idstat:.4f}, p = {self._idp:.4f}")
        if self._widstat is not None:
            lines.append(f"Cragg-Donald F = {self._widstat:.2f}")
        if self._first_stage_f is not None:
            lines.append(f"First-stage F = {self._first_stage_f:.2f}")
        if self._j_stat is not None and self._j_stat != 0:
            lines.append(f"Sargan/Hansen J = {self._j_stat:.4f}, p = {self._j_pval:.4f}")
        if self._kappa is not None:
            lines.append(f"kappa = {self._kappa:.6f}")

        return "\n".join(lines)

    def _wald_chi2(self):
        mask = [i for i, n in enumerate(self._var_names) if n != "_cons"]
        if not mask:
            return 0.0
        b = self._coef[mask]
        V = self._V_sub(mask)
        try:
            chi2 = float(b.T @ np.linalg.solve(V, b))
        except Exception:
            chi2 = 0.0
        return chi2

    def _wald_pval(self):
        from scipy import stats as sp_stats
        chi2 = self._wald_chi2()
        if self._df_m <= 0:
            return 1.0
        return float(1 - sp_stats.chi2.cdf(chi2, self._df_m))

    def _V_sub(self, mask):
        if self._var_beta is not None:
            return self._var_beta[np.ix_(mask, mask)]
        return np.diag(self._std_err[mask] ** 2)

    def save(self, path):
        with open(path, "w") as f:
            f.write(self.summary())
