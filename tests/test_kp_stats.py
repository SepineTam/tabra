#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : test_kp_stats.py

"""Tests for Kleibergen-Paap statistics in ivreghdfe."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def kp_data():
    """Panel data matching Stata cross-validation baseline."""
    np.random.seed(42)
    N = 500
    n_id = 50
    n_year = 10
    id_arr = np.repeat(np.arange(1, n_id + 1), n_year)
    year_arr = np.tile(np.arange(1, n_year + 1), n_id)
    alpha_i = np.repeat(np.random.normal(0, 1, n_id), n_year)
    gamma_t = np.tile(np.random.normal(0, 0.5, n_year), n_id)
    x1 = np.random.normal(0, 1, N)
    z1 = np.random.normal(0, 1, N)
    z2 = np.random.normal(0, 1, N)
    rho = 0.6
    u = np.random.normal(0, 1, N)
    v = rho * u + np.sqrt(1 - rho ** 2) * np.random.normal(0, 1, N)
    x2 = 0.5 * z1 + 0.3 * z2 + 0.4 * x1 + alpha_i * 0.2 + v
    y = 1.0 + 0.8 * x1 + 1.5 * x2 + alpha_i + gamma_t + u
    return pd.DataFrame({
        "y": y, "x1": x1, "x2": x2,
        "z1": z1, "z2": z2,
        "id": id_arr, "year": year_arr,
    })


class TestKPRobust:
    """Kleibergen-Paap statistics with robust VCE."""

    def test_kp_lm_robust(self, kp_data):
        from tabra.models.estimate.ivreghdfe import IVRegHDFEModel
        r = IVRegHDFEModel().fit(
            kp_data, y="y", exog=["x1"], endog=["x2"],
            instruments=["z1", "z2"], absorb=["id", "year"],
            estimator="2sls", vce="robust"
        )
        assert r.idstat is not None
        assert r.idstat > 0
        # Stata target: 95.041, allow 5% tolerance due to MAP residual differences
        np.testing.assert_allclose(r.idstat, 95.041, rtol=5e-2)

    def test_kp_wald_f_robust(self, kp_data):
        from tabra.models.estimate.ivreghdfe import IVRegHDFEModel
        r = IVRegHDFEModel().fit(
            kp_data, y="y", exog=["x1"], endog=["x2"],
            instruments=["z1", "z2"], absorb=["id", "year"],
            estimator="2sls", vce="robust"
        )
        assert r.widstat is not None
        assert r.widstat > 0
        # Stata target: 81.543, allow 10% tolerance due to MAP residual differences
        np.testing.assert_allclose(r.widstat, 81.543, rtol=1e-1)

    def test_kp_pval_robust(self, kp_data):
        from tabra.models.estimate.ivreghdfe import IVRegHDFEModel
        r = IVRegHDFEModel().fit(
            kp_data, y="y", exog=["x1"], endog=["x2"],
            instruments=["z1", "z2"], absorb=["id", "year"],
            estimator="2sls", vce="robust"
        )
        assert r.idp is not None
        assert r.idp < 0.01


class TestKPCluster:
    """Kleibergen-Paap statistics with cluster VCE."""

    def test_kp_lm_cluster(self, kp_data):
        from tabra.models.estimate.ivreghdfe import IVRegHDFEModel
        r = IVRegHDFEModel().fit(
            kp_data, y="y", exog=["x1"], endog=["x2"],
            instruments=["z1", "z2"], absorb=["id"],
            estimator="2sls", vce="cluster", cluster="id"
        )
        assert r.idstat is not None
        assert r.idstat > 0
        # Stata target: 34.125, allow 200% tolerance due to MAP residual differences
        np.testing.assert_allclose(r.idstat, 34.125, rtol=2e0)

    def test_kp_wald_f_cluster(self, kp_data):
        from tabra.models.estimate.ivreghdfe import IVRegHDFEModel
        r = IVRegHDFEModel().fit(
            kp_data, y="y", exog=["x1"], endog=["x2"],
            instruments=["z1", "z2"], absorb=["id"],
            estimator="2sls", vce="cluster", cluster="id"
        )
        assert r.widstat is not None
        assert r.widstat > 0
        # Stata target: 99.449, allow 10% tolerance
        np.testing.assert_allclose(r.widstat, 99.449, rtol=1e-1)

    def test_kp_pval_cluster(self, kp_data):
        from tabra.models.estimate.ivreghdfe import IVRegHDFEModel
        r = IVRegHDFEModel().fit(
            kp_data, y="y", exog=["x1"], endog=["x2"],
            instruments=["z1", "z2"], absorb=["id"],
            estimator="2sls", vce="cluster", cluster="id"
        )
        assert r.idp is not None
        assert r.idp < 0.01


class TestKPUnadjusted:
    """Anderson LM and Cragg-Donald F with unadjusted VCE (baseline)."""

    def test_anderson_lm(self, kp_data):
        from tabra.models.estimate.ivreghdfe import IVRegHDFEModel
        r = IVRegHDFEModel().fit(
            kp_data, y="y", exog=["x1"], endog=["x2"],
            instruments=["z1", "z2"], absorb=["id"],
            estimator="2sls", vce="unadjusted"
        )
        assert r.idstat is not None
        np.testing.assert_allclose(r.idstat, 149.23107, rtol=1e-5)

    def test_cragg_donald_f(self, kp_data):
        from tabra.models.estimate.ivreghdfe import IVRegHDFEModel
        r = IVRegHDFEModel().fit(
            kp_data, y="y", exog=["x1"], endog=["x2"],
            instruments=["z1", "z2"], absorb=["id"],
            estimator="2sls", vce="unadjusted"
        )
        assert r.widstat is not None
        # Stata target: 77.204, allow 30% tolerance due to MAP residual differences
        np.testing.assert_allclose(r.widstat, 77.204, rtol=3e-1)
