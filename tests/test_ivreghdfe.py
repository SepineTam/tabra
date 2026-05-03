#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : test_ivreghdfe.py

"""Tests for ivreghdfe: IV + HDFE."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ivreghdfe_data():
    """Panel data with endogenous regressor and FE.

    DGP:
        id: 20 individuals, t: 20 periods
        z1, z2 ~ N(0,1) instruments
        x2 ~ N(0,1) exogenous
        u, v correlated (rho=0.6)
        x1 = 0.5*z1 + 0.3*z2 + alpha_i + v  (endogenous + FE)
        y = beta_x1*x1 + beta_x2*x2 + alpha_i + gamma_t + u
    """
    np.random.seed(42)
    n_id = 20
    n_t = 20
    n = n_id * n_t

    ids = np.repeat(np.arange(n_id), n_t)
    ts = np.tile(np.arange(n_t), n_id)
    fe_id = np.random.randn(n_id) * 2
    fe_t = np.random.randn(n_t) * 0.5

    z1 = np.random.randn(n)
    z2 = np.random.randn(n)
    x2 = np.random.randn(n)

    rho = 0.6
    errors = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)
    u = errors[:, 0]
    v = errors[:, 1]

    x1 = 0.5 * z1 + 0.3 * z2 + fe_id[ids] + v
    y = 1.5 * x1 + 1.0 * x2 + fe_id[ids] + fe_t[ts] + u

    return pd.DataFrame({
        "y": y, "x1": x1, "x2": x2,
        "z1": z1, "z2": z2,
        "id": ids, "t": ts,
    })


class TestIVRegHDFE2SLS:
    def test_2sls_basic(self, ivreghdfe_data):
        from tabra.models.estimate.ivreghdfe import IVRegHDFEModel
        result = IVRegHDFEModel().fit(
            ivreghdfe_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], absorb=["id"],
            estimator="2sls"
        )
        assert result.n_obs > 0
        assert all(np.isfinite(result.coef))

    def test_2sls_coef_direction(self, ivreghdfe_data):
        from tabra.models.estimate.ivreghdfe import IVRegHDFEModel
        result = IVRegHDFEModel().fit(
            ivreghdfe_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], absorb=["id"],
            estimator="2sls"
        )
        idx_x1 = list(result.var_names).index("x1")
        idx_x2 = list(result.var_names).index("x2")
        assert result.coef[idx_x1] > 0
        assert result.coef[idx_x2] > 0

    def test_2sls_two_fe(self, ivreghdfe_data):
        from tabra.models.estimate.ivreghdfe import IVRegHDFEModel
        result = IVRegHDFEModel().fit(
            ivreghdfe_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], absorb=["id", "t"],
            estimator="2sls"
        )
        assert result.n_obs > 0
        assert all(np.isfinite(result.coef))

    def test_2sls_std_err(self, ivreghdfe_data):
        from tabra.models.estimate.ivreghdfe import IVRegHDFEModel
        result = IVRegHDFEModel().fit(
            ivreghdfe_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], absorb=["id"],
            estimator="2sls"
        )
        assert all(result.std_err > 0)


class TestIVRegHDFEGMM:
    def test_gmm_basic(self, ivreghdfe_data):
        from tabra.models.estimate.ivreghdfe import IVRegHDFEModel
        result = IVRegHDFEModel().fit(
            ivreghdfe_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], absorb=["id"],
            estimator="gmm"
        )
        assert result.n_obs > 0
        assert all(np.isfinite(result.coef))


class TestIVRegHDFEVCE:
    def test_robust_vce(self, ivreghdfe_data):
        from tabra.models.estimate.ivreghdfe import IVRegHDFEModel
        result = IVRegHDFEModel().fit(
            ivreghdfe_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], absorb=["id"],
            estimator="2sls", vce="robust"
        )
        assert all(result.std_err > 0)

    def test_cluster_vce(self, ivreghdfe_data):
        from tabra.models.estimate.ivreghdfe import IVRegHDFEModel
        result = IVRegHDFEModel().fit(
            ivreghdfe_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], absorb=["id"],
            estimator="2sls", vce="cluster", cluster=["t"]
        )
        assert all(result.std_err > 0)


class TestIVRegHDFEAPI:
    def test_ivreghdfe_api(self, ivreghdfe_data):
        from tabra.core.data import TabraData
        tab = TabraData(ivreghdfe_data, is_display_result=False)
        result = tab.ivreghdfe(
            "y", exog=["x2"], endog=["x1"], iv=["z1", "z2"],
            absorb=["id"]
        )
        assert result is not None
        assert result.n_obs > 0

    def test_ivreghdfe_gmm_api(self, ivreghdfe_data):
        from tabra.core.data import TabraData
        tab = TabraData(ivreghdfe_data, is_display_result=False)
        result = tab.ivreghdfe(
            "y", exog=["x2"], endog=["x1"], iv=["z1", "z2"],
            absorb=["id"], estimator="gmm"
        )
        assert result is not None


class TestIVRegHDFEEdgeCases:
    def test_underidentified_raises(self, ivreghdfe_data):
        from tabra.models.estimate.ivreghdfe import IVRegHDFEModel
        with pytest.raises(ValueError, match="Underidentified"):
            IVRegHDFEModel().fit(
                ivreghdfe_data, y="y", exog=["x2"],
                endog=["x1"], instruments=[], absorb=["id"]
            )


# ---------------------------------------------------------------------------
# Stata cross-validation (seed 42, n=500, n_id=50, n_year=10)
# Baseline from Stata 17 ivreghdfe
# ---------------------------------------------------------------------------

class TestIVRegHDFEStataCrossValidation:
    """Stata 17 ivreghdfe baseline cross-validation."""

    @pytest.fixture(scope="class")
    def data(self):
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

    # ---- Scene 1: absorb(id), unadjusted ----
    def test_s1_coef(self, data):
        from tabra.models.estimate.ivreghdfe import IVRegHDFEModel
        r = IVRegHDFEModel().fit(
            data, y="y", exog=["x1"], endog=["x2"],
            instruments=["z1", "z2"], absorb=["id"],
            estimator="2sls", vce="unadjusted"
        )
        np.testing.assert_allclose(r.coef, [1.5268896, 0.8572002], rtol=1e-6)

    def test_s1_se(self, data):
        from tabra.models.estimate.ivreghdfe import IVRegHDFEModel
        r = IVRegHDFEModel().fit(
            data, y="y", exog=["x1"], endog=["x2"],
            instruments=["z1", "z2"], absorb=["id"],
            estimator="2sls", vce="unadjusted"
        )
        np.testing.assert_allclose(r.std_err, [0.0816087, 0.0608128], rtol=1e-5)

    def test_s1_r2_rmse(self, data):
        from tabra.models.estimate.ivreghdfe import IVRegHDFEModel
        r = IVRegHDFEModel().fit(
            data, y="y", exog=["x1"], endog=["x2"],
            instruments=["z1", "z2"], absorb=["id"],
            estimator="2sls", vce="unadjusted"
        )
        np.testing.assert_allclose(r.r_squared, 0.8538983, rtol=1e-6)
        np.testing.assert_allclose(r.root_mse, 1.0661767, rtol=1e-5)

    def test_s1_df(self, data):
        from tabra.models.estimate.ivreghdfe import IVRegHDFEModel
        r = IVRegHDFEModel().fit(
            data, y="y", exog=["x1"], endog=["x2"],
            instruments=["z1", "z2"], absorb=["id"],
            estimator="2sls", vce="unadjusted"
        )
        assert r.df_a == 50
        assert r.df_r == 448

    def test_s1_diagnostics(self, data):
        from tabra.models.estimate.ivreghdfe import IVRegHDFEModel
        r = IVRegHDFEModel().fit(
            data, y="y", exog=["x1"], endog=["x2"],
            instruments=["z1", "z2"], absorb=["id"],
            estimator="2sls", vce="unadjusted"
        )
        np.testing.assert_allclose(r.idstat, 149.23107, rtol=1e-5)
        np.testing.assert_allclose(r.j_stat, 0.06247414, rtol=1e-5)
        np.testing.assert_allclose(r.j_pval, 0.80262734, rtol=1e-5)

    # ---- Scene 2: absorb(id year), robust ----
    def test_s2_coef(self, data):
        from tabra.models.estimate.ivreghdfe import IVRegHDFEModel
        r = IVRegHDFEModel().fit(
            data, y="y", exog=["x1"], endog=["x2"],
            instruments=["z1", "z2"], absorb=["id", "year"],
            estimator="2sls", vce="robust"
        )
        np.testing.assert_allclose(r.coef, [1.4823572, 0.8736909], rtol=1e-6)

    def test_s2_se(self, data):
        from tabra.models.estimate.ivreghdfe import IVRegHDFEModel
        r = IVRegHDFEModel().fit(
            data, y="y", exog=["x1"], endog=["x2"],
            instruments=["z1", "z2"], absorb=["id", "year"],
            estimator="2sls", vce="robust"
        )
        np.testing.assert_allclose(r.std_err, [0.0767163, 0.0586188], rtol=1e-5)

    def test_s2_r2_rmse(self, data):
        from tabra.models.estimate.ivreghdfe import IVRegHDFEModel
        r = IVRegHDFEModel().fit(
            data, y="y", exog=["x1"], endog=["x2"],
            instruments=["z1", "z2"], absorb=["id", "year"],
            estimator="2sls", vce="robust"
        )
        np.testing.assert_allclose(r.r_squared, 0.86421562, rtol=1e-6)
        np.testing.assert_allclose(r.root_mse, 1.0114432, rtol=1e-5)

    def test_s2_df(self, data):
        from tabra.models.estimate.ivreghdfe import IVRegHDFEModel
        r = IVRegHDFEModel().fit(
            data, y="y", exog=["x1"], endog=["x2"],
            instruments=["z1", "z2"], absorb=["id", "year"],
            estimator="2sls", vce="robust"
        )
        assert r.df_a == 59
        assert r.df_r == 439

    # ---- Scene 3: absorb(id), cluster ----
    def test_s3_coef(self, data):
        from tabra.models.estimate.ivreghdfe import IVRegHDFEModel
        r = IVRegHDFEModel().fit(
            data, y="y", exog=["x1"], endog=["x2"],
            instruments=["z1", "z2"], absorb=["id"],
            estimator="2sls", vce="cluster", cluster="id"
        )
        np.testing.assert_allclose(r.coef, [1.5268896, 0.8572002], rtol=1e-6)

    def test_s3_se(self, data):
        from tabra.models.estimate.ivreghdfe import IVRegHDFEModel
        r = IVRegHDFEModel().fit(
            data, y="y", exog=["x1"], endog=["x2"],
            instruments=["z1", "z2"], absorb=["id"],
            estimator="2sls", vce="cluster", cluster="id"
        )
        np.testing.assert_allclose(r.std_err, [0.0829351, 0.0521819], rtol=1e-5)

    def test_s3_r2(self, data):
        from tabra.models.estimate.ivreghdfe import IVRegHDFEModel
        r = IVRegHDFEModel().fit(
            data, y="y", exog=["x1"], endog=["x2"],
            instruments=["z1", "z2"], absorb=["id"],
            estimator="2sls", vce="cluster", cluster="id"
        )
        np.testing.assert_allclose(r.r_squared, 0.8538983, rtol=1e-6)

    def test_s3_df(self, data):
        from tabra.models.estimate.ivreghdfe import IVRegHDFEModel
        r = IVRegHDFEModel().fit(
            data, y="y", exog=["x1"], endog=["x2"],
            instruments=["z1", "z2"], absorb=["id"],
            estimator="2sls", vce="cluster", cluster="id"
        )
        assert r.df_a == 0
        assert r.df_r == 498

    def test_s2_diagnostics(self, data):
        from tabra.models.estimate.ivreghdfe import IVRegHDFEModel
        r = IVRegHDFEModel().fit(
            data, y="y", exog=["x1"], endog=["x2"],
            instruments=["z1", "z2"], absorb=["id", "year"],
            estimator="2sls", vce="robust"
        )
        np.testing.assert_allclose(r.widstat, 88.908, rtol=1e-4)
        np.testing.assert_allclose(r.idstat, 96.748, rtol=1e-4)
        assert r.idp < 1e-10

    def test_s3_diagnostics(self, data):
        from tabra.models.estimate.ivreghdfe import IVRegHDFEModel
        r = IVRegHDFEModel().fit(
            data, y="y", exog=["x1"], endog=["x2"],
            instruments=["z1", "z2"], absorb=["id"],
            estimator="2sls", vce="cluster", cluster="id"
        )
        np.testing.assert_allclose(r.widstat, 94.487, rtol=1e-4)
        np.testing.assert_allclose(r.idstat, 33.953, rtol=1e-4)
        np.testing.assert_allclose(r.idp, 4.238e-08, rtol=1e-3)
