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
