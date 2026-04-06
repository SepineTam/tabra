#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : test_ivtobit.py

"""Tests for ivtobit: MLE and twostep estimators."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ivtobit_data():
    """Generate data with censored outcome and endogenous regressor.

    DGP:
        z1, z2 ~ N(0,1) instruments
        x2 ~ N(0,1) exogenous
        rho = 0.5 correlation between u and v
        v ~ N(0,1), u = rho*v + sqrt(1-rho^2)*eps, eps ~ N(0,1)
        x1 = 0.5*z1 + 0.3*z2 + v  (endogenous)
        y* = 1.0 + 1.5*x1 + 1.0*x2 + u
        y = max(y*, 0)  (left-censored at 0)
    """
    np.random.seed(42)
    n = 500
    z1 = np.random.randn(n)
    z2 = np.random.randn(n)
    x2 = np.random.randn(n)

    rho = 0.5
    v = np.random.randn(n)
    eps = np.random.randn(n)
    u = rho * v + np.sqrt(1 - rho ** 2) * eps

    x1 = 0.5 * z1 + 0.3 * z2 + v
    y_star = 1.0 + 1.5 * x1 + 1.0 * x2 + u
    y = np.maximum(y_star, 0.0)  # left-censored at 0

    return pd.DataFrame({
        "y": y, "x1": x1, "x2": x2,
        "z1": z1, "z2": z2,
    })


class TestIVTobitMLE:
    def test_mle_basic(self, ivtobit_data):
        from tabra.models.estimate.ivtobit import IVTobitModel
        result = IVTobitModel().fit(
            ivtobit_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], ll=0.0, method="mle"
        )
        assert result.n_obs == 500
        assert result.method == "mle"
        assert all(np.isfinite(result.coef))

    def test_mle_coef_direction(self, ivtobit_data):
        from tabra.models.estimate.ivtobit import IVTobitModel
        result = IVTobitModel().fit(
            ivtobit_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], ll=0.0, method="mle"
        )
        idx_x1 = list(result.var_names).index("x1")
        idx_x2 = list(result.var_names).index("x2")
        assert result.coef[idx_x1] > 0
        assert result.coef[idx_x2] > 0

    def test_mle_converged(self, ivtobit_data):
        from tabra.models.estimate.ivtobit import IVTobitModel
        result = IVTobitModel().fit(
            ivtobit_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], ll=0.0, method="mle"
        )
        assert result.converged is True

    def test_mle_ll(self, ivtobit_data):
        from tabra.models.estimate.ivtobit import IVTobitModel
        result = IVTobitModel().fit(
            ivtobit_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], ll=0.0, method="mle"
        )
        assert result.ll is not None
        assert np.isfinite(result.ll)

    def test_mle_std_err(self, ivtobit_data):
        from tabra.models.estimate.ivtobit import IVTobitModel
        result = IVTobitModel().fit(
            ivtobit_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], ll=0.0, method="mle"
        )
        assert all(result.std_err > 0)

    def test_mle_censor_counts(self, ivtobit_data):
        from tabra.models.estimate.ivtobit import IVTobitModel
        result = IVTobitModel().fit(
            ivtobit_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], ll=0.0, method="mle"
        )
        assert hasattr(result, 'n_lc')
        assert result.n_lc > 0  # should have some left-censored
        assert result.n_unc > 0

    def test_mle_endog_test(self, ivtobit_data):
        from tabra.models.estimate.ivtobit import IVTobitModel
        result = IVTobitModel().fit(
            ivtobit_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], ll=0.0, method="mle"
        )
        assert result.endog_test_stat is not None
        assert result.endog_test_pval is not None


class TestIVTobitTwostep:
    def test_twostep_basic(self, ivtobit_data):
        from tabra.models.estimate.ivtobit import IVTobitModel
        result = IVTobitModel().fit(
            ivtobit_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], ll=0.0, method="twostep"
        )
        assert result.n_obs == 500
        assert result.method == "twostep"
        assert all(np.isfinite(result.coef))

    def test_twostep_coef_direction(self, ivtobit_data):
        from tabra.models.estimate.ivtobit import IVTobitModel
        result = IVTobitModel().fit(
            ivtobit_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], ll=0.0, method="twostep"
        )
        idx_x1 = list(result.var_names).index("x1")
        idx_x2 = list(result.var_names).index("x2")
        assert result.coef[idx_x1] > 0
        assert result.coef[idx_x2] > 0


class TestIVTobitRobust:
    def test_robust_vce(self, ivtobit_data):
        from tabra.models.estimate.ivtobit import IVTobitModel
        result = IVTobitModel().fit(
            ivtobit_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], ll=0.0, method="mle", vce="robust"
        )
        assert all(result.std_err > 0)
        assert all(np.isfinite(result.std_err))


class TestIVTobitAPI:
    def test_ivtobit_api(self, ivtobit_data):
        from tabra.core.data import TabraData
        tab = TabraData(ivtobit_data, is_display_result=False)
        result = tab.ivtobit("y", exog=["x2"], endog=["x1"],
                             iv=["z1", "z2"], ll=0.0, method="mle")
        assert result is not None
        assert result.n_obs == 500

    def test_ivtobit_twostep_api(self, ivtobit_data):
        from tabra.core.data import TabraData
        tab = TabraData(ivtobit_data, is_display_result=False)
        result = tab.ivtobit("y", exog=["x2"], endog=["x1"],
                             iv=["z1", "z2"], ll=0.0, method="twostep")
        assert result.method == "twostep"


class TestIVTobitEdgeCases:
    def test_no_constant(self, ivtobit_data):
        from tabra.models.estimate.ivtobit import IVTobitModel
        result = IVTobitModel().fit(
            ivtobit_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], ll=0.0, method="mle", is_con=False
        )
        assert "_cons" not in result.var_names

    def test_underidentified_raises(self, ivtobit_data):
        from tabra.models.estimate.ivtobit import IVTobitModel
        with pytest.raises(ValueError, match="Underidentified"):
            IVTobitModel().fit(
                ivtobit_data, y="y", exog=["x2"],
                endog=["x1"], instruments=[], ll=0.0, method="mle"
            )

    def test_right_censored(self, ivtobit_data):
        """Test with right-censoring instead of left."""
        from tabra.models.estimate.ivtobit import IVTobitModel
        df = ivtobit_data.copy()
        df["y_rc"] = np.minimum(df["y"], 2.0)
        result = IVTobitModel().fit(
            df, y="y_rc", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], ul=2.0, method="mle"
        )
        assert result.n_obs == 500
        assert all(np.isfinite(result.coef))
