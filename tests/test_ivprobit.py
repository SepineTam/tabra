#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : test_ivprobit.py

"""Tests for ivprobit: MLE and twostep estimators."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ivprobit_data():
    """Generate data with binary outcome and endogenous regressor.

    DGP:
        x2 ~ N(0,1) exogenous
        z1, z2 ~ N(0,1) instruments
        v, eps ~ N(0,1) correlated via rho=0.6
        x1 = 0.5*z1 + 0.3*z2 + v  (endogenous)
        y* = -0.5 + 1.0*x1 + 1.5*x2 + eps
        y = 1(y* > 0)
    """
    np.random.seed(42)
    n = 500
    z1 = np.random.randn(n)
    z2 = np.random.randn(n)
    x2 = np.random.randn(n)

    rho = 0.6
    cov = np.array([[1, rho], [rho, 1]])
    errors = np.random.multivariate_normal([0, 0], cov, n)
    eps = errors[:, 0]
    v = errors[:, 1]

    x1 = 0.5 * z1 + 0.3 * z2 + v
    y_star = -0.5 + 1.0 * x1 + 1.5 * x2 + eps
    y = (y_star > 0).astype(float)

    return pd.DataFrame({
        "y": y, "x1": x1, "x2": x2,
        "z1": z1, "z2": z2,
    })


class TestIVProbitMLE:
    def test_mle_basic(self, ivprobit_data):
        from tabra.models.estimate.ivprobit import IVProbitModel
        result = IVProbitModel().fit(
            ivprobit_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], method="mle"
        )
        assert result.n_obs == 500
        assert result.method == "mle"
        assert all(np.isfinite(result.coef))
        assert len(result.coef) == len(result.var_names)

    def test_mle_coef_direction(self, ivprobit_data):
        """Coefficients should have correct sign direction."""
        from tabra.models.estimate.ivprobit import IVProbitModel
        result = IVProbitModel().fit(
            ivprobit_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], method="mle"
        )
        idx_x1 = list(result.var_names).index("x1")
        idx_x2 = list(result.var_names).index("x2")
        # Both should be positive in the DGP
        assert result.coef[idx_x1] > 0
        assert result.coef[idx_x2] > 0

    def test_mle_converged(self, ivprobit_data):
        from tabra.models.estimate.ivprobit import IVProbitModel
        result = IVProbitModel().fit(
            ivprobit_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], method="mle"
        )
        assert result.converged is True

    def test_mle_log_likelihood(self, ivprobit_data):
        from tabra.models.estimate.ivprobit import IVProbitModel
        result = IVProbitModel().fit(
            ivprobit_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], method="mle"
        )
        assert result.ll is not None
        assert np.isfinite(result.ll)

    def test_mle_std_err_positive(self, ivprobit_data):
        from tabra.models.estimate.ivprobit import IVProbitModel
        result = IVProbitModel().fit(
            ivprobit_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], method="mle"
        )
        assert all(result.std_err > 0)

    def test_mle_rho(self, ivprobit_data):
        """rho should be estimable and not zero when endogeneity exists."""
        from tabra.models.estimate.ivprobit import IVProbitModel
        result = IVProbitModel().fit(
            ivprobit_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], method="mle"
        )
        assert hasattr(result, 'rho')
        assert result.rho is not None
        assert -1 < result.rho < 1

    def test_mle_endog_test(self, ivprobit_data):
        """Endogeneity test (Wald H0: rho=0) should be present."""
        from tabra.models.estimate.ivprobit import IVProbitModel
        result = IVProbitModel().fit(
            ivprobit_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], method="mle"
        )
        assert result.endog_test_stat is not None
        assert result.endog_test_pval is not None
        assert 0 <= result.endog_test_pval <= 1


class TestIVProbitTwostep:
    def test_twostep_basic(self, ivprobit_data):
        from tabra.models.estimate.ivprobit import IVProbitModel
        result = IVProbitModel().fit(
            ivprobit_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], method="twostep"
        )
        assert result.n_obs == 500
        assert result.method == "twostep"
        assert all(np.isfinite(result.coef))

    def test_twostep_coef_direction(self, ivprobit_data):
        from tabra.models.estimate.ivprobit import IVProbitModel
        result = IVProbitModel().fit(
            ivprobit_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], method="twostep"
        )
        idx_x1 = list(result.var_names).index("x1")
        idx_x2 = list(result.var_names).index("x2")
        assert result.coef[idx_x1] > 0
        assert result.coef[idx_x2] > 0

    def test_twostep_no_vhat_in_output(self, ivprobit_data):
        """Newey twostep reports structural coefficients (no v_hat in output)."""
        from tabra.models.estimate.ivprobit import IVProbitModel
        result = IVProbitModel().fit(
            ivprobit_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], method="twostep"
        )
        # Newey min-chi-squared reports structural coef, not v_hat
        assert "vhat_x1" not in result.var_names


class TestIVProbitRobust:
    def test_robust_vce(self, ivprobit_data):
        from tabra.models.estimate.ivprobit import IVProbitModel
        result = IVProbitModel().fit(
            ivprobit_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], method="mle", vce="robust"
        )
        assert all(result.std_err > 0)
        assert all(np.isfinite(result.std_err))


class TestIVProbitAPI:
    def test_ivprobit_api(self, ivprobit_data):
        from tabra.core.data import TabraData
        tab = TabraData(ivprobit_data, is_display_result=False)
        result = tab.ivprobit("y", exog=["x2"], endog=["x1"],
                              iv=["z1", "z2"], method="mle")
        assert result is not None
        assert result.n_obs == 500

    def test_ivprobit_twostep_api(self, ivprobit_data):
        from tabra.core.data import TabraData
        tab = TabraData(ivprobit_data, is_display_result=False)
        result = tab.ivprobit("y", exog=["x2"], endog=["x1"],
                              iv=["z1", "z2"], method="twostep")
        assert result.method == "twostep"


class TestIVProbitEdgeCases:
    def test_no_constant(self, ivprobit_data):
        from tabra.models.estimate.ivprobit import IVProbitModel
        result = IVProbitModel().fit(
            ivprobit_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], method="mle", is_con=False
        )
        assert "_cons" not in result.var_names

    def test_underidentified_raises(self, ivprobit_data):
        """More endogenous variables than instruments should raise."""
        from tabra.models.estimate.ivprobit import IVProbitModel
        with pytest.raises(ValueError, match="Underidentified"):
            IVProbitModel().fit(
                ivprobit_data, y="y", exog=["x2"],
                endog=["x1"], instruments=[], method="mle"
            )


# ---------------------------------------------------------------------------
# Stata cross-validation: every scalar must match Stata output to at least
# 1e-4 relative tolerance (last-decimal rounding only).
# ---------------------------------------------------------------------------

class TestIVProbitStataCrossValidation:
    """Stata 17 ivprobit baseline cross-validation (seed 42, n=500)."""

    @pytest.fixture(scope="class")
    def data(self):
        np.random.seed(42)
        n = 500
        z1 = np.random.randn(n)
        z2 = np.random.randn(n)
        x2 = np.random.randn(n)
        rho = 0.6
        cov = np.array([[1, rho], [rho, 1]])
        errors = np.random.multivariate_normal([0, 0], cov, n)
        eps = errors[:, 0]
        v = errors[:, 1]
        x1 = 0.5 * z1 + 0.3 * z2 + v
        y_star = -0.5 + 1.0 * x1 + 1.5 * x2 + eps
        y = (y_star > 0).astype(float)
        return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "z1": z1, "z2": z2})

    # ---- MLE default ----
    def test_mle_default_coef(self, data):
        from tabra.models.estimate.ivprobit import IVProbitModel
        r = IVProbitModel().fit(
            data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], method="mle", vce="unadjusted"
        )
        assert r.var_names == ["x1", "x2", "_cons"]
        np.testing.assert_allclose(r.coef,
                                   [1.113627, 1.550236, -0.430839], rtol=1e-5)

    def test_mle_default_se(self, data):
        from tabra.models.estimate.ivprobit import IVProbitModel
        r = IVProbitModel().fit(
            data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], method="mle", vce="unadjusted"
        )
        np.testing.assert_allclose(r.std_err,
                                   [0.2246492, 0.1839215, 0.0950356], rtol=1e-5)

    def test_mle_default_z(self, data):
        from tabra.models.estimate.ivprobit import IVProbitModel
        r = IVProbitModel().fit(
            data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], method="mle", vce="unadjusted"
        )
        np.testing.assert_allclose(r.z_stat,
                                   [4.957181, 8.428791, -4.533445], rtol=1e-5)

    def test_mle_default_p(self, data):
        from tabra.models.estimate.ivprobit import IVProbitModel
        r = IVProbitModel().fit(
            data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], method="mle", vce="unadjusted"
        )
        np.testing.assert_allclose(r.p_value,
                                   [7.152339e-07, 0.0, 5.802939e-06], rtol=1e-5)

    def test_mle_default_rho(self, data):
        from tabra.models.estimate.ivprobit import IVProbitModel
        r = IVProbitModel().fit(
            data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], method="mle", vce="unadjusted"
        )
        np.testing.assert_allclose(r.rho, 0.5975633, rtol=1e-5)
        np.testing.assert_allclose(r.rho_se, 0.1006255, rtol=1e-5)

    def test_mle_default_ll_chi2(self, data):
        from tabra.models.estimate.ivprobit import IVProbitModel
        r = IVProbitModel().fit(
            data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], method="mle", vce="unadjusted"
        )
        np.testing.assert_allclose(r.ll, -803.26182, rtol=1e-5)
        np.testing.assert_allclose(r.chi2, 84.69, rtol=1e-3)

    def test_mle_default_endog_test(self, data):
        from tabra.models.estimate.ivprobit import IVProbitModel
        r = IVProbitModel().fit(
            data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], method="mle", vce="unadjusted"
        )
        np.testing.assert_allclose(r.endog_test_stat, 19.40, rtol=1e-3)
        np.testing.assert_allclose(r.endog_test_pval, 0.0, atol=1e-4)

    # ---- MLE robust ----
    def test_mle_robust_coef(self, data):
        from tabra.models.estimate.ivprobit import IVProbitModel
        r = IVProbitModel().fit(
            data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], method="mle", vce="robust"
        )
        np.testing.assert_allclose(r.coef,
                                   [1.113627, 1.550236, -0.430839], rtol=1e-5)

    def test_mle_robust_se(self, data):
        from tabra.models.estimate.ivprobit import IVProbitModel
        r = IVProbitModel().fit(
            data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], method="mle", vce="robust"
        )
        np.testing.assert_allclose(r.std_err,
                                   [0.2335241, 0.1824299, 0.0972864], rtol=1e-5)

    def test_mle_robust_chi2(self, data):
        from tabra.models.estimate.ivprobit import IVProbitModel
        r = IVProbitModel().fit(
            data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], method="mle", vce="robust"
        )
        np.testing.assert_allclose(r.chi2, 94.30, rtol=1e-3)

    def test_mle_robust_endog_test(self, data):
        from tabra.models.estimate.ivprobit import IVProbitModel
        r = IVProbitModel().fit(
            data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], method="mle", vce="robust"
        )
        np.testing.assert_allclose(r.endog_test_stat, 21.83, rtol=1e-3)

    # ---- twostep ----
    def test_twostep_coef(self, data):
        from tabra.models.estimate.ivprobit import IVProbitModel
        r = IVProbitModel().fit(
            data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], method="twostep", vce="unadjusted"
        )
        np.testing.assert_allclose(r.coef,
                                   [1.387155, 1.921722, -0.534330], rtol=1e-5)

    def test_twostep_se(self, data):
        from tabra.models.estimate.ivprobit import IVProbitModel
        r = IVProbitModel().fit(
            data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], method="twostep", vce="unadjusted"
        )
        np.testing.assert_allclose(r.std_err,
                                   [0.1962004, 0.1955903, 0.1117416], rtol=1e-5)

    def test_twostep_chi2(self, data):
        from tabra.models.estimate.ivprobit import IVProbitModel
        r = IVProbitModel().fit(
            data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], method="twostep", vce="unadjusted"
        )
        np.testing.assert_allclose(r.chi2, 101.17, rtol=1e-3)

    def test_twostep_endog_test(self, data):
        from tabra.models.estimate.ivprobit import IVProbitModel
        r = IVProbitModel().fit(
            data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], method="twostep", vce="unadjusted"
        )
        np.testing.assert_allclose(r.endog_test_stat, 15.74, rtol=1e-3)
