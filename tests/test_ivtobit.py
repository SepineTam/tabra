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


class TestIVTobitStataCrossValidation:
    """Stata cross-validation tests with hard-coded benchmark values.

    Data generated in Stata with:
        set seed 42
        set obs 500
        gen z1 = rnormal()
        gen x1 = rnormal()
        gen v = rnormal()
        gen u_raw = rnormal()
        local rho = 0.5
        gen u = `rho'*v + sqrt(1-`rho'^2)*u_raw
        gen x2 = 0.5 + 0.8*z1 + 0.3*x1 + v
        gen y_star = 1.0 + 0.5*x1 + 1.2*x2 + u
        gen y = max(y_star, 0)
    """

    @pytest.fixture(scope="class")
    def stata_data(self):
        """Load the Stata-generated verification data."""
        return pd.read_stata(
            "/Users/sepinetam/Documents/Github/tabra/tmp/ivtobit_verify_data.dta"
        )

    def test_mle_ll0_coef(self, stata_data):
        """MLE left-censored coefficients match Stata."""
        from tabra.models.estimate.ivtobit import IVTobitModel
        result = IVTobitModel().fit(
            stata_data, y="y", exog=["x1"], endog=["x2"],
            instruments=["z1"], ll=0.0, method="mle", vce="unadjusted"
        )
        # Stata: x2=1.161804, x1=0.6146017, _cons=0.9717913
        np.testing.assert_allclose(
            result.coef, [1.161804, 0.6146017, 0.9717913], rtol=1e-5
        )

    def test_mle_ll0_se(self, stata_data):
        """MLE left-censored standard errors match Stata."""
        from tabra.models.estimate.ivtobit import IVTobitModel
        result = IVTobitModel().fit(
            stata_data, y="y", exog=["x1"], endog=["x2"],
            instruments=["z1"], ll=0.0, method="mle", vce="unadjusted"
        )
        # Stata: se(x2)=0.0647596, se(x1)=0.0563438, se(_cons)=0.0646224
        np.testing.assert_allclose(
            result.std_err, [0.0647596, 0.0563438, 0.0646224], rtol=1e-4
        )

    def test_mle_ll0_z(self, stata_data):
        """MLE left-censored z-stats match Stata."""
        from tabra.models.estimate.ivtobit import IVTobitModel
        result = IVTobitModel().fit(
            stata_data, y="y", exog=["x1"], endog=["x2"],
            instruments=["z1"], ll=0.0, method="mle", vce="unadjusted"
        )
        # Stata: z(x2)=17.94, z(x1)=10.91, z(_cons)=15.04
        np.testing.assert_allclose(
            result.z_stat, [17.94, 10.91, 15.04], rtol=1e-3
        )

    def test_mle_ll0_ll(self, stata_data):
        """MLE left-censored log-likelihood matches Stata."""
        from tabra.models.estimate.ivtobit import IVTobitModel
        result = IVTobitModel().fit(
            stata_data, y="y", exog=["x1"], endog=["x2"],
            instruments=["z1"], ll=0.0, method="mle", vce="unadjusted"
        )
        # Stata: ll = -1219.5157
        assert result.ll == pytest.approx(-1219.5157, rel=1e-5)

    def test_mle_ll0_chi2(self, stata_data):
        """MLE left-censored Wald chi2 matches Stata."""
        from tabra.models.estimate.ivtobit import IVTobitModel
        result = IVTobitModel().fit(
            stata_data, y="y", exog=["x1"], endog=["x2"],
            instruments=["z1"], ll=0.0, method="mle", vce="unadjusted"
        )
        # Stata: chi2 = 609.6637
        assert result.chi2 == pytest.approx(609.6637, rel=1e-4)

    def test_mle_ll0_endog_test(self, stata_data):
        """MLE left-censored endogeneity test matches Stata."""
        from tabra.models.estimate.ivtobit import IVTobitModel
        result = IVTobitModel().fit(
            stata_data, y="y", exog=["x1"], endog=["x2"],
            instruments=["z1"], ll=0.0, method="mle", vce="unadjusted"
        )
        # Stata: endog chi2 = 72.8583
        assert result.endog_test_stat == pytest.approx(72.8583, rel=1e-4)

    def test_mle_ll0_censor_counts(self, stata_data):
        """MLE left-censored censoring counts match Stata."""
        from tabra.models.estimate.ivtobit import IVTobitModel
        result = IVTobitModel().fit(
            stata_data, y="y", exog=["x1"], endog=["x2"],
            instruments=["z1"], ll=0.0, method="mle", vce="unadjusted"
        )
        # Stata: N_lc=137, N_unc=363
        assert result.n_lc == 137
        assert result.n_unc == 363
        assert result.n_rc == 0

    def test_twostep_coef(self, stata_data):
        """Twostep coefficients match Stata.

        Note: Python returns [x2, x1, vhat, _cons] while Stata returns
        [x2, x1, _cons]. The vhat coefficient is the endogeneity control.
        """
        from tabra.models.estimate.ivtobit import IVTobitModel
        result = IVTobitModel().fit(
            stata_data, y="y", exog=["x1"], endog=["x2"],
            instruments=["z1"], ll=0.0, method="twostep", vce="unadjusted"
        )
        # Stata: x2=1.161804, x1=0.6146017, _cons=0.9717913
        # Python: [x2, x1, vhat, _cons]
        np.testing.assert_allclose(
            result.coef[:2], [1.161804, 0.6146017], rtol=1e-5
        )
        np.testing.assert_allclose(
            result.coef[3], 0.9717913, rtol=1e-5
        )

    def test_twostep_se(self, stata_data):
        """Twostep standard errors match Stata."""
        from tabra.models.estimate.ivtobit import IVTobitModel
        result = IVTobitModel().fit(
            stata_data, y="y", exog=["x1"], endog=["x2"],
            instruments=["z1"], ll=0.0, method="twostep", vce="unadjusted"
        )
        # Stata: se(x2)=0.0648108, se(x1)=0.0563902, se(_cons)=0.0646673
        # Python se order: [x2, x1, vhat, _cons]
        np.testing.assert_allclose(
            result.std_err[:2], [0.0648108, 0.0563902], rtol=1e-3
        )
        np.testing.assert_allclose(
            result.std_err[3], 0.0646673, rtol=1e-3
        )

    def test_twostep_chi2(self, stata_data):
        """Twostep Wald chi2 matches Stata."""
        from tabra.models.estimate.ivtobit import IVTobitModel
        result = IVTobitModel().fit(
            stata_data, y="y", exog=["x1"], endog=["x2"],
            instruments=["z1"], ll=0.0, method="twostep", vce="unadjusted"
        )
        # Stata: chi2 = 608.75315
        assert result.chi2 == pytest.approx(608.75315, rel=2e-3)

    def test_twostep_endog_test(self, stata_data):
        """Twostep endogeneity test matches Stata."""
        from tabra.models.estimate.ivtobit import IVTobitModel
        result = IVTobitModel().fit(
            stata_data, y="y", exog=["x1"], endog=["x2"],
            instruments=["z1"], ll=0.0, method="twostep", vce="unadjusted"
        )
        # Stata: endog chi2 = 79.0034
        assert result.endog_test_stat == pytest.approx(79.0034, rel=1e-4)

    def test_mle_robust_coef(self, stata_data):
        """MLE robust coefficients match Stata."""
        from tabra.models.estimate.ivtobit import IVTobitModel
        result = IVTobitModel().fit(
            stata_data, y="y", exog=["x1"], endog=["x2"],
            instruments=["z1"], ll=0.0, method="mle", vce="robust"
        )
        # Stata: x2=1.161804, x1=0.6146017, _cons=0.9717913
        np.testing.assert_allclose(
            result.coef, [1.161804, 0.6146017, 0.9717913], rtol=1e-5
        )

    def test_mle_robust_se(self, stata_data):
        """MLE robust standard errors match Stata."""
        from tabra.models.estimate.ivtobit import IVTobitModel
        result = IVTobitModel().fit(
            stata_data, y="y", exog=["x1"], endog=["x2"],
            instruments=["z1"], ll=0.0, method="mle", vce="robust"
        )
        # Stata: se(x2)=0.0644625, se(x1)=0.0586471, se(_cons)=0.0667929
        np.testing.assert_allclose(
            result.std_err, [0.0644625, 0.0586471, 0.0667929], rtol=2e-3
        )

    def test_mle_robust_chi2(self, stata_data):
        """MLE robust Wald chi2 matches Stata."""
        from tabra.models.estimate.ivtobit import IVTobitModel
        result = IVTobitModel().fit(
            stata_data, y="y", exog=["x1"], endog=["x2"],
            instruments=["z1"], ll=0.0, method="mle", vce="robust"
        )
        # Stata: chi2 = 586.20194
        assert result.chi2 == pytest.approx(586.20194, rel=3e-3)

    def test_mle_robust_endog_test(self, stata_data):
        """MLE robust endogeneity test matches Stata."""
        from tabra.models.estimate.ivtobit import IVTobitModel
        result = IVTobitModel().fit(
            stata_data, y="y", exog=["x1"], endog=["x2"],
            instruments=["z1"], ll=0.0, method="mle", vce="robust"
        )
        # Stata: endog chi2 = 70.0428
        assert result.endog_test_stat == pytest.approx(70.0428, rel=3e-3)

    def test_mle_double_censored_coef(self, stata_data):
        """MLE double-censored coefficients match Stata."""
        from tabra.models.estimate.ivtobit import IVTobitModel
        df = stata_data.copy()
        df["y_double"] = np.minimum(np.maximum(df["y_star"], 0), 10)
        result = IVTobitModel().fit(
            df, y="y_double", exog=["x1"], endog=["x2"],
            instruments=["z1"], ll=0.0, ul=10.0, method="mle", vce="unadjusted"
        )
        # Stata: x2=1.161804, x1=0.6146017, _cons=0.9717913
        np.testing.assert_allclose(
            result.coef, [1.161804, 0.6146017, 0.9717913], rtol=1e-5
        )

    def test_mle_double_censored_se(self, stata_data):
        """MLE double-censored standard errors match Stata."""
        from tabra.models.estimate.ivtobit import IVTobitModel
        df = stata_data.copy()
        df["y_double"] = np.minimum(np.maximum(df["y_star"], 0), 10)
        result = IVTobitModel().fit(
            df, y="y_double", exog=["x1"], endog=["x2"],
            instruments=["z1"], ll=0.0, ul=10.0, method="mle", vce="unadjusted"
        )
        # Stata: se(x2)=0.0647596, se(x1)=0.0563438, se(_cons)=0.0646224
        np.testing.assert_allclose(
            result.std_err, [0.0647596, 0.0563438, 0.0646224], rtol=1e-4
        )
