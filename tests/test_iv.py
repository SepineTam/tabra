#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : test_iv.py

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def iv_numpy_data():
    """Deterministic data + numpy oracle for 2SLS."""
    np.random.seed(42)
    n = 200
    z1 = np.random.randn(n)
    z2 = np.random.randn(n)
    x2 = np.random.randn(n)
    u = np.random.randn(n) * 0.5
    x1 = 0.5 * z1 + 0.3 * z2 + 0.4 * u
    y = 1.0 + 2.0 * x1 + 1.5 * x2 + u

    # Oracle: X = [const, x1, x2], Z = [const, z1, z2, x2]
    Z_oracle = np.column_stack([np.ones(n), z1, z2, x2])
    X_oracle = np.column_stack([np.ones(n), x1, x2])
    PZ = Z_oracle @ np.linalg.inv(Z_oracle.T @ Z_oracle) @ Z_oracle.T
    X_hat = PZ @ X_oracle
    beta_2sls = np.linalg.inv(X_hat.T @ X_hat) @ X_hat.T @ y
    resid = y - X_oracle @ beta_2sls
    sigma2 = resid @ resid / (n - X_oracle.shape[1])
    se_2sls = np.sqrt(sigma2 * np.diag(np.linalg.inv(X_hat.T @ X_hat)))

    oracle = {
        "_cons": (beta_2sls[0], se_2sls[0]),
        "x1": (beta_2sls[1], se_2sls[1]),
        "x2": (beta_2sls[2], se_2sls[2]),
    }

    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "z1": z1, "z2": z2})
    return df, oracle


def _get_val(result, name, field="coef"):
    names = list(result.var_names)
    idx = names.index(name)
    if field == "coef":
        return result.coef[idx]
    elif field == "se":
        return result.std_err[idx]
    raise ValueError(f"Unknown field: {field}")


class Test2SLS:
    def test_2sls_numpy_oracle(self, iv_numpy_data):
        from tabra.models.estimate.iv import IVModel
        df, oracle = iv_numpy_data
        result = IVModel().fit(
            df, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], estimator="2sls"
        )
        for var_name in ("x1", "x2", "_cons"):
            assert abs(_get_val(result, var_name) - oracle[var_name][0]) < 0.01

    def test_2sls_standard_errors(self, iv_numpy_data):
        from tabra.models.estimate.iv import IVModel
        df, oracle = iv_numpy_data
        result = IVModel().fit(
            df, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], estimator="2sls"
        )
        for var_name in ("x1", "x2", "_cons"):
            assert abs(_get_val(result, var_name, "se") - oracle[var_name][1]) < 0.01

    def test_2sls_n_obs(self, iv_numpy_data):
        from tabra.models.estimate.iv import IVModel
        df, _ = iv_numpy_data
        result = IVModel().fit(
            df, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], estimator="2sls"
        )
        assert result.n_obs == 200

    def test_2sls_var_names(self, iv_numpy_data):
        from tabra.models.estimate.iv import IVModel
        df, _ = iv_numpy_data
        result = IVModel().fit(
            df, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], estimator="2sls"
        )
        assert "x1" in result.var_names
        assert "x2" in result.var_names
        assert "_cons" in result.var_names

    def test_2sls_multiple_endog(self):
        """2SLS with multiple endogenous variables."""
        from tabra.models.estimate.iv import IVModel
        np.random.seed(123)
        n = 300
        z1, z2, z3 = np.random.randn(n), np.random.randn(n), np.random.randn(n)
        x3 = np.random.randn(n)
        u = np.random.randn(n) * 0.3
        x1 = 0.5 * z1 + 0.3 * z2 + 0.3 * u
        x2 = 0.4 * z3 + 0.2 * u
        y = 1.0 + 1.5 * x1 + 1.0 * x2 + 0.8 * x3 + u

        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3,
                           "z1": z1, "z2": z2, "z3": z3})
        result = IVModel().fit(
            df, y="y", exog=["x3"], endog=["x1", "x2"],
            instruments=["z1", "z2", "z3"], estimator="2sls"
        )
        assert result.n_obs == 300
        # x1 and x2 should be close to true values
        assert abs(_get_val(result, "x1") - 1.5) < 0.5
        assert abs(_get_val(result, "x2") - 1.0) < 0.5


class TestGMM:
    def test_gmm_basic(self, iv_numpy_data):
        from tabra.models.estimate.iv import IVModel
        df, oracle = iv_numpy_data
        result = IVModel().fit(
            df, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], estimator="gmm"
        )
        for var_name in ("x1", "x2", "_cons"):
            assert abs(_get_val(result, var_name) - oracle[var_name][0]) < 0.5

    def test_gmm_n_obs(self, iv_numpy_data):
        from tabra.models.estimate.iv import IVModel
        df, _ = iv_numpy_data
        result = IVModel().fit(
            df, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], estimator="gmm"
        )
        assert result.n_obs == 200


class TestLIML:
    def test_liml_basic(self, iv_numpy_data):
        from tabra.models.estimate.iv import IVModel
        df, oracle = iv_numpy_data
        result = IVModel().fit(
            df, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], estimator="liml"
        )
        for var_name in ("x1", "x2", "_cons"):
            assert abs(_get_val(result, var_name) - oracle[var_name][0]) < 1.0

    def test_liml_n_obs(self, iv_numpy_data):
        from tabra.models.estimate.iv import IVModel
        df, _ = iv_numpy_data
        result = IVModel().fit(
            df, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], estimator="liml"
        )
        assert result.n_obs == 200


class TestIVDiagnostics:
    def test_first_stage_f(self, iv_numpy_data):
        from tabra.models.estimate.iv import IVModel
        df, _ = iv_numpy_data
        result = IVModel().fit(
            df, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], estimator="2sls"
        )
        assert result.first_stage_f > 0
        assert result.first_stage_f > 10

    def test_overid_j_stat(self, iv_numpy_data):
        from tabra.models.estimate.iv import IVModel
        df, _ = iv_numpy_data
        result = IVModel().fit(
            df, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], estimator="2sls"
        )
        assert result.j_stat is not None
        assert result.j_pval is not None
        assert 0 <= result.j_pval <= 1


class TestIVEdgeCases:
    def test_underidentified_raises(self, iv_numpy_data):
        from tabra.models.estimate.iv import IVModel
        df, _ = iv_numpy_data
        with pytest.raises(ValueError, match="[Uu]nderidentif|fewer|instrument"):
            IVModel().fit(
                df, y="y", exog=["x2"], endog=["x1", "x2"],
                instruments=["z1"], estimator="2sls"
            )

    def test_invalid_estimator_raises(self, iv_numpy_data):
        from tabra.models.estimate.iv import IVModel
        df, _ = iv_numpy_data
        with pytest.raises(ValueError):
            IVModel().fit(
                df, y="y", exog=["x2"], endog=["x1"],
                instruments=["z1", "z2"], estimator="invalid"
            )

    def test_exactly_identified_no_j(self):
        """Exactly identified: no overid test."""
        from tabra.models.estimate.iv import IVModel
        np.random.seed(42)
        n = 200
        z1 = np.random.randn(n)
        x2 = np.random.randn(n)
        u = np.random.randn(n) * 0.5
        x1 = 0.5 * z1 + 0.4 * u
        y = 1.0 + 2.0 * x1 + 1.5 * x2 + u
        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "z1": z1})
        result = IVModel().fit(
            df, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1"], estimator="2sls"
        )
        assert result.j_stat is None or result.j_stat == 0.0

    def test_robust_vce(self, iv_numpy_data):
        from tabra.models.estimate.iv import IVModel
        df, _ = iv_numpy_data
        result_unadj = IVModel().fit(
            df, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], estimator="2sls", vce="unadjusted"
        )
        result_robust = IVModel().fit(
            df, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2"], estimator="2sls", vce="robust"
        )
        assert all(result_unadj.std_err > 0)
        assert all(result_robust.std_err > 0)


class TestIVAPI:
    def test_ivreg_api(self):
        from tabra.core.data import TabraData
        np.random.seed(42)
        n = 200
        z1 = np.random.randn(n)
        z2 = np.random.randn(n)
        x2 = np.random.randn(n)
        u = np.random.randn(n) * 0.5
        x1 = 0.5 * z1 + 0.3 * z2 + 0.4 * u
        y = 1.0 + 2.0 * x1 + 1.5 * x2 + u
        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "z1": z1, "z2": z2})
        tab = TabraData(df, is_display_result=False)
        result = tab.ivreg(
            "y", exog=["x2"], endog=["x1"], iv=["z1", "z2"],
            estimator="2sls"
        )
        assert result is not None
        assert result.n_obs == 200
        assert abs(_get_val(result, "x1") - 2.0) < 0.5
