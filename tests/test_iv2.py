#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : test_iv2.py

"""Tests for ivreg2 enhancements: CUE, Fuller, k-class, cluster VCE, diagnostics."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def iv_data():
    np.random.seed(42)
    n = 300
    z1 = np.random.randn(n)
    z2 = np.random.randn(n)
    z3 = np.random.randn(n)
    x2 = np.random.randn(n)
    u = np.random.randn(n) * 0.5
    x1 = 0.5 * z1 + 0.3 * z2 + 0.3 * u
    y = 1.0 + 2.0 * x1 + 1.5 * x2 + u

    df = pd.DataFrame({
        "y": y, "x1": x1, "x2": x2,
        "z1": z1, "z2": z2, "z3": z3,
        "cluster_id": np.random.randint(0, 20, n),
    })
    return df


class TestCUE:
    def test_cue_basic(self, iv_data):
        from tabra.models.estimate.iv import IVModel
        result = IVModel().fit(
            iv_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2", "z3"], estimator="cue"
        )
        assert result.n_obs == 300
        assert result.estimator == "cue"
        assert all(np.isfinite(result.coef))
        # x1 should be close to 2.0
        idx_x1 = list(result.var_names).index("x1")
        assert abs(result.coef[idx_x1] - 2.0) < 1.0

    def test_cue_has_diagnostics(self, iv_data):
        from tabra.models.estimate.iv import IVModel
        result = IVModel().fit(
            iv_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2", "z3"], estimator="cue"
        )
        assert result.first_stage_f > 0


class TestFuller:
    def test_fuller_basic(self, iv_data):
        from tabra.models.estimate.iv import IVModel
        result = IVModel().fit(
            iv_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2", "z3"], estimator="fuller",
            fuller_alpha=1.0
        )
        assert result.n_obs == 300
        assert result.estimator == "fuller"
        assert all(np.isfinite(result.coef))

    def test_fuller_close_to_liml(self, iv_data):
        """Fuller with small alpha should be close to LIML."""
        from tabra.models.estimate.iv import IVModel
        r_liml = IVModel().fit(
            iv_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2", "z3"], estimator="liml"
        )
        r_fuller = IVModel().fit(
            iv_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2", "z3"], estimator="fuller",
            fuller_alpha=0.01
        )
        for i in range(len(r_liml.coef)):
            assert abs(r_liml.coef[i] - r_fuller.coef[i]) < 0.5


class TestKClass:
    def test_kclass_1_is_2sls(self, iv_data):
        """k-class with k=1 should equal 2SLS."""
        from tabra.models.estimate.iv import IVModel
        r_2sls = IVModel().fit(
            iv_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2", "z3"], estimator="2sls"
        )
        r_kclass = IVModel().fit(
            iv_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2", "z3"], estimator="kclass",
            kclass_k=1.0
        )
        for i in range(len(r_2sls.coef)):
            assert abs(r_2sls.coef[i] - r_kclass.coef[i]) < 0.01

    def test_kclass_basic(self, iv_data):
        from tabra.models.estimate.iv import IVModel
        result = IVModel().fit(
            iv_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2", "z3"], estimator="kclass",
            kclass_k=0.5
        )
        assert result.n_obs == 300
        assert all(np.isfinite(result.coef))


class TestClusterVCE:
    def test_cluster_vce(self, iv_data):
        from tabra.models.estimate.iv import IVModel
        result = IVModel().fit(
            iv_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2", "z3"], estimator="2sls",
            vce="cluster", cluster=["cluster_id"]
        )
        assert result.vce_type == "cluster"
        assert all(result.std_err > 0)
        assert all(np.isfinite(result.std_err))


class TestDiagnostics:
    def test_anderson_lm(self, iv_data):
        from tabra.models.estimate.iv import IVModel
        result = IVModel().fit(
            iv_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2", "z3"], estimator="2sls"
        )
        assert hasattr(result, 'idstat')
        assert result.idstat is not None
        assert result.idstat > 0

    def test_cragg_donald_f(self, iv_data):
        from tabra.models.estimate.iv import IVModel
        result = IVModel().fit(
            iv_data, y="y", exog=["x2"], endog=["x1"],
            instruments=["z1", "z2", "z3"], estimator="2sls"
        )
        assert hasattr(result, 'widstat')
        assert result.widstat is not None
        assert result.widstat > 0


class TestIVReg2API:
    def test_ivreg2_api(self, iv_data):
        from tabra.core.data import TabraData
        tab = TabraData(iv_data, is_display_result=False)
        result = tab.ivreg2(
            "y", exog=["x2"], endog=["x1"], iv=["z1", "z2", "z3"],
            estimator="2sls"
        )
        assert result is not None
        assert result.n_obs == 300

    def test_ivreg2_cue_api(self, iv_data):
        from tabra.core.data import TabraData
        tab = TabraData(iv_data, is_display_result=False)
        result = tab.ivreg2(
            "y", exog=["x2"], endog=["x1"], iv=["z1", "z2", "z3"],
            estimator="cue"
        )
        assert result.estimator == "cue"

    def test_ivreg2_cluster_api(self, iv_data):
        from tabra.core.data import TabraData
        tab = TabraData(iv_data, is_display_result=False)
        result = tab.ivreg2(
            "y", exog=["x2"], endog=["x1"], iv=["z1", "z2", "z3"],
            vce="cluster", cluster=["cluster_id"]
        )
        assert result.vce_type == "cluster"
