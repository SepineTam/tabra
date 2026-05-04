#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Smoke test: verify the installed package can be imported and basic operations work.

import numpy as np
import pandas as pd


def test_import():
    import tabra
    from tabra.core.data import TabraData
    assert hasattr(tabra, "load_data")
    assert hasattr(TabraData, "est")


def test_basic_reg():
    from tabra.core.data import TabraData
    np.random.seed(0)
    df = pd.DataFrame({
        "y": np.random.randn(20) + 2.0,
        "x": np.random.randn(20),
    })
    tab = TabraData(df, is_display_result=False)
    result = tab.est.reg("y", ["x"], is_con=True)
    assert result.n_obs == 20
    assert len(result.coef) == 2  # x + _cons
    assert np.isfinite(result.coef).all()


if __name__ == "__main__":
    test_import()
    test_basic_reg()
    print("Smoke test passed.")
