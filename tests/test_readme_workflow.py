"""Acceptance tests for the non-plotting workflow documented in README.md."""

import numpy as np
import pandas as pd

from tabra import load_data
from tabra.results.base import BaseResult


def _readme_data() -> pd.DataFrame:
    random = np.random.default_rng(20260722)
    n_obs = 80
    foreign = np.repeat([0, 1], n_obs // 2)
    rep78 = np.tile(np.arange(1, 6), n_obs // 5)
    mpg = random.normal(22, 4, n_obs)
    weight = random.normal(3000, 450, n_obs)
    fixed_effect = rep78 * 180 + foreign * 300
    price = 2500 - 65 * mpg + 2.1 * weight + fixed_effect
    price += random.normal(0, 250, n_obs)
    return pd.DataFrame(
        {
            "price": price,
            "mpg": mpg,
            "weight": weight,
            "rep78": rep78,
            "foreign": foreign,
        }
    )


def test_readme_non_plotting_workflow():
    tab = load_data(_readme_data(), is_display_result=False)

    about = tab.about(is_display=False)
    assert about.python_version

    assert tab.setting.plot(block=False) is tab.setting

    summary = tab.data.sum()
    assert isinstance(summary, BaseResult)
    assert "price" in summary.var_names

    assert tab.data.gen("p__sq", "price ^ 2") is tab.data
    assert np.allclose(tab._df["p__sq"], tab._df["price"] ** 2)

    hdfe_result = tab.est.reghdfe(
        "price", ["mpg", "weight"], absorb=["rep78"]
    )
    assert isinstance(hdfe_result, BaseResult)
    assert hdfe_result.display_command().startswith("reghdfe price mpg weight")

    ols_result = tab.est.reg("price", ["mpg", "weight"])
    assert isinstance(ols_result, BaseResult)
    assert tab.result is ols_result
    assert tab.results.last is ols_result
    assert tab.model is not None

    tab.xtset("foreign")
    panel_result = tab.est.xtreg("price", ["mpg", "weight"], model="fe")
    assert isinstance(panel_result, BaseResult)
    assert panel_result.display_command() == "xtreg price mpg weight, fe"
    assert tab.results.last is panel_result
    assert len(tab.results.history) == 4


def test_accessors_are_stable_session_objects():
    tab = load_data(_readme_data(), is_display_result=False)

    assert tab.data is tab.data
    assert tab.est is tab.est
    assert tab.plot is tab.plot
    assert tab.about is tab.about
