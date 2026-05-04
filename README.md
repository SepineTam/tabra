# Tabra
Tabra is a Python toolkit for empirical research workflows, designed for data processing, model estimation, result export, and reproducible analysis.

[![Publish to PyPI](https://github.com/SepineTam/tabra/actions/workflows/publish.yml/badge.svg)](https://github.com/SepineTam/tabra/actions/workflows/publish.yml)
[![PyPI version](https://img.shields.io/pypi/v/tabra.svg)](https://pypi.org/project/tabra/)
[![PyPI Downloads](https://static.pepy.tech/badge/tabra)](https://pepy.tech/projects/tabra)
[![Issue](https://img.shields.io/badge/Issue-report-green.svg)](https://github.com/sepinetam/tabra/issues/new)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/SepineTam/tabra)

## Quickly Start

```bash
uv add 'tabra[all]'
```

```python
import pandas as pd

from tabra import load_data
from tabra.plot import PlotKind
from tabra.plot.template import AER


tab = load_data(pd.read_stata(
    "https://www.stata-press.com/data/r19/auto.dta",
    convert_categoricals=False,
    storage_options={"User-Agent": "Mozilla/5.0"}
))

tab.about()

tab.setting.plot(block=False)

tab.data.sum()
tab.data.gen("p__sq", "price ^ 2")

tab.est.reghdfe("price", ["mpg", "weight"], absorb=["rep78"])

tab.plot.hist(
    "p__sq",
    bins=20,
    title="Price Distribution"
).save(".local/figs/demo_hist_of_price_square", formats=["png", "jpg"])

mix_figure = tab.plot.mix(
    [
        {PlotKind.scatter: {"x": "mpg", "y": "price"}},
        {PlotKind.lfitci: {"x": "mpg", "y": "price"}},
    ],
    title="Price vs. Mpg",
    xtitle="mpg",
    ytitle="price and price^2",
    template=AER
).show()

tab.est.reg("price", ["mpg", "weight"])
tab.plot.coefplot().save(".local/figs/demo_coefplot", formats=["pdf"])

tab.xtset("foreign")
tab.est.xtreg("price", ["mpg", "weight"], model="fe")

```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development workflow.

