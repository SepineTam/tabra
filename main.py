import pandas as pd

from tabra import load_data
from tabra.plot import PlotKind
from tabra.plot.template import AER


tab = load_data(pd.read_stata(
    "https://www.stata-press.com/data/r19/auto.dta",
    convert_categoricals=False,
    storage_options={"User-Agent": "Mozilla/5.0"}
))

tab.data.sum()
tab.data.gen("p__sq", "price ^ 2")

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

tab.reg("price", ["mpg", "weight"])
tab.plot.coefplot().save(".local/figs/demo_coefplot", formats=["pdf"])

tab.xtset("foreign")
tab.xtreg("price", ["mpg", "weight"], model="fe")
