import pandas as pd

from tabra import load_data


df = pd.read_stata("/Applications/StataNow/auto.dta")

tab = load_data(df)

result = tab.reg("price", ["weight", "mpg"], is_con=True)

