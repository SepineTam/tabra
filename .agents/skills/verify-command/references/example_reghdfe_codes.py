# reghdfe cross-validation: Python side

import pandas as pd

from tabra import load_data

auto = pd.read_stata(".local/data/auto.dta", convert_categoricals=False)
nlswork = pd.read_stata(".local/data/nlswork.dta", convert_categoricals=False)

if isinstance(nlswork, pd.io.stata.StataReader):
    nlswork = nlswork.read()
nlswork = nlswork.dropna(subset=["ln_wage"])

# Case 1: auto, absorb(rep78) unadjusted
tab = load_data(auto)
tab.reghdfe("price", ["weight", "mpg"], absorb=["rep78"])

# Case 2: auto, absorb(rep78) robust
tab.reghdfe("price", ["weight", "mpg"], absorb=["rep78"], vce="robust")

# Case 3: auto, absorb(rep78 foreign) unadjusted
tab.reghdfe("price", ["weight", "mpg"], absorb=["rep78", "foreign"])

del tab
# Case 4: nlswork, absorb(idcode year) unadjusted
tab = load_data(nlswork)
tab.reghdfe("ln_wage", ["age", "tenure", "ttl_exp", "grade"],
            absorb=["idcode", "year"])

# Case 5: nlswork, absorb(idcode year) robust
tab.reghdfe("ln_wage", ["age", "tenure", "ttl_exp", "grade"],
            absorb=["idcode", "year"], vce="robust")

# Case 6: nlswork, absorb(idcode year) cluster(idcode)
tab.reghdfe("ln_wage", ["age", "tenure", "ttl_exp", "grade"],
            absorb=["idcode", "year"], vce="cluster", cluster=["idcode"])
