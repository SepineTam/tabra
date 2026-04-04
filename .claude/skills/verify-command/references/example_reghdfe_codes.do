// reghdfe cross-validation: Stata side

clear all
set more off

// Case 1: auto, absorb(rep78) unadjusted
sysuse auto, clear
reghdfe price weight mpg, absorb(rep78)

// Case 2: auto, absorb(rep78) robust
sysuse auto, clear
reghdfe price weight mpg, absorb(rep78) vce(robust)

// Case 3: auto, absorb(rep78 foreign) unadjusted
sysuse auto, clear
reghdfe price weight mpg, absorb(rep78 foreign)

// Case 4: nlswork, absorb(idcode year) unadjusted
webuse nlswork, clear
reghdfe ln_wage age tenure ttl_exp grade, absorb(idcode year)

// Case 5: nlswork, absorb(idcode year) robust
webuse nlswork, clear
reghdfe ln_wage age tenure ttl_exp grade, absorb(idcode year) vce(robust)

// Case 6: nlswork, absorb(idcode year) cluster(idcode)
webuse nlswork, clear
reghdfe ln_wage age tenure ttl_exp grade, absorb(idcode year) vce(cluster idcode)
