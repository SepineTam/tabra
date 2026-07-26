version 16.0
clear all
set more off

args data_path output_path

capture confirm file "`data_path'"
if _rc {
    display as error "Validation data file not found: `data_path'"
    exit 601
}

quietly import delimited using "`data_path'", clear varnames(1) asdouble
confirm variable y x1 x2

file open results using "`output_path'", write replace text
file write results "case_id" _tab "engine_version" _tab "n_obs" _tab ///
    "df_model" _tab "df_resid" _tab "term" _tab "estimate" _tab ///
    "std_error" _tab "statistic" _tab "p_value" _tab "r_squared" _tab ///
    "r_squared_adj" _tab "f_stat" _tab "f_p_value" _tab ///
    "ss_residual" _tab "ss_model" _tab "ss_total" _tab "root_mse" _n

foreach case_id in basic no_constant {
    if "`case_id'" == "basic" {
        quietly regress y x1 x2
        local terms x1 x2 _cons
    }
    else {
        quietly regress y x1 x2, noconstant
        local terms x1 x2
    }

    scalar validation_tss = e(rss) + e(mss)

    foreach term of local terms {
        scalar validation_t = _b[`term'] / _se[`term']
        scalar validation_p = 2 * ttail(e(df_r), abs(validation_t))
        scalar validation_fp = Ftail(e(df_m), e(df_r), e(F))

        file write results "`case_id'" _tab "`c(stata_version)'" _tab ///
            %24.17e (e(N)) _tab %24.17e (e(df_m)) _tab ///
            %24.17e (e(df_r)) _tab "`term'" _tab ///
            %24.17e (_b[`term']) _tab %24.17e (_se[`term']) _tab ///
            %24.17e (validation_t) _tab %24.17e (validation_p) _tab ///
            %24.17e (e(r2)) _tab %24.17e (e(r2_a)) _tab ///
            %24.17e (e(F)) _tab %24.17e (validation_fp) _tab ///
            %24.17e (e(rss)) _tab %24.17e (e(mss)) _tab ///
            %24.17e (validation_tss) _tab %24.17e (e(rmse)) _n
    }
}

file close results
exit 0
