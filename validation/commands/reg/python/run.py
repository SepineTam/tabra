#!/usr/bin/env python3
"""Run Tabra reg cases and emit the normalized validation schema."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path

import pandas as pd


REG_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REG_DIRECTORY))

from runner_common import (  # noqa: E402
    load_case_specifications,
    make_document,
    verify_dataset,
    write_normalized_result,
)
from tabra.models.estimate.ols import OLS  # noqa: E402


def normalize_case(dataframe: pd.DataFrame, case_specification: dict) -> dict:
    result = OLS().fit(
        dataframe,
        y=case_specification["dependent"],
        x=case_specification["independent"],
        is_con=case_specification["include_constant"],
    )
    parameters = [
        {
            "name": name,
            "estimate": float(result.coef[index]),
            "std_error": float(result.std_err[index]),
            "statistic": float(result.t_stat[index]),
            "p_value": float(result.p_value[index]),
        }
        for index, name in enumerate(result.var_names)
    ]
    return {
        "case_id": case_specification["case_id"],
        "status": "completed",
        "converged": True,
        "n_obs": int(result.n_obs),
        "df_model": int(result._df_model),
        "df_resid": int(result._df_resid),
        "parameters": parameters,
        "statistics": {
            "r_squared": float(result.r_squared),
            "r_squared_adj": float(result.r_squared_adj),
            "f_stat": float(result.f_stat),
            "f_p_value": float(result.f_pval),
            "ss_residual": float(result.SSR),
            "ss_model": float(result.SSE),
            "ss_total": float(result.SST),
            "root_mse": float(result._root_mse),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()

    dataset_sha256, n_rows = verify_dataset(arguments.data)
    dataframe = pd.read_csv(arguments.data)
    if len(dataframe) != n_rows:
        raise RuntimeError("Parsed row count differs from the verified CSV row count")

    cases = [
        normalize_case(dataframe, case_specification)
        for case_specification in load_case_specifications()
    ]
    try:
        engine_version = importlib.metadata.version("tabra")
    except importlib.metadata.PackageNotFoundError:
        engine_version = "source-tree"
    document = make_document(
        engine_name="python",
        engine_version=engine_version,
        dataset_path=arguments.data,
        dataset_sha256=dataset_sha256,
        n_rows=n_rows,
        cases=cases,
    )
    write_normalized_result(arguments.output, document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
