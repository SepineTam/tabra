#!/usr/bin/env python3
"""Run Tabra reg cases and emit the normalized validation schema."""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import sys
from pathlib import Path

import pandas as pd

REG_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REG_DIRECTORY))

from runner_common import (
    hash_included_row_ids,
    load_case_specifications,
    load_raw_row_ids,
    make_document,
    verify_dataset,
    write_normalized_result,
)

from tabra.models.estimate.ols import OLS


def normalize_case(
    dataframe: pd.DataFrame,
    raw_row_ids: list[str],
    case_specification: dict,
) -> dict:
    estimation_columns = [
        case_specification["dependent"],
        *case_specification["independent"],
    ]
    complete_case_mask = dataframe[estimation_columns].notna().all(axis=1)
    estimation_dataframe = dataframe.loc[complete_case_mask]
    result = OLS().fit(
        estimation_dataframe,
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
    if int(complete_case_mask.sum()) != int(result.n_obs):
        raise RuntimeError("Python estimation sample does not match result.n_obs")
    included_row_ids = [
        row_id
        for row_id, included in zip(raw_row_ids, complete_case_mask, strict=True)
        if included
    ]
    requested_parameters = [*case_specification["independent"]]
    if case_specification["include_constant"]:
        requested_parameters.append("_cons")
    returned_parameters = set(result.var_names)
    if not returned_parameters.issubset(requested_parameters):
        raise RuntimeError("Python returned an unexpected parameter")
    return {
        "case_id": case_specification["case_id"],
        "included_row_sha256": hash_included_row_ids(included_row_ids),
        "omitted_parameters": [
            parameter
            for parameter in requested_parameters
            if parameter not in returned_parameters
        ],
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

    raw_row_ids = load_raw_row_ids(arguments.data)
    if len(raw_row_ids) != n_rows:
        raise RuntimeError("Raw row_id count differs from the verified CSV row count")
    cases = [
        normalize_case(dataframe, raw_row_ids, case_specification)
        for case_specification in load_case_specifications()
    ]
    try:
        engine_version = importlib.metadata.version("tabra")
    except importlib.metadata.PackageNotFoundError:
        engine_version = "source-tree"
    document = make_document(
        engine_name="python",
        engine_version=platform.python_version(),
        dataset_path=arguments.data,
        dataset_sha256=dataset_sha256,
        n_rows=n_rows,
        cases=cases,
    )
    document["metadata"]["tabra_version"] = engine_version
    write_normalized_result(arguments.output, document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
