#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : importers.py

from os import PathLike
from pathlib import Path

import pandas as pd

from tabra.core.data import TabraData


_PATH_READERS = {
    ".csv": pd.read_csv,
    ".dta": pd.read_stata,
}


def _read_data_path(source: str | PathLike) -> pd.DataFrame:
    path = Path(source).expanduser()
    display_path = path.resolve(strict=False)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {display_path}")
    if not path.is_file():
        raise ValueError(f"Data path must point to a file: {display_path}")

    reader = _PATH_READERS.get(path.suffix.lower())
    if reader is None:
        supported = ", ".join(sorted(_PATH_READERS))
        raise ValueError(
            f"Unsupported data file format '{path.suffix or '<none>'}' for "
            f"{display_path}. Supported formats: {supported}"
        )
    return reader(path)


def load_data(df, *, style="stata", is_display_result=True):
    """Load a DataFrame or supported data file into a TabraData instance.

    Args:
        df (pd.DataFrame | str | os.PathLike): Input DataFrame or path to a
            CSV or Stata file.
        style (str): Output display style. Default "stata".
        is_display_result (bool): Whether to print estimation results immediately. Default True.

    Returns:
        TabraData: A wrapped data object.

    Example:
        >>> import pandas as pd
        >>> dta = load_data(pd.DataFrame({"x": [1, 2, 3]}))
    """
    if isinstance(df, pd.DataFrame):
        dataframe = df
    elif isinstance(df, (str, PathLike)):
        dataframe = _read_data_path(df)
    else:
        raise TypeError(
            "df must be a pandas DataFrame or a supported file path, "
            f"got {type(df).__name__}"
        )

    return TabraData(
        dataframe,
        style=style,
        is_display_result=is_display_result,
    )
