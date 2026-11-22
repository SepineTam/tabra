#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Song Tan, Inc. All Rights Reserved
#
# @Author : Song Tan
# @Email  : sepinetam@gmail.com
# @File   : tabra/data/io/_io.py

from pathlib import Path
from typing import overload

from ..dataset import Dataset


@overload
def load_data(str) -> Dataset:
    ...

@overload
def load_data(Path) -> Dataset:
    ...

def load_data(Path) -> Dataset:
    ...


def export_data(
    dataset: Dataset,
    target: Path | None = None,
    *,
    replace: bool = False,
) -> Path:
    if target is None:
        target = Path.cwd() / f"{dataset.name}.dta"

    if target.is_file():
        if replace:
            target.unlink()
        else:
            raise FileExistsError(f"{target} already exists, you should set replace as `True`")

    _extension = Path(dataset.name).suffix
    ...

    return target
