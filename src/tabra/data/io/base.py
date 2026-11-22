#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Song Tan, Inc. All Rights Reserved
#
# @Author : Song Tan
# @Email  : sepinetam@gmail.com
# @File   : tabra/data/io/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

# from ..dataset import Dataset

class IOBase(ABC):
    @abstractmethod
    def load_data(self, path: Path) -> 'Dataset':
        pass

    @abstractmethod
    def save_data(self, path: Path, data: 'Dataset'):
        pass
