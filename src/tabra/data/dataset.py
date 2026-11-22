#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Song Tan, Inc. All Rights Reserved
#
# @Author : Song Tan
# @Email  : sepinetam@gmail.com
# @File   : tabra/data/dataset.py

class Dataset:
    def __init__(self):
        self.name: str = "Untitled"
        self._variables = {}
        self._observations = 0

    @property
    def observations(self):
        return self._observations

    @property
    def n_vars(self):
        return len(self._variables)

    @property
    def frames(self) -> tuple[int, int]:
        return self.observations, self.n_vars

    def get_variable(self, name):
        return self._variables[name]

    def add_variable(self, variable):
        ...

    def drop_variable(self, name):
        ...

    def rename_variable(self, old, new):
        ...
