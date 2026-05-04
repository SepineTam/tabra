#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : about.py

from __future__ import annotations

import platform
import shutil
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd


try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def _get_ram_info():
    if not _HAS_PSUTIL:
        return None, None, None
    mem = psutil.virtual_memory()  # type: ignore[reportPossiblyUnboundVariable]
    total_gb = mem.total / (1024 ** 3)
    used_gb = mem.used / (1024 ** 3)
    available_gb = mem.available / (1024 ** 3)
    return total_gb, used_gb, available_gb


def _get_disk_info():
    disk = shutil.disk_usage("/")
    total_gb = disk.total / (1024 ** 3)
    used_gb = disk.used / (1024 ** 3)
    free_gb = disk.free / (1024 ** 3)
    return total_gb, used_gb, free_gb


def _get_cpu_info():
    if _HAS_PSUTIL:
        physical = psutil.cpu_count(logical=False)  # type: ignore[reportPossiblyUnboundVariable]
        logical = psutil.cpu_count(logical=True)  # type: ignore[reportPossiblyUnboundVariable]
        return physical or "unknown", logical or "unknown"
    return "unknown", "unknown"


@dataclass
class AboutInfo:
    """System and runtime environment information."""

    tabra_version: str
    python_version: str
    os_name: str
    os_version: str
    os_release: str
    machine: str
    processor: str
    cpu_physical: int | str
    cpu_logical: int | str
    ram_total_gb: float | None
    ram_used_gb: float | None
    ram_available_gb: float | None
    disk_total_gb: float
    disk_used_gb: float
    disk_free_gb: float
    pandas_version: str
    numpy_version: str

    def __str__(self) -> str:
        lines = [
            "Tabra Environment",
            "=" * 50,
            f"  Tabra version      : {self.tabra_version}",
            f"  Python version     : {self.python_version}",
            "-" * 50,
            f"  OS                 : {self.os_name}",
            f"  OS version         : {self.os_version}",
            f"  OS release         : {self.os_release}",
            f"  Machine            : {self.machine}",
            f"  Processor          : {self.processor}",
            "-" * 50,
            f"  CPU (physical)     : {self.cpu_physical}",
            f"  CPU (logical)      : {self.cpu_logical}",
        ]
        if self.ram_total_gb is not None:
            lines += [
                "-" * 50,
                f"  RAM total          : {self.ram_total_gb:.2f} GB",
                f"  RAM used           : {self.ram_used_gb:.2f} GB",
                f"  RAM available      : {self.ram_available_gb:.2f} GB",
            ]
        lines += [
            "-" * 50,
            f"  Disk total         : {self.disk_total_gb:.2f} GB",
            f"  Disk used          : {self.disk_used_gb:.2f} GB",
            f"  Disk free          : {self.disk_free_gb:.2f} GB",
            "-" * 50,
            f"  Pandas version     : {self.pandas_version}",
            f"  NumPy version      : {self.numpy_version}",
            "=" * 50,
        ]
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks."""
        rows = [
            ("Tabra version", self.tabra_version),
            ("Python version", self.python_version),
            ("OS", f"{self.os_name} {self.os_version}"),
            ("OS release", self.os_release),
            ("Machine", self.machine),
            ("Processor", self.processor),
            ("CPU (physical)", str(self.cpu_physical)),
            ("CPU (logical)", str(self.cpu_logical)),
        ]
        if self.ram_total_gb is not None:
            rows += [
                ("RAM total", f"{self.ram_total_gb:.2f} GB"),
                ("RAM used", f"{self.ram_used_gb:.2f} GB"),
                ("RAM available", f"{self.ram_available_gb:.2f} GB"),
            ]
        rows += [
            ("Disk total", f"{self.disk_total_gb:.2f} GB"),
            ("Disk used", f"{self.disk_used_gb:.2f} GB"),
            ("Disk free", f"{self.disk_free_gb:.2f} GB"),
            ("Pandas version", self.pandas_version),
            ("NumPy version", self.numpy_version),
        ]
        html_rows = "".join(
            f"<tr><td style='padding:3px 14px 3px 0;color:#555;font-weight:600;'>"
            f"{k}</td><td style='padding:3px 0;'>{v}</td></tr>"
            for k, v in rows
        )
        return (
            "<div style='font-family:monospace;font-size:13px;'>"
            "<table style='border-collapse:collapse;'>"
            f"{html_rows}"
            "</table></div>"
        )


class About:
    """Accessor for retrieving system and runtime information."""

    def __init__(self, tabra):
        self._tabra = tabra

    def __call__(self, is_display: bool = True) -> AboutInfo:
        """Return system and runtime environment information.

        Args:
            is_display (bool): If True, print the information immediately.
                Default True.

        Returns:
            AboutInfo: A dataclass containing system and runtime info.
        """
        info = self._collect()
        if is_display:
            print(info)
        return info

    def _collect(self) -> AboutInfo:
        try:
            from importlib.metadata import version
            tabra_version = version("tabra")
        except Exception:
            tabra_version = "unknown"

        ram_total, ram_used, ram_avail = _get_ram_info()
        disk_total, disk_used, disk_free = _get_disk_info()
        cpu_phys, cpu_log = _get_cpu_info()

        return AboutInfo(
            tabra_version=tabra_version,
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            os_name=platform.system(),
            os_version=platform.version(),
            os_release=platform.release(),
            machine=platform.machine(),
            processor=platform.processor() or "unknown",
            cpu_physical=cpu_phys,
            cpu_logical=cpu_log,
            ram_total_gb=ram_total,
            ram_used_gb=ram_used,
            ram_available_gb=ram_avail,
            disk_total_gb=disk_total,
            disk_used_gb=disk_used,
            disk_free_gb=disk_free,
            pandas_version=pd.__version__,
            numpy_version=np.__version__,
        )
