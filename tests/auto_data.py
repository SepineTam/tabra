"""Utilities for loading Stata auto dataset in tests."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
import pytest

_AUTO_URL = "https://www.stata-press.com/data/r19/auto2.dta"
_CACHE_PATH = Path(".local/data/auto.dta")


def load_auto_df(*, convert_categoricals=True):
    """Load Stata auto dataset with local-cache-first strategy.

    Priority:
    1) local repository cache at .local/data/auto.dta
    2) download from official Stata Press URL and cache locally
    """
    if _CACHE_PATH.exists():
        return pd.read_stata(_CACHE_PATH, convert_categoricals=convert_categoricals)

    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(_AUTO_URL, timeout=20) as response:
            payload = response.read()
    except Exception as exc:
        pytest.skip(f"Auto dataset unavailable in this environment: {exc}")
    data = BytesIO(payload)
    df = pd.read_stata(data, convert_categoricals=convert_categoricals)
    _CACHE_PATH.write_bytes(payload)
    return df
