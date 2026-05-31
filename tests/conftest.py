from __future__ import annotations

from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "unit: fast unit tests")
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "plot: plotting and visualization tests")
    config.addinivalue_line("markers", "ops: numerical operation tests")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        path = Path(str(item.fspath)).as_posix()
        filename = Path(path).name

        if "/tests/ops/" in path:
            item.add_marker(pytest.mark.ops)
            item.add_marker(pytest.mark.unit)
            continue

        if "plot" in filename or "heatmap" in filename or "kdensity" in filename:
            item.add_marker(pytest.mark.plot)
            item.add_marker(pytest.mark.integration)
            continue

        if filename in {"smoke_test.py", "test_api.py", "test_save.py"}:
            item.add_marker(pytest.mark.integration)
            continue

        item.add_marker(pytest.mark.unit)
