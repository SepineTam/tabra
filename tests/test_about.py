import numpy as np
import pandas as pd
import pytest

from tabra import load_data
from tabra.core.about import AboutInfo, _get_cpu_info, _get_disk_info, _get_ram_info


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({"y": np.random.randn(n), "x1": np.random.randn(n)})


class TestAbout:
    def test_about_returns_dataclass(self, sample_df):
        data = load_data(sample_df)
        info = data.about(is_display=False)
        assert isinstance(info, AboutInfo)

    def test_about_info_has_expected_fields(self, sample_df):
        data = load_data(sample_df)
        info = data.about(is_display=False)
        assert info.os_name in ("Darwin", "Linux", "Windows")
        assert info.python_version.count(".") >= 1
        assert info.disk_total_gb > 0
        assert info.disk_used_gb >= 0
        assert info.disk_free_gb >= 0

    def test_about_info_str_not_empty(self, sample_df):
        data = load_data(sample_df)
        info = data.about(is_display=False)
        s = str(info)
        assert "Tabra Environment" in s
        assert info.os_name in s

    def test_about_info_html_not_empty(self, sample_df):
        data = load_data(sample_df)
        info = data.about(is_display=False)
        html = info._repr_html_()
        assert "<table" in html
        assert info.os_name in html

    def test_about_display_true_prints(self, sample_df, capsys):
        data = load_data(sample_df)
        info = data.about(is_display=True)
        captured = capsys.readouterr()
        assert "Tabra Environment" in captured.out
        assert isinstance(info, AboutInfo)

    def test_about_display_false_no_print(self, sample_df, capsys):
        data = load_data(sample_df)
        info = data.about(is_display=False)
        captured = capsys.readouterr()
        assert captured.out == ""
        assert isinstance(info, AboutInfo)


class TestAboutHelpers:
    def test_get_ram_info_returns_tuple(self):
        total, used, avail = _get_ram_info()
        if total is not None:
            assert isinstance(total, float)
            assert isinstance(used, float)
            assert isinstance(avail, float)
            assert total > 0

    def test_get_disk_info_returns_tuple(self):
        total, used, free = _get_disk_info()
        assert isinstance(total, float)
        assert isinstance(used, float)
        assert isinstance(free, float)
        assert total > 0

    def test_get_cpu_info_returns_tuple(self):
        phys, log = _get_cpu_info()
        assert phys is not None
        assert log is not None
