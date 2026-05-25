import pandas as pd
import pytest

from tabra.core.est_accessor import EstAccessor


class DummyResult:
    def __init__(self):
        self.style = None
        self.display_called = False
        self.command = None

    def set_style(self, style):
        self.style = style

    def set_display(self, flag):
        if flag:
            self.display_called = True

    def set_command(self, command):
        self.command = command


class DummyModel:
    def __init__(self, recorder):
        self.recorder = recorder

    def fit(self, *args, **kwargs):
        self.recorder["args"] = args
        self.recorder["kwargs"] = kwargs
        return DummyResult()


class DummyTabra:
    def __init__(self, *, is_display_result):
        self._df = pd.DataFrame({"y": [1.0, 2.0], "x": [0.1, 0.2]})
        self._style = "stata"
        self._is_display_result = is_display_result
        self._result = None
        self._panel_var = None


def test_reg_assigns_style_updates_tabra_result_and_display(monkeypatch):
    recorder = {}

    def _factory():
        return DummyModel(recorder)

    monkeypatch.setattr("tabra.core.est_accessor.OLS", _factory)

    tabra = DummyTabra(is_display_result=True)
    accessor = EstAccessor(tabra)

    result = accessor.reg("y", ["x"], is_con=False)

    assert recorder["args"][0] is tabra._df
    assert recorder["args"][1] == "y"
    assert recorder["args"][2] == ["x"]
    assert recorder["kwargs"]["is_con"] is False
    assert result.style == "stata"
    assert result.display_called is True
    assert result.command == "reg y x, nocons"
    assert tabra._result is result


def test_reg_no_display_when_disabled(monkeypatch):
    recorder = {}

    def _factory():
        return DummyModel(recorder)

    monkeypatch.setattr("tabra.core.est_accessor.OLS", _factory)

    tabra = DummyTabra(is_display_result=False)
    accessor = EstAccessor(tabra)

    result = accessor.reg("y", ["x"])

    assert result.display_called is False
    assert result.command == "reg y x"


def test_xtreg_requires_xtset_before_fit():
    tabra = DummyTabra(is_display_result=False)
    accessor = EstAccessor(tabra)

    with pytest.raises(ValueError, match=r"Call xtset\(\) first"):
        accessor.xtreg("y", ["x"])


def test_xtreg_passes_panel_var_and_model_options(monkeypatch):
    recorder = {}

    def _factory():
        return DummyModel(recorder)

    monkeypatch.setattr("tabra.core.est_accessor.PanelModel", _factory)

    tabra = DummyTabra(is_display_result=False)
    tabra._panel_var = ("id", "year")
    accessor = EstAccessor(tabra)

    result = accessor.xtreg("y", ["x"], model="re", is_con=False)

    assert recorder["args"][0] is tabra._df
    assert recorder["args"][1] == "y"
    assert recorder["args"][2] == ["x"]
    assert recorder["args"][3] == ("id", "year")
    assert recorder["kwargs"]["model"] == "re"
    assert recorder["kwargs"]["is_con"] is False
    assert result.style == "stata"
    assert result.command == "xtreg y x, re nocons"
