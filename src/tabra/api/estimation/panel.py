"""Panel-data estimation commands."""

from tabra.models.estimate.panel import PanelModel


class PanelCommands:
    """Commands that use panel settings from ``tab.xtset``."""

    def xtreg(
        self,
        y: str,
        x: list[str],
        model: str = "fe",
        is_con: bool = True,
    ):
        if self._tabra._panel_var is None:
            raise ValueError("Call xtset() first to set panel variables")
        options = [model]
        if not is_con:
            options.append("nocons")
        command = self._command("xtreg", y, x, options)
        return self._run_model(
            PanelModel(),
            command,
            y,
            x,
            self._tabra._panel_var,
            model=model,
            is_con=is_con,
        )
