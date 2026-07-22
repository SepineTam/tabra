"""Linear and quantile estimation commands."""

from tabra.models.estimate.ols import OLS
from tabra.models.estimate.qreg import QuantileRegression
from tabra.models.estimate.reghdfe import RegHDFE


class LinearCommands:
    """Commands for linear outcome models."""

    def reg(self, y: str, x: list[str], is_con: bool = True):
        options = [] if is_con else ["nocons"]
        command = self._command("reg", y, x, options)
        return self._run_model(OLS(), command, y, x, is_con=is_con)

    def reghdfe(
        self,
        y: str,
        x: list[str],
        absorb: list[str],
        vce: str = "unadjusted",
        cluster: list[str] = None,
        is_con: bool = True,
    ):
        options = [f"absorb({' '.join(absorb)})"]
        if vce != "unadjusted":
            options.append(f"vce({vce})")
        if cluster:
            options.append(f"cluster({' '.join(cluster)})")
        if not is_con:
            options.append("nocons")
        command = self._command("reghdfe", y, x, options)
        return self._run_model(
            RegHDFE(),
            command,
            y,
            x,
            absorb=absorb,
            vce=vce,
            cluster=cluster,
            is_con=is_con,
        )

    def qreg(
        self,
        y: str,
        x: list[str],
        quantile: float = 0.5,
        is_con: bool = True,
    ):
        options = []
        if quantile != 0.5:
            options.append(f"quantile({quantile})")
        if not is_con:
            options.append("nocons")
        command = self._command("qreg", y, x, options)
        return self._run_model(
            QuantileRegression(),
            command,
            y,
            x,
            quantile=quantile,
            is_con=is_con,
        )
