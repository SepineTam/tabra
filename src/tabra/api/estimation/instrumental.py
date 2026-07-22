"""Instrumental-variable estimation commands."""

from tabra.models.estimate.iv import IVModel
from tabra.models.estimate.ivprobit import IVProbitModel
from tabra.models.estimate.ivreghdfe import IVRegHDFEModel
from tabra.models.estimate.ivtobit import IVTobitModel


class InstrumentalCommands:
    """Commands for linear and nonlinear instrumental-variable models."""

    @staticmethod
    def _iv_equation(endog: list[str], instruments: list[str]) -> str:
        return f"({' '.join(endog)} = {' '.join(instruments)})"

    def ivreg(
        self,
        y: str,
        exog: list[str],
        endog: list[str],
        iv: list[str],
        estimator: str = "2sls",
        vce: str = "unadjusted",
        is_con: bool = True,
    ):
        options = []
        if estimator != "2sls":
            options.append(estimator)
        if vce != "unadjusted":
            options.append(f"vce({vce})")
        if not is_con:
            options.append("nocons")
        command = self._command(
            "ivreg", y, [*exog, self._iv_equation(endog, iv)], options
        )
        return self._run_model(
            IVModel(),
            command,
            y,
            exog=exog,
            endog=endog,
            instruments=iv,
            estimator=estimator,
            vce=vce,
            is_con=is_con,
        )

    def ivreg2(
        self,
        y: str,
        exog: list[str],
        endog: list[str],
        iv: list[str],
        estimator: str = "2sls",
        vce: str = "unadjusted",
        cluster: list[str] = None,
        is_con: bool = True,
        fuller_alpha: float = 1.0,
        kclass_k: float = None,
    ):
        options = [estimator]
        if vce != "unadjusted":
            options.append(f"vce({vce})")
        if cluster:
            options.append(f"cluster({' '.join(cluster)})")
        if not is_con:
            options.append("nocons")
        command = self._command(
            "ivreg2", y, [*exog, self._iv_equation(endog, iv)], options
        )
        return self._run_model(
            IVModel(),
            command,
            y,
            exog=exog,
            endog=endog,
            instruments=iv,
            estimator=estimator,
            vce=vce,
            cluster=cluster,
            is_con=is_con,
            fuller_alpha=fuller_alpha,
            kclass_k=kclass_k,
        )

    def ivprobit(
        self,
        y: str,
        exog: list[str],
        endog: list[str],
        iv: list[str],
        method: str = "mle",
        vce: str = "unadjusted",
        is_con: bool = True,
    ):
        options = []
        if method != "mle":
            options.append(method)
        if vce != "unadjusted":
            options.append(f"vce({vce})")
        if not is_con:
            options.append("nocons")
        command = self._command(
            "ivprobit", y, [*exog, self._iv_equation(endog, iv)], options
        )
        return self._run_model(
            IVProbitModel(),
            command,
            y,
            exog=exog,
            endog=endog,
            instruments=iv,
            method=method,
            vce=vce,
            is_con=is_con,
        )

    def ivtobit(
        self,
        y: str,
        exog: list[str],
        endog: list[str],
        iv: list[str],
        ll=None,
        ul=None,
        method: str = "mle",
        vce: str = "unadjusted",
        is_con: bool = True,
    ):
        options = []
        if ll is not None:
            options.append(f"ll({ll})")
        if ul is not None:
            options.append(f"ul({ul})")
        if method != "mle":
            options.append(method)
        if vce != "unadjusted":
            options.append(f"vce({vce})")
        if not is_con:
            options.append("nocons")
        command = self._command(
            "ivtobit", y, [*exog, self._iv_equation(endog, iv)], options
        )
        return self._run_model(
            IVTobitModel(),
            command,
            y,
            exog=exog,
            endog=endog,
            instruments=iv,
            ll=ll,
            ul=ul,
            method=method,
            vce=vce,
            is_con=is_con,
        )

    def ivreghdfe(
        self,
        y: str,
        exog: list[str],
        endog: list[str],
        iv: list[str],
        absorb: list[str],
        estimator: str = "2sls",
        vce: str = "unadjusted",
        cluster: list[str] = None,
        is_con: bool = True,
    ):
        options = [f"absorb({' '.join(absorb)})"]
        if estimator != "2sls":
            options.append(estimator)
        if vce != "unadjusted":
            options.append(f"vce({vce})")
        if cluster:
            options.append(f"cluster({' '.join(cluster)})")
        if not is_con:
            options.append("nocons")
        command = self._command(
            "ivreghdfe", y, [*exog, self._iv_equation(endog, iv)], options
        )
        return self._run_model(
            IVRegHDFEModel(),
            command,
            y,
            exog=exog,
            endog=endog,
            instruments=iv,
            absorb=absorb,
            estimator=estimator,
            vce=vce,
            cluster=cluster,
            is_con=is_con,
        )
