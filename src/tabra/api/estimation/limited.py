"""Limited-dependent-variable estimation commands."""

from tabra.models.estimate.heckman import HeckmanModel
from tabra.models.estimate.tobit import TobitModel


class LimitedCommands:
    """Commands for censoring and sample-selection models."""

    def heckman(
        self,
        y: str,
        x: list[str],
        select_x: list[str],
        select_var: str = None,
        is_con: bool = True,
        method: str = "mle",
    ):
        select_terms = " ".join(select_x)
        if select_var:
            select_terms = f"{select_var} = {select_terms}"
        options = [f"select({select_terms})"]
        if method != "mle":
            options.append(method)
        if not is_con:
            options.append("nocons")
        command = self._command("heckman", y, x, options)
        return self._run_model(
            HeckmanModel(),
            command,
            y,
            x,
            select_x=select_x,
            select_var=select_var,
            is_con=is_con,
            method=method,
        )

    def tobit(
        self,
        y: str,
        x: list[str],
        ll=None,
        ul=None,
        vce: str = "unadjusted",
        is_con: bool = True,
    ):
        options = []
        if ll is not None:
            options.append(f"ll({ll})")
        if ul is not None:
            options.append(f"ul({ul})")
        if vce != "unadjusted":
            options.append(f"vce({vce})")
        if not is_con:
            options.append("nocons")
        command = self._command("tobit", y, x, options)
        return self._run_model(
            TobitModel(),
            command,
            y,
            x,
            ll=ll,
            ul=ul,
            vce=vce,
            is_con=is_con,
        )
