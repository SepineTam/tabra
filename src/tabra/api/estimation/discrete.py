"""Discrete-choice and generalized linear model commands."""

from tabra.models.estimate.binary_choice import LogitModel, ProbitModel
from tabra.models.estimate.glm import GLMModel
from tabra.models.estimate.mlogit import MultinomialLogitModel
from tabra.models.estimate.ordered_choice import OrderedLogitModel, OrderedProbitModel


class DiscreteCommands:
    """Commands for binary, ordered, multinomial, and GLM outcomes."""

    def probit(self, y: str, x: list[str], is_con: bool = True):
        options = [] if is_con else ["nocons"]
        command = self._command("probit", y, x, options)
        return self._run_model(ProbitModel(), command, y, x, is_con=is_con)

    def logit(self, y: str, x: list[str], is_con: bool = True):
        options = [] if is_con else ["nocons"]
        command = self._command("logit", y, x, options)
        return self._run_model(LogitModel(), command, y, x, is_con=is_con)

    def oprobit(self, y: str, x: list[str], is_con: bool = True):
        options = [] if is_con else ["nocons"]
        command = self._command("oprobit", y, x, options)
        return self._run_model(OrderedProbitModel(), command, y, x, is_con=is_con)

    def ologit(self, y: str, x: list[str], is_con: bool = True):
        options = [] if is_con else ["nocons"]
        command = self._command("ologit", y, x, options)
        return self._run_model(OrderedLogitModel(), command, y, x, is_con=is_con)

    def glm(
        self,
        y: str,
        x: list[str],
        family: str = "gaussian",
        link: str = None,
        is_con: bool = True,
    ):
        options = [f"family({family})"]
        if link:
            options.append(f"link({link})")
        if not is_con:
            options.append("nocons")
        command = self._command("glm", y, x, options)
        return self._run_model(
            GLMModel(),
            command,
            y,
            x,
            family=family,
            link=link,
            is_con=is_con,
        )

    def mlogit(
        self,
        y: str,
        x: list[str],
        base_outcome=None,
        is_con: bool = True,
    ):
        options = []
        if base_outcome is not None:
            options.append(f"baseoutcome({base_outcome})")
        if not is_con:
            options.append("nocons")
        command = self._command("mlogit", y, x, options)
        return self._run_model(
            MultinomialLogitModel(),
            command,
            y,
            x,
            base_outcome=base_outcome,
            is_con=is_con,
        )
