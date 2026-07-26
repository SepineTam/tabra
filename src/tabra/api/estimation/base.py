"""Shared execution support for estimation command accessors."""


class EstimationCommandBase:
    """Run estimators through the session's common result lifecycle."""

    def __init__(self, tabra):
        self._tabra = tabra

    @staticmethod
    def _command(name: str, y: str, variables=None, options=None) -> str:
        parts = [name, y]
        if variables:
            parts.extend(variables)
        command = " ".join(str(part) for part in parts)
        option_list = [str(option) for option in (options or []) if option]
        if option_list:
            command += ", " + " ".join(option_list)
        return command

    def _run_model(self, estimator_instance, command: str, *args, **kwargs):
        result = estimator_instance.fit(self._tabra._df, *args, **kwargs)
        return self._tabra._finalize_result(
            result,
            command=command,
            model=estimator_instance,
        )
