"""Variable creation and replacement commands."""

from tabra.data.expressions import ExpressionEngine


class VariableCommands:
    """Commands that mutate variables in a ``TabraData`` session."""

    def __init__(self, tabra, owner):
        self._tabra = tabra
        self._owner = owner

    def gen(self, var: str, expr: str, replace: bool = False):
        """Generate a variable from a restricted Stata-like expression."""
        if var in self._tabra._df.columns and not replace:
            raise ValueError(
                f"Variable '{var}' already exists. Use replace=True to overwrite, "
                "or use tab.data.replace() / tab.data.drop() first."
            )

        frame = self._tabra._df.copy()
        frame[var] = ExpressionEngine.evaluate(expr, frame)
        self._tabra._df = frame
        return self._owner

    def replace(self, var: str, expr: str, cond: str = None):
        """Replace all or selected values of an existing variable."""
        if var not in self._tabra._df.columns:
            raise KeyError(f"Variable '{var}' not found in DataFrame")

        if cond is None:
            self._tabra._df[var] = ExpressionEngine.evaluate(
                expr, self._tabra._df
            )
        else:
            mask = self._tabra._df.eval(cond)
            self._tabra._df.loc[mask, var] = ExpressionEngine.evaluate(
                expr,
                self._tabra._df.loc[mask],
            )
        return self._owner
