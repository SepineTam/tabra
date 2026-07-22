"""Public ``tab.est`` accessor assembled from domain command groups."""

from tabra.api.estimation.base import EstimationCommandBase
from tabra.api.estimation.discrete import DiscreteCommands
from tabra.api.estimation.instrumental import InstrumentalCommands
from tabra.api.estimation.limited import LimitedCommands
from tabra.api.estimation.linear import LinearCommands
from tabra.api.estimation.panel import PanelCommands


class EstAccessor(
    LinearCommands,
    PanelCommands,
    DiscreteCommands,
    LimitedCommands,
    InstrumentalCommands,
    EstimationCommandBase,
):
    """Stata-like estimation commands grouped internally by model family."""

    pass
