"""
Implements the `DecentralizingAgent` which creates ony buy and one sell controller for each time-step and relinquishes
control of negotiations to buy/sell the required number of items of its input/output product.
"""
from scml.scml2020.components import (
    SupplyDrivenProductionStrategy,
    MovingRangeNegotiationManager,
)
from ..components.trading import PredictionBasedTradingStrategy

__all__ = ["MovingRangeAgent"]

from ..world import SCML2020Agent


class MovingRangeAgent(
    MovingRangeNegotiationManager,
    PredictionBasedTradingStrategy,
    SupplyDrivenProductionStrategy,
    SCML2020Agent,
):
    pass
