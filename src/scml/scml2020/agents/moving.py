"""
Implements the `DecentralizingAgent` which creates ony buy and one sell controller for each time-step and relinquishes
control of negotiations to buy/sell the required number of items of its input/output product.
"""
from scml.scml2020.components import MovingRangeNegotiationManager
from scml.scml2020.components import SupplyDrivenProductionStrategy

from ..components import KeepOnlyGoodPrices
from ..components.trading import PredictionBasedTradingStrategy
from ..world import SCML2020Agent

__all__ = ["MovingRangeAgent"]



class MovingRangeAgent(
    KeepOnlyGoodPrices,
    MovingRangeNegotiationManager,
    PredictionBasedTradingStrategy,
    SupplyDrivenProductionStrategy,
    SCML2020Agent,
):
    pass
