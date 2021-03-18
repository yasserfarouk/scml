"""
Implements the `DecentralizingAgent` which creates ony buy and one sell controller for each time-step and relinquishes
control of negotiations to buy/sell the required number of items of its input/output product.
"""
from scml.scml2020.components import MovingRangeNegotiationManager
from scml.scml2020.components import SupplyDrivenProductionStrategy

from ..components import KeepOnlyGoodPrices
from ..components.trading import PredictionBasedTradingStrategy
from ..components.prediction import MarketAwareTradePredictionStrategy
from ..world import SCML2020Agent

__all__ = ["MovingRangeAgent", "MarketAwareMovingRangeAgent"]


class MovingRangeAgent(
    MovingRangeNegotiationManager,
    PredictionBasedTradingStrategy,
    SupplyDrivenProductionStrategy,
    SCML2020Agent,
):
    pass


class MarketAwareMovingRangeAgent(MarketAwareTradePredictionStrategy, MovingRangeAgent):
    def __init__(
        self,
        *args,
        min_price_margin=0.5,
        max_price_margin=0.5,
        **kwargs
    ):
        super().__init__(
            *args,
            min_price_margin=min_price_margin,
            max_price_margin=max_price_margin,
            **kwargs
        )

