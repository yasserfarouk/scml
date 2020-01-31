"""
Implements the `DecentralizingAgent` which creates ony buy and one sell controller for each time-step and relinquishes
control of negotiations to buy/sell the required number of items of its input/output product.
"""
import numpy as np
from typing import Tuple

from negmas import LinearUtilityFunction

from scml.scml2020.components import FixedERPStrategy
from scml.scml2020.components import (
    SupplyDrivenProductionStrategy,
    StepNegotiationManager,
    MovingRangeNegotiationManager,
)
from .do_nothing import DoNothingAgent
from ..components.trading import PredictionBasedTradingStrategy

__all__ = ["MovingRangeAgent"]


class MovingRangeAgent(
    MovingRangeNegotiationManager,
    PredictionBasedTradingStrategy,
    SupplyDrivenProductionStrategy,
    DoNothingAgent,
):
    pass
