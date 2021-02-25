"""
Implements the `DecentralizingAgent` which creates ony buy and one sell controller for each time-step and relinquishes
control of negotiations to buy/sell the required number of items of its input/output product.
"""
from typing import Tuple

import numpy as np
from negmas import LinearUtilityFunction

from scml.scml2020.components import IndependentNegotiationsManager
from scml.scml2020.components import StepNegotiationManager
from scml.scml2020.components import SupplyDrivenProductionStrategy

from ..components.signing import KeepOnlyGoodPrices
from ..components.trading import PredictionBasedTradingStrategy
from ..components.prediction import MarketAwareTradePredictionStrategy
from ..world import SCML2020Agent

__all__ = [
    "DecentralizingAgent",
    "MarketAwareDecentralizingAgent",
    "IndDecentralizingAgent",
    "MarketAwareIndDecentralizingAgent",
    "DecentralizingAgentWithLogging",
]


class _NegotiationCallbacks:
    def acceptable_unit_price(self, step: int, sell: bool) -> int:
        production_cost = np.max(self.awi.profile.costs[:, self.awi.my_input_product])
        if sell:
            return production_cost + self.input_cost[step]
        return self.output_price[step] - production_cost

    def target_quantity(self, step: int, sell: bool) -> int:
        if sell:
            needed, secured = self.outputs_needed, self.outputs_secured
        else:
            needed, secured = self.inputs_needed, self.inputs_secured

        return needed[step] - secured[step]

    def target_quantities(self, steps: Tuple[int, int], sell: bool) -> np.ndarray:
        """Implemented for speed but not really required"""

        if sell:
            needed, secured = self.outputs_needed, self.outputs_secured
        else:
            needed, secured = self.inputs_needed, self.inputs_secured

        return needed[steps[0] : steps[1]] - secured[steps[0] : steps[1]]


class DecentralizingAgent(
    KeepOnlyGoodPrices,
    _NegotiationCallbacks,
    StepNegotiationManager,
    PredictionBasedTradingStrategy,
    SupplyDrivenProductionStrategy,
    SCML2020Agent,
):
    pass


class MarketAwareDecentralizingAgent(
    MarketAwareTradePredictionStrategy,
    KeepOnlyGoodPrices,
    _NegotiationCallbacks,
    StepNegotiationManager,
    PredictionBasedTradingStrategy,
    SupplyDrivenProductionStrategy,
    SCML2020Agent,
):
    pass


class DecentralizingAgentWithLogging(
    KeepOnlyGoodPrices,
    _NegotiationCallbacks,
    StepNegotiationManager,
    PredictionBasedTradingStrategy,
    SupplyDrivenProductionStrategy,
    SCML2020Agent,
):
    def __init__(self, *args, buying_margin=0.33, selling_margin=0.33, **kwargs):
        super().__init__(
            *args,
            buying_margin=buying_margin,
            selling_margin=selling_margin,
            logdebug=True,
            **kwargs
        )


class IndDecentralizingAgent(
    KeepOnlyGoodPrices,
    _NegotiationCallbacks,
    IndependentNegotiationsManager,
    PredictionBasedTradingStrategy,
    SupplyDrivenProductionStrategy,
    SCML2020Agent,
):
    def create_ufun(self, is_seller: bool, issues=None, outcomes=None):
        if is_seller:
            return LinearUtilityFunction((1, 1, 10))
        return LinearUtilityFunction((1, -1, -10))


class MarketAwareIndDecentralizingAgent(
    MarketAwareTradePredictionStrategy,
    IndDecentralizingAgent,
):
    pass
