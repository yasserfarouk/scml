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
from ..components.trading import (
    MarketAwarePredictionBasedTradingStrategy,
    PredictionBasedTradingStrategy,
)
from ..components.prediction import MarketAwareTradePredictionStrategy
from ..world import SCML2020Agent

__all__ = [
    "DecentralizingAgent",
    "IndDecentralizingAgent",
    "DecentralizingAgentWithLogging",
    "MarketAwareDecentralizingAgent",
    "MarketAwareIndDecentralizingAgent",
]


class _NegotiationCallbacks:
    def acceptable_unit_price(self, step: int, sell: bool) -> int:
        production_cost = np.max(self.awi.profile.costs[:, self.awi.my_input_product])
        if sell:
            return min(production_cost + self.input_cost[step:].min(), self.output_price[step])
        return max(self.output_price[step:].max() - production_cost, self.input_cost[step])

    def target_quantity(self, step: int, sell: bool) -> int:
        if sell:
            needed, secured = self.outputs_needed[step:].sum(), self.outputs_secured[step:].sum()
        else:
            needed, secured = self.inputs_needed[:step].sum(), self.inputs_secured[:step].sum()

        return min(self.awi.n_lines, needed - secured)

    def target_quantities(self, steps: Tuple[int, int], sell: bool) -> np.ndarray:
        """Implemented for speed but not really required"""

        if sell:
            needed, secured = self.outputs_needed, self.outputs_secured
        else:
            needed, secured = self.inputs_needed, self.inputs_secured

        return needed[steps[0] : steps[1]] - secured[steps[0] : steps[1]]


class DecentralizingAgent(
    _NegotiationCallbacks,
    StepNegotiationManager,
    PredictionBasedTradingStrategy,
    SupplyDrivenProductionStrategy,
    SCML2020Agent,
):
    pass


class MarketAwareDecentralizingAgent(
    MarketAwareTradePredictionStrategy,
    _NegotiationCallbacks,
    StepNegotiationManager,
    PredictionBasedTradingStrategy,
    KeepOnlyGoodPrices,
    SupplyDrivenProductionStrategy,
    SCML2020Agent,
):
    def __init__(
        self,
        *args,
        buying_margin=None,
        selling_margin=None,
        min_price_margin=0.5,
        max_price_margin=0.5,
        **kwargs
    ):
        super().__init__(
            *args,
            buying_margin=buying_margin,
            selling_margin=selling_margin,
            min_price_margin=min_price_margin,
            max_price_margin=max_price_margin,
            **kwargs
        )


class DecentralizingAgentWithLogging(
    _NegotiationCallbacks,
    StepNegotiationManager,
    PredictionBasedTradingStrategy,
    SupplyDrivenProductionStrategy,
    SCML2020Agent,
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, logdebug=True, **kwargs)


class IndDecentralizingAgent(
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
    KeepOnlyGoodPrices,
    MarketAwareTradePredictionStrategy,
    IndDecentralizingAgent,
):
    def __init__(
        self,
        *args,
        buying_margin=None,
        selling_margin=None,
        min_price_margin=0.5,
        max_price_margin=0.5,
        **kwargs
    ):
        super().__init__(
            *args,
            buying_margin=buying_margin,
            selling_margin=selling_margin,
            min_price_margin=min_price_margin,
            max_price_margin=max_price_margin,
            **kwargs
        )
