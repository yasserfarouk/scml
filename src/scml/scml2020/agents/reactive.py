"""
Implements the `DecentralizingAgent` which creates ony buy and one sell controller for each time-step and relinquishes
control of negotiations to buy/sell the required number of items of its input/output product.
"""
from typing import Tuple

import numpy as np

from scml.scml2020.components import FixedTradePredictionStrategy
from scml.scml2020.components import StepNegotiationManager
from scml.scml2020.components.production import TradeDrivenProductionStrategy

from ..components.signing import KeepOnlyGoodPrices
from ..components.trading import ReactiveTradingStrategy
from ..world import SCML2020Agent

__all__ = ["ReactiveAgent", "MarketAwareReactiveAgent"]


class ReactiveAgent(
    StepNegotiationManager,
    ReactiveTradingStrategy,
    TradeDrivenProductionStrategy,
    FixedTradePredictionStrategy,
    SCML2020Agent,
):
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

class MarketAwareReactiveAgent(KeepOnlyGoodPrices, ReactiveAgent):
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

