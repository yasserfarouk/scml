"""
Implements the `DecentralizingAgent` which creates ony buy and one sell controller for each time-step and relinquishes
control of negotiations to buy/sell the required number of items of its input/output product.
"""
import numpy as np
from typing import Tuple

from scml.scml2020.components import FixedTradePredictionStrategy
from scml.scml2020.components import StepNegotiationManager
from scml.scml2020.components.production import TradeDrivenProductionStrategy
from ..components.trading import ReactiveTradingStrategy

__all__ = ["ReactiveAgent"]

from ..world import SCML2020Agent


class ReactiveAgent(
    StepNegotiationManager,
    ReactiveTradingStrategy,
    TradeDrivenProductionStrategy,
    FixedTradePredictionStrategy,
    SCML2020Agent
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

        return needed[steps[0]: steps[1]] - secured[steps[0]: steps[1]]
