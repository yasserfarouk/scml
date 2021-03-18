"""
Implements the base class for agents that negotiate independently with different partners.

These agents do not take production capacity, availability of materials or any other aspects of the simulation into
account. They are to serve only as baselines.

Assumptions
-----------

The main assumptions of the agents based on `IndependentNegotiationsAgent` are:

1. All production processes take one input time and generate one output type.

"""

import numpy as np

from ..components.negotiation import IndependentNegotiationsManager
from ..components.prediction import FixedTradePredictionStrategy
from ..components.signing import KeepOnlyGoodPrices
from ..components.trading import ReactiveTradingStrategy
from ..world import SCML2020Agent

__all__ = ["IndependentNegotiationsAgent", "MarketAwareIndependentNegotiationsAgent"]


class IndependentNegotiationsAgent(
    IndependentNegotiationsManager,
    FixedTradePredictionStrategy,
    ReactiveTradingStrategy,
    SCML2020Agent,
):
    """
    Implements the base class for agents that negotiate independently with different partners.

    These agents do not take production capacity, availability of materials or any other aspects of the simulation into
    account. They are to serve only as baselines.

    Remarks:

        - `IndependentNegotiationsAgent` agents assume that each production process has one input type with the same
           index as itself and one output type with one added to the index (i.e. process $i$ takes product $i$ as input
           and creates product $i+1$ as output.
        - It does not assume that all lines have the same production cost (it uses the average cost though).
        - It does not assume that the agent has a single production process.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def acceptable_unit_price(self, step: int, sell: bool) -> int:
        production_cost = np.max(self.awi.profile.costs[:, self.awi.my_input_product])
        if sell:
            return max(production_cost + self.input_cost[step], self.output_price[step])
        return min(self.output_price[step] - production_cost, self.input_cost[step])

    def target_quantity(self, step: int, sell: bool) -> int:
        if sell:
            needed, secured = self.outputs_needed, self.outputs_secured
        else:
            needed, secured = self.inputs_needed, self.inputs_secured

        return needed[step] - secured[step]

class MarketAwareIndependentNegotiationsAgent(KeepOnlyGoodPrices, IndependentNegotiationsAgent):
    """
    Implements the base class for agents that negotiate independently with different partners using trading/catalog
    prices to control signing

    These agents do not take production capacity, availability of materials or any other aspects of the simulation into
    account. They are to serve only as baselines.

    Remarks:

        - `IndependentNegotiationsAgent` agents assume that each production process has one input type with the same
           index as itself and one output type with one added to the index (i.e. process $i$ takes product $i$ as input
           and creates product $i+1$ as output.
        - It does not assume that all lines have the same production cost (it uses the average cost though).
        - It does not assume that the agent has a single production process.

    """
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
