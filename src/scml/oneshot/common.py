from dataclasses import dataclass

from ..scml2020.common import QUANTITY
from ..scml2020.common import TIME
from ..scml2020.common import UNIT_PRICE

__all__ = [
    "QUANTITY",
    "UNIT_PRICE",
    "TIME",
    "OneShotState",
    "OneShotExogenousContract",
    "OneShotProfile",
]


@dataclass
class OneShotState:
    """State of a one-shot agent"""

    __slots__ = [
        "exogenous_input_quantity",
        "exogenous_input_price",
        "exogenous_output_quantity",
        "exogenous_output_price",
        "disposal_cost",
        "shortfall_penalty",
        "current_balance",
    ]

    exogenous_input_quantity: int
    """Exogenous input quantity for the current step"""
    exogenous_input_price: int
    """Exogenous input unit price for the current step"""
    exogenous_output_quantity: int
    """Exogenous output quantity for the current step"""
    exogenous_output_price: int
    """Exogenous output unit price for the current step"""
    disposal_cost: float
    """Current unit disposal cost"""
    shortfall_penalty: float
    """Current unit shortfall penalty"""
    current_balance: int
    """Current balance of the agent."""


@dataclass
class OneShotExogenousContract:
    """Exogenous contract information"""

    __slots__ = [
        "quantity",
        "unit_price",
        "product",
        "seller",
        "buyer",
        "time",
        "revelation_time",
    ]

    quantity: int
    """Contract quantity"""
    unit_price: int
    """Contract unit price"""
    product: int
    """Product index"""
    seller: str
    """Seller ID (when passing contrtacts to the constructor of SCML2020OneShotWorld, you can also pass an interged index referring to the agent's index in the `agent_types` list)"""
    buyer: str
    """Buyer ID (when passing contrtacts to the constructor of SCML2020OneShotWorld, you can also pass an interged index referring to the agent's index in the `agent_types` list)"""
    time: int
    """Simulation step at which the contract is exceucted"""
    revelation_time: int
    """Simulation step at which the contract is revealed to its owner. Should not exceed `time` and the default `generate()` method sets it to time"""


@dataclass
class OneShotProfile:
    """Defines all private information of a factory"""

    __slots__ = [
        "cost",
        "n_lines",
        "input_product",
        "shortfall_penalty_mean",
        "disposal_cost_mean",
        "shortfall_penalty_dev",
        "disposal_cost_dev",
    ]
    cost: float
    """The cost of production"""
    input_product: int
    """The index of the input product (x for $L_x$ factories)"""
    n_lines: int
    """Number of lines for this factory"""
    shortfall_penalty_mean: float
    """A positive number specifying the average penalty for selling too much."""
    disposal_cost_mean: float
    """A positive number specifying the average penalty buying too much."""
    shortfall_penalty_dev: float
    """A positive number specifying the std. dev.  of penalty for selling too much."""
    disposal_cost_dev: float
    """A positive number specifying the std. dev. penalty buying too much."""

    @property
    def level(self):
        return self.input_product

    @property
    def output_product(self):
        return self.input_product + 1

    @property
    def process(self):
        return self.input_product
