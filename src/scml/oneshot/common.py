from dataclasses import dataclass

from negmas import DEFAULT_EDGE_TYPES  # NoContractExecutionMixin,

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
        "storage_cost",
        "delivery_penalty",
    ]

    exogenous_input_quantity: int
    exogenous_input_price: int
    exogenous_output_quantity: int
    exogenous_output_price: int
    storage_cost: float
    delivery_penalty: float


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
    unit_price: int
    product: int
    seller: str
    buyer: str
    time: int
    revelation_time: int


@dataclass
class OneShotProfile:
    """Defines all private information of a factory"""

    __slots__ = [
        "cost",
        "n_lines",
        "input_product",
        "delivery_penalty_mean",
        "storage_cost_mean",
        "delivery_penalty_dev",
        "storage_cost_dev",
    ]
    cost: float
    """The cost of production"""
    input_product: int
    """The index of the input product (x for $L_x$ factories)"""
    n_lines: int
    """Number of lines for this factory"""
    delivery_penalty_mean: float
    """A positive number specifying the average penalty for selling too much."""
    storage_cost_mean: float
    """A positive number specifying the average penalty buying too much."""
    delivery_penalty_dev: float
    """A positive number specifying the std. dev.  of penalty for selling too much."""
    storage_cost_dev: float
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
