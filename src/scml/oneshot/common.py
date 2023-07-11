from dataclasses import dataclass
from typing import Literal

from attr import define
from negmas import SAONMI
from negmas.common import define
from negmas.outcomes import DiscreteCartesianOutcomeSpace
from negmas.sao.common import SAOState

from ..scml2019.common import FinancialReport
from ..scml2020.common import QUANTITY, TIME, UNIT_PRICE

__all__ = [
    "QUANTITY",
    "UNIT_PRICE",
    "TIME",
    "OneShotState",
    "OneShotExogenousContract",
    "OneShotProfile",
]


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


@define
class OneShotProfile:
    """Defines all private information of a factory"""

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


@define(frozen=True)
class NegotiationDetails:
    buyer: str
    seller: str
    product: int
    nmi: SAONMI


@define(frozen=True)
class OneShotState:
    """State of a one-shot agent"""

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
    total_sales: int
    """Total quantity registered as sales using `awi.register_sale`."""
    total_supplies: int
    """Total quantity registered as supplies using `awi.register_supply`."""

    n_products: int
    """ Number of products in the production chain."""
    n_processes: int
    """ Number of processes in the production chain."""
    n_competitors: int
    """ Number of other factories on the same production level."""
    all_suppliers: list[list[str]]
    """ A list of all suppliers by product."""
    all_consumers: list[list[str]]
    """ A list of all consumers by product."""
    bankrupt_agents: list[str]
    """list of bankrupt agents"""
    catalog_prices: list[float]
    """A list of the catalog prices (by product)."""
    price_multiplier: float
    """The multiplier multiplied by the trading/catalog price when the negotiation agendas are created to decide the maximum and lower quantities. """
    is_exogenous_forced: bool
    """exogenous contracts always forced or can the agent decide not to sign them. """
    current_step: int
    """Current simulation step (inherited from `negmas.situated.AgentWorldInterface` )."""
    n_steps: int
    """Number of simulation steps (inherited from `negmas.situated.AgentWorldInterface` )."""
    relative_simulation_time: float
    """Fraction of the simulation completed (inherited from `negmas.situated.AgentWorldInterface`)."""
    profile: OneShotProfile
    """Gives the agent profile including its production cost, number of
    production lines, input product index, mean of its delivery penalties,
    mean of its disposal costs, standard deviation of its shortfall penalties
    and standard deviation of its disposal costs. See `OneShotProfile` for full
    description. This information is private information and no other agent knows it."""
    n_lines: int
    """The number of production lines in the factory (private information)."""
    is_first_level: bool
    """Is the agent in the first production level (i.e. it is an input agent that buys the raw material)."""
    is_last_level: bool
    """Is the agent in the last production level (i.e. it is an output agent that sells the final product)."""
    is_middle_level: bool
    """Is the agent neither a first level nor a last level agent"""
    my_input_product: int
    """The input product to the factory controlled by the agent."""
    my_output_product: int
    """The output product from the factory controlled by the agent."""
    level: int
    """The production level which is numerically the same as the input product."""
    my_suppliers: list[str]
    """A list of IDs for all suppliers to the agent (i.e. agents that can sell the input product of the agent)."""
    my_consumers: list[str]
    """A list of IDs for all consumers to the agent (i.e. agents that can buy the output product of the agent)."""
    my_partners: list[str]
    """A list of IDs for all negotiation partners of the agent (in the order suppliers then consumers)."""
    penalties_scale: Literal["trading", "catalog", "unit", "none"]
    """The scale at which to calculate disposal cost/delivery penalties.
    "trading" and "catalog" mean trading and catalog prices. "unit" means the
    contract's unit price while "none" means that disposal cost/shortfall penalty are absolute."""
    n_input_negotiations: int
    """Number of negotiations with suppliers."""
    n_output_negotiations: int
    """Number of negotiations with consumers."""
    trading_prices: list[float]
    """The trading prices of all products. This information is only available if `publish_trading_prices` is set in the world."""
    exogenous_contract_summary: list[tuple[int, int]]
    """A list of n_products lists each giving the total quantity and average price of exogenous contracts for a product. This information is only available if `publish_exogenous_summary` is set in the world."""
    reports_of_agents: dict[str, dict[int, FinancialReport]]
    """Gives all past financial reports of a given agent. See `FinancialReport` for details."""
    current_input_outcome_space: DiscreteCartesianOutcomeSpace
    """The current issues for all negotiations to buy the input product of the agent. If the agent is at level zero, this will be empty. This is exactly the same as current_input_outcome_space.issues"""
    current_output_outcome_space: DiscreteCartesianOutcomeSpace
    """The current issues for all negotiations to buy the output product of
    the agent. If the agent is at level n_products - 1, this will be empty.
    This is exactly the same as current_output_outcome_space.issues"""
    current_negotiation_details: dict[str, dict[str, NegotiationDetails]]
    """Details on all current negotiations separated into "buy" and "sell" dictionaries."""
    sales: dict[str, int]
    """Today's sales per customer so far."""
    supplies: dict[str, int]
    """Today's supplies per supplier so far."""
    needed_sales: int
    """Today's needed sales as of now (exogenous input - exogenous output - total sales so far)."""
    needed_supplies: int
    """Today's needed supplies  as of now (exogenous output - exogenous input - total supplies)."""
    # running_negotiations: dict[str, SAOState]
    # """Maps partner ID to the state of the running negotiation with it (if any)"""
