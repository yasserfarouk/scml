import random
import sys
from dataclasses import dataclass
from typing import Literal

from attr import define
from negmas import make_issue, make_os
from negmas.outcomes import DiscreteCartesianOutcomeSpace, Outcome
from negmas.sao import SAONMI
from negmas.sao.common import SAOState

__all__ = [
    "QUANTITY",
    "UNIT_PRICE",
    "TIME",
    "OneShotState",
    "OneShotExogenousContract",
    "OneShotProfile",
    "FinancialReport",
    "is_system_agent",
    "INFINITE_COST",
    "SYSTEM_BUYER_ID",
    "SYSTEM_SELLER_ID",
    "is_system_agent",
]


QUANTITY = 0
"""Index of quantity in negotiation issues"""


TIME = 1
"""Index of time in negotiation issues"""


UNIT_PRICE = 2
"""Index of unit price in negotiation issues"""

INFINITE_COST = sys.maxsize // 2
"""A constant indicating an invalid cost for lines incapable of running some process"""

SYSTEM_SELLER_ID = "SELLER"
"""ID of the system seller agent"""

SYSTEM_BUYER_ID = "BUYER"
"""ID of the system buyer agent"""

COMPENSATION_ID = "COMPENSATOR"
"""ID of the takeover agent"""


def is_system_agent(aid: str) -> bool:
    """
    Checks whether an agent is a system agent or not

    Args:

        aid: Agent ID

    Returns:

        True if the ID is for a system agent.
    """
    return (
        aid.startswith(SYSTEM_SELLER_ID)
        or aid.startswith(SYSTEM_BUYER_ID)
        or aid.startswith(COMPENSATION_ID)
    )


@dataclass
class FinancialReport:
    """A report published periodically by the system showing the financial standing of an agent"""

    __slots__ = [
        "agent_id",
        "step",
        "cash",
        "assets",
        "breach_prob",
        "breach_level",
        "is_bankrupt",
        "agent_name",
    ]
    agent_id: str
    """Agent ID"""
    step: int
    """Simulation step at the beginning of which the report was published."""
    cash: int
    """Cash in the agent's wallet. Negative numbers indicate liabilities."""
    assets: int
    """Value of the products in the agent's inventory @ catalog prices. """
    breach_prob: float
    """Number of times the agent breached a contract over the total number of contracts it signed."""
    breach_level: float
    """Sum of the agent's breach levels so far divided by the number of contracts it signed."""
    is_bankrupt: bool
    """Whether the agent is already bankrupt (i.e. incapable of doing any more transactions)."""
    agent_name: str
    """Agent name for printing purposes"""

    def __str__(self):
        bankrupt = "BANKRUPT" if self.is_bankrupt else ""
        return (
            f"{self.agent_name} @ {self.step} {bankrupt}: Cash: {self.cash}, Assets: {self.assets}, "
            f"breach_prob: {self.breach_prob}, breach_level: {self.breach_level} "
            f"{'(BANKRUPT)' if self.is_bankrupt else ''}"
        )


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
    storage_cost_mean: float
    """A positive number specifying the average cost for keeping inventory for one step. This is only used if the products are not `perishable`."""
    storage_cost_dev: float
    """A positive number specifying the std. dev.  cost for keeping inventory for one step. This is only used if the products are not `perishable`."""

    @property
    def level(self):
        return self.input_product

    @property
    def output_product(self):
        return self.input_product + 1

    @property
    def process(self):
        return self.input_product

    @classmethod
    def random(cls, input_product: int, oneshot: bool) -> "OneShotProfile":
        scm = random.random() * 0.02
        scv = random.random() * 0.01
        dcm = dcv = 0
        if oneshot:
            scm, scv, dcm, dcv = dcm, dcv, scm, scv

        return OneShotProfile(
            cost=random.randint(1, 4),
            input_product=input_product,
            n_lines=10,
            shortfall_penalty_mean=random.random() * 0.2,
            shortfall_penalty_dev=random.random() * 0.02,
            disposal_cost_mean=dcm,
            disposal_cost_dev=dcv,
            storage_cost_mean=scm,
            storage_cost_dev=scv,
        )


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
    """Total quantity registered as sales today using `awi.register_sale`."""
    total_supplies: int
    """Total quantity registered as supplies today using `awi.register_supply`."""
    total_future_sales: int
    """Total quantity registered as sales in the future using `awi.register_sale`."""
    total_future_supplies: int
    """Total quantity registered as supplies in the future using `awi.register_supply`."""

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
    production_capacities: list[int]
    """ A list of total production capacity per production level."""
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
    """A list of n_products lists each giving the total quantity and average price of exogenous
    contracts for a product. This information is only available if `publish_exogenous_summary` is set in the world."""
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
    """Today supplies per supplier so far."""
    needed_sales: int
    """Today's needed sales as of now (exogenous input - exogenous output - total sales so far)."""
    needed_supplies: int
    """Today needed supplies  as of now (exogenous output - exogenous input - total supplies)."""
    perishable: bool = True
    """Is this a perishable domain (oneshot) of not (std) """
    allow_zero_quantity: bool = False
    """Does this world allow zero quantity in negotiated offers"""
    storage_cost: float = 0.0
    """Current unit storage cost. Only used in standard worlds where products are not perishable"""

    @property
    def running_buy_states(self) -> dict[str, SAOState]:
        """All running buy negotiations as a mapping from partner ID to current negotiation state"""
        return {  # type: ignore
            partner: info.nmi.state
            for partner, info in self.current_negotiation_details["buy"].items()
        }

    @property
    def current_sell_states(self) -> dict[str, SAOState]:
        """All running sell negotiations as a mapping from partner ID to current negotiation state"""
        return {  # type: ignore
            partner: info.nmi.state
            for partner, info in self.current_negotiation_details["sell"].items()
        }

    @property
    def current_states(self) -> dict[str, SAOState]:
        """All running negotiations as a mapping from partner ID to current negotiation state"""
        d = self.running_buy_states | self.current_sell_states
        return d

    @property
    def current_buy_nmis(self) -> dict[str, SAONMI]:
        """All running buy negotiations as a mapping from partner ID to current negotiation nmi"""
        return {  # type: ignore
            partner: info.nmi
            for partner, info in self.current_negotiation_details["buy"].items()
        }

    @property
    def current_sell_nmis(self) -> dict[str, SAONMI]:
        """All running sell negotiations as a mapping from partner ID to current negotiation nmi"""
        return {  # type: ignore
            partner: info.nmi
            for partner, info in self.current_negotiation_details["sell"].items()
        }

    @property
    def current_nmis(self) -> dict[str, SAONMI]:
        """All running negotiations as a mapping from partner ID to current negotiation state"""
        d = self.current_buy_nmis
        d.update(self.current_sell_nmis)
        return d

    @property
    def current_buy_offers(self) -> dict[str, Outcome]:
        """All current buy negotiations as a mapping from partner ID to current offer"""
        return {  # type: ignore
            partner: info.nmi.state.current_offer  # type: ignore
            for partner, info in self.current_negotiation_details["buy"].items()
            if info.nmi.state.running and info.nmi.state.started
        }

    @property
    def current_sell_offers(self) -> dict[str, Outcome]:
        """All current sell negotiations as a mapping from partner ID to current offer"""
        return {  # type: ignore
            partner: info.nmi.state.current_offer  # type: ignore
            for partner, info in self.current_negotiation_details["sell"].items()
            if info.nmi.state.running and info.nmi.state.started
        }

    @property
    def current_offers(self) -> dict[str, Outcome]:
        """All current negotiations as a mapping from partner ID to current offer"""
        d = self.current_buy_offers | self.current_sell_offers
        return d

    @classmethod
    def random(cls, oneshot: bool | None = None) -> "OneShotState":  # type: ignore
        if oneshot is None:
            oneshot = random.randint(0, 1) > 0
        storage_cost, disposal_cost = 0.0, 0.2 * random.random() + 0.1
        if not oneshot:
            storage_cost, disposal_cost = disposal_cost / 5, storage_cost
        n_processes = 2 if oneshot else random.randint(2, 4)
        level = random.randint(0, n_processes - 1)
        n_agents_per_process = [random.randint(2, 8) for _ in range(n_processes)]
        names, nxt = [], 0
        namesof = dict()
        for p in range(n_processes):
            namesof[p] = [
                f"{_:02}"
                + random.choice("ABCDEFGZY")
                + random.choice("abdioxfwl")
                + f"@{p:02}"
                for _ in range(nxt, nxt + n_agents_per_process[p])
            ]
            names += namesof[p]
            nxt += n_agents_per_process[p]
        if level == 0:
            ein, eout = random.randint(5, 10), 0
        elif level == n_processes - 1:
            eout, ein = random.randint(5, 10), 0
        else:
            eout, ein = 0, 0

        ip = random.randint(10, 20)
        my_suppliers = namesof[level - 1] if level > 0 else ["SELLER"]
        my_consumers = names[level + 1] if level < n_processes - 1 else ["BUYER"]
        n_steps = random.randint(50, 200)
        step = random.randint(0, n_steps)
        esummary: list[tuple[int, int]] = [(0, 0) for _ in range(n_processes)]
        esummary[0] = (random.randint(6, 10), random.randint(10, 23))
        esummary[-1] = (random.randint(6, 10), random.randint(45, 67))
        return OneShotState(
            exogenous_input_quantity=ein,
            exogenous_input_price=ip,
            exogenous_output_quantity=eout,
            exogenous_output_price=random.randint(10, 30) + ip,
            disposal_cost=disposal_cost,
            shortfall_penalty=random.random() * 0.4 + 0.1,
            current_balance=random.randint(10000, 20000),
            total_sales=random.randint(0, 20),
            total_supplies=random.randint(0, 20),
            total_future_sales=random.randint(0, 200),
            total_future_supplies=random.randint(0, 200),
            n_products=n_processes + 1,
            n_processes=n_processes,
            n_competitors=n_agents_per_process[level] - 1,
            all_suppliers=[["SELLER"]] + [namesof[k] for k in range(len(namesof) - 1)],
            all_consumers=[namesof[k] for k in range(len(namesof) - 1)] + [["BUYER"]],
            bankrupt_agents=(
                [random.choice(names) for _ in range(x)]
                if (x := random.randint(0, 10)) != 0
                else []
            ),
            catalog_prices=[
                random.random() * (i + 1) * 10 for i in range(n_processes + 1)
            ],
            price_multiplier=random.random() * 0.5 + 1.5,
            is_exogenous_forced=True,
            current_step=step,
            n_steps=n_steps,
            relative_simulation_time=step / n_steps,
            profile=OneShotProfile.random(level, oneshot),
            n_lines=10,
            is_first_level=level == 0,
            is_last_level=level == n_processes - 1,
            is_middle_level=0 < level < n_processes - 1,
            my_input_product=level,
            my_output_product=level + 1,
            level=level,
            my_suppliers=my_suppliers,
            my_consumers=my_consumers,
            my_partners=my_suppliers + my_consumers,
            penalties_scale=random.choice(["trading", "catalog", "unit", "none"]),
            n_input_negotiations=n_agents_per_process[level - 1] if level > 0 else 0,
            n_output_negotiations=(
                n_agents_per_process[level + 1] if level < n_processes - 1 else 0
            ),
            trading_prices=[
                random.random() * 50 + 10.0 for _ in range(n_processes + 1)
            ],
            exogenous_contract_summary=esummary,
            reports_of_agents=dict(),
            current_input_outcome_space=make_os(  # type: ignore
                [
                    make_issue((1, 10)),
                    (
                        make_issue(step, step)
                        if oneshot
                        else make_issue(step, step + random.randint(0, 4))
                    ),
                    make_issue((20.0, 50.0)),
                ]
            ),
            current_output_outcome_space=make_os(  # type: ignore
                [
                    make_issue((1, 10)),
                    (
                        make_issue(step, step)
                        if oneshot
                        else make_issue(step, step + random.randint(0, 4))
                    ),
                    make_issue((40.0, 90.0)),
                ]
            ),
            current_negotiation_details=dict(),
            sales=dict(),
            supplies=dict(),
            needed_sales=random.randint(0, 10) if level < n_processes - 1 else 0,
            needed_supplies=random.randint(0, 10) if level > 0 else 0,
            perishable=oneshot,
            storage_cost=storage_cost,
        )
