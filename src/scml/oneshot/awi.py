"""
Implements the one shot version of the Agent-World Interface.

"""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import TYPE_CHECKING, Literal, Any
import numpy as np
from negmas import ContiguousIssue
from negmas.outcomes import DiscreteCartesianOutcomeSpace, Outcome, make_os
from negmas.sao import SAONMI, SAOState
from negmas.situated import AgentWorldInterface

from .common import (
    FinancialReport,
    NegotiationDetails,
    OneShotProfile,
    OneShotState,
    is_system_agent,
)

if TYPE_CHECKING:
    from scml.oneshot.world import SCMLBaseWorld

    from .agent import OneShotAgent

__all__ = ["OneShotAWI"]


def defaultdict_int() -> dict[Any, int]:
    return defaultdict(int)


class OneShotAWI(AgentWorldInterface):
    """
    The agent world interface for the one-shot game.

    This class contains all the methods needed to access the simulation to
    extract information which are divided into 4 groups:

    Static World Information:
        Information about the world and the agent that does not change over
        time. These include:

        A. Market Information:
          - **n_products**: Number of products in the production chain.
          - **n_processes**: Number of processes in the production chain.
          - **n_competitors**: Number of other factories on the same production level.
          - **all_suppliers**: A list of all suppliers by product.
          - **all_consumers**: A list of all consumers by product.
          - **proudction_capacities**: The total production capacity (i.e. number of lines)
                                     for each production level (i.e. manufacturing process).
          - **is_system**: Is the given system ID corresponding to a system agent?
          - **is_bankrupt**: Is the given agent bankrupt? None asks about self
          - **catalog_prices**: A list of the catalog prices (by product).
          - **price_multiplier**: The multiplier multiplied by the trading/catalog price
            when the negotiation agendas are created to decide the maximum and lower quantities.
          - **is_exogenous_forced**: Are exogenous contracts always forced or can the
            agent decide not to sign them.
          - **current_step**: Current simulation step (inherited from `negmas.situated.AgentWorldInterface` ).
          - **n_steps**: Number of simulation steps (inherited from `negmas.situated.AgentWorldInterface` ).
          - **relative_time**: fraction of the simulation completed (inherited from `negmas.situated.AgentWorldInterface`).
          - **state**: The full state of the agent ( `OneShotState` ).
          - **settings* The system settings (inherited from `negmas.situated.AgentWorldInterface` ).
          - **quantity_range* The maximum quantity in all negotiation agendas (new in 0.6.1)
          - **price_range* The maximum number of different prices in any negotiation agenda (new in 0.6.1)

        B. Agent Information:
          - **profile**: Gives the agent profile including its production cost, number
            of production lines, input product index, mean of its delivery
            penalties, mean of its disposal costs, standard deviation of its
            shortfall penalties and standard deviation of its disposal costs.
            See `OneShotProfile` for full description. This information is private
            information and no other agent knows it.
          - **n_lines**: the number of production lines in the factory (private information).
          - **is_first_level**: Is the agent in the first production level (i.e. it is an
            input agent that buys the raw material).
          - **is_last_level**: Is the agent in the last production level (i.e. it is an
            output agent that sells the final product).
          - **is_middle_level**: Is the agent neither a first level nor a last level agent
          - **my_input_product**: The input product to the factory controlled by the agent.
          - **my_output_product**: The output product from the factory controlled by the agent.
          - **level**: The production level which is numerically the same as the input product.
          - **my_suppliers**: A list of IDs for all suppliers to the agent (i.e. agents
            that can sell the input product of the agent).
          - **my_consumers**: A list of IDs for all consumers to the agent (i.e. agents
            that can buy the output product of the agent).
          - **penalties_scale**: The scale at which to calculate disposal cost/delivery
            penalties. "trading" and "catalog" mean trading and
            catalog prices. "unit" means the contract's unit price
            while "none" means that disposal cost/shortfall penalty
            are absolute.
          - **n_input_negotiations**: Number of negotiations with suppliers.
          - **n_output_negotiations**: Number of negotiations with consumers.

    Dynamic World Information:
        Information about the world and the agent that changes over time.

        A. Market Information:
          - **trading_prices**: The trading prices of all products. This information
            is only available if `publish_trading_prices` is
            set in the world.
          - **exogenous_contract_summary**: A list of n_products tuples each giving
            the total quantity and average price of
            exogenous contracts for a product. This
            information is only available if
            `publish_exogenous_summary` is set in
            the world.
          - **is_perishable**: Are all products perishable?

        B. Other Agents' Information:
          - **reports_of_agent**: Gives all past financial reports of a given agent.
            See `FinancialReport` for details.
          - **reports_at_step**: Gives all reports of all agents at a given step.
            See `FinancialReport` for details.

        C. Current Negotiations Information:
          - **current_input_outcome_space**: The current outcome-space for all negotiations to buy
            the input product of the agent. If the agent is at level zero, this will have no issues.
          - **current_output_outcome_space**: The current outcome-space for all negotiations to buy
            the output product of the agent. If the agent
            is at level n_products - 1, this will have no issues.
          - **current_negotiation_details**: Details on all current negotiations separated into "buy"
            and "sell" dictionaries.

          Useful helpers about current negotiations:

          - **current_input_issues**: The current issues for all negotiations to buy
            the input product of the agent. If the agent
            is at level zero, this will be empty.
            This is exactly the same as current_input_outcome_space.issues
          - **current_output_issues**: The current issues for all negotiations to buy
            the output product of the agent. If the agent
            is at level n_products - 1, this will be empty.
            This is exactly the same as current_output_outcome_space.issues
          - **current_buy_nmis**: All NMIs for current buy negotiations.
          - **current_sell_nmis**: All NMIs for current sell negotiations.
          - **current_nmis**: All states for current negotiations.
          - **current_buy_states**: All states for current buy negotiations.
          - **current_sell_states**: All states for current sell negotiations.
          - **current_states**: All states for current negotiations.
          - **current_buy_offers**: All offers for current buy negotiations.
          - **current_sell_offers**: All offers for current sell negotiations.
          - **current_offers**: All offers for current negotiations.
          - **running_buy_nmis**: All NMIs for running buy negotiations.
          - **running_sell_nmis**: All NMIs for running sell negotiations.
          - **running_nmis**: All states for running negotiations.
          - **running_buy_states**: All states for running buy negotiations.
          - **running_sell_states**: All states for running sell negotiations.
          - **running_states**: All states for running negotiations.

        D. Agent Information:
          - **current_exogenous_input_quantity**: The total quantity the agent have
            in its input exogenous contract.
          - **current_exogenous_input_price**: The total price of the agent's
            input exogenous contract.
          - **current_exogenous_output_quantity**: The total quantity the agent have
            in its output exogenous contract.
          - **current_exogenous_output_price**: The total price of the agent's
            output exogenous contract
          - **current_disposal_cost**: The disposal cost per unit item in the current
            step.
          - **current_shortfall_penalty**: The shortfall penalty per unit item in the current
            step.
          - **current_balance**: The current balance of the agent
          - **current_score**: The current score (balance / initial balance) of the agent
          - **current_inventory_input**: The total quantity remaining in the inventory of the input product
          - **current_inventory_output**: The total quantity remaining in the inventory of the output product
          - **current_inventory**: The total quantity remaining in the inventory of the input and output product

        E. Sales and Supplies (quantities) for today:
          - **sales**: Today's sales per customer so far.
          - **supplies**: Today's supplies per supplier so far.
          - **total_sales**: Today's total sales so far.
          - **total_supplies**: Today's total supplies so far.
          - **needed_sales**: Today's needed sales as of now (exogenous input + total supplies - exogenous output - total sales so far).
          - **needed_supplies**: Today's needed supplies  as of now (exogenous output + total sales - exogenous input - total supplies so far).


    Services (All inherited from `negmas.situated.AgentWorldInterface`):
      - **logdebug/loginfo/logwarning/logerror**: Logs to the world log at the given log level.
      - **logdebug_agent/loginf_agnet/...**: Logs to the agent specific log at the given log level.
      - **bb_query**: Queries the bulletin-board.
      - **bb_read**: Read a section of the bulletin-board.

    """

    def __init__(self, world: SCMLBaseWorld, agent: OneShotAgent):
        super().__init__(world, agent)  # type: ignore
        self._world = world
        self.agent = agent
        self._future_sales: dict[int, dict[str, int]] = defaultdict(defaultdict_int)
        self._future_supplies: dict[int, dict[str, int]] = defaultdict(defaultdict_int)
        self._future_sales_cost: dict[int, dict[str, int]] = defaultdict(
            defaultdict_int
        )
        self._future_supplies_cost: dict[int, dict[str, int]] = defaultdict(
            defaultdict_int
        )

    # ================================================================
    # Static World Information (does not change during the simulation)
    # ================================================================

    # Market information
    # ------------------
    @property
    def max_n_lines(self) -> int:
        """Maximum number of lines in the whole system"""
        return self._world._max_n_lines

    @property
    def quantity_range(self) -> int:
        """The maximum cardinality of the quantity issue in all negotiations"""
        return self.max_n_lines

    @property
    def price_range(self) -> int:
        """The maximum cardinality of the quantity issue in all negotiations"""
        return 2

    @property
    def n_products(self) -> int:
        """Returns the number of products in the system"""
        return len(self._world.catalog_prices)

    @property
    def n_competitors(self) -> int:
        """Returns the number of factories/agents in the same production level"""
        return len(self._world.consumers[self.my_input_product]) - 1

    @property
    def n_processes(self) -> int:
        """Returns the number of processes in the system"""
        return self.n_products - 1

    @property
    def all_suppliers(self) -> list[list[str]]:
        """Returns a list of agent IDs for all suppliers for every product"""
        return self._world.suppliers

    @property
    def production_capacities(self) -> list[int]:
        """Returns the total production capacity in the market for each process"""
        return self._world.production_capacity

    @property
    def all_consumers(self) -> list[list[str]]:
        """Returns a list of agent IDs for all consumers for every product"""
        return self._world.consumers

    def is_system(self, aid: str) -> bool:
        """
        Checks whether an agent is a system agent or not

        Args:
            aid: Agent ID
        """
        return is_system_agent(aid)

    def is_bankrupt(self, aid: str | None = None) -> bool:
        """
        Checks whether an agent is a system agent or not

        Args:
            aid: Agent ID
        """
        if not aid:
            aid = self.agent.id
        return self._world.is_bankrupt[aid]

    @property
    def horizon(self) -> int:
        """Horizon for negotiations"""
        return self._world.horizon

    @property
    def catalog_prices(self) -> np.ndarray:
        """Returns the catalog prices of all products"""
        return self._world.catalog_prices

    @property
    def price_multiplier(self) -> float:
        """
        Controls the minimum and maximum prices in the negotiation agendas

        Remarks:
            - The base price is either the catalog price if trading price information
              is not public or the trading price.
            - The minimum unit price in any negotiation agenda is the base price of
              the previous product in the chain ***divided* by the multiplier. If that is
              less than 1, the minimum unit price becomes 1.
            - The maximum unit price in any negotiation agenda is the base price of
              the previous product in the chain ***multiplied* by the multiplier. If that is
              less than 1, the minimum unit price becomes 1.
        """
        return self._world.price_multiplier

    @property
    def is_exogenous_forced(self) -> bool:
        """
        Are exogenous contracts forced in the sense that the agent cannot decide
        not to sign them?
        """
        return self.bb_read("settings", "force_signing") or self.bb_read(  # type: ignore
            "settings", "force_exogenous"
        )

    @property
    def allow_zero_quantity(self) -> bool:
        """
        Does negotiations allow zero quantity?
        """
        return self._world.allow_zero_quantity

    # ================================================================
    # Static Agent Information (does not change during the simulation)
    # ================================================================

    @property
    def profile(self) -> OneShotProfile:
        """Gets the profile (static private information) associated with the agent"""
        return self._world.agent_profiles[self.agent.id]

    @property
    def n_lines(self) -> int:
        """The number of lines in the corresponding factory.
        You can read `state` to get this among other information"""
        return self.profile.n_lines if self.profile else 0

    @property
    def n_input_negotiations(self) -> int:
        """
        Number of negotiations with suppliers at every step
        """
        if self.is_first_level:
            return 0
        return len(self.my_suppliers)

    @property
    def n_output_negotiations(self) -> int:
        """
        Number of negotiations with consumers at every step
        """
        if self.is_last_level:
            return 0
        return len(self.my_consumers)

    @property
    def is_first_level(self):
        """
        Whether this agent is in the first production level
        """
        return self.my_input_product == 0

    @property
    def is_last_level(self):
        """
        Whether this agent is in the last production level
        """
        return self.my_output_product == self.n_products - 1

    @property
    def level(self):
        """The production level which is the index of the process for
        this factory (or the index of its input product)"""
        return self.my_input_product

    @property
    def is_middle_level(self):
        """
        Whether this agent is in neither in the first nor in the last level
        """
        return 0 < self.my_input_product < self.n_products - 2

    @property
    def my_input_product(self) -> int:
        """the product I need to buy"""
        return self.profile.input_product if self.profile else -10

    @property
    def my_output_product(self) -> int:
        """the product I need to sell"""
        return self.profile.output_product if self.profile else -10

    @property
    def my_competitors(self) -> list[str]:
        """Returns the names of all factories in the same level as me"""
        return [
            _
            for _ in self._world.consumers[self.my_input_product]
            if _ != self.agent.id
        ]

    @property
    def my_suppliers(self) -> list[str]:
        """Returns a list of IDs for all of the agent's suppliers
        (agents that can supply the product I need).
        """
        return self.all_suppliers[self.level]

    @property
    def my_consumers(self) -> list[str]:
        """Returns a list of IDs for all the agent's consumers
        (agents that can consume at least one product it may produce).

        """
        return self.all_consumers[self.level + 1]

    @property
    def my_partners(self) -> list[str]:
        """Returns a list of IDs for all of the agent's partners starting with suppliers"""
        return [
            _
            for _ in itertools.chain(self.my_suppliers, self.my_consumers)
            if not self.is_system(_)
        ]

    @property
    def penalties_scale(self) -> Literal["trading", "catalog", "unit", "none"]:
        return self._world.penalties_scale  # type: ignore

    # =========================================================
    # Dynamic Agent Information (changes during the simulation)
    # =========================================================
    @property
    def state(self) -> OneShotState:
        all_agents = [_ for _ in self._world.agents.keys() if self.is_system(_)]

        return OneShotState(
            perishable=self.is_perishable,
            allow_zero_quantity=self.allow_zero_quantity,
            exogenous_input_quantity=self.current_exogenous_input_quantity,
            exogenous_input_price=self.current_exogenous_input_price,
            exogenous_output_quantity=self.current_exogenous_output_quantity,
            exogenous_output_price=self.current_exogenous_output_price,
            disposal_cost=self.current_disposal_cost,
            storage_cost=self.current_storage_cost,
            shortfall_penalty=self.current_shortfall_penalty,
            current_balance=self.current_balance,
            total_sales=self.total_sales,
            total_supplies=self.total_supplies,
            total_future_sales=self.total_future_sales,
            total_future_supplies=self.total_future_supplies,
            n_products=self.n_products,
            n_processes=self.n_processes,
            n_competitors=self.n_competitors,
            all_suppliers=self.all_suppliers,
            all_consumers=self.all_consumers,
            my_partners=self.my_partners,
            production_capacities=self.production_capacities,
            catalog_prices=self.catalog_prices.tolist(),
            price_multiplier=self.price_multiplier,
            is_exogenous_forced=self.is_exogenous_forced,
            current_step=self.current_step,
            n_steps=self.n_steps,
            relative_simulation_time=self.relative_time,
            profile=self.profile,
            n_lines=self.n_lines,
            is_first_level=self.is_first_level,
            is_last_level=self.is_last_level,
            is_middle_level=self.is_middle_level,
            my_input_product=self.my_input_product,
            my_output_product=self.my_output_product,
            level=self.level,
            my_suppliers=self.my_suppliers,
            my_consumers=self.my_consumers,
            penalties_scale=self.penalties_scale,
            n_input_negotiations=self.n_input_negotiations,
            n_output_negotiations=self.n_output_negotiations,
            trading_prices=self.trading_prices.tolist(),
            exogenous_contract_summary=self.exogenous_contract_summary,
            current_input_outcome_space=self.current_input_outcome_space,
            current_output_outcome_space=self.current_output_outcome_space,
            current_negotiation_details=self.current_negotiation_details,
            sales=self.sales,
            supplies=self.supplies,
            needed_sales=self.needed_sales,
            needed_supplies=self.needed_supplies,
            bankrupt_agents=[_ for _ in all_agents if self.is_bankrupt(_)],
            reports_of_agents=dict(
                zip(all_agents, [self.reports_of_agent(_) for _ in all_agents])
            ),
            # running_negotiations=dict(),
        )

    @property
    def current_balance(self):
        return self._world.current_balance(self.agent.id)

    @property
    def current_score(self) -> float:
        """Returns the current score (profit) of the agent"""
        return self._world.scores()[self.agent.id]

    @property
    def current_inventory(self) -> tuple[int, int]:
        """Current input and output inventory quantity"""
        return (
            self._world._inventory_input[self.agent.id],
            self._world._inventory_output[self.agent.id],
        )

    @property
    def current_inventory_input(self) -> int:
        """Current input inventory quantity"""
        return self._world._inventory_input[self.agent.id]

    @property
    def current_inventory_output(self) -> int:
        """Current output inventory quantity"""
        return self._world._inventory_output[self.agent.id]

    @property
    def current_exogenous_input_quantity(self) -> int:
        """
        The exogenous contracts for the input (this step)
        """
        return self._world.exogenous_qin[self.agent.id]

    @property
    def current_exogenous_input_price(self) -> int:
        """
        The exogenous contracts for the input (this step)
        """
        return self._world.exogenous_pin[self.agent.id]

    @property
    def current_exogenous_output_quantity(self) -> int:
        """
        The exogenous contracts for the input (this step)
        """
        return self._world.exogenous_qout[self.agent.id]

    @property
    def current_exogenous_output_price(self) -> int:
        """
        The exogenous contracts for the input (this step)
        """
        return self._world.exogenous_pout[self.agent.id]

    def penalty_multiplier(self, is_input: bool, unit_price: float | None) -> float:
        """
        Returns the penalty multiplier for a contract with the give unit price.

        Remarks:
            - The unit price is only needed if the penalties_scale is unit. For
              all other options (trading, catalog, none), the penalty scale does
              not depend on the unit price.
        """
        if self.penalties_scale.startswith("n"):
            return 1
        if self.penalties_scale.startswith("t"):
            return self.trading_prices[
                self.my_input_product if is_input else self.my_output_product
            ]
        if self.penalties_scale.startswith("c"):
            return self.catalog_prices[
                self.my_input_product if is_input else self.my_output_product
            ]
        if unit_price is None:
            raise ValueError(
                "Must pass unit price to the penalty multiplier if the scale does not start with n, t or c"
            )
        return unit_price

    @property
    def is_perishable(self) -> bool:
        """Are all products perishable (original design of OneShot)"""
        return self._world.perishable

    @property
    def current_disposal_cost(self) -> float:
        """Cost of storing one unit (penalizes buying too much/ selling too little)"""
        return self._world.agent_disposal_cost[self.agent.id][self._world.current_step]

    @property
    def current_storage_cost(self) -> float:
        """Cost of storing one unit (penalizes buying too much/ selling too little)"""
        return self._world.agent_storage_cost[self.agent.id][self._world.current_step]

    @property
    def current_shortfall_penalty(self) -> float:
        """Cost of failure to deliver one unit (penalizes buying too little / selling too much)"""
        return self._world.agent_shortfall_penalty[self.agent.id][
            self._world.current_step
        ]

    # =========================================================
    # Dynamic World Information (changes during the simulation)
    # =========================================================

    # Public Market Condition Information
    # -----------------------------------

    @property
    def trading_prices(self) -> np.ndarray:
        """Returns the current trading prices of all products"""
        return (
            self._world.trading_prices
            if self._world.publish_trading_prices
            else self.catalog_prices
        )

    @property
    def exogenous_contract_summary(self) -> list[tuple[int, int]]:
        """
        The exogenous contracts in the current step for all products

        Returns:
            A list of tuples giving the total quantity and total price of
            all revealed exogenous contracts of all products at the current
            step.
        """
        return (  # type: ignore
            self._world.exogenous_contracts_summary
            if self._world.publish_exogenous_summary
            else None
        )

    # Other agents' information
    # -------------------------

    def reports_of_agent(self, aid: str) -> dict[int, FinancialReport]:
        """Returns a dictionary mapping time-steps to financial reports of
        the given agent"""
        return self.bb_read("reports_agent", aid)  # type: ignore

    def reports_at_step(self, step: int) -> dict[str, FinancialReport]:
        """Returns a dictionary mapping agent ID to its financial report for
        the given time-step"""
        result = self.bb_read("reports_time", str(step))
        if result is not None:
            return result
        steps = sorted(
            int(i)
            for i in self.bb_query("reports_time", None, query_keys=True).keys()  # type: ignore
        )
        for s, prev in zip(steps[1:], steps[:-1]):
            if s > step:
                return self.bb_read("reports_time", prev)  # type: ignore
        return self.bb_read("reports_time", str(steps[-1]))  # type: ignore

    # Negotiation set information
    # ---------------------------

    @property
    def current_input_issues(self) -> list[ContiguousIssue]:
        return self._world._current_issues[self.my_input_product]

    @property
    def current_output_issues(self) -> list[ContiguousIssue]:
        return self._world._current_issues[self.my_output_product]

    @property
    def current_input_outcome_space(self) -> DiscreteCartesianOutcomeSpace:
        return make_os(self._world._current_issues[self.my_input_product])  # type: ignore

    @property
    def current_output_outcome_space(self) -> DiscreteCartesianOutcomeSpace:
        return make_os(self._world._current_issues[self.my_output_product])  # type: ignore

    @property
    def current_negotiation_details(self) -> dict[str, dict[str, NegotiationDetails]]:
        """
        Details of current negotiations separated as two dicts for buying and selling.

        Remarks:
            - current_negotiation_details["buy"] gives details on all negotiations for buying
            - current_negotiation_details["sell"] gives details on all negotiations for selling
        """
        return self._world._agent_negotiations.get(
            self.agent.id, dict(buy=dict(), sell=dict())
        )

    @property
    def current_buy_states(self) -> dict[str, SAOState]:
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
        return self.current_buy_states | self.current_sell_states

    @property
    def current_buy_nmis(self) -> dict[str, SAONMI]:
        """All running buy negotiations as a mapping from partner ID to current negotiation nmi"""
        return {  # type: ignore
            partner: info.nmi
            for partner, info in self.current_negotiation_details["buy"].items()
        }

    @property
    def current_sell_nmis(self) -> dict[str, SAONMI]:
        """All running negotiations as a mapping from partner ID to current negotiation state"""
        return {  # type: ignore
            partner: info.nmi
            for partner, info in self.current_negotiation_details["sell"].items()
        }

    @property
    def current_nmis(self) -> dict[str, SAONMI]:
        """All running negotiations as a mapping from partner ID to current negotiation nmi"""
        return self.current_buy_nmis | self.current_sell_nmis

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
        return {
            partner: info.nmi.state.current_offer  # type: ignore
            for partner, info in self.current_negotiation_details["sell"].items()
            if info.nmi.state.running and info.nmi.state.started
        }

    @property
    def current_offers(self) -> dict[str, Outcome]:
        """All current negotiations as a mapping from partner ID to current offer"""
        return self.current_buy_offers | self.current_sell_offers

    @property
    def running_buy_states(self) -> dict[str, SAOState]:
        """All running buy negotiations as a mapping from partner ID to current negotiation state"""
        return {  # type: ignore
            partner: info.nmi.state
            for partner, info in self.current_negotiation_details["buy"].items()
            if not info.nmi.state.ended
        }

    @property
    def running_sell_states(self) -> dict[str, SAOState]:
        """All running sell negotiations as a mapping from partner ID to current negotiation state"""
        return {  # type: ignore
            partner: info.nmi.state
            for partner, info in self.current_negotiation_details["sell"].items()
            if not info.nmi.state.ended
        }

    @property
    def running_states(self) -> dict[str, SAOState]:
        """All running negotiations as a mapping from partner ID to current negotiation state"""
        return self.running_sell_states | self.running_buy_states

    @property
    def running_sell_nmis(self) -> dict[str, SAONMI]:
        """All running sell negotiations as a mapping from partner ID to current negotiation nmi"""
        return {  # type: ignore
            partner: info.nmi
            for partner, info in self.current_negotiation_details["sell"].items()
            if not info.nmi.state.ended
        }

    @property
    def running_buy_nmis(self) -> dict[str, SAONMI]:
        """All running buy negotiations as a mapping from partner ID to current negotiation nmi"""
        return {  # type: ignore
            partner: info.nmi
            for partner, info in self.current_negotiation_details["buy"].items()
            if not info.nmi.state.ended
        }

    @property
    def running_nmis(self) -> dict[str, SAONMI]:
        """All running negotiations as a mapping from partner ID to current negotiation nmi"""
        return self.running_sell_nmis | self.running_buy_nmis

    # Sales and supplies

    @property
    def sales(self) -> dict[str, int]:
        """Sales (quantity) per customer so far (this day)"""
        return self._future_sales.get(self.current_step, dict())

    @property
    def supplies(self) -> dict[str, int]:
        """Supplies (quantity) per supplier so far (this day)"""
        return self._future_supplies.get(self.current_step, dict())

    @property
    def sales_cost(self) -> dict[str, int]:
        """Sales (total price) per customer so far (this day)"""
        return self._future_sales_cost.get(self.current_step, dict())

    @property
    def supplies_cost(self) -> dict[str, int]:
        """Supplies (total price) per supplier so far (this day)"""
        return self._future_supplies_cost.get(self.current_step, dict())

    @property
    def future_sales(self) -> dict[int, dict[str, int]]:
        """Future sales (quantity) per customer so far (excluding this day)"""
        return {t: v for t, v in self._future_sales.items() if t > self.current_step}

    @property
    def future_supplies(self) -> dict[int, dict[str, int]]:
        """Future supplies (quantity) per supplier so far (excluding this day)"""
        return {t: v for t, v in self._future_supplies.items() if t > self.current_step}

    @property
    def future_sales_cost(self) -> dict[int, dict[str, int]]:
        """Future sales (total price) per customer so far (excluding this day)"""
        return {
            t: v for t, v in self._future_sales_cost.items() if t > self.current_step
        }

    @property
    def future_supplies_cost(self) -> dict[int, dict[str, int]]:
        """Future supplies (total price) per supplier so far (excluding this day)"""
        return {
            t: v for t, v in self._future_supplies_cost.items() if t > self.current_step
        }

    @property
    def total_sales(self) -> int:
        """Total sales so far (this day)"""
        return sum(self.sales.values())

    @property
    def total_supplies(self) -> int:
        """Total supplies so far (this day)"""
        return sum(self.supplies.values())

    @property
    def total_future_sales(self) -> int:
        """Total sales so far (this day)"""
        return sum(sum(_.values()) for _ in self.future_sales.values())

    def total_sales_from(self, start: int) -> int:
        """Total sales starting at start and ending at end (inclusive). Past days are ignored"""
        return sum(
            sum(_.values())
            for i, _ in self._future_sales.items()
            if max(start, self.current_step) <= i <= self.n_steps - 1
        )

    def total_supplies_from(self, start: int) -> int:
        """Total supplies starting at start and ending at end (inclusive). Past days are ignored"""
        return sum(
            sum(_.values())
            for i, _ in self._future_supplies.items()
            if max(start, self.current_step) <= i <= self.n_steps - 1
        )

    def total_sales_between(self, start: int, end: int) -> int:
        """Total sales starting at start and ending at end (inclusive). Past days are ignored"""
        return sum(
            sum(_.values())
            for i, _ in self._future_sales.items()
            if max(start, self.current_step) <= i <= end
        )

    def total_supplies_between(self, start: int, end: int) -> int:
        """Total supplies starting at start and ending at end (inclusive). Past days are ignored"""
        return sum(
            sum(_.values())
            for i, _ in self._future_supplies.items()
            if max(start, self.current_step) <= i <= end
        )

    def total_supplies_until(self, step: int) -> int:
        """Total supplies starting today until the given step (inclusive). Past days are ignored"""
        if step < self.current_step:
            return 0
        return self.total_supplies_between(self.current_step, step)

    def total_sales_until(self, step: int) -> int:
        """Total sales starting today until the given step (inclusive). Past days are ignored"""
        if step < self.current_step:
            return 0
        return self.total_sales_between(self.current_step, step)

    def total_sales_at(self, step: int) -> int:
        """Total sales already signed at a future step"""
        if step < self.current_step:
            return 0
        return sum(self._future_sales.get(step, dict()).values())

    def total_supplies_at(self, step: int) -> int:
        """Total supplies already signed at a future step"""
        if step < self.current_step:
            return 0
        return sum(self._future_supplies.get(step, dict()).values())

    @property
    def total_future_supplies(self) -> int:
        """Total supplies so far (this day)"""
        return sum(sum(_.values()) for _ in self.future_supplies.values())

    @property
    def needed_sales(self) -> int:
        """Sales that need to be secured (exogenous input + total supplies - exogenous output - total sales so far)"""
        if self.is_last_level:
            return 0
        x = (
            self.current_exogenous_input_quantity
            + self.current_inventory_input
            + self.total_supplies
            - self.total_sales
            - self.current_exogenous_output_quantity
            - self.current_inventory_output
        )
        return min(self.n_lines, x) if self.is_perishable else x

    @property
    def needed_supplies(self) -> int:
        """Supplies that need to be secured (exogenous output + total sales - exogenous input - total supplies so far)"""
        if self.is_first_level:
            return 0
        x = (
            self.current_exogenous_output_quantity
            + self.current_inventory_output
            + self.total_sales
            - self.total_supplies
            - self.current_exogenous_input_quantity
            - self.current_inventory_input
        )
        return min(self.n_lines, x) if self.is_perishable else x

    # helper operations (sales and supplies) -- you do not need to call these.
    def _register_sale(
        self, customer: str, quantity: int, unit_price: int, step: int
    ) -> None:
        # assert (
        #     quantity == 0
        #     or not self.is_perishable
        #     or step != self.current_step
        #     or self._future_sales[step][customer] == 0
        #     or (self._world.one_time_per_negotiation and self._world.horizon)
        # ), f"{self.agent.id} Cannot have more than one sale to {customer} ({self.sales[customer]=}, {quantity=})"
        self._future_sales[step][customer] += quantity
        self._future_sales_cost[step][customer] += quantity * unit_price

    def _register_supply(
        self, supplier: str, quantity: int, unit_price: int, step: int
    ) -> None:
        # assert (
        #     quantity == 0
        #     or not self.is_perishable
        #     or step != self.current_step
        #     or self._future_supplies[step][supplier] == 0
        #     or (self._world.one_time_per_negotiation and self._world.horizon)
        # ), f"{self.agent.id} Cannot have more than one supply to {supplier} ({self.supplies[supplier]=}, {quantity=})"
        self._future_supplies[step][supplier] += quantity
        self._future_supplies_cost[step][supplier] += quantity * unit_price

    def _reset_sales_and_supplies(self) -> None:
        for d in (
            self._future_supplies,
            self._future_supplies_cost,
            self._future_sales,
            self._future_sales_cost,
        ):
            to_remove = []
            for t in d.keys():
                if t < self.current_step:
                    to_remove.append(t)
            for t in to_remove:
                del d[t]
