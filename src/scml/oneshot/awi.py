"""
Implements the one shot version of the Agent-World Interface.

"""
from typing import Any, Dict, List, Tuple

import numpy as np
from negmas import AgentWorldInterface
from negmas.outcomes import Issue

from ..scml2020 import FinancialReport, is_system_agent

from .common import OneShotProfile, OneShotState

__all__ = ["OneShotAWI"]


class OneShotAWI(AgentWorldInterface):
    """
    The agent world interface for the one-shot game.

    This class contains all the methods needed to access the simulation to
    extract information which are divided into 4 groups:

    Static World Information:
        Information about the world and the agent that does not change over
        time. These include:

        A. Market Information:
          - *n_products*: Number of products in the production chain.
          - *n_processes*: Number of processes in the production chain.
          - *n_competitors*: Number of other factories on the same production level.
          - *all_suppliers*: A list of all suppliers by product.
          - *all_consumers*: A list of all consumers by product.
          - *is_system*: Is the given system ID corresponding to a system agent?
          - *catalog_prices*: A list of the catalog prices (by product).
          - *price_multiplier*: The multiplier multiplied by the trading/catalog price
            when the negotiation agendas are created to decide the maximum and lower quantities.
          - *is_exogenous_forced*: Are exogenous contracts always forced or can the
            agent decide not to sign them.
          - *current_step*: Current simulation step (inherited from `negmas.situated.AgentWorldInterface` ).
          - *n_steps*: Number of simulation steps (inherited from `negmas.situated.AgentWorldInterface` ).
          - *relative_time*: fraction of the simulation completed (inherited from `negmas.situated.AgentWorldInterface`).
          - *state*: The full state of the agent ( `OneShotState` ).
          - *settings* The system settings (inherited from `negmas.situated.AgentWorldInterface` ).

        B. Agent Information:
          - *profile*: Gives the agent profile including its production cost, number
            of production lines, input product index, mean of its delivery
            penalties, mean of its disposal costs, standard deviation of its
            shortfall penalties and standard deviation of its disposal costs.
            See `OneShotProfile` for full description. This information is private
            information and no other agent knows it.
          - *n_lines*: the number of production lines in the factory (private information).
          - *is_first_level*: Is the agent in the first production level (i.e. it is an
            input agent that buys the raw material).
          - *is_last_level*: Is the agent in the last production level (i.e. it is an
            output agent that sells the final product).
          - *is_middle_level*: Is the agent neither a first level nor a last level agent
          - *my_input_product*: The input product to the factory controlled by the agent.
          - *my_output_product*: The output product from the factory controlled by the agent.
          - *level*: The production level which is numerically the same as the input product.
          - *my_suppliers*: A list of IDs for all suppliers to the agent (i.e. agents
            that can sell the input product of the agent).
          - *my_consumers*: A list of IDs for all consumers to the agent (i.e. agents
            that can buy the output product of the agent).
          - *penalties_scale*: The scale at which to calculate disposal cost/delivery
            penalties. "trading" and "catalog" mean trading and
            catalog prices. "unit" means the contract's unit price
            while "none" means that disposal cost/shortfall penalty
            are absolute.
          - *n_input_negotiations*: Number of negotiations with suppliers.
          - *n_output_negotiations*: Number of negotiations with consumers.

    Dynamic World Information:
        Information about the world and the agent that changes over time.

        A. Market Information:
          - *trading_prices*: The trading prices of all products. This information
            is only available if `publish_trading_prices` is
            set in the world.
          - *exogenous_contract_summary*: A list of n_products tuples each giving
            the total quantity and average price of
            exogenous contracts for a product. This
            information is only available if
            `publish_exogenous_summary` is set in
            the world.

        B. Other Agents' Information:
          - *reports_of_agent*: Gives all past financial reports of a given agent.
            See `FinancialReport` for details.
          - *reports_at_step*: Gives all reports of all agents at a given step.
            See `FinancialReport` for details.

        C. Current Negotiations Information:
          - *current_input_issues*: The current issues for all negotiations to buy
            the input product of the agent. If the agent
            is at level zero, this will be empty.
          - *current_output_issues*: The current issues for all negotiations to buy
            the output product of the agent. If the agent
            is at level n_products - 1, this will be empty.

        D. Agent Information:
          - *current_exogenous_input_quantity*: The total quantity the agent have
            in its input exogenous contract.
          - *current_exogenous_input_price*: The total price of the agent's
            input exogenous contract.
          - *current_exogenous_output_quantity*: The total quantity the agent have
            in its output exogenous contract.
          - *current_exogenous_output_price*: The total price of the agent's
            output exogenous contract.
          - *current_disposal_cost*: The disposal cost per unit item in the current
            step.
          - *current_shortfall_penalty*: The shortfall penalty per unit item in the current
            step.
          - *current_balance*: The current balance of the agent

    Services (All inherited from `negmas.situated.AgentWorldInterface`):
      - *logdebug/loginfo/logwarning/logerror*: Logs to the world log at the given log level.
      - *logdebug_agent/loginf_agnet/...*: Logs to the agent specific log at the given log level.
      - *bb_query*: Queries the bulletin-board.
      - *bb_read*: Read a section of the bulletin-board.

    """

    # ================================================================
    # Static World Information (does not change during the simulation)
    # ================================================================

    # Market information
    # ------------------

    @property
    def n_products(self) -> int:
        """Returns the number of products in the system"""
        return len(self._world.catalog_prices)

    @property
    def n_competitors(self) -> int:
        """Returns the number of factories/agents in the same production level"""
        return len(self._world.consumers[self.my_output_product]) - 1

    @property
    def n_processes(self) -> int:
        """Returns the number of processes in the system"""
        return self.n_products - 1

    @property
    def all_suppliers(self) -> List[List[str]]:
        """Returns a list of agent IDs for all suppliers for every product"""
        return self._world.suppliers

    @property
    def all_consumers(self) -> List[List[str]]:
        """Returns a list of agent IDs for all consumers for every product"""
        return self._world.consumers

    def is_system(self, aid: str) -> bool:
        """
        Checks whether an agent is a system agent or not

        Args:
            aid: Agent ID
        """
        return is_system_agent(aid)

    @property
    def catalog_prices(self) -> np.ndarray:
        """Returns the catalog prices of all products"""
        return self._world.catalog_prices

    @property
    def price_multiplier(self):
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
    def is_exogenous_forced(self):
        """
        Are exogenous contracts forced in the sense that the agent cannot decide
        not to sign them?
        """
        return self.bb_read("settings", "force_signing") or self.bb_read(
            "settings", "force_exogenous"
        )

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
    def is_middle_level(self):
        """
        Whether this agent is in neither in the first nor in the last level
        """
        return 0 < self.my_input_product < self.n_products - 2

    @property
    def my_input_product(self) -> int:
        """the product I need to buy"""
        return self.profile.input_product if self.profile else -10

    level = my_input_product
    """The production level of the agent"""

    @property
    def my_output_product(self) -> int:
        """the product I need to sell"""
        return self.profile.output_product if self.profile else -10

    @property
    def my_suppliers(self) -> List[str]:
        """Returns a list of IDs for all of the agent's suppliers
        (agents that can supply the product I need).
        """
        return self.all_suppliers[self.level]

    @property
    def my_consumers(self) -> List[str]:
        """Returns a list of IDs for all the agent's consumers
        (agents that can consume at least one product it may produce).

        """
        return self.all_consumers[self.level + 1]

    @property
    def penalties_scale(self) -> str:
        return self._world.penalties_scale

    # =========================================================
    # Dynamic Agent Information (changes during the simulation)
    # =========================================================

    def state(self) -> Any:
        return OneShotState(
            exogenous_input_quantity=self.current_exogenous_input_quantity,
            exogenous_input_price=self.current_exogenous_input_price,
            exogenous_output_quantity=self.current_exogenous_output_quantity,
            exogenous_output_price=self.current_exogenous_output_price,
            disposal_cost=self.current_disposal_cost,
            shortfall_penalty=self.current_shortfall_penalty,
            current_balance=self.current_balance,
        )

    @property
    def current_balance(self):
        return self._world.current_balance(self.agent.id)

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

    def penalty_multiplier(self, is_input: bool, unit_price: float) -> float:
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
        return unit_price

    @property
    def current_disposal_cost(self) -> float:
        """Cost of storing one unit (penalizes buying too much/ selling too little)"""
        return self._world.agent_disposal_cost[self.agent.id][self._world.current_step]

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
            self._world.trading_prices if self._world.publish_trading_prices else None
        )

    @property
    def exogenous_contract_summary(self) -> List[Tuple[int, int]]:
        """
        The exogenous contracts in the current step for all products

        Returns:
            A list of tuples giving the total quantity and total price of
            all revealed exogenous contracts of all products at the current
            step.
        """
        return (
            self._world.exogenous_contracts_summary
            if self._world.publish_exogenous_summary
            else None
        )

    # Other agents' information
    # -------------------------

    def reports_of_agent(self, aid: str) -> Dict[int, FinancialReport]:
        """Returns a dictionary mapping time-steps to financial reports of
        the given agent"""
        return self.bb_read("reports_agent", aid)

    def reports_at_step(self, step: int) -> Dict[str, FinancialReport]:
        """Returns a dictionary mapping agent ID to its financial report for
        the given time-step"""
        result = self.bb_read("reports_time", str(step))
        if result is not None:
            return result
        steps = sorted(
            [
                int(i)
                for i in self.bb_query("reports_time", None, query_keys=True).keys()
            ]
        )
        for (s, prev) in zip(steps[1:], steps[:-1]):
            if s > step:
                return self.bb_read("reports_time", prev)
        return self.bb_read("reports_time", str(steps[-1]))

    # Negotiation set information
    # ---------------------------

    @property
    def current_input_issues(self) -> List[Issue]:
        return self._world._current_issues[self.my_input_product]

    @property
    def current_output_issues(self) -> List[Issue]:
        return self._world._current_issues[self.my_output_product]
