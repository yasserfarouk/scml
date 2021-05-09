"""
Implements the Agent-World-Interface for SCML2020 worlds
"""
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from negmas import AgentWorldInterface
from negmas import Issue
from negmas import Negotiator
from negmas import PassThroughSAONegotiator
from negmas import SAOController
from negmas import SAONegotiator
from .common import ANY_LINE
from .common import ANY_STEP
from .common import is_system_agent
from .common import FactoryState, FinancialReport, FactoryProfile

__all__ = [
    "AWI",
]


class AWI(AgentWorldInterface):
    """
    The Agent SCML2020World Interface for SCML2020 world.

    This class contains all the methods needed to access the simulation to
    extract information which are divided into 5 groups:

    Static World Information:
        Information about the world and the agent that does not change over
        time. These include:

        A. Market Information:
          - *n_products*: Number of products in the production chain.
          - *n_processes*: Number of processes in the production chain.
          - *n_competitors*: Number of other factories on the same production level.
          - *all_suppliers*: A list of all suppliers by product.
          - *all_consumers*: A list of all consumers by product.
          - *catalog_prices*: A list of the catalog prices (by product).
          - *inputs*: Inputs to every manufacturing process.
          - *outputs*: Outputs to every manufacturing process.
          - *is_system*: Is the given system ID corresponding to a system agent?
          - *current_step*: Current simulation step (inherited from `negmas.situated.AgentWorldInterface` ).
          - *n_steps*: Number of simulation steps (inherited from `negmas.situated.AgentWorldInterface` ).
          - *relative_time*: fraction of the simulation completed (inherited from `negmas.situated.AgentWorldInterface`).
          - *settings*: The system settings (inherited from `negmas.situated.AgentWorldInterface` ).

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
          - *my_input_products*: All input products of a factory controlled by the agent.
            Currently, it is always a list of one item. For future compatibility.
          - *my_output_products*: All output products of a factory controlled by the agent.
            Currently, it is always a list of one item. For future compatibility.
          - *available_for_production*: Returns the line-step slots available for production.
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
          - *state*: The full state of the agent ( `FactoryState` ).
          - *current_balance*: The current balance of the agent
          - *current_inventory*: The current inventory of the agent (quantity per product)

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

    Actions:
        A. Negotiation Control:
          - *request_negotiations*: Requests a set of negotiations controlled by a 
            single controller.
          - *request_negotiation*: Requests a negotiation controlled by a single 
            negotiator.

        B. Production Control:
          - *schedule_production*: Schedules production using one of the predefined 
            scheduling strategies.
          - *order_production*: Orders production directly for the current step.
          - *set_commands*: Sets production commands directly on the factory.
          - *cancel_production*: Cancels a scheduled production command.

    Services (All inherited from `negmas.situated.AgentWorldInterface`):
      - *logdebug/loginfo/logwarning/logerror*: Logs to the world log at the given log level.
      - *logdebug_agent/loginf_agnet/...*: Logs to the agent specific log at the given log level.
      - *bb_query*: Queries the bulletin-board.
      - *bb_read*: Read a section of the bulletin-board.

    """

    # --------
    # Actions
    # --------

    def request_negotiations(
        self,
        is_buy: bool,
        product: int,
        quantity: Union[int, Tuple[int, int]],
        unit_price: Union[int, Tuple[int, int]],
        time: Union[int, Tuple[int, int]],
        controller: Optional[SAOController] = None,
        negotiators: List[Negotiator] = None,
        partners: List[str] = None,
        extra: Dict[str, Any] = None,
        copy_partner_id=True,
    ) -> bool:
        """
        Requests a negotiation

        Args:

            is_buy: If True the negotiation is about buying otherwise selling.
            product: The product to negotiate about
            quantity: The minimum and maximum quantities. Passing a single value q is equivalent to passing (q,q)
            unit_price: The minimum and maximum unit prices. Passing a single value u is equivalent to passing (u,u)
            time: The minimum and maximum delivery step. Passing a single value t is equivalent to passing (t,t)
            controller: The controller to manage the complete set of negotiations
            negotiators: An optional list of negotiators to use for negotiating with the given partners (in the same
                         order).
            partners: ID of all the partners to negotiate with.
            extra: Extra information accessible through the negotiation annotation to the caller
            copy_partner_id: If true, the partner ID will be copied to the negotiator ID


        Returns:

            `True` if the partner accepted and the negotiation is ready to start

        Remarks:

            - You can either use controller or negotiators. One of them must be None.
            - All negotiations will use the following issues **in order**: quantity, time, unit_price
            - Negotiations with bankrupt agents or on invalid products (see next point) will be automatically rejected
            - Valid products for a factory are the following (any other products are not valid):
                1. Buying an input product (i.e. product $\\in$ `my_input_products` ) and an output product if the world
                   settings allows it (see `allow_buying_output`)
                1. Selling an output product (i.e. product $\\in$ `my_output_products` ) and an input product if the
                   world settings allows it (see `allow_selling_input`)


        """
        if controller is not None and negotiators is not None:
            raise ValueError(
                "You cannot pass both controller and negotiators to request_negotiations"
            )
        if controller is None and negotiators is None:
            raise ValueError(
                "You MUST pass either controller or negotiators to request_negotiations"
            )
        if extra is None:
            extra = dict()
        buyable, sellable = self.my_input_products, self.my_output_products
        if self._world.allow_selling_input:
            sellable = set(sellable + self.my_input_products)
        if self._world.allow_buying_output:
            buyable = set(buyable + self.my_output_products)
        if (product not in buyable and is_buy) or (
            product not in sellable and not is_buy
        ):
            self._world.logwarning(
                f"{self.agent.name} requested ({'buying' if is_buy else 'selling'}) on {product}. This is not allowed"
            )
            return False
        if partners is None:
            partners = (
                self.all_suppliers[product] if is_buy else self.all_consumers[product]
            )
        partners = [_ for _ in partners if not is_system_agent(_)]
        if not partners:
            return False

        if negotiators is None:
            negotiators = [
                PassThroughSAONegotiator(
                    name=_ if copy_partner_id else None,
                    id=_ if copy_partner_id else None,
                )
                for _ in partners
            ]
        results = [
            self.request_negotiation(
                is_buy, product, quantity, unit_price, time, partner, negotiator, extra
            )
            if not self._world.a2f[partner].is_bankrupt
            and self._world.can_negotiate(partner, self.agent.id)
            else False
            for partner, negotiator in zip(partners, negotiators)
        ]
        # for r, n in zip(results, negotiators):
        #     if not r:
        #         controller.kill_negotiator(n.id, force=True)
        for p, neg, r in zip(partners, negotiators, results):
            if not r:
                continue
            controller.add_negotiator(neg)
            self._world._registered_negs[tuple(sorted([p, self.agent.id]))] += 1
        return any(results)

    def request_negotiation(
        self,
        is_buy: bool,
        product: int,
        quantity: Union[int, Tuple[int, int]],
        unit_price: Union[int, Tuple[int, int]],
        time: Union[int, Tuple[int, int]],
        partner: str,
        negotiator: SAONegotiator,
        extra: Dict[str, Any] = None,
    ) -> bool:
        """
        Requests a negotiation

        Args:

            is_buy: If True the negotiation is about buying otherwise selling.
            product: The product to negotiate about
            quantity: The minimum and maximum quantities. Passing a single value q is equivalent to passing (q,q)
            unit_price: The minimum and maximum unit prices. Passing a single value u is equivalent to passing (u,u)
            time: The minimum and maximum delivery step. Passing a single value t is equivalent to passing (t,t)
            partner: ID of the partner to negotiate with.
            negotiator: The negotiator to use for this negotiation (if the partner accepted to negotiate)
            extra: Extra information accessible through the negotiation annotation to the caller

        Returns:

            `True` if the partner accepted and the negotiation is ready to start

        Remarks:

            - All negotiations will use the following issues **in order**: quantity, time, unit_price
            - Negotiations with bankrupt agents or on invalid products (see next point) will be automatically rejected
            - Valid products for a factory are the following (any other products are not valid):
                1. Buying an input product (i.e. product $\\in$ `my_input_products` ) and an output product if the world
                   settings allows it (see `allow_buying_output`)
                1. Selling an output product (i.e. product $\\in$ `my_output_products` ) and an input product if the
                   world settings allows it (see `allow_selling_input`)


        """
        if self._world.a2f[partner].is_bankrupt or not self._world.can_negotiate(
            partner, self.agent.id
        ):
            return False

        if extra is None:
            extra = dict()
        buyable, sellable = self.my_input_products, self.my_output_products
        if self._world.allow_selling_input:
            sellable = set(sellable + self.my_input_products)
        if self._world.allow_buying_output:
            buyable = set(buyable + self.my_output_products)
        if (product not in buyable and is_buy) or (
            product not in sellable and not is_buy
        ):
            self._world.logwarning(
                f"{self.agent.name} requested ({'buying' if is_buy else 'selling'}) on {product}. This is not allowed"
            )
            return False

        def values(x: Union[int, Tuple[int, int]]):
            if not isinstance(x, Iterable):
                return int(x), int(x)
            return int(x[0]), int(x[1])

        self._world.logdebug(
            f"{self.agent.name} requested to {'buy' if is_buy else 'sell'} {product} to {partner}"
            f" q: {quantity}, u: {unit_price}, t: {time}"
        )

        annotation = {
            "product": product,
            "is_buy": is_buy,
            "buyer": self.agent.id if is_buy else partner,
            "seller": partner if is_buy else self.agent.id,
            "caller": self.agent.id,
        }
        issues = [
            Issue(values(quantity), name="quantity", value_type=int),
            Issue(values(time), name="time", value_type=int),
            Issue(values(unit_price), name="unit_price", value_type=int),
        ]
        partners = [self.agent.id, partner]
        extra["negotiator_id"] = negotiator.id
        req_id = self.agent.create_negotiation_request(
            issues=issues,
            partners=partners,
            negotiator=negotiator,
            annotation=annotation,
            extra=dict(**extra),
        )
        result = self.request_negotiation_about(
            issues=issues, partners=partners, req_id=req_id, annotation=annotation
        )
        if result:
            self._world._registered_negs[tuple(sorted([partner, self.agent.id]))] += 1
        return result

    def schedule_production(
        self,
        process: int,
        repeats: int,
        step: Union[int, Tuple[int, int]] = ANY_STEP,
        line: int = ANY_LINE,
        override: bool = True,
        method: str = "latest",
        partial_ok: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Orders the factory to run the given process at the given line at the given step

        Args:

            process: The process to run
            repeats: How many times to repeat the process
            step: The simulation step or a range of steps. The special value ANY_STEP gives the factory the freedom to
                  schedule production at any step in the present or future.
            line: The production line. The special value ANY_LINE gives the factory the freedom to use any line
            override: Whether to override existing production commands or not
            method: When to schedule the command if step was set to a range. Options are latest, earliest
            partial_ok: If true, allows partial scheduling

        Returns:
            Tuple[int, int] giving the steps and lines at which production is scheduled.

        Remarks:

            - The step cannot be in the past. Production can only be ordered for current and future steps
            - ordering production of process -1 is equivalent of `cancel_production` only if both step and line are
              given
        """
        return self._world.a2f[self.agent.id].schedule_production(
            process, repeats, step, line, override, method, partial_ok
        )

    def order_production(
        self, process: int, steps: np.ndarray, lines: np.ndarray
    ) -> None:
        """
        Orders production of the given process

        Args:
            process: The process to run
            steps: The time steps to run the process at as an np.ndarray
            lines: The corresponding lines to run the process at

        Remarks:

            - len(steps) must equal len(lines)
            - No checks are done in this function. It is expected to be used after calling `available_for_production`
        """
        return self._world.a2f[self.agent.id].order_production(process, steps, lines)

    def available_for_production(
        self,
        repeats: int,
        step: Union[int, Tuple[int, int]] = ANY_STEP,
        line: int = ANY_LINE,
        override: bool = True,
        method: str = "latest",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Finds available times and lines for scheduling production.

        Args:

            repeats: How many times to repeat the process
            step: The simulation step or a range of steps. The special value ANY_STEP gives the factory the freedom to
                  schedule production at any step in the present or future.
            line: The production line. The special value ANY_LINE gives the factory the freedom to use any line
            override: Whether to override any existing commands at that line at that time.
            method: When to schedule the command if step was set to a range. Options are latest, earliest, all

        Returns:

            Tuple[np.ndarray, np.ndarray] The steps and lines at which production is scheduled.

        Remarks:

            - You cannot order production in the past or in the current step
            - Ordering production, will automatically update inventory and balance for all simulation steps assuming
              that this production will be carried out. At the indicated `step` if production was not possible (due
              to insufficient funds or insufficient inventory of the input product), the predictions for the future
              will be corrected.

        """
        return self._world.a2f[self.agent.id].available_for_production(
            repeats, step, line, override, method
        )

    def set_commands(self, commands: np.ndarray, step: int = -1) -> None:
        """
        Sets the production commands for all lines in the given step

        Args:

            commands: n_lines vector of commands. A command is either a process number to run or `NO_COMMAND` to keep
                      the line idle
            step: The step to set the commands at. If < 0, it means current step
        """
        if step < 0:
            step = self._world.current_step
        self._world.a2f[self.agent.id].commands[step, :] = commands

    def cancel_production(self, step: int, line: int) -> bool:
        """
        Cancels any production commands on that line at this step

        Args:
            step: The step to cancel production at (must be in the future).
            line: The production line

        Returns:

            success/failure

        Remarks:

            - The step cannot be in the past or the current step. Cancellation can only be ordered for future steps
        """
        return self._world.a2f[self.agent.id].cancel_production(step, line)

    # ---------------------
    # Information Gathering
    # ---------------------

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

    @property
    def state(self) -> FactoryState:
        """Receives the factory state"""
        return self._world.a2f[self.agent.id].state
    
    @property
    def current_balance(self):
        """Current balance of the agent"""
        return self.state.balance

    @property
    def current_inventory(self):
        """Current inventory of the agent"""
        return self.state.inventory

    def reports_of_agent(self, aid: str) -> Dict[int, FinancialReport]:
        """Returns a dictionary mapping time-steps to financial reports of the given agent"""
        return self.bb_read("reports_agent", aid)

    def reports_at_step(self, step: int) -> Dict[str, FinancialReport]:
        """Returns a dictionary mapping agent ID to its financial report for the given time-step"""
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

    @property
    def profile(self) -> FactoryProfile:
        """Gets the profile (static private information) associated with the agent"""
        profile = self._world.a2f[self.agent.id].profile
        return FactoryProfile(profile.costs)

    @property
    def all_suppliers(self) -> List[List[str]]:
        """Returns a list of agent IDs for all suppliers for every product"""
        return self._world.suppliers

    @property
    def all_consumers(self) -> List[List[str]]:
        """Returns a list of agent IDs for all consumers for every product"""
        return self._world.consumers

    @property
    def catalog_prices(self) -> np.ndarray:
        """Returns the catalog prices of all products"""
        return self._world.catalog_prices

    @property
    def inputs(self) -> np.ndarray:
        """Returns the number of inputs to every production process"""
        return self._world.process_inputs

    @property
    def outputs(self) -> np.ndarray:
        """Returns the number of outputs to every production process"""
        return self._world.process_outputs

    @property
    def n_products(self) -> int:
        """Returns the number of products in the system"""
        return len(self._world.catalog_prices)

    @property
    def n_processes(self) -> int:
        """Returns the number of processes in the system"""
        return self.n_products - 1

    @property
    def n_competitors(self) -> int:
        """Returns the number of factories/agents in the same production level"""
        return len(self._world.consumers[self.my_output_product]) - 1

    @property
    def my_input_product(self) -> int:
        """Returns a list of products that are inputs to at least one process the agent can run"""
        products = self._world.agent_inputs.get(self.agent.id, np.empty(0, dtype=int))
        if len(products) < 1:
            return -1
        return products[0]

    @property
    def my_output_product(self) -> int:
        """Returns a list of products that are outputs to at least one process the agent can run"""
        products = self._world.agent_outputs.get(self.agent.id, np.empty(0, dtype=int))
        if len(products) < 1:
            return self.n_products
        return products[0]

    @property
    def my_input_products(self) -> np.ndarray:
        """Returns a list of products that are inputs to at least one process the agent can run"""
        return self._world.agent_inputs.get(self.agent.id, np.empty(0, dtype=int))

    @property
    def my_output_products(self) -> np.ndarray:
        """Returns a list of products that are outputs to at least one process the agent can run"""
        return self._world.agent_outputs.get(self.agent.id, np.empty(0, dtype=int))

    @property
    def my_suppliers(self) -> List[str]:
        """Returns a list of IDs for all of the agent's suppliers (agents that can supply at least one product it may
        need).

        Remarks:

            - If the agent have multiple input products, suppliers of a specific product $p$ can be found using:
              **self.all_suppliers[p]**.
        """
        return self._world.agent_suppliers.get(self.agent.id, [])

    @property
    def my_consumers(self) -> List[str]:
        """Returns a list of IDs for all the agent's consumers (agents that can consume at least one product it may
        produce).

        Remarks:

            - If the agent have multiple output products, consumers of a specific product $p$ can be found using:
              **self.all_consumers[p]**.
        """
        return self._world.agent_consumers.get(self.agent.id, [])

    @property
    def n_lines(self) -> int:
        """The number of lines in the corresponding factory. You can read `state` to get this among other information"""
        return self.state.n_lines

    @property
    def n_products(self) -> int:
        """Number of products in the world"""
        return self.state.n_products

    @property
    def n_processes(self) -> int:
        """Number of processes in the world"""
        return self.state.n_processes

    def is_system(self, aid: str) -> bool:
        """
        Checks whether an agent is a system agent or not

        Args:
            aid: Agent ID
        """
        return is_system_agent(aid)
