"""Implements the world class for the SCML2020 world """
import copy
import functools
import itertools
import logging
import math
import numbers
import random
from abc import abstractmethod
from collections import defaultdict, namedtuple
import sys
from dataclasses import dataclass, field
from typing import (
    Optional,
    Dict,
    Any,
    Union,
    Tuple,
    Callable,
    List,
    Set,
    Collection,
    Type,
    Iterable,
)
import numpy as np

from negmas import (
    Contract,
    Action,
    Breach,
    AgentWorldInterface,
    Agent,
    RenegotiationRequest,
    Negotiator,
    AgentMechanismInterface,
    MechanismState,
    Issue,
    Entity,
    SAONegotiator,
    SAOController,
    PassThroughSAONegotiator,
)
from negmas.helpers import instantiate, unique_name
from negmas.situated import World, TimeInAgreementMixin, BreachProcessing

__all__ = [
    "FactoryState",
    "SCML2020Agent",
    "SCML2020AWI",
    "SCML2020World",
    "FinancialReport",
    "FactoryProfile",
    "INVALID_COST",
    "NO_COMMAND",
]

INVALID_COST = sys.maxsize
"""A constant indicating an invalid cost for lines incapable of running some process"""
NO_COMMAND = -1
"""A constant indicating no command is scheduled on a factory line"""

@dataclass
class FinancialReport:
    """A report published periodically by the system showing the financial standing of an agent"""
    __slots__ = ["agent_id", "step", "cash", "assets", "breach_prob", "breach_level", "is_bankrupt", "agent_name"]
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
        return f"{self.agent_name} @ {self.step} {bankrupt}: Cash: {self.cash}, Assets: {self.assets}, " \
               f"breach_prob: {self.breach_prob}, breach_level: {self.breach_level}"


@dataclass
class FactoryProfile:
    """Defines all private information of a factory"""
    __slots__ = [
        "process", "cost", "guaranteed_sales", "guaranteed_supplies"
        , "guaranteed_sale_prices", "guaranteed_supply_prices"

    ]
    process: int
    """The single process executable by the factory"""
    cost: int
    """The cost of production"""
    guaranteed_sales: np.ndarray
    """A n_steps  vector giving guaranteed sales of output product"""
    guaranteed_supplies: np.ndarray
    """A n_steps vector giving guaranteed supplies of input product"""
    guaranteed_sale_prices: np.ndarray
    """A n_steps vector giving guaranteed unit prices for the `guaranteed_sales`"""
    guaranteed_supply_prices: np.ndarray
    """A n_steps vector giving guaranteed unit prices for the `guaranteed_supplies` """


@dataclass
class Failure:
    """A production failure"""
    __slots__ = ["is_inventory", "line", "step", "process"]
    is_inventory: bool
    """True if the cause of failure was insufficient inventory. If False, the cause was insufficient funds. Note that
    if both conditions were true, only insufficient funds (is_inventory=False) will be reported."""
    line: int
    """The line at which the failure happened"""
    step: int
    """The step at which the failure happened"""


@dataclass
class FactoryState:
    __slots__ = ["input_amount", "output_amount", "balance", "input_change", "output_change", "balance_change", "commands"]
    input_amount: int
    """Amount of input product in storage"""
    output_amount: int
    """Amount of input product in storage"""
    balance: int
    """Current balance in the wallet"""
    input_change: int
    """Changes in the inventory of the input"""
    output_change: int
    """Changes in the inventory of the output"""
    balance_change: int
    """Change in the balance in the last step"""
    commands: np.ndarray
    """n_steps * n_lines array giving the process scheduled on each line at every step for the 
    whole simulation"""


class Factories:
    """All simulated factories"""

    def __init__(
        self,
        processes: np.ndarray,
        costs: np.ndarray,
        n_steps: int,
        n_lines: int,
        initial_balance: int,
        gsales: np.ndarray,
        gsupplies: np.ndarray,
        gsale_prices: np.ndarray,
        gsupply_prices: np.ndarray,
        inputs: np.ndarray,
        min_balance: int,
    ):
        assert len(processes) == len(costs)
        n_processes = len(inputs)
        n_products = n_processes + 1

        self.gsales, self.gsale_prices = gsales, gsale_prices
        """An n_factories * n_steps array  specifying guaranteed sales"""

        self.gsupplies, self.gsupply_prices = gsupplies, gsupply_prices
        """An n_factories * n_steps array  specifying guaranteed supplies"""

        self.n_factories = n_factories = len(processes)
        """Number of factories"""

        self.current_step = -1
        """Current simulation step"""

        self.processes = processes
        """The process that can be run by every factory"""
        self.costs = costs
        """Cost of running the process for every factory"""

        self.commands = NO_COMMAND * np.ones((n_factories, n_steps, n_lines), dtype=int)
        """An n_factory n_steps * n_lines array giving the process scheduled for each line at every step. 
         NO_COMMAND indicates an empty line. """

        self.balances = initial_balance * np.ones((n_factories))
        """Current balances of all factories"""
        self.inventory = np.zeros(n_products, dtype=int)
        """Current inventory of all factories"""

        self.inputs = inputs

        self.inventory_changes = np.zeros((n_factories, n_products), dtype=int)
        """Changes in the inventory in the last step"""
        self.balance_changes = np.zeros(n_factories, dtype=int)
        """Change in the balance in the last step"""
        self.min_balance = min_balance
        """The minimum balance possible"""

    def state(self, factory: int) -> FactoryState:
        p = self.processes[factory]
        return FactoryState(
            self.inventory[factory, p],
            self.inventory[factory, p+1],
            self.balances[factory],
            self.inventory_changes[factory, p],
            self.inventory_changes[factory, p + 1],
            self.balance_change[factory],
            self.commands[factory],
        )

    def profile(self, factory: int) -> FactoryProfile:
        p = self.processes[factory]
        return FactoryProfile(
            p, self.costs[factory], self.gsales[factory], self.gsupplies[factory],
            self.gsale_prices[factory], self.gsupply_prices[factory]
        )

    def schedule_production(
        self, factory: int, step: int = -1, line: int = -1, override: bool = True
    ) -> bool:
        """
        Orders production of the given process on the given step and line.

        Args:

            factory: The factory to produce
            step: The step at which to do the production. If < 0, any empty time in the future will be used
            line: The line to do the production. If < 0, any empty line will be used
            override: Whether to override any existing commands at that line at that time.

        Returns:

            `True` if successful.

        Remarks:

            - You cannot order production in the past or in the current step
            - Ordering production, will automatically update inventory and balance for all simulation steps assuming
              that this production will be carried out. At the indicated `step` if production was not possible (due
              to insufficient funds or insufficient inventory of the input product), the predictions for the future
              will be corrected.

        """
        if 0 <= step < self.current_step + 1:
            return False
        if line < 0 and step < 0:
            steps, lines = np.nonzero(self.commands[factory, self.current_step + 1 :, :] < 0)
            if len(steps) < 0 and not override:
                return False
            step, line = steps[0], lines[0]
        elif line < 0:
            line = np.argmax(self.commands[factory, step, :] < 0)
        elif step < 0:
            step = np.argmax(self.commands[factory, self.current_step + 1 :, line] < 0)
        if self.commands[factory, step, line] >= 0 and not override:
            return False
        self.commands[factory, step, line] = self.processes[factory]
        return True

    def cancel_production(self, factory: int, step: int, line: int) -> bool:
        """
        Cancels pre-ordered production given that it did not start yet.

        Args:
            factory: The factory
            step: Step to cancel at
            line: Line to cancel at

        Returns:

            True if step >= self.current_step

        Remarks:

            - Cannot cancel a process in the past or present.
        """
        self.commands[factory, step, line] = NO_COMMAND
        return True

    def step(
        self, accepted_sales: np.ndarray, accepted_supplies: np.ndarray
    ) -> List[Failure]:
        """
        Override this method to modify stepping logic.

        Args:
            accepted_sales: Sales per factory
            accepted_supplies: Supplies per factory

        Returns:

        """
        self.current_step += 1
        step = self.current_step
        failures = [[] for _ in range(self.n_factories)]
        initial_balances = self.balances.copy()
        initial_inventories = self.inventory.copy()

        # buy guaranteed supplies as much as possible
        supply_cost = self.gsupplies[:, step] * self.gsupply_prices[:, step]


        # Sell guaranteed sales as much as possible
        inventory = self.inventory - accepted_sales
        if np.min(inventory) < 0:
            real_sales = accepted_sales[inventory >= 0]
            self.balance += np.sum(real_sales * profile.guaranteed_sale_prices[step, inventory >= 0])
            self.inventory[inventory >= 0] -= real_sales
        else:
            self.balance += np.sum(profile.guaranteed_sale_prices[step, :] * accepted_sales)
            self.inventory -= accepted_sales

        # do production
        for line in np.nonzero(self.commands[step, :] >= 0)[0]:
            p = self.commands[step, line]
            cost = profile.costs[line, p]
            ins, outs = self.inputs[p], self.outputs[p]
            if self.balance < cost or cost == INVALID_COST:
                failures.append(
                    Failure(is_inventory=False, line=line, step=step, process=p)
                )
                # self._register_failure(step, p, cost, ins, outs)
                continue
            inp, outp = p, p + 1
            if self.inventory[inp] < ins or self.inventory[outp] < outs:
                failures.append(
                    Failure(is_inventory=True, line=line, step=step, process=p)
                )
                # self._register_failure(step, p, cost, ins, outs)
                continue

        assert self.balance >= min(self.min_balance, initial_balance)
        assert np.min(self.inventory) >= 0
        self.inventory_changes = self.inventory - initial_inventory
        self.balance_change = self.balance - initial_balance
        return failures

    def transaction(self, product: int, quantity: int, price: int) -> None:
        """
        Registers a transaction (a buy/sell)

        Args:
            product: The product transacted on
            quantity: The quantity
            price: The total price
        """
        self.inventory[product] += quantity
        self.inventory_changes[product] += quantity
        self.pay(price)

    def pay(self, money: int) -> None:
        """
        Pays money

        Args:
            money: amount to pay
        """
        self.balance -= money
        self.balance_change -= money
        assert self.balance > self.min_balance, (
            f"Factory {self.id}'s balance is {self.balance} "
            f"(min is {self.min_balance})"
        )


class SCML2020AWI(AgentWorldInterface):
    """The Agent World Interface for SCML2020 world"""

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
        controller: SAOController,
        partners: List[str] = None,
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
            partners: ID of all the partners to negotiate with.

        Returns:

            `True` if the partner accepted and the negotiation is ready to start

        """
        if partners is None:
            partners = (
                self.all_suppliers[product] if is_buy else self.all_consumers[product]
            )
        negotiators = [
            controller.create_negotiator(PassThroughSAONegotiator) for _ in partners
        ]
        results = [
            self.request_negotiation(
                is_buy, product, quantity, unit_price, time, partner, negotiator
            )
            for partner, negotiator in zip(partners, negotiators)
        ]
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

        Returns:

            `True` if the partner accepted and the negotiation is ready to start

        """

        def values(x: Union[int, Tuple[int, int]]):
            if not isinstance(x, Iterable):
                x = (x, x)
            return (x[0], x[1] + 1)

        annotation = {
            "product": product,
            "is_buy": is_buy,
            "buyer": self.agent.id if is_buy else partner,
            "seller": partner if is_buy else self.agent.id,
        }
        issues = [
            Issue(values(quantity), name="quantity"),
            Issue(values(unit_price), name="unit_price"),
            Issue(values(time), name="time"),
        ]
        partners = [self.agent.id, partner]
        req_id = self.agent.create_negotiation_request(
            issues=issues,
            partners=partners,
            negotiator=negotiator,
            annotation=annotation,
            extra=None,
        )
        return self.request_negotiation_about(
            issues=issues, partners=partners, req_id=req_id, annotation=annotation
        )

    def schedule_production(
        self, process: int, step: int, line: int, override: bool = True
    ) -> bool:
        """
        Orders the factory to run the given process at the given line at the given step

        Args:

            process: The process to run
            step: The simulation step. Must be in the future
            line: The production line
            override: Whether to override existing production commands or not

        Returns:
            success/failure

        Remarks:

            - The step cannot be in the past or the current step. Production can only be ordered for future steps
            - ordering production of process -1 is equivalent of `cancel_production`
        """
        return self._world.a2f[self.agent.id].schedule_production(
            process, step, line, override
        )

    def cancel_production(self, step: int, line: int) -> bool:
        """
        Cancels any production commands on that line at this step

        Args:
            step: The step to cancel production at (must be in the future).
            line: The production line

        Returns:

            success/failure

        Remarks:

            - The step cannot be in the past or the current step. Production can only be ordered for future steps
        """
        return self._world.a2f[self.agent.id].cancel_production(step, line)

    # ---------------------
    # Information Gathering
    # ---------------------

    def state(self) -> FactoryState:
        """Receives the factory state"""
        return self._world.a2f[self.agent.id].state

    @property
    @functools.lru_cache(maxsize=1)
    def profile(self) -> FactoryProfile:
        """Gets the profile (static private information) associated with the agent"""
        return self._world.a2f[self.agent.id].profile

    @property
    @functools.lru_cache(maxsize=1)
    def all_suppliers(self) -> List[List[str]]:
        """Returns a list of agent IDs for all suppliers for every product"""
        return self._world.suppliers

    @property
    @functools.lru_cache(maxsize=1)
    def all_consumers(self) -> List[List[str]]:
        """Returns a list of agent IDs for all consumers for every product"""
        return self._world.consumers

    @property
    @functools.lru_cache(maxsize=1)
    def my_input_products(self) -> np.ndarray:
        """Returns a list of products that are inputs to at least one process the agent can run"""
        return self._world.agent_inputs[self.agent.id]

    @property
    @functools.lru_cache(maxsize=1)
    def my_output_products(self) -> np.ndarray:
        """Returns a list of products that are outputs to at least one process the agent can run"""
        return self._world.agent_outputs[self.agent.id]

    @property
    @functools.lru_cache(maxsize=1)
    def my_suppliers(self) -> List[str]:
        """Returns a list of IDs for all of the agent's suppliers (agents that can supply at least one product it may
        need).

        Remarks:

            - If the agent have multiple input products, suppliers of a specific product $p$ can be found using:
              **self.all_suppliers[p]**.
        """
        return list(
            itertools.chain(self.all_suppliers[_] for _ in self.my_input_products)
        )

    @property
    @functools.lru_cache(maxsize=1)
    def my_consumers(self) -> List[str]:
        """Returns a list of IDs for all the agent's consumers (agents that can consume at least one product it may
        produce).

        Remarks:

            - If the agent have multiple output products, consumers of a specific product $p$ can be found using:
              **self.all_consumers[p]**.
        """
        return list(
            itertools.chain(self.all_consumers[_] for _ in self.my_output_products)
        )

    @property
    @functools.lru_cache(maxsize=1)
    def catalog_prices(self) -> np.ndarray:
        """Returns the catalog prices of all products"""
        return self._world.catalog_prices

    @property
    @functools.lru_cache(maxsize=1)
    def inputs(self) -> np.ndarray:
        """Returns the number of inputs to every production process"""
        return self._world.process_inputs

    @property
    @functools.lru_cache(maxsize=1)
    def outputs(self) -> np.ndarray:
        """Returns the number of outputs to every production process"""
        return self._world.process_outputs

    @property
    @functools.lru_cache(maxsize=1)
    def n_products(self) -> int:
        """Returns the number of products in the system"""
        return len(self._world.catalog_prices)

    @property
    @functools.lru_cache(maxsize=1)
    def n_processes(self) -> int:
        """Returns the number of processes in the system"""
        return self.n_products - 1


class SCML2020Agent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _respond_to_negotiation_request(
        self,
        initiator: str,
        partners: List[str],
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
        role: Optional[str],
        req_id: Optional[str],
    ) -> Optional[Negotiator]:
        return self.respond_to_negotiation_request(
            initiator, issues, annotation, mechanism
        )

    def set_renegotiation_agenda(
        self, contract: Contract, breaches: List[Breach]
    ) -> Optional[RenegotiationRequest]:
        return None

    def respond_to_renegotiation_request(
        self, contract: Contract, breaches: List[Breach], agenda: RenegotiationRequest
    ) -> Optional[Negotiator]:
        return None

    def on_neg_request_rejected(self, req_id: str, by: Optional[List[str]]):
        pass

    def on_neg_request_accepted(self, req_id: str, mechanism: AgentMechanismInterface):
        pass

    @abstractmethod
    def on_contract_nullified(
        self, contract: Contract, compensation_money: int, compensation_fraction: float
    ) -> None:
        """
        Called whenever a contract is nullified (because the partner is bankrupt)

        Args:

            contract: The contract being nullified
            compensation_money: The compensation money that is already added to the agent's wallet
            compensation_fraction: The fraction of the contract's total to be compensated. The rest is lost.

        """

    @abstractmethod
    def on_failures(self, failures: List[Failure]) -> None:
        """
        Called whenever there are failures either in production or in execution of guaranteed transactions

        Args:

            failures: A list of `Failure` s.
        """

    @abstractmethod
    def confirm_guaranteed_sales(
        self, quantities: np.ndarray, unit_prices: np.ndarray
    ) -> np.ndarray:
        """
        Called to confirm the amount of guaranteed sales the agent is willing to accept

        Args:

            quantities: An n_products vector giving the maximum quantity that can sold (without negotiation)
            unit_prices: An n_products vector giving the guaranteed unit prices

        Returns:

            An n_products vector specifying the quantities to be sold (up to the given `quantities` limit).
        """

    @abstractmethod
    def confirm_guaranteed_supplies(
        self, quantities: np.ndarray, unit_prices: np.ndarray
    ) -> np.ndarray:
        """
        Called to confirm the amount of guaranteed supplies the agent is willing to accept

        Args:

            quantities: An n_products vector giving the maximum quantity that can bought (without negotiation)
            unit_prices: An n_products vector giving the guaranteed unit prices

        Returns:

            An n_products vector specifying the quantities to be bought (up to the given `quantities` limit).
        """

    @abstractmethod
    def respond_to_negotiation_request(
        self,
        initiator: str,
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
    ) -> Optional[Negotiator]:
        """
        Called whenever another agent requests a negotiation with this agent.

        Args:
            initiator: The ID of the agent that requested this negotiation
            issues: Negotiation issues
            annotation: Annotation attached with this negotiation
            mechanism: The `AgentMechanismInterface` interface to the mechanism to be used for this negotiation.

        Returns:
            None to reject the negotiation, otherwise a negotiator
        """


ContractInfo = namedtuple(
    "ContractInfo", ["q", "u", "product", "is_seller", "partner", "contract"]
)
CompensationRecord = namedtuple(
    "CompensationRecord", ["product", "quantity", "money", "seller_bankrupt", "factory"]
)


class SCML2020World(TimeInAgreementMixin, World):
    """A Supply Chain World Simulation as described for the SCML league of ANAC @ IJCAI 2020.

        Args:

            process_inputs: An n_processes vector specifying the number of inputs from each product needed to execute
                            each process.
            process_outputs: An n_processes vector specifying the number of inputs from each product generated by
                            executing each process.
            catalog_prices: An n_products vector (i.e. n_processes+1 vector) giving the catalog price of all products
            profiles: An n_agents list of `FactoryProfile` objects specifying the private profile of the factory
                      associated with each agent.
            agent_types: An n_agents list of strings/ `SCML2020Agent` classes specifying the type of each agent
            agent_params: An n_agents dictionaries giving the parameters of each agent
            initial_balance: The initial balance in each agent's wallet. All agents will start with this same value.
            breach_penalty: The total penalty paid upon a breach will be calculated as (breach_level * breach_penalty *
                            contract_quantity * contract_unit_price).
            supply_limit: An n_steps * n_products array giving the total supply available of each product over time.
                          Only affects guaranteed supply.
            sales_limit: An n_steps * n_products array giving the total sales to happen for each product over time.
                         Only affects guaranteed sales.
            financial_report_period: The number of steps between financial reports. If < 1, it is a fraction of n_steps
            borrow_on_breach: If true, agents will be forced to borrow money on breach as much as possible to honor the
                              contract
            interest_rate: The interest at which loans grow over time (it only affect a factory when its balance is
                           negative)
            borrow_limit: The maximum amount that be be borrowed (including interest). The balance of any factory cannot
                          go lower than - borrow_limit or the agent will go bankrupt immediately
            compensation_fraction: Fraction of a contract to be compensated (at most) if a partner goes bankrupt. Notice
                                   that this fraction is not guaranteed because the bankrupt agent may not have enough
                                   assets to pay all of its standing contracts to this level of compensation. In such
                                   cases, a smaller fraction will be used.
            compensate_immediately: If true, compensation will happen immediately when an agent goes bankrupt and in
                                    in money. This means that agents with contracts involving the bankrupt agent will
                                    just have these contracts be nullified and receive monetary compensation immediately
                                    . If false, compensation will not happen immediately but at the contract execution
                                    time. In this case, agents with contracts involving the bankrupt agent will be
                                    informed of the compensation fraction (instead of the compensation money) at the
                                    time of bankruptcy and will receive the compensation in kind (money if they are
                                    sellers and products if they are buyers) at the normal execution time of the
                                    contract. In the special case of no-compensation (i.e. `compensation_fraction` is
                                    zero or the bankrupt agent has no assets), the two options will behave similarity.
            compensate_before_past_debt: If true, then compensations will be paid before past debt is considered,
                                         otherwise, the money from liquidating bankrupt agents will first be used to
                                         pay past debt then whatever remains will be used for compensation. Notice that
                                         in all cases, the trigger of bankruptcy will be paid before compensation and
                                         past debts.
            force_max_guaranteed_transactions: If true, agents are not asked to confirm guaranteed transactions and they
                                               are carried out up to bankruptcy
            compact: If True, no logs will be kept and the whole simulation will use a smaller memory footprint
            n_steps: Number of simulation steps (can be considered as days).
            time_limit: Total time allowed for the complete simulation in seconds.
            neg_n_steps: Number of negotiation steps allowed for all negotiations.
            neg_time_limit: Total time allowed for a complete negotiation in seconds.
            neg_step_time_limit: Total time allowed for a single step of a negotiation. in seconds.
            negotiation_speed: The number of negotiation steps that pass in every simulation step. If 0, negotiations
                               will be guaranteed to finish within a single simulation step
            signing_delay: The number of simulation steps to pass between a contract is concluded and signed
            name: The name of the simulations
            **kwargs: Other parameters that are passed directly to `World` constructor.

    """

    def __init__(
        self,
        # SCML2020 specific parameters
        process_inputs: np.ndarray,
        process_outputs: np.ndarray,
        catalog_prices: np.ndarray,
        profiles: List[FactoryProfile],
        agent_types: List[Type[SCML2020Agent]],
        agent_params: List[Dict[str, Any]] = None,
        initial_balance: int = 1000,
        breach_penalty=0.15,
        financial_report_period=5,
        borrow_on_breach=False,
        interest_rate=0.05,
        borrow_limit=-0.2,
        supply_limit: np.ndarray = None,
        sales_limit: np.ndarray = None,
        compensation_fraction=1.0,
        compensate_immediately=False,
        compensate_before_past_debt=False,
        force_max_guaranteed_transactions=True,
        # General World Parameters
        compact=False,
        n_steps=1000,
        time_limit=60 * 90,
        # mechanism params
        neg_n_steps=100,
        neg_time_limit=2 * 60,
        neg_step_time_limit=60,
        negotiation_speed=101,
        # simulation parameters
        signing_delay=1,
        name: str = None,
        **kwargs,
    ):
        self.compensation_fraction = compensation_fraction
        if compact:
            kwargs["log_screen_level"] = logging.CRITICAL
            kwargs["log_file_level"] = logging.ERROR
            kwargs["save_mechanism_state_in_contract"] = False
            kwargs["log_negotiations"] = False
            kwargs["log_ufuns"] = False
            kwargs["save_cancelled_contracts"] = False
            kwargs["save_resolved_breaches"] = False
            kwargs["save_negotiations"] = False
        self.compact = compact
        if negotiation_speed == 0:
            negotiation_speed = neg_n_steps + 1
        super().__init__(
            bulletin_board=None,
            breach_processing=BreachProcessing.NONE,
            awi_type="scml.scml2020.SCML2020AWI",
            start_negotiations_immediately=False,
            mechanisms={"negmas.sao.SAOMechanism": {}},
            default_signing_delay=signing_delay,
            n_steps=n_steps,
            time_limit=time_limit,
            negotiation_speed=negotiation_speed,
            neg_n_steps=neg_n_steps,
            neg_time_limit=neg_time_limit,
            neg_step_time_limit=neg_step_time_limit,
            name=name,
            **kwargs,
        )
        TimeInAgreementMixin.init(self, time_field="time")
        self.breach_penalty = breach_penalty
        self.bulletin_board.add_section("cfps")
        self.bulletin_board.add_section("reports_time")
        self.bulletin_board.add_section("reports_agent")

        if not isinstance(agent_types, Iterable):
            agent_types = [agent_types] * len(profiles)

        assert len(profiles) == len(agent_types)
        self.profiles = profiles
        self.catalog_prices = catalog_prices
        self.process_inputs = process_inputs
        self.process_outputs = process_outputs
        self.n_products = len(catalog_prices)
        self.n_processes = len(process_inputs)
        self.borrow_on_breach = borrow_on_breach
        self.interest_rate = interest_rate
        self.force_max_guaranteed_transactions = force_max_guaranteed_transactions
        self.compensate_before_past_debt = compensate_before_past_debt
        self.financial_reports_period = (
            financial_report_period
            if financial_report_period >= 1
            else int(0.5 + financial_report_period * n_steps)
        )
        self.compensation_fraction = compensation_fraction
        self.compensate_immediately = compensate_immediately
        self.borrow_limit = (
            borrow_limit
            if borrow_limit > 1
            else int(0.5 + borrow_limit * initial_balance)
        )
        assert self.n_products == self.n_processes + 1

        if supply_limit is None:
            self.supply_limit = sys.maxsize * np.ones(
                (n_steps, self.n_products), dtype=int
            )
        else:
            self.supply_limit = supply_limit
        if sales_limit is None:
            self.sales_limit = sys.maxsize * np.ones(
                (n_steps, self.n_products), dtype=int
            )
        else:
            self.sales_limit = sales_limit

        self.factories = [
            Factory(
                profile=profile,
                initial_balance=initial_balance,
                id=f"f{i}",
                inputs=process_inputs.copy(),
                outputs=process_outputs.copy(),
            )
            for i, profile in enumerate(profiles)
        ]
        if agent_params is None:
            agent_params = [dict(name=f"{_.__name__[:3]}{i:03}") for i, _ in enumerate(agent_types)]
        agents = []
        for i, (atype, aparams) in enumerate(zip(agent_types, agent_params)):
            a = instantiate(atype, **aparams)
            self.join(a, i)
            agents.append(a)
        n_agents = len(agents)
        self.factories = [
            Factory(p, initial_balance, process_inputs, process_outputs)
            for p in profiles
        ]
        self.a2f = dict(zip((_.id for _ in agents), self.factories))
        self.f2a = dict(zip((_.id for _ in self.factories), agents))
        self.afp = list(zip(agents, self.factories, profiles))
        self.a2i = dict(zip((_.id for _ in agents), range(n_agents)))
        self.i2a = agents
        self.f2i = dict(zip((_.id for _ in self.factories), range(n_agents)))
        self.i2f = self.factories

        self.breach_prob = dict(zip((_.id for _ in agents), itertools.repeat(0.0)))
        self.breach_level = dict(zip((_.id for _ in agents), itertools.repeat(0.0)))
        self.agent_n_contracts = dict(zip((_.id for _ in agents), itertools.repeat(0)))

        n_processes = len(process_inputs)
        n_products = n_processes + 1

        self.suppliers: List[List[str]] = [[] for _ in range(n_products)]
        self.consumers: List[List[str]] = [[] for _ in range(n_products)]
        self.agent_processes: Dict[str, List[int]] = defaultdict(list)
        self.agent_inputs: Dict[str, List[int]] = defaultdict(list)
        self.agent_outputs: Dict[str, List[int]] = defaultdict(list)

        for p in range(n_processes):
            for agent_id, profile in zip(self.agents.keys(), profiles):
                if np.all(profile.costs[:, p] == INVALID_COST):
                    continue
                self.suppliers[p + 1].append(agent_id)
                self.consumers[p].append(agent_id)
                self.agent_processes[agent_id].append(p)
                self.agent_inputs[agent_id].append(p)
                self.agent_outputs[agent_id].append(p + 1)

        self.agent_processes = {k: np.array(v) for k, v in self.agent_processes.items()}
        self.agent_inputs = {k: np.array(v) for k, v in self.agent_inputs.items()}
        self.agent_outputs = {k: np.array(v) for k, v in self.agent_outputs.items()}

        self._n_production_failures = 0
        self.__n_nullified = 0
        self.__n_bankrupt = 0
        self.penalties = 0
        self.is_bankrupt: Dict[str, bool] = dict(
            zip(self.agents.keys(), itertools.repeat(False))
        )
        self.agent_contracts: Dict[str, List[List[ContractInfo]]] = {
            aid: [[] for _ in range(n_steps)] for aid in self.agents.keys()
        }
        self.compensation_balance = 0
        self.compensation_records: Dict[str, List[CompensationRecord]] = defaultdict(
            list
        )
        self.compensation_factory = Factory(
            FactoryProfile(
                np.zeros((n_steps, n_processes), dtype=int),
                np.zeros((n_steps, n_products), dtype=int),
                np.zeros((n_steps, n_products), dtype=int),
                np.zeros((n_steps, n_products), dtype=int),
                np.zeros((n_steps, n_products), dtype=int),
            ),
            initial_balance=0,
            inputs=self.process_inputs,
            outputs=self.process_outputs,
        )

    @classmethod
    def generate(
        cls,
        agent_types: List[Type[SCML2020Agent]],
        agent_params: List[Dict[str, Any]] = None,
        depth=4,
        process_inputs: Union[np.ndarray, Tuple[int, int]] = (1, 4),
        process_outputs: Union[np.ndarray, Tuple[int, int]] = (1, 1),
        profit_mean=0.15,
        profit_stddev=0.05,
        initial_balance: int = 1000,
        breach_penalty=0.15,
        n_steps: int = 100,
        n_lines: int = 10,
        n_agents_per_process: int = 1,
        agent_name_reveals_location=False,
        signing_delay: int = 1,
    ):
        n_products = depth + 1
        n_processes = depth
        if not isinstance(agent_types, Iterable):
            agent_types = [agent_types] * (n_agents_per_process * n_processes)
        n_agents = len(agent_types)
        assert n_agents >= n_processes
        assert n_agents_per_process <= n_agents
        if isinstance(process_inputs, np.ndarray):
            process_inputs = process_inputs.flatten()
            assert (
                len(process_inputs) == depth
            ), f"depth is {depth} but got {len(process_inputs)} inputs"
            inputs = process_inputs
        else:
            inputs = np.random.randint(*process_inputs, size=depth)

        if isinstance(process_outputs, np.ndarray):
            process_outputs = process_outputs.flatten()
            assert (
                len(process_outputs) == depth
            ), f"depth is {depth} but got {len(process_outputs)} outputs"
            outputs = process_outputs
        else:
            outputs = np.random.randint(*process_outputs, size=depth)

        min_cost, max_cost = 1, 10
        min_unit, max_unit = 1, 10
        if n_agents_per_process < 1:
            costs = np.random.randint(
                min_cost, max_cost, (n_agents, n_lines, n_processes), dtype=float
            )
            n_agents_per_process = n_agents
        elif n_agents >= n_agents_per_process * n_processes:
            costs = float("inf") * np.ones(
                (n_agents, n_lines, n_processes), dtype=float
            )
            for p in range(n_processes):
                costs[
                    p * n_agents_per_process : (p + 1) * n_agents_per_process, :, p
                ] = np.random.randint(min_cost, max_cost, size=n_agents_per_process)
            if not agent_name_reveals_location:
                costs = np.random.permutation(costs)
        else:
            costs = INVALID_COST * np.ones((n_agents, n_lines, n_processes), dtype=int)
            for p in range(n_processes):
                # guarantee that at least one agent is running each process (we know that n_agents>=n_processes)
                costs[p, :, p] = random.randrange(min_cost, max_cost)
                # generate random new agents for the process
                agents = np.random.randint(
                    0, n_agents - 2, size=n_agents_per_process - 1
                )
                # make sure that the agent with the same index as the process is not chosen twice
                agents[agents >= p] += 1
                # assign agents to processes
                costs[agents, :, p] = np.random.randint(
                    min_cost, max_cost, size=(n_agents_per_process, 1)
                )
            # permute costs so that agent i is not really guaranteed to run process i
            if not agent_name_reveals_location:
                costs = np.random.permutation(costs)

        # generate guaranteed contracts
        # -----------------------------
        # select the amount of random raw material available to every agent that can consume it
        quantity = np.zeros((n_products, n_agents, n_steps), dtype=int)
        a0 = np.nonzero(~np.isinf(costs[:, 0, 0].flatten()))[0].flatten()
        quantity[0, a0, :-1] = np.random.randint(
            0, process_inputs[0] * n_lines, size=(1, len(a0), n_steps - 1)
        )
        # initialize unit prices randomly within the allowed negotiation range
        price = np.random.randint(min_unit, max_unit, size=quantity.shape)
        for p in range(1, n_processes):
            # find the agents / lines that can run this process
            ins = quantity[p - 1, :, :-1]

    def generate_initial_contracts(
        self
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pass

    def get_private_state(self, agent: "Agent") -> dict:
        return vars(self.a2f[agent.id].state)

    def add_financial_report(
        self, agent: SCML2020Agent, factory: Factory, reports_agent, reports_time
    ) -> None:
        """
        Records a financial report for the given agent in the agent indexed reports and time indexed reports

        Args:
            agent: The agent
            factory: Its factory
            reports_agent: A dictionary of financial reports indexed by agent id
            reports_time: A dictionary of financial reports indexed by time

        Returns:

        """
        bankrupt = self.is_bankrupt[agent.id]
        inventory = (
            int(np.sum(self.catalog_prices * factory.inventory)) if not bankrupt else 0
        )
        report = FinancialReport(
            agent_id=agent.id,
            step=self.current_step,
            cash=factory.balance,
            assets=inventory,
            breach_prob=self.breach_prob[agent.id],
            breach_level=self.breach_level[agent.id],
            is_bankrupt=bankrupt,
            agent_name = agent.name
        )
        repstr = str(report).replace("\n", " ")
        self.logdebug(f"{agent.name}: {repstr}")
        if reports_agent.get(agent.id, None) is None:
            reports_agent[agent.id] = {}
        reports_agent[agent.id][self.current_step] = report
        if reports_time.get(self.current_step, None) is None:
            reports_time[self.current_step] = {}
        reports_time[self.current_step][agent.id] = report

    def simulation_step(self):
        s = self.current_step

        # pay interests for negative balances
        # -----------------------------------
        if self.interest_rate > 0.0:
            for agent, factory, _ in self.afp:
                if factory.balance < 0:
                    to_pay = -int(math.ceil(self.interest_rate * factory.balance))
                    if factory.balance - to_pay < -self.borrow_limit:
                        to_pay = self.__make_bankrupt(agent.id, factory, to_pay)
                    factory.pay(to_pay)

        # publish financial reports
        # -------------------------
        if self.current_step % self.financial_reports_period == 0:
            reports_agent = self.bulletin_board.data["reports_agent"]
            reports_time = self.bulletin_board.data["reports_time"]
            for agent, factory, _ in self.afp:
                self.add_financial_report(agent, factory, reports_agent, reports_time)

        # do guaranteed transactions and step factories
        # ---------------------------------------------
        if self.force_max_guaranteed_transactions:
            for a, f, p in self.afp:
                f.step(p.guaranteed_supplies[s, :], p.guaranteed_sales[s, :])
        else:
            afp_randomized = [
                self.afp[_] for _ in np.random.permutation(np.arange(len(self.afp)))
            ]
            for a, f, p in afp_randomized:
                a: SCML2020Agent
                f: Factory
                supply = a.confirm_guaranteed_supplies(
                    p.guaranteed_supplies[s].copy(), p.guaranteed_supply_prices[s].copy()
                )
                sales = a.confirm_guaranteed_sales(
                    p.guaranteed_sales[s].copy(), p.guaranteed_sale_prices[s].copy()
                )
                f.step(sales, supply)

    def contract_size(self, contract: Contract) -> float:
        return contract.agreement["quantity"] * contract.agreement["unit_price"]

    def contract_record(self, contract: Contract) -> Dict[str, Any]:
        c = {
            "id": contract.id,
            "seller_name": self.agents[contract.annotation["seller"]].name,
            "buyer_name": self.agents[contract.annotation["buyer"]].name,
            "seller_type": self.agents[
                contract.annotation["seller"]
            ].__class__.__name__,
            "buyer_type": self.agents[contract.annotation["buyer"]].__class__.__name__,
            "delivery_time": contract.agreement["time"],
            "quantity": contract.agreement["quantity"],
            "unit_price": contract.agreement["unit_price"],
            "signed_at": contract.signed_at if contract.signed_at is not None else -1,
            "nullified_at": contract.nullified_at
            if contract.nullified_at is not None
            else -1,
            "concluded_at": contract.concluded_at,
            "signatures": "|".join(str(_) for _ in contract.signatures),
            "issues": contract.issues if not self.compact else None,
            "seller": contract.annotation["seller"],
            "buyer": contract.annotation["buyer"],
        }
        if not self.compact:
            c.update(contract.annotation)
        c["n_neg_steps"] = contract.mechanism_state.step
        return c

    def breach_record(self, breach: Breach) -> Dict[str, Any]:
        return {
            "perpetrator": breach.perpetrator,
            "perpetrator_name": breach.perpetrator,
            "level": breach.level,
            "type": breach.type,
            "time": breach.step,
        }

    def execute_action(
        self, action: Action, agent: "Agent", callback: Callable = None
    ) -> bool:
        if action.type == "schedule":
            return self.a2f[agent.id].schedule_production(
                process=action.params["process"],
                step=action.params.get("step", -1),
                line=action.params.get("line", -1),
                override=action.params.get("override", True),
            )
        elif action.type == "cancel":
            return self.a2f[agent.id].cancel_production(
                step=action.params.get("step", -1), line=action.params.get("line", -1)
            )

    def post_step_stats(self):
        self._stats["n_contracts_nullified"].append(self.__n_nullified)
        self._stats["n_bankrupt"].append(self.__n_bankrupt)
        market_size = 0
        self._stats[f"_balance_society"].append(self.penalties)
        internal_market_size = self.penalties
        for a, f, _ in self.afp:
            self._stats[f"balance_{a.name}"].append(f.balance)
            self._stats[f"storage_{a.name}"].append(f.inventory.sum())
            market_size += f.balance
        self._stats["market_size"].append(market_size)
        self._stats["production_failures"].append(
            self._n_production_failures / len(self.factories)
            if len(self.factories) > 0
            else np.nan
        )
        self._stats["_market_size_total"].append(market_size + internal_market_size)

    def pre_step_stats(self):
        self._n_production_failures = 0
        self.__n_nullified = 0
        self.__n_bankrupt = 0

    @property
    def business_size(self) -> float:
        """The total business size defined as the total money transferred within the system"""
        return sum(self.stats["activity_level"])

    @property
    def agreement_rate(self) -> float:
        """Fraction of negotiations ending in agreement and leading to signed contracts"""
        n_negs = sum(self.stats["n_negotiations"])
        n_contracts = len(self._saved_contracts)
        return n_contracts / n_negs if n_negs != 0 else np.nan

    @property
    def cancellation_rate(self) -> float:
        """Fraction of negotiations ending in agreement and leading to signed contracts"""
        n_negs = sum(self.stats["n_negotiations"])
        n_contracts = len(self._saved_contracts)
        n_signed_contracts = len(
            [_ for _ in self._saved_contracts.values() if _["signed"]]
        )
        return (1.0 - n_signed_contracts / n_contracts) if n_contracts != 0 else np.nan

    @property
    def n_negotiation_rounds_successful(self) -> float:
        """Average number of rounds in a successful negotiation"""
        n_negs = sum(self.stats["n_contracts_concluded"])
        if n_negs == 0:
            return np.nan
        return sum(self.stats["n_negotiation_rounds_successful"]) / n_negs

    @property
    def n_negotiation_rounds_failed(self) -> float:
        """Average number of rounds in a successful negotiation"""
        n_negs = sum(self.stats["n_negotiations"]) - sum(
            self.stats["n_contracts_concluded"]
        )
        if n_negs == 0:
            return np.nan
        return sum(self.stats["n_negotiation_rounds_failed"]) / n_negs

    @property
    def contract_execution_fraction(self) -> float:
        """Fraction of signed contracts successfully executed"""
        n_executed = sum(self.stats["n_contracts_executed"])
        n_signed_contracts = len(
            [_ for _ in self._saved_contracts.values() if _["signed"]]
        )
        return n_executed / n_signed_contracts if n_signed_contracts > 0 else np.nan

    @property
    def breach_rate(self) -> float:
        """Fraction of signed contracts that led to breaches"""
        n_breaches = sum(self.stats["n_breaches"])
        n_signed_contracts = len(
            [_ for _ in self._saved_contracts.values() if _["signed"]]
        )
        if n_signed_contracts != 0:
            return n_breaches / n_signed_contracts
        return np.nan

    def order_contracts_for_execution(
        self, contracts: Collection[Contract]
    ) -> Collection[Contract]:
        return sorted(contracts, key=lambda x: x.annotation["product"])

    def _execute(
        self,
        product: int,
        q: int,
        p: int,
        u: int,
        buyer_factory: Factory,
        seller_factory: Factory,
        has_breaches: bool,
    ):
        """Executes the contract"""
        if has_breaches:
            money = min(buyer_factory.balance, p)
            quantity = min(seller_factory.inventory[product], q)
            p, q = min(money, quantity * u), min(quantity, int(money / u))
            assert q * u == p
        buyer_factory.transaction(product, q, -p)
        seller_factory.transaction(product, -q, p)

    def __register_contract(self, agent_id: str, level: float) -> None:
        """Registers execution of the contract in the agent's stats"""
        n_contracts = self.agent_n_contracts[agent_id] - 1
        self.breach_prob[agent_id] = (
            self.breach_prob[agent_id] * n_contracts + (level > 0)
        ) / (n_contracts + 1)
        self.breach_level[agent_id] = (
            self.breach_prob[agent_id] * n_contracts + level
        ) / (n_contracts + 1)

    def __make_bankrupt(self, agent_id: str, factory: Factory, required: int) -> int:
        """
        Bankruptcy processing for the given agent

        Args:
            agent_id: The agent to be made bankrupt
            factory: The associated factory
            required: The money required after the bankruptcy is processed

        Returns:
            The amount of money to pay back to the entity that should have been paid `money`

        """
        # sell everything on the agent's inventory
        total = int(np.sum(factory.inventory * self.catalog_prices))
        pay_back = min(required, total)
        available = total - required
        # If past debt is paid before compensation pay it
        original_balance = factory.balance
        if not self.compensate_before_past_debt:
            available += original_balance

        # get all future contracts of the bankrupt agent that are not exeucted
        contracts = list(
            itertools.chain(
                *(
                    self.agent_contracts[agent_id][s]
                    for s in range(self.current_step, self.n_steps)
                )
            )
        )
        owed = 0
        total_owed = 0
        nulled_contracts = []
        for contract in contracts:
            total_owed += contract.q * contract.u
            if (
                self.is_bankrupt[contract.partner]
                or contract.contract.nullified_at is not None
            ):
                continue
            nulled_contracts.append(contract)
            owed += contract.q * contract.u

        if available <= 0:
            self.__record_bankrupt(
                agent_id, factory, original_balance, required, available, total_owed
            )
            return pay_back

        # give the liquidation money to the bankrupt agent to pay compensations
        factory.balance = available

        if owed <= 0:
            self.__record_bankrupt(
                agent_id, factory, original_balance, required, available, total_owed
            )
            return pay_back

        # calculate compensation fraction
        if available >= owed:
            fraction = self.compensation_fraction
        else:
            fraction = self.compensation_fraction * available / owed

        # calculate compensation and pay it as needed
        for contract in nulled_contracts:
            victim = self.agents[contract.partner]
            victim_factory = self.a2f.get(victim.id, None)
            # calculate compensation (as money)
            compensation = min(factory.balance, fraction * contract.q * contract.u)
            if compensation < 0:
                self.nullify_contract(contract)
                continue
            if self.compensate_immediately:
                # pay immediate compensation if indicated
                victim_factory.pay(-compensation)
                factory.pay(compensation)
            else:
                # add the required products/money to the internal compensation inventory/funds to be paid at the
                # contract execution time.
                if contract.is_seller:
                    self.compensation_records[contract.contract.id].append(
                        CompensationRecord(
                            contract.product,
                            int((compensation // contract.u) * contract.u),
                            0,
                            True,
                            victim_factory,
                        )
                    )
                else:
                    self.compensation_records[contract.contract.id].append(
                        CompensationRecord(-1, 0, compensation, False, victim_factory)
                    )

            victim.on_contract_nullified(
                contract=contract,
                bankrupt_partner=agent_id,
                compensation_money=compensation,
                compensation_fraction=fraction,
            )
            self.nullify_contract(contract)
            self.__record_bankrupt(
                agent_id, factory, original_balance, required, available, total_owed
            )
            return pay_back

    def __record_bankrupt(
        self,
        agent_id: str,
        factory: Factory,
        original_balance: int,
        required: int,
        available: int,
        owed: int,
    ) -> None:
        """
        Records agent bankruptcy

        Args:

            agent_id: ID of the bankrupt agent
            factory: Its factory
            original_balance: The original balance in its wallet at the time of bankruptcy
            required: The money required from it at the time of bankruptcy
            available: total liquidation money minus the money required (available to pay `owed` money)
            owed: total owed money

        """

        # set the balance to the worst of the borrow limit and what remains after liquidation and paying all debts
        factory.balance = min(-self.borrow_limit, original_balance + available - owed)
        # no remaining inventory
        factory.inventory = np.zeros(self.n_products, dtype=int)

        # record bankruptcy
        self.is_bankrupt[agent_id] = True

        # announce bankruptcy
        reports_agent = self.bulletin_board.data["reports_agent"]
        reports_time = self.bulletin_board.data["reports_time"]
        self.add_financial_report(
            self.agents[agent_id], factory, reports_agent, reports_time
        )
        self.__n_bankrupt += 1

    def on_contract_signed(self, contract: Contract):
        super().on_contract_signed(contract)
        t = contract.agreement["time"]
        u, q = contract.agreement["unit_price"], contract.agreement["quantity"]
        product = contract.annotation["partner"]
        agent, partner = contract.partners
        is_seller = agent == contract.annotation["seller"]
        self.agent_contracts[agent][t].append(
            ContractInfo(q, u, product, partner, is_seller, contract)
        )
        self.agent_contracts[partner][t].append(
            ContractInfo(q, u, product, agent, not is_seller, contract)
        )

    def nullify_contract(self, contract: Contract):
        self.__n_nullified += 1
        contract.nullified_at = self.current_step

    def __register_breach(
        self, agent_id: str, level: float, contract_total: float, factory: Factory
    ) -> bool:
        """
        Registers a breach of the given level on the given agent. Assume that the contract is already added
        to the agent_contracts

        Args:
            agent_id: The perpetrator of the breach
            level: The breach level
            contract_total: The total of the contract breached (quantity * unit_price)
            factory: The factory corresponding to the perpetrator

        Returns:
            indicates whether the agent should go bankrupt
        """
        bankrupt = False
        if level <= 0:
            return bankrupt
        penalty = int(math.ceil(level * contract_total))
        if factory.balance - penalty < -self.borrow_limit:
            penalty = self.borrow_limit + factory.balance
            bankrupt = True
        if penalty > 0:
            factory.pay(penalty)
            self.penalties += penalty
        return bankrupt

    def start_contract_execution(self, contract: Contract) -> Set[Breach]:

        # get contract info
        breaches = set()
        product = contract.annotation["product"]
        buyer_id, seller_id = (
            contract.annotation["buyer"],
            contract.annotation["seller"],
        )
        buyer, buyer_factory = self.agents[buyer_id], self.a2f[buyer_id]
        seller, seller_factory = self.agents[seller_id], self.a2f[seller_id]
        q, u, t = (
            contract.agreement["quantity"],
            contract.agreement["unit_price"],
            contract.agreement["time"],
        )
        if q <= 0 or u <= 0:
            self.logwarning(f"Contract {str(contract)} has zero quantity of unit price!!! will be ignored")
            return breaches
        p = q * u
        assert t == self.current_step
        self.agent_n_contracts[buyer_id] += 1
        self.agent_n_contracts[seller_id] += 1
        missing_product = q - seller_factory.inventory[product]
        missing_money = p - buyer_factory.balance
        buyer_bankrupt = seller_bankrupt = False

        # if the contract is already nullified, take care of it
        if contract.nullified_at is not None:
            self.compensation_factory.inventory[product] = 0
            self.compensation_factory.balance = 0
            for c in self.compensation_records.get(contract.id, []):
                if c.product >= 0 and c.quantity > 0:
                    assert c.product == product
                    self.compensation_factory.inventory[product] += c.quantity
                self.compensation_factory.balance += c.money
                if c.seller_bankrupt:
                    seller_factory = self.compensation_factory
                else:
                    buyer_factory = self.compensation_factory
        if seller_factory == buyer_factory:
            self.logwarning(
                f"Seller factory {seller_factory.id} and Buyer factory {buyer_factory.id} are the same."
                f" This is most likely happening because you have two compensation records for the "
                f"same contract!!"
            )
            return breaches
        # if there are no breaches, just execute the contract
        if missing_money <= 0 and missing_product <= 0:
            self._execute(
                product, q, p, u, buyer_factory, seller_factory, has_breaches=False
            )
            self.__register_contract(seller_id, 0)
            self.__register_contract(buyer_id, 0)
            return breaches

        # if there is a product breach (the seller does not have enough products), register it
        if missing_product <= 0:
            self.__register_contract(seller_id, 0)
        else:
            product_breach_level = missing_product / q
            breaches.add(
                Breach(
                    contract=contract,
                    perpetrator=seller_id,
                    victims=buyer_id,
                    level=product_breach_level,
                    type="product",
                )
            )
            self.__register_contract(seller_id, product_breach_level)
            seller_bankrupt = self.__register_breach(
                seller_id, product_breach_level, p, seller_factory
            )
            if self.borrow_on_breach:
                # calculate the amount to be paid by the perpetrator (including the penalty)
                to_pay = math.ceil(missing_product * u * (1 + self.breach_penalty))
                q_payable = missing_product
                # it is not possible to borrow the whole amount. Borrow as much as possible
                if seller_factory.balance - to_pay < -self.borrow_limit:
                    can_pay = self.borrow_limit + seller_factory.balance
                    q_payable = int(can_pay // u)
                    assert q_payable < missing_product
                    to_pay = q_payable * u
                    seller_bankrupt = True
                # borrow as much as possible
                seller_factory.pay(to_pay)
                seller_factory.inventory[product] += q_payable
                missing_product = 0

        # if there is a money breach (the buyer does not have enough money), register it
        if missing_money < 0:
            self.__register_contract(buyer_id, 0)
        else:
            money_breach_level = missing_money / p
            breaches.add(
                Breach(
                    contract=contract,
                    perpetrator=buyer_id,
                    victims=seller_id,
                    level=money_breach_level,
                    type="money",
                )
            )
            self.__register_contract(buyer_id, money_breach_level)
            buyer_bankrupt = self.__register_breach(
                buyer_id, money_breach_level, p, buyer_factory
            )
            if self.borrow_on_breach:
                # find out the amount to be paid to borrow the needed money
                paid_for = missing_money
                to_pay = math.ceil(missing_money * (1 + self.breach_penalty))
                if buyer_factory.balance - to_pay < -self.borrow_limit:
                    can_pay = self.borrow_limit + seller_factory.balance
                    assert can_pay < to_pay
                    to_pay = int(can_pay // u) * u
                    paid_for = int(to_pay // (1 + self.breach_penalty))
                    buyer_bankrupt = True
                # do borrow
                assert to_pay >= paid_for
                buyer_factory.pay(to_pay - paid_for)

        # execute the contract to the limit possible
        self._execute(
            product,
            q,
            p,
            u,
            buyer_factory,
            seller_factory,
            has_breaches=missing_product > 0 or missing_money > 0,
        )
        if seller_bankrupt:
            self.__make_bankrupt(seller_id, seller_factory)
        if buyer_bankrupt:
            self.__make_bankrupt(buyer_id, buyer_factory)
        # return the list of breaches
        return breaches

    def complete_contract_execution(
        self, contract: Contract, breaches: List[Breach], resolution: Contract
    ) -> None:
        raise RuntimeError(
            "complete_contract_execution should never be called in SCML2020 as there is no breach"
            " resolution allowed"
        )
