"""Implements the world class for the SCM2020 world """
import copy
import functools
import itertools
import logging
import math
import random
from abc import abstractmethod
from collections import defaultdict
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
    SAONegotiator)
from negmas.helpers import instantiate, unique_name
from negmas.situated import World, TimeInAgreementMixin, BreachProcessing

__all__ = [
    "FactoryState", "SCML2020Agent", "SCML2020AWI", "SCM2020World"
]

@dataclass
class FactoryProfile:
    costs: np.ndarray
    """An n_lines * n_processes array giving the cost of executing any process (may be infinite)"""
    guaranteed_sales: np.ndarray
    """A n_steps * n_products array giving guaranteed sales of different products for the whole simulation time"""
    guaranteed_supplies: np.ndarray
    """A n_steps * n_products array giving guaranteed sales of different products for the whole simulation time"""
    guaranteed_sale_prices: np.ndarray
    """A n_steps * n_products array giving guaranteed unit prices for the `guaranteed_quantities` . It will be zero
    for times and products for which there are no guaranteed quantities (i.e. (guaranteed_quantities[...] == 0) =>
     (guaranteed_prices[...] == 0) )"""
    guaranteed_supply_prices: np.ndarray
    """A n_steps * n_products array giving guaranteed unit prices for the `guaranteed_quantities` . It will be zero
    for times and products for which there are no guaranteed quantities (i.e. (guaranteed_quantities[...] == 0) =>
     (guaranteed_prices[...] == 0) )"""
    n_products: int = field(init=False)
    """Number of products in the world"""
    n_processes: int = field(init=False)
    """Number of processes in the world"""
    n_lines: int = field(init=False)
    """Number of lines in this factory"""
    n_steps: int = field(init=False)
    """Number of simulation steps"""

    def __post_init__(self):
        self.n_lines, self.n_processes = self.costs.shape
        self.n_products = self.n_processes + 1
        self.n_steps = self.guaranteed_sales.shape[0]


@dataclass
class Failure:
    """A production failure"""

    is_inventory: bool
    """True if the cause of failure was insufficient inventory. If False, the cause was insufficient funds. Note that
    if both conditions were true, only insufficient funds (is_inventory=False) will be reported."""
    line: int
    """The line at which the failure happened"""
    step: int
    """The step at which the failure happened"""
    process: int
    """The process that failed to execute (if `guaranteed_contract_failure` and `is_inventory` , then this will be the 
    process that would have generated the needed product. and if `guaranteed_contract_failure` and not `is_inventory` 
    , then this will be the product that was not received because of unavailable funds)"""
    is_guaranteed_transaction_failure: bool = False
    """Is the failure resulting from a guaranteed contract? This can happen if the agent does not have enough 
    money to buy some guaranteed purchases. In this case, it will buy as much as it can then the prediction of the 
    future inventory and balance will be updated accordingly."""

@dataclass
class FactoryState:
    inventory: np.ndarray
    """An n_products vector giving current quantity of every product in storage"""
    balance: int
    """Current balance in the wallet"""
    commands: np.ndarray
    """n_steps * n_lines array giving the process scheduled on each line at every step for the 
    whole simulation"""


class Factory:
    """A simulated factory"""

    def __init__(
        self,
        profile: FactoryProfile,
        initial_balance: int,
        inputs: np.ndarray,
        outputs: np.ndarray,
        id: Optional[str] = None,
    ):
        self.__profile = profile
        self.current_step = -1
        """Current simulation step"""
        self.profile = copy.deepcopy(profile)
        """The readonly factory profile (See `FactoryProfile` )"""
        self.commands = -1 * np.ones((profile.n_steps, profile.n_lines))
        """An n_steps * n_lines array giving the process scheduled for each line at every step. -1 indicates an empty
        line. """
        # self.predicted_inventory = profile.guaranteed_quantities.copy()
        """An n_steps * n_products array giving the inventory content at different steps. For steps in the past and 
        present, this is the *actual* value of the inventory at that time. For steps in the future, this is a 
        *prediction* of the inventory at that step."""
        self.balance = initial_balance
        """Current balance"""
        self.inventory = np.zeros(profile.n_products)
        """Current inventory"""
        # self.predicted_balance = initial_balance - np.sum(
        #     profile.guaranteed_quantities * profile.guaranteed_prices, axis=-1
        # )
        """An n_steps vector giving the wallet balance at different steps. For steps in the past and 
        present, this is the *actual* value of the balance at that time. For steps in the future, this is a 
        *prediction* of the balance at that step."""
        self.id = id
        """A unique ID for the factory"""
        self.inputs = inputs
        """An n_process array giving the number of inputs needed for each process 
        (of the product with the same index)"""
        self.outputs = outputs
        """An n_process array giving the number of outputs produced by each process 
        (of the product with the next index)"""

    @property
    def state(self) -> FactoryState:
        return FactoryState(self.inventory.copy(), self.balance, self.commands.copy())

    @property
    def current_inventory(self) -> np.ndarray:
        """Current inventory contents"""
        return self.inventory

    @property
    def current_balance(self) -> float:
        """Current wallet balance"""
        return self.balance

    def schedule_production(
        self, process: int, step: int = -1, line: int = -1, override: bool = True
    ) -> bool:
        """
        Orders production of the given process on the given step and line.

        Args:

            process: The process index
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
            steps, lines = np.nonzero(self.commands[self.current_step + 1 :, :] < 0)
            if len(steps) < 0 and not override:
                return False
            step, line = steps[0], lines[0]
        elif line < 0:
            line = np.argmax(self.commands[step, :] < 0)
        elif step < 0:
            step = np.argmax(self.commands[:, line] < 0)
        if self.commands[step, line] >= 0 and not override:
            return False
        self.commands[step, line] = process
        # self.predicted_inventory[step:, process] -= self.inputs[process]
        # self.predicted_inventory[step + 1:, process + 1] += self.outputs[process]
        # self.predicted_balance[step:] -= self.__profile.costs[line, process]
        return True

    def cancel_production(self, step: int, line: int) -> bool:
        """
        Cancels pre-ordered production given that it did not start yet.

        Args:

            step: Step to cancel at
            line: Line to cancel at

        Returns:

            True if step >= self.current_step

        Remarks:

            - Cannot cancel a process in the past or present.
        """
        if step <= self.current_step:
            return False
        self.commands[step, line] = -1
        return True

    def step(self, accepted_sales: np.ndarray, accepted_supplies: np.ndarray) -> List[Failure]:
        """
        Override this method to modify stepping logic.

        Args:
            accepted_sales: Sales per product accepted by the factory manager
            accepted_supplies: Supplies per product accepted by the factory manager

        Returns:

        """
        self.current_step += 1
        step = self.current_step
        profile = self.__profile
        failures = []
        initial_balance = self.balance

        # buy guaranteed supplies as much as possible
        supply_money = np.sum(profile.guaranteed_supply_prices[step, :] * accepted_supplies)
        missing_money = max(0, supply_money - self.balance)
        if missing_money > 0:
            failures.append(
                Failure(is_inventory=False, line=-1, step=step, process=-1, is_guaranteed_transaction_failure=True)
            )
            for p, q in accepted_supplies:
                u = profile.guaranteed_supply_prices[step, p]
                price = u * q
                if price > self.balance:
                    if u < self.balance:
                        q = math.floor(self.balance / u)
                        self.balance -= u * q
                        self.inventory[p] += q
                    continue
                self.balance -= price
                self.inventory[p] += q
        else:
            self.balance -= supply_money
            self.inventory += accepted_supplies

            # Sell guaranteed sales as much as possible
            sale_money = np.sum(profile.guaranteed_sale_prices[step, :] * accepted_sales)
            inventory = self.inventory - accepted_sales
            failed_sales = np.nonzero(inventory < 0)[0]
            if len(failed_sales) > 0:
                for p in failed_sales:
                    failures.append(
                        Failure(is_inventory=True, line=-1, step=step, process=p - 1,
                                is_guaranteed_transaction_failure=True)
                    )
                self.balance += np.sum(sale_money[inventory > 0])
                self.inventory -= accepted_sales[inventory > 0]
            else:
                self.balance += sale_money
                self.inventory -= accepted_sales

        # do production
        for line in np.nonzero(self.commands[step, :] >= 0)[0]:
            p = self.commands[step, line]
            cost = profile.costs[line, p]
            ins, outs = self.inputs[p], self.outputs[p]
            if self.balance < cost:
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

        assert self.balance >= min(0, initial_balance)
        assert np.min(self.inventory) >= 0
        return failures


class SCML2020AWI(AgentWorldInterface):

    # --------
    # Actions
    # --------

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
            return list(range(x[0], x[1]+1))

        annotation = {
            "product": product,
            "is_buy": is_buy,
            "buyer": self.agent.id if is_buy else partner,
            "seller": partner if is_buy else self.agent.id,
        }
        return self.request_negotiation_about(issues=[
            Issue(values(quantity), name="quantity"), Issue(values(unit_price), name="unit_price"), Issue(values(time), name="time")
        ], partners=[self.agent.id, partner], req_id=unique_name(""), annotation=annotation)

    def schedule_production(self, process: int, step: int, line: int, override: bool=True) -> bool:
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
        return self._world.a2f[self.agent.id].schedule_production(process, step, line, override)

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
    def my_input_products(self) -> List[int]:
        """Returns a list of products that are inputs to at least one process the agent can run"""
        return self._world.agent_inputs[self.agent.id]

    @property
    @functools.lru_cache(maxsize=1)
    def my_output_products(self) -> List[int]:
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
        return list(itertools.chain(self.all_suppliers[_] for _ in self.my_input_products))

    @property
    @functools.lru_cache(maxsize=1)
    def my_consumers(self) -> List[str]:
        """Returns a list of IDs for all the agent's consumers (agents that can consume at least one product it may
        produce).

        Remarks:

            - If the agent have multiple output products, consumers of a specific product $p$ can be found using:
              **self.all_consumers[p]**.
        """
        return list(itertools.chain(self.all_consumers[_] for _ in self.my_output_products))


class SCML2020Agent(Agent):
    def __init__(self, **kwargs):
        super().__init__(kwargs)

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

    def sign_contract(self, contract: Contract) -> Optional[str]:
        return self.id

    def on_neg_request_rejected(self, req_id: str, by: Optional[List[str]]):
        pass

    def on_neg_request_accepted(self, req_id: str, mechanism: AgentMechanismInterface):
        pass

    @abstractmethod
    def on_contract_nullified(self, contract: Contract, compensation: int) -> None:
        """
        Called whenever a contract is nullified (because the partner is bankrupt)

        Args:

            contract: The contract being nullified
            compensation: The compensation money that is already added to the agent's wallet

        """

    @abstractmethod
    def on_failures(self, failures: List[Failure]) -> None:
        """
        Called whenever there are failures either in production or in execution of guaranteed transactions

        Args:

            failures: A list of `Failure` s.
        """

    @abstractmethod
    def confirm_guaranteed_sales(self, quantities: np.ndarray, unit_prices: np.ndarray) -> np.ndarray:
        """
        Called to confirm the amount of guaranteed sales the agent is willing to accept

        Args:

            quantities: An n_products vector giving the maximum quantity that can sold (without negotiation)
            unit_prices: An n_products vector giving the guaranteed unit prices

        Returns:

            An n_products vector specifying the quantities to be sold (up to the given `quantities` limit).
        """

    @abstractmethod
    def confirm_guaranteed_supplies(self, quantities: np.ndarray, unit_prices: np.ndarray) -> np.ndarray:
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


class DoNothingAgent(SCML2020Agent):

    def respond_to_negotiation_request(
        self,
        initiator: str,
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
    ) -> Optional[Negotiator]:
        return None

    def confirm_guaranteed_sales(self, quantities: np.ndarray, unit_prices: np.ndarray) -> np.ndarray:
        return np.zeros_like(quantities)

    def confirm_guaranteed_supplies(self, quantities: np.ndarray, unit_prices: np.ndarray) -> np.ndarray:
        return np.zeros_like(quantities)

    def on_negotiation_failure(self, partners: List[str], annotation: Dict[str, Any],
                               mechanism: AgentMechanismInterface, state: MechanismState) -> None:
        pass

    def on_negotiation_success(self, contract: Contract, mechanism: AgentMechanismInterface) -> None:
        pass

    def on_contract_signed(self, contract: Contract) -> None:
        pass

    def on_contract_cancelled(self, contract: Contract, rejectors: List[str]) -> None:
        pass

    def on_contract_executed(self, contract: Contract) -> None:
        pass

    def on_contract_breached(self, contract: Contract, breaches: List[Breach], resolution: Optional[Contract]) -> None:
        pass

    def on_contract_nullified(self, contract: Contract, compensation: int) -> None:
        pass

    def on_failures(self, failures: List[Failure]) -> None:
        pass

    def step(self):
        pass

    def init(self):
        pass


class SCM2020World(World, TimeInAgreementMixin):
    """A Supply Chain World Simulation as described for the SCML league of ANAC 2020 @ IJCAI.

        Args:

            process_inputs: An n_processes vector specifying the number of inputs from each product needed to execute
                            each process.
            process_outputs: An n_processes vector specifying the number of inputs from each product generated by
                            executing each process.
            catalog_prices: An n_products vector (i.e. n_processes+1 vector) giving the catalog price of all products
            profiles: An n_agents list of `FactoryProfile` objects specifying the private profile of the factory
                      associated with each agent.
            agent_types: An n_agents list of strings/ `SCM2020Agent` classes specifying the type of each agent
            agent_params: An n_agents dictionaries giving the parameters of each agent
            initial_balance: The initial balance in each agent's wallet. All agents will start with this same value.
            breach_penalty: The total penalty paid upon a breach will be calculated as (breach_level * breach_penalty *
                            contract_quantity * contract_unit_price).
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
            awi_type="negmas.apps.scml2020.SCM2020AWI",
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

        assert len(profiles) == len(agent_types)
        self.profiles = profiles
        self.catalog_prices = catalog_prices
        self.process_inputs = process_inputs
        self.process_outputs = process_outputs
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
            agent_params = [dict() for _ in range(len(agent_types))]
        for i, (atype, aparams) in enumerate(zip(agent_types, agent_params)):
            self.join(instantiate(atype, aparams), i)
        agents = list(self.agents.values())
        self.factories = [Factory(p, initial_balance, process_inputs, process_outputs) for p in profiles]
        self.a2f = dict(zip((_.id for _ in agents), self.factories))
        self.f2a = dict(zip((_.id for _ in self.factories), agents))

        n_processes = len(process_inputs)

        self.suppliers: List[List[str]] = [[] for _ in range(n_processes)]
        self.consumers: List[List[str]] = [[] for _ in range(n_processes)]
        self.agent_processes: Dict[str, List[str]] = defaultdict(list)
        self.agent_inputs: Dict[str, List[str]] = defaultdict(list)
        self.agent_outputs: Dict[str, List[str]] = defaultdict(list)

        for p in range(n_processes):
            for agent_id, profile in zip(self.agents.keys(), profiles):
                if np.all(np.isinf(profile.costs[:, p])):
                    continue
                self.suppliers[p + 1].append(agent_id)
                self.consumers[p].append(agent_id)
                self.agent_processes[agent_id].append(p)
                self.agent_inputs[agent_id].append(p)
                self.agent_outputs[agent_id].append(p)

    @staticmethod
    def generate(
        cls,
        depth=4,
        agent_types: List[Type[SCML2020Agent]] = DoNothingAgent,
        agent_params: List[Dict[str, Any]] = None,
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
            costs = float("inf") * np.ones(
                (n_agents, n_lines, n_processes), dtype=float
            )
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
        quantity = np.zeros((n_products, n_agents, n_steps))
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

    def post_step_stats(self):
        pass

    def pre_step_stats(self):
        pass

    def order_contracts_for_execution(
        self, contracts: Collection[Contract]
    ) -> Collection[Contract]:
        pass

    def contract_record(self, contract: Contract) -> Dict[str, Any]:
        pass

    def breach_record(self, breach: Breach) -> Dict[str, Any]:
        pass

    def start_contract_execution(self, contract: Contract) -> Set[Breach]:
        pass

    def complete_contract_execution(
        self, contract: Contract, breaches: List[Breach], resolution: Contract
    ) -> None:
        pass

    def execute_action(
        self, action: Action, agent: "Agent", callback: Callable = None
    ) -> bool:
        pass

    def get_private_state(self, agent: "Agent") -> dict:
        pass

    def simulation_step(self):
        pass

    def contract_size(self, contract: Contract) -> float:
        pass
