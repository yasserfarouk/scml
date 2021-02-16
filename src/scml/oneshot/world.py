"""
Implements the one shot version of SCML
"""
import sys
import random
import logging
import itertools
from collections import defaultdict
import pandas as pd
import copy
from dataclasses import dataclass
from typing import (
    List,
    Dict,
    Any,
    Optional,
    Iterable,
    Collection,
    Type,
    Tuple,
    Union,
    Callable,
)
import numpy as np
from matplotlib.axis import Axis
import networkx as nx
from negmas import (
    World,
    Agent,
    AgentWorldInterface,
    Issue,
    AgentMechanismInterface,
    RenegotiationRequest,
    Negotiator,
    Contract,
    MechanismState,
    Breach,
    PassThroughSAONegotiator,
    SAOController,
    NoContractExecutionMixin,
    Operations,
    BreachProcessing,
    DEFAULT_EDGE_TYPES,
    SAONegotiator,
)
from negmas.helpers import get_full_type_name, get_class, instantiate, unique_name
from ..common import integer_cut, intin, realin, make_array
from .ufun import OneShotUFun
from ..scml2020.common import (
    SYSTEM_BUYER_ID,
    SYSTEM_SELLER_ID,
    is_system_agent,
    INFINITE_COST,
)
from ..scml2020 import FinancialReport


__all__ = ["SCML2020OneShotWorld", "OneShotAWI"]


@dataclass
class OneShotState:
    """State of a one-shot agent"""

    __slots__ = [
        "input_quantity",
        "input_price",
        "output_quantity",
        "output_price",
        "exogenous_input_quantity",
        "exogenous_input_price",
        "exogenous_output_quantity",
        "exogenous_output_price",
        "storage_cost",
        "delivery_penalty",
        "utilities",
    ]

    input_quantity: int
    input_price: int
    output_quantity: int
    output_price: int
    exogenous_input_quantity: int
    exogenous_input_price: int
    exogenous_output_quantity: int
    exogenous_output_price: int
    storage_cost: float
    delivery_penalty: float
    utilities: List[float]


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


class OneShotAWI(AgentWorldInterface):
    """The agent world interface for the one-shot game"""

    # Accessing sections on the bulletin board directly
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
    def n_products(self) -> int:
        """Returns the number of products in the system"""
        return len(self._world.catalog_prices)

    @property
    def n_processes(self) -> int:
        """Returns the number of processes in the system"""
        return self.n_products - 1

    # information about the agent location in the market

    @property
    def my_input_product(self) -> int:
        """the product I need to buy"""
        return self.profile.input_product if self.profile else -10

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
        return self.all_consumers[self.level]

    # private information
    # ===================
    @property
    def n_lines(self) -> int:
        """The number of lines in the corresponding factory.
        You can read `state` to get this among other information"""
        return self.profile.n_lines if self.profile else -10

    @property
    def profile(self) -> OneShotProfile:
        """Gets the profile (static private information) associated with the agent"""
        return self.agent.profile

    def state(self) -> Any:
        return OneShotState(
            input_quantity=self._world._input_quantity[self.agent.id],
            input_price=self._world._input_price[self.agent.id],
            output_quantity=self._world._output_quantity[self.agent.id],
            output_price=self._world._output_price[self.agent.id],
            exogenous_input_quantity=self.current_exogenous_input_quantity,
            exogenous_input_price=self.current_exogenous_input_price,
            exogenous_output_quantity=self.current_exogenous_output_quantity,
            exogenous_output_price=self.current_exogenous_output_price,
            storage_cost=self.current_storage_cost,
            delivery_penalty=self.current_delivery_penalty,
            utilities=self.utilities,
        )

    @property
    def utilities(self) -> List[float]:
        """The utilities received by the agent so far"""
        return self._world._profits[self.agent.id]

    @property
    def current_breach_level(self) -> List[float]:
        """
        The breach level of the agent indicating how accurately is it matching
        inputs to outputs.
        """
        return sum(self._world._breach_levels[self.agent.id]) / (
            self._world.current_step + 1
        )

    @property
    def current_breach_prob(self) -> List[float]:
        """
        The breach probability of the agent so far (i.e. fraction of steps
        in which it did not exactly match inputs to outputs.)
        """
        return sum(self._world._breaches_of[self.agent.id]) / len(
            self._world._breaches_of[self.agent.id]
        )

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

    @property
    def current_storage_cost(self) -> float:
        """Cost of storing one unit (penalizes buying too much/ selling too little)"""
        return self._world.agent_storage_cost[self.agent.id]

    @property
    def current_delivery_penalty(self) -> float:
        """Cost of failure to deliver one unit (penalizes buying too little / selling too much)"""
        return self._world.agent_delivery_penalty[self.agent.id]


class _OneShotBaseAgent(Agent):
    """The base class of all one-shot agents"""

    def on_contract_executed(self, contract: Contract) -> None:
        pass

    def on_contract_breached(
        self, contract: Contract, breaches: List[Breach], resolution: Optional[Contract]
    ) -> None:
        pass

    def __init__(
        self,
        controller_type: Union[str, Type["OneShotAgent"]],
        controller_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.controller_type = controller_type
        self.controller_params = dict() if not controller_params else controller_params
        self.controller = None
        self.profile = None

    def make_ufun(self):
        awi: OneShotAWI = self.awi
        qin, pin = (
            awi.current_exogenous_input_quantity,
            awi.current_exogenous_input_price,
        )
        qout, pout = (
            awi.current_exogenous_output_quantity,
            awi.current_exogenous_output_price,
        )
        return OneShotUFun(
            owner=self,
            p_in=pin,
            p_out=pout,
            q_in=qin,
            q_out=qout,
            cost=awi.profile.cost,
            storage_cost=awi.current_storage_cost,
            delivery_penalty=awi.current_delivery_penalty,
        )

    def make_controller(self) -> Optional[SAOController]:
        self.controller = instantiate(
            self.controller_type,
            owner=self,
            ufun=self.make_ufun(),
            **self.controller_params,
        )
        return self.controller

    def init(self):
        self.controller = self.make_controller()
        self.controller.init()

    def step(self):
        self.controller.step()

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type_name,
            "level": self.awi.my_input_product if self.awi else None,
            "levels": [self.awi.my_input_product] if self.awi else None,
        }

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
        partner = [_ for _ in partners if _ != self.id][0]
        if not self.controller:
            self.controller = self.make_controller()
        neg = self.controller.create_negotiator(PassThroughSAONegotiator, name=partner)
        return neg

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

    @property
    def internal_state(self) -> Dict[str, Any]:
        """Returns the internal state of the agent for debugging purposes"""
        return self.controller.internal_state if self.controller else dict()

    def on_negotiation_failure(self, *args, **kwargs) -> None:
        """Called whenever a negotiation ends without agreement"""
        return (
            self.controller.on_negotiation_failure(*args, **kwargs)
            if self.controller
            else None
        )

    def on_negotiation_success(self, *args, **kwargs) -> None:
        """Called whenever a negotiation ends with agreement"""
        return (
            self.controller.on_negotiation_success(*args, **kwargs)
            if self.controller
            else None
        )

    def sign_all_contracts(self, contracts: List[Contract]) -> List[Optional[str]]:
        """Signs all contracts"""
        return [self.id] * len(contracts)


class _SystemAgent(_OneShotBaseAgent):
    """Implements an agent for handling system operations"""

    def __init__(self, *args, role, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = role
        self.name = role

    def respond_to_negotiation_request(
        self,
        initiator: str,
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
    ) -> Optional[Negotiator]:
        pass

    def step(self):
        pass

    def init(self):
        pass

    def on_negotiation_failure(
        self,
        partners: List[str],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
        state: MechanismState,
    ) -> None:
        pass

    def on_negotiation_success(
        self, contract: Contract, mechanism: AgentMechanismInterface
    ) -> None:
        pass

    def sign_all_contracts(self, contracts: List[Contract]) -> List[Optional[str]]:
        """Signs all contracts"""
        return [self.id] * len(contracts)


class SCML2020OneShotWorld(NoContractExecutionMixin, World):
    """Implements the SCML-OneShot variant of the SCM world.

    Args:
        catalog_prices: An n_products vector (i.e. n_processes+1 vector) giving the catalog price of all products
        profiles: An n_agents list of `OneShotFactoryProfile` objects specifying the private profile of the factory
                  associated with each agent.
        agent_types: An n_agents list of strings/ `OneShotAgent` classes specifying the type of each agent
        agent_params: An n_agents dictionaries giving the parameters of each agent
        catalog_quantities: The quantities in the past for which catalog_prices are the average unit prices. This
                            is used when updating the trading prices. If set to zero then the trading price will
                            follow the market price and will not use the catalog_price (except for products that are
                            never sold in the market for which the trading price will take the default value of the
                            catalog price). If set to a large value (e.g. 10000), the price at which a product is sold
                            will not affect the trading price
        financial_report_period: The number of steps between financial reports. If < 1, it is a fraction of n_steps
        exogenous_force_max: If true, exogenous contracts are forced to be signed independent of the setting of
                             `force_signing`
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
        **kwargs: Other parameters that are passed directly to `SCML2020World` constructor.

    """

    def __init__(
        self,
        # SCML2020 specific parameters
        catalog_prices: np.ndarray,
        profiles: List[OneShotProfile],
        agent_types: List[Type["OneShotAgent"]],
        agent_params: List[Dict[str, Any]] = None,
        catalog_quantities: Union[int, np.ndarray] = 50,
        # breach processing parameters
        financial_report_period=5,
        bankruptcy_limit=0.0,
        # external contracts parameters
        exogenous_contracts: Collection[OneShotExogenousContract] = tuple(),
        exogenous_dynamic: bool = False,
        exogenous_force_max: bool = False,
        # factory parameters
        initial_balance: Union[np.ndarray, Tuple[int, int], int] = 1000,
        # General SCML2020World Parameters
        compact=False,
        no_logs=False,
        n_steps=1000,
        time_limit=60 * 90,
        # mechanism params
        neg_n_steps=20,
        neg_time_limit=2 * 60,
        neg_step_time_limit=60,
        negotiation_speed=0,
        # public information
        publish_exogenous_summary=True,
        publish_trading_prices=True,
        # negotiation params,
        price_multiplier=2.0,
        # trading price parameters
        trading_price_discount=0.9,
        # simulation parameters
        signing_delay=0,
        force_signing=False,
        batch_signing=True,
        name: str = None,
        # debugging parameters
        agent_name_reveals_position: bool = True,
        agent_name_reveals_type: bool = True,
        # evaluation paramters
        **kwargs,
    ):
        self._profits: Dict[str, List[float]] = defaultdict(list)
        self._breach_levels: Dict[str, List[float]] = defaultdict(list)
        self._breaches_of: Dict[str, List[bool]] = defaultdict(list)
        self.trading_price_discount = trading_price_discount
        self.catalog_quantities = catalog_quantities
        self.publish_exogenous_summary = publish_exogenous_summary
        self.price_multiplier = price_multiplier
        self.publish_trading_prices = publish_trading_prices
        self.agent_storage_cost: Dict[str, List[float]] = dict()
        self.agent_delivery_penalty: Dict[str, List[float]] = dict()
        kwargs["log_to_file"] = not no_logs
        if compact:
            kwargs["log_screen_level"] = logging.CRITICAL
            kwargs["log_file_level"] = logging.ERROR
            kwargs["log_negotiations"] = False
            kwargs["log_ufuns"] = False
            # kwargs["save_mechanism_state_in_contract"] = False
            kwargs["save_cancelled_contracts"] = False
            kwargs["save_resolved_breaches"] = False
            kwargs["save_negotiations"] = False
        self.compact = compact
        if negotiation_speed == 0:
            negotiation_speed = neg_n_steps + 1
        mechanisms = kwargs.pop("mechanisms", {})
        super().__init__(
            bulletin_board=None,
            breach_processing=BreachProcessing.NONE,
            awi_type="scml.oneshot.OneShotAWI",
            mechanisms={
                "negmas.sao.SAOMechanism": mechanisms.get(
                    "negmas.sao.SAOMechanism",
                    dict(
                        end_on_no_response=True,
                        avoid_ultimatum=True,
                        dynamic_entry=False,
                        max_wait=len(agent_types),
                    ),
                )
            },
            default_signing_delay=signing_delay,
            n_steps=n_steps,
            time_limit=time_limit,
            negotiation_speed=negotiation_speed,
            neg_n_steps=neg_n_steps,
            neg_time_limit=neg_time_limit,
            neg_step_time_limit=neg_step_time_limit,
            force_signing=force_signing,
            batch_signing=batch_signing,
            negotiation_quota_per_step=float("inf"),
            negotiation_quota_per_simulation=float("inf"),
            no_logs=no_logs,
            operations=(
                Operations.StatsUpdate,
                Operations.SimulationStep,
                Operations.Negotiations,
                Operations.ContractSigning,
                Operations.ContractExecution,
                Operations.AgentSteps,
                Operations.SimulationStep,
                Operations.StatsUpdate,
            ),
            name=name,
            **kwargs,
        )
        self.bulletin_board.record(
            "settings", financial_report_period, "financial_report_period"
        )
        self.bulletin_board.record(
            "settings", exogenous_force_max, "exogenous_force_max"
        )
        # self.bulletin_board.record("settings", storage_cost, "ufun_storage_cost")
        # self.bulletin_board.record(
        #     "settings", delivery_penalty, "ufun_delivery_penalty"
        # )
        self.bulletin_board.record("settings", True, "has_exogenous_contracts")
        self.bulletin_board.record("settings", bankruptcy_limit, "bankruptcy_limit")

        if self.info is None:
            self.info = {}
        n_products = len(catalog_prices)
        n_processes = n_products - 1
        process_inputs = [[i] for i in range(n_processes)]
        process_outputs = [[i + 1] for i in range(n_processes)]
        self.exogenous_dynamic = exogenous_dynamic
        self.info.update(
            process_inputs=process_inputs,
            process_outputs=process_outputs,
            catalog_prices=catalog_prices,
            agent_types_final=[get_full_type_name(_) for _ in agent_types],
            agent_params_final=agent_params,
            initial_balance_final=initial_balance,
            bankruptcy_limit=bankruptcy_limit,
            financial_report_period=financial_report_period,
            exogenous_force_max=exogenous_force_max,
            compact=compact,
            no_logs=no_logs,
            n_steps=n_steps,
            time_limit=time_limit,
            neg_n_steps=neg_n_steps,
            neg_time_limit=neg_time_limit,
            neg_step_time_limit=neg_step_time_limit,
            negotiation_speed=negotiation_speed,
            signing_delay=signing_delay,
            agent_name_reveals_position=agent_name_reveals_position,
            agent_name_reveals_type=agent_name_reveals_type,
            # storage_cost=storage_cost,
            # delivery_penalty=delivery_penalty,
            exogenous_dynamic=exogenous_dynamic,
            publish_exogenous_summary=publish_exogenous_summary,
            publish_trading_prices=publish_trading_prices,
            selected_price_multiplier=price_multiplier,
        )
        self.bulletin_board.add_section("reports_agent")
        self.bulletin_board.add_section("reports_time")
        if self.publish_exogenous_summary:
            self.bulletin_board.add_section("exogenous_contracts_summary")
        if self.publish_trading_prices:
            self.bulletin_board.add_section("trading_prices")

        initial_balance = make_array(initial_balance, len(profiles), dtype=int)
        self.bankruptcy_limit = (
            -bankruptcy_limit
            if isinstance(bankruptcy_limit, int)
            else -int(0.5 + bankruptcy_limit * initial_balance.mean())
        )

        if not isinstance(agent_types, Iterable):
            agent_types = [agent_types] * len(profiles)

        assert len(profiles) == len(agent_types)
        self.profiles = profiles
        self.catalog_prices = catalog_prices
        self.process_inputs = process_inputs
        self.process_outputs = process_outputs
        self.n_products = len(catalog_prices)
        self.n_processes = len(process_inputs)
        self.exogenous_force_max = exogenous_force_max
        self.financial_reports_period = (
            financial_report_period
            if financial_report_period >= 1
            else int(0.5 + financial_report_period * n_steps)
        )
        # breakpoint()
        agent_types = [get_class(_) for _ in agent_types]
        self.controller_types = [
            get_class(_["controller_type"])._type_name()
            if _["controller_type"]
            else "system_agent"
            for _ in agent_params
        ]
        assert (
            self.n_products == self.n_processes + 1
        ), f"{self.n_products, self.n_processes}"

        n_agents = len(profiles)
        if agent_name_reveals_position or agent_name_reveals_type:
            default_names = [f"{_:02}" for _ in range(n_agents)]
        else:
            default_names = [unique_name("", add_time=False) for _ in range(n_agents)]
        if agent_name_reveals_type:
            for i, at in enumerate(self.controller_types):
                default_names[i] += f"{get_class(at).__name__[:3]}"
        agent_levels = [p.level for p in profiles]
        if agent_name_reveals_position:
            for i, l in enumerate(agent_levels):
                default_names[i] += f"@{l:01}"
        if agent_params is None:
            agent_params = [dict(name=name) for i, name in enumerate(default_names)]
        elif isinstance(agent_params, dict):
            a = copy.copy(agent_params)
            agent_params = []
            for i, name in enumerate(default_names):
                b = copy.deepcopy(a)
                b["name"] = name
                agent_params.append(b)
        elif len(agent_params) == 1:
            a = copy.copy(agent_params[0])
            agent_params = []
            for i, _ in enumerate(default_names):
                b = copy.deepcopy(a)
                b["name"] = name
                agent_params.append(b)
        else:
            if agent_name_reveals_type or agent_name_reveals_position:
                for i, (ns, ps) in enumerate(zip(default_names, agent_params)):
                    agent_params[i] = dict(**ps)
                    agent_params[i]["name"] = ns
        agent_types += [_SystemAgent, _SystemAgent]
        agent_params += [
            {"role": SYSTEM_SELLER_ID, "controller_type": None},
            {"role": SYSTEM_BUYER_ID, "controller_type": None},
        ]
        profiles.append(
            OneShotProfile(
                cost=INFINITE_COST,
                input_product=-1,
                n_lines=0,
                storage_cost_mean=0.0,
                delivery_penalty_mean=0.0,
                storage_cost_dev=0.0,
                delivery_penalty_dev=0.0,
            )
        )
        profiles.append(
            OneShotProfile(
                cost=INFINITE_COST,
                input_product=n_processes,
                n_lines=0,
                storage_cost_mean=0.0,
                delivery_penalty_mean=0.0,
                storage_cost_dev=0.0,
                delivery_penalty_dev=0.0,
            )
        )
        initial_balance = initial_balance.tolist() + [
            sys.maxsize // 4,
            sys.maxsize // 4,
        ]
        agents = []
        for i, (atype, aparams) in enumerate(zip(agent_types, agent_params)):
            a = instantiate(atype, **aparams)
            a.id = a.name
            self.join(a, i)
            agents.append(a)
        self.agent_types = [
            get_class(_["controller_type"])._type_name()
            if _["controller_type"]
            else "system_agent"
            for _ in agent_params
        ]
        self.agent_params = [
            {k: v for k, v in _.items() if k != "name" and k != "controller_type"}
            for _ in agent_params
        ]
        self.agent_unique_types = [
            f"{t}{hash(str(p))}" if len(p) > 0 else t
            for t, p in zip(self.agent_types, self.agent_params)
        ]

        self.agent_n_contracts = dict(zip((_.id for _ in agents), itertools.repeat(0)))

        self.suppliers: List[List[str]] = [[] for _ in range(n_products)]
        self.consumers: List[List[str]] = [[] for _ in range(n_products)]
        self.agent_processes: Dict[str, List[int]] = defaultdict(list)
        self.agent_inputs: Dict[str, List[int]] = defaultdict(list)
        self.agent_outputs: Dict[str, List[int]] = defaultdict(list)
        self.agent_consumers: Dict[str, List[str]] = defaultdict(list)
        self.agent_suppliers: Dict[str, List[str]] = defaultdict(list)
        self.agent_profiles: Dict[str, OneShotProfile] = defaultdict(list)

        self.consumers[n_products - 1].append(SYSTEM_BUYER_ID)
        self.agent_processes[SYSTEM_BUYER_ID] = []
        self.agent_inputs[SYSTEM_BUYER_ID] = [n_products - 1]
        self.agent_outputs[SYSTEM_BUYER_ID] = []
        self.suppliers[0].append(SYSTEM_SELLER_ID)
        self.agent_processes[SYSTEM_SELLER_ID] = []
        self.agent_inputs[SYSTEM_SELLER_ID] = []
        self.agent_outputs[SYSTEM_SELLER_ID] = [0]

        for agent_id, profile in zip(self.agents.keys(), profiles):
            # if is_system_agent(agent_id):
            #     continue
            if profile.cost == INFINITE_COST:
                continue
            p = profile.level
            self.suppliers[p + 1].append(agent_id)
            self.consumers[p].append(agent_id)
            self.agent_processes[agent_id].append(p)
            self.agent_inputs[agent_id].append(p)
            self.agent_outputs[agent_id].append(p + 1)
            self.agent_profiles[agent_id] = profile
            self.agents[agent_id].profile = profile

        for aid, agent in self.agents.items():
            if is_system_agent(aid):
                continue
            profile: OneShotProfile = agent.profile
            self.agent_storage_cost[aid] = (
                np.random.randn(self.n_steps) * profile.storage_cost_dev
                + profile.storage_cost_mean
            )
            self.agent_delivery_penalty[aid] = (
                np.random.randn(self.n_steps) * profile.delivery_penalty_dev
                + profile.delivery_penalty_mean
            )

        for p in range(n_products):
            for a in self.suppliers[p]:
                self.agent_consumers[a] = self.consumers[p]
            for a in self.consumers[p]:
                self.agent_suppliers[a] = self.suppliers[p]

        self.agent_processes = {k: np.array(v) for k, v in self.agent_processes.items()}
        self.agent_inputs = {k: np.array(v) for k, v in self.agent_inputs.items()}
        self.agent_outputs = {k: np.array(v) for k, v in self.agent_outputs.items()}
        assert all(
            len(v) == 1 or is_system_agent(self.agents[k].id)
            for k, v in self.agent_outputs.items()
        ), f"Not all agent outputs are singular:\n{self.agent_outputs}"
        assert all(
            len(v) == 1 or is_system_agent(self.agents[k].id)
            for k, v in self.agent_inputs.items()
        ), f"Not all agent inputs are singular:\n{self.agent_outputs}"
        assert all(
            is_system_agent(k) or self.agent_inputs[k][0] == self.agent_outputs[k] - 1
            for k in self.agent_inputs.keys()
        ), f"Some agents have outputs != input+1\n{self.agent_outputs}\n{self.agent_inputs}"
        self.is_bankrupt: Dict[str, bool] = dict(
            zip(self.agents.keys(), itertools.repeat(False))
        )
        self.exogenous_contracts: Dict[int : List[Contract]] = defaultdict(list)
        for c in exogenous_contracts:
            seller_id = agents[c.seller].id if c.seller >= 0 else SYSTEM_SELLER_ID
            buyer_id = agents[c.buyer].id if c.buyer >= 0 else SYSTEM_BUYER_ID
            contract = Contract(
                agreement={
                    "time": c.time,
                    "quantity": c.quantity,
                    "unit_price": c.unit_price,
                },
                partners=[buyer_id, seller_id],
                issues=[],
                signatures=dict(),
                signed_at=-1,
                to_be_signed_at=c.time,
                annotation={
                    "seller": seller_id,
                    "buyer": buyer_id,
                    "caller": SYSTEM_SELLER_ID
                    if seller_id == SYSTEM_SELLER_ID
                    else SYSTEM_BUYER_ID,
                    "is_buy": True,
                    "product": c.product,
                },
            )
            self.exogenous_contracts[c.time].append(contract)
        self._traded_quantity = np.ones(n_products) * self.catalog_quantities
        self._real_price = np.nan * np.ones((n_products, n_steps + 1))
        self._sold_quantity = np.zeros((n_products, n_steps + 1), dtype=int)
        self._real_price[:, 0] = self.catalog_prices
        self._trading_price = np.tile(
            self._real_price[:, 0].reshape((n_products, 1)), (1, n_steps + 1)
        )
        self._betas = np.ones(n_steps + 1)
        self._betas[1] = self.trading_price_discount
        self._betas[1:] = np.cumprod(self._betas[1:])
        self._betas_sum = self.catalog_quantities * np.ones((n_products, n_steps + 1))
        if self.publish_trading_prices:
            self.bulletin_board.record("trading_prices", self._trading_price[:, 1])
        # temporary variables for calculating scores
        self._input_quantity = defaultdict(int)
        self._input_price = defaultdict(int)
        self._output_quantity = defaultdict(int)
        self._output_price = defaultdict(int)
        self.exogenous_qout = defaultdict(int)
        self.exogenous_qin = defaultdict(int)
        self.exogenous_pout = defaultdict(int)
        self.exogenous_pin = defaultdict(int)
        self.exogenous_contracts_summary = None

        self.initial_balances = dict(
            zip(self.agents.keys(), initial_balance)
        )

    @classmethod
    def generate(
        cls,
        agent_types: List[Union[str, Type["OneShotAgent"]]],
        agent_params: List[Dict[str, Any]] = None,
        n_steps: Union[Tuple[int, int], int] = (50, 200),
        n_processes: Union[Tuple[int, int], int] = 2,
        n_lines: Union[np.ndarray, Tuple[int, int], int] = 100,
        n_agents_per_process: Union[np.ndarray, Tuple[int, int], int] = 3,
        process_inputs: Union[np.ndarray, Tuple[int, int], int] = 1,
        process_outputs: Union[np.ndarray, Tuple[int, int], int] = 1,
        production_costs: Union[np.ndarray, Tuple[int, int], int] = (1, 10),
        profit_means: Union[np.ndarray, Tuple[float, float], float] = (0.1, 0.2),
        profit_stddevs: Union[np.ndarray, Tuple[float, float], float] = 0.05,
        max_productivity: Union[np.ndarray, Tuple[float, float], float] = 1.0,
        cost_increases_with_level=True,
        equal_exogenous_supply=False,
        equal_exogenous_sales=False,
        exogenous_control: Union[Tuple[float, float], float] = -1,
        force_signing=True,
        profit_basis=np.mean,
        storage_cost: Union[np.ndarray, Tuple[float, float], float] = (0.0, 0.2),
        delivery_penalty: Union[np.ndarray, Tuple[float, float], float] = (0.0, 0.2),
        storage_cost_dev: Union[np.ndarray, Tuple[float, float], float] = (0.01, 0.02),
        delivery_penalty_dev: Union[np.ndarray, Tuple[float, float], float] = (
            0.01,
            0.02,
        ),
        exogenous_price_dev: Union[np.ndarray, Tuple[float, float], float] = (0.1, 0.2),
        price_multiplier: Union[np.ndarray, Tuple[float, float], float] = (1.5, 2.0),
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generates the configuration for a world

        Args:

            agent_types: All agent types
            agent_params: Agent parameters used to initialize them
            n_steps: Number of simulation steps
            n_processes: Number of processes in the production chain
            n_lines: Number of lines per factory
            process_inputs: Number of input units per process
            process_outputs: Number of output units per process
            production_costs: Production cost per factory
            profit_means: Mean profitability per production level (i.e. process).
            profit_stddevs:  Std. Dev. of the profitability of every level (i.e. process).
            max_productivity:  Maximum possible productivity per level (i.e. process).
            initial_balance: The initial balance of all agents
            n_agents_per_process: Number of agents per process
            cost_increases_with_level: If true, production cost will be higher for processes nearer to the final
                                       product.
            profit_basis: The statistic used when controlling catalog prices by profit arguments. It can be np.mean,
                          np.median, np.min, np.max or any Callable[[list[float]], float] and is used to summarize
                          production costs at every level.
            horizon: The horizon used for revealing external supply/sales as a fraction of n_steps
            equal_exogenous_supply: If true, external supply will be distributed equally among all agents in the first
                                   layer
            equal_exogenous_sales: If true, external sales will be distributed equally among all agents in the last
                                   layer
            force_signing: Whether to force contract signatures (exogenous contracts are treated in the same way).
            exogenous_control: How much control does the agent have over exogenous contract signing. Only effective if
                               force_signing is False and use_exogenous_contracts is True
            storage_cost: A range to sample mean-storage costs for all factories from
            delivery_penalty: A range to sample mean-delivery penalty for all factories from
            storage_cost_dev: A range to sample std. dev of storage costs for all factories from
            delivery_penalty_dev: A range to sample std. dev of delivery penalty for all factories from
            exogenous_price_dev: The standard deviation of exogenous contract prices relative to the mean price
            price_multiplier: A value to multiply with trading/catalog price to get the upper limit on prices for all negotiations
            **kwargs:

        Returns:

            world configuration as a Dict[str, Any]. A world can be generated from this dict by calling SCML2020World(**d)

        Remarks:

            - Most parameters (i.e. `process_inputs` , `process_outputs` , `n_agents_per_process` , `costs` ) can
              take a single value, a tuple of two values, or a list of values.
              If it has a single value, it is repeated for all processes/factories as appropriate. If it is a
              tuple of two numbers $(i, j)$, each process will take a number sampled from a uniform distribution
              supported on $[i, j]$ inclusive. If it is a list of values, of the length `n_processes` , it is used as
              it is otherwise, it is used to sample values for each process.

        """
        info = dict(
            n_steps=n_steps,
            n_processes=n_processes,
            n_lines=n_lines,
            force_signing=force_signing,
            n_agents_per_process=n_agents_per_process,
            process_inputs=process_inputs,
            process_outputs=process_outputs,
            production_costs=production_costs,
            profit_means=profit_means,
            profit_stddevs=profit_stddevs,
            max_productivity=max_productivity,
            cost_increases_with_level=cost_increases_with_level,
            equal_exogenous_sales=equal_exogenous_sales,
            equal_exogenous_supply=equal_exogenous_supply,
            price_multiplier=price_multiplier,
            exogenous_price_dev=exogenous_price_dev,
            cash_availability=float("inf"),
            profit_basis="min"
            if profit_basis == np.min
            else "mean"
            if profit_basis == np.mean
            else "max"
            if profit_basis == np.max
            else "median"
            if profit_basis == np.median
            else "unknown",
        )
        exogenous_price_dev = realin(exogenous_price_dev)
        price_multiplier = realin(price_multiplier)
        n_processes = intin(n_processes)
        n_steps = intin(n_steps)
        exogenous_control = realin(exogenous_control)
        np.errstate(divide="ignore")
        n_startup = n_processes
        if n_steps <= n_startup:
            raise ValueError(
                f"Cannot generate a world with n_steps <= n_processes: {n_steps} <= {n_startup}"
            )

        process_inputs = make_array(process_inputs, n_processes, dtype=int)
        process_outputs = make_array(process_outputs, n_processes, dtype=int)
        n_agents_per_process = make_array(n_agents_per_process, n_processes, dtype=int)
        profit_means = make_array(profit_means, n_processes, dtype=float)
        profit_stddevs = make_array(profit_stddevs, n_processes, dtype=float)
        max_productivity = make_array(
            max_productivity, n_processes * n_steps, dtype=float
        ).reshape((n_processes, n_steps))
        n_agents = n_agents_per_process.sum()
        assert n_agents >= n_processes
        n_products = n_processes + 1
        production_costs = make_array(production_costs, n_agents, dtype=int)
        if not isinstance(agent_types, Iterable):
            agent_types = [agent_types] * n_agents
            if agent_params is None:
                agent_params = dict()
            if isinstance(agent_params, dict):
                agent_params = [copy.copy(agent_params) for _ in range(n_agents)]
            else:
                assert len(agent_params) == 1
                agent_params = [copy.copy(agent_params[0]) for _ in range(n_agents)]
        elif len(agent_types) != n_agents:
            if agent_params is None:
                agent_params = [dict()] * len(agent_types)
            if isinstance(agent_params, dict):
                agent_params = [
                    copy.copy(agent_params) for _ in range(len(agent_types))
                ]
            assert len(agent_types) == len(agent_params)
            tp = random.choices(list(range(len(agent_types))), k=n_agents)
            agent_types = [copy.copy(agent_types[_]) for _ in tp]
            agent_params = [copy.copy(agent_params[_]) for _ in tp]
        else:
            if agent_params is None:
                agent_params = [dict()] * len(agent_types)
            if isinstance(agent_params, dict):
                agent_params = [
                    copy.copy(agent_params) for _ in range(len(agent_types))
                ]
            agent_types = list(agent_types)
            agent_params = list(agent_params)
            assert len(agent_types) == len(agent_params)

        for t, p in zip(agent_types, agent_params):
            p["controller_type"] = t
        agent_types = [_OneShotBaseAgent] * len(agent_types)
        # generate production costs making sure that every agent can do exactly one process
        n_agents_cumsum = n_agents_per_process.cumsum().tolist()
        first_agent = [0] + n_agents_cumsum[:-1]
        last_agent = n_agents_cumsum[:-1] + [n_agents]
        costs = INFINITE_COST * np.ones((n_agents, n_lines, n_processes), dtype=int)
        for p, (f, l) in enumerate(zip(first_agent, last_agent)):
            costs[f:l, :, p] = production_costs[f:l].reshape((l - f), 1)

        process_of_agent = np.empty(n_agents, dtype=int)
        for i, (f, l) in enumerate(zip(first_agent, last_agent)):
            process_of_agent[f:l] = i
            if cost_increases_with_level:
                production_costs[f:l] = np.round(
                    production_costs[f:l] * (i + 1)  # math.sqrt(i + 1)
                ).astype(int)

        # generate external contract amounts (controlled by productivity):

        # - generate total amount of input to the market (it will end up being an n_products list of n_steps vectors)
        quantities = [
            np.round(n_lines * n_agents_per_process[0] * max_productivity[0, :]).astype(
                int
            )
        ]
        # - make sure there is a cool-down period at the end in which no more input is added that cannot be converted
        #   into final products in time
        quantities[0][-n_startup:] = 0
        # - for each level, find the amount of the output product that can be produced given the input amount and
        #   productivity
        for p in range(n_processes):
            agents = n_agents_per_process[p]
            lines = n_lines * agents
            quantities.append(
                np.minimum(
                    (quantities[-1] // process_outputs[p]) * process_inputs[p],
                    (
                        np.round(lines * max_productivity[p, :]).astype(int)
                        // process_inputs[p]
                    )
                    * process_outputs[p],
                )
            )
            # * shift quantities one step to account for the one step needed to move the produce to the next level. This
            #   step results from having production happen after contract execution.
            quantities[-1][1:] = quantities[-1][:-1]
            quantities[-1][0] = 0
            assert quantities[-1][-1] == 0 or p >= n_startup - 1
            assert quantities[-1][0] == 0
            # assert np.sum(quantities[-1] == 0) >= n_startup
        # - divide the quantity at every level between factories
        if equal_exogenous_supply:
            exogenous_supplies = np.maximum(
                1, np.round(quantities[0] / n_agents_per_process[0]).astype(int)
            ).tolist()
            exogenous_supplies = [
                np.array([exogenous_supplies[p]] * n_agents_per_process[p])
                for p in range(n_processes)
            ]
        else:
            exogenous_supplies = []
            for s in range(n_steps):
                exogenous_supplies.append(
                    integer_cut(quantities[0][s], n_agents_per_process[0], 0)
                )
                assert sum(exogenous_supplies[-1]) == quantities[0][s]
        if equal_exogenous_sales:
            exogenous_sales = np.maximum(
                1, np.round(quantities[-1] / n_agents_per_process[-1]).astype(int)
            ).tolist()
            exogenous_sales = [
                np.array([exogenous_sales[p]] * n_agents_per_process[p])
                for p in range(n_processes)
            ]
        else:
            exogenous_sales = []
            for s in range(n_steps):
                exogenous_sales.append(
                    integer_cut(quantities[-1][s], n_agents_per_process[-1], 0)
                )
                assert sum(exogenous_sales[-1]) == quantities[-1][s]
        # - now exogenous_supplies and exogenous_sales are both n_steps lists
        #   of n_agents_per_process[p] vectors (jagged)

        # assign prices to the quantities given the profits
        catalog_prices = np.zeros(n_products, dtype=int)
        catalog_prices[0] = 10
        supply_prices = np.zeros((n_agents_per_process[0], n_steps), dtype=int)
        supply_prices[:, :] = catalog_prices[0]
        sale_prices = np.zeros((n_agents_per_process[-1], n_steps), dtype=int)

        manufacturing_costs = np.zeros((n_processes, n_steps), dtype=int)
        for p in range(n_processes):
            manufacturing_costs[p, :] = profit_basis(
                costs[first_agent[p] : last_agent[p], :, p]
            )
            manufacturing_costs[p, :p] = 0
            manufacturing_costs[p, p - n_startup :] = 0

        profits = np.zeros((n_processes, n_steps))
        for p in range(n_processes):
            profits[p, :] = np.random.randn() * profit_stddevs[p] + profit_means[p]

        input_costs = np.zeros((n_processes, n_steps), dtype=int)
        for step in range(n_steps):
            input_costs[0, step] = np.sum(
                exogenous_supplies[step] * supply_prices[:, step][:]
            )

        input_quantity = np.zeros((n_processes, n_steps), dtype=int)
        input_quantity[0, :] = quantities[0]

        active_lines = np.hstack(
            [(n_lines * n_agents_per_process).reshape((n_processes, 1))] * n_steps
        )
        assert active_lines.shape == (n_processes, n_steps)
        active_lines[0, :] = input_quantity[0, :] // process_inputs[0]

        output_quantity = np.zeros((n_processes, n_steps), dtype=int)
        output_quantity[0, :] = active_lines[0, :] * process_outputs[0]

        manufacturing_costs[0, :-n_startup] *= active_lines[0, :-n_startup]

        total_costs = input_costs + manufacturing_costs

        output_total_prices = np.ceil(total_costs * (1 + profits)).astype(int)

        for p in range(1, n_processes):
            input_costs[p, p:] = output_total_prices[p - 1, p - 1 : -1]
            input_quantity[p, p:] = output_quantity[p - 1, p - 1 : -1]
            active_lines[p, :] = input_quantity[p, :] // process_inputs[p]
            output_quantity[p, :] = active_lines[p, :] * process_outputs[p]
            manufacturing_costs[p, p : p - n_startup] *= active_lines[
                p, p : p - n_startup
            ]
            total_costs[p, :] = input_costs[p, :] + manufacturing_costs[p, :]
            output_total_prices[p, :] = np.ceil(
                total_costs[p, :] * (1 + profits[p, :])
            ).astype(int)
        sale_prices[:, n_startup:] = np.ceil(
            output_total_prices[-1, n_startup - 1 : -1]
            / output_quantity[-1, n_startup - 1 : -1]
        ).astype(int)
        product_prices = np.zeros((n_products, n_steps))
        product_prices[0, :-n_startup] = catalog_prices[0]
        product_prices[1:, 1:] = np.ceil(
            np.divide(
                output_total_prices.astype(float),
                output_quantity.astype(float),
                out=np.zeros_like(output_total_prices, dtype=float),
                where=output_quantity != 0,
            )
        ).astype(int)[:, :-1]
        catalog_prices = np.ceil(
            [
                profit_basis(product_prices[p, p : p + n_steps - n_startup])
                for p in range(n_products)
            ]
        ).astype(int)
        profile_info: List[
            Tuple[OneShotProfile, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = []
        nxt = 0
        for l in range(n_processes):
            for a in range(n_agents_per_process[l]):
                esales = np.zeros((n_steps, n_products), dtype=int)
                esupplies = np.zeros((n_steps, n_products), dtype=int)
                esale_prices = np.zeros((n_steps, n_products), dtype=int)
                esupply_prices = np.zeros((n_steps, n_products), dtype=int)
                if l == 0:
                    esupplies[:, 0] = [exogenous_supplies[s][a] for s in range(n_steps)]
                    esupply_prices[:, 0] = supply_prices[a, :]
                if l == n_processes - 1:
                    esales[:, -1] = [exogenous_sales[s][a] for s in range(n_steps)]
                    esale_prices[:, -1] = sale_prices[a, :]
                profile_info.append(
                    (
                        OneShotProfile(
                            cost=[_ for _ in costs[nxt][0] if _ != INFINITE_COST][0],
                            input_product=l,
                            n_lines=n_lines,
                            storage_cost_mean=realin(storage_cost),
                            delivery_penalty_mean=realin(delivery_penalty),
                            storage_cost_dev=realin(storage_cost_dev),
                            delivery_penalty_dev=realin(delivery_penalty_dev),
                        ),
                        esales,
                        esale_prices,
                        esupplies,
                        esupply_prices,
                    )
                )
                nxt += 1
        max_income = (
            output_quantity * catalog_prices[1:].reshape((n_processes, 1)) - total_costs
        )

        assert nxt == n_agents
        info.update(
            dict(
                product_prices=product_prices,
                active_lines=active_lines,
                input_quantities=input_quantity,
                output_quantities=output_quantity,
                expected_productivity=float(np.sum(active_lines))
                / np.sum(n_lines * n_steps * n_agents_per_process),
                expected_n_products=np.sum(active_lines, axis=-1),
                expected_income=max_income,
                expected_welfare=float(np.sum(max_income)),
                expected_income_per_step=max_income.sum(axis=0),
                expected_income_per_process=max_income.sum(axis=-1),
                expected_mean_profit=float(np.sum(max_income)),
                expected_profit_sum=float(n_agents * np.sum(max_income)),
            )
        )

        exogenous = []
        for (
            indx,
            (profile, esales, esale_prices, esupplies, esupply_prices),
        ) in enumerate(profile_info):
            input_product = process_of_agent[indx]
            for step, (sale, price) in enumerate(
                zip(esales[:, input_product + 1], esale_prices[:, input_product + 1])
            ):
                if sale == 0:
                    continue
                thisprice = int(
                    0.5 + price + np.random.randn() * exogenous_price_dev * price
                )
                if force_signing or exogenous_control <= 0.0:
                    exogenous.append(
                        OneShotExogenousContract(
                            product=input_product + 1,
                            quantity=sale,
                            unit_price=thisprice,
                            time=step,
                            revelation_time=step,
                            seller=indx,
                            buyer=-1,
                        )
                    )
                else:
                    n_contracts = int(1 + exogenous_control * (sale - 1))
                    per_contract = integer_cut(sale, n_contracts, 0)
                    for q in per_contract:
                        if q == 0:
                            continue
                        thisprice = int(
                            0.5
                            + price
                            + np.random.randn() * exogenous_price_dev * price
                        )
                        exogenous.append(
                            OneShotExogenousContract(
                                product=input_product + 1,
                                quantity=q,
                                unit_price=thisprice,
                                time=step,
                                revelation_time=step,
                                seller=indx,
                                buyer=-1,
                            )
                        )
            for step, (supply, price) in enumerate(
                zip(esupplies[:, input_product], esupply_prices[:, input_product])
            ):
                if supply == 0:
                    continue
                thisprice = int(
                    0.5 + price + np.random.randn() * exogenous_price_dev * price
                )
                if force_signing or exogenous_control <= 0.0:
                    exogenous.append(
                        OneShotExogenousContract(
                            product=input_product,
                            quantity=supply,
                            unit_price=thisprice,
                            time=step,
                            revelation_time=step,
                            seller=-1,
                            buyer=indx,
                        )
                    )
                else:
                    n_contracts = int(1 + exogenous_control * (supply - 1))
                    per_contract = integer_cut(supply, n_contracts, 0)
                    for q in per_contract:
                        if q == 0:
                            continue
                        thisprice = int(
                            0.5
                            + price
                            + np.random.randn() * exogenous_price_dev * price
                        )
                        exogenous.append(
                            OneShotExogenousContract(
                                product=input_product,
                                quantity=q,
                                unit_price=thisprice,
                                time=step,
                                revelation_time=step,
                                seller=-1,
                                buyer=indx,
                            )
                        )
        return dict(
            # process_inputs=process_inputs,
            # process_outputs=process_outputs,
            catalog_prices=catalog_prices,
            profiles=[_[0] for _ in profile_info],
            exogenous_contracts=exogenous,
            agent_types=agent_types,
            agent_params=agent_params,
            n_steps=n_steps,
            info=info,
            price_multiplier=price_multiplier,
            force_signing=force_signing,
            **kwargs,
        )

    def add_financial_report(
        self, agent: _OneShotBaseAgent, reports_agent, reports_time
    ) -> None:
        """
        Records a financial report for the given agent in the agent indexed reports and time indexed reports

        Args:
            agent: The agent
            reports_agent: A dictionary of financial reports indexed by agent id
            reports_time: A dictionary of financial reports indexed by time

        Returns:

        """
        current_balance = sum(self._profits[agent.id]) + self.initial_balances[agent.id]
        self.is_bankrupt[agent.id] = (
            current_balance < self.bankruptcy_limit or self.is_bankrupt[agent.id]
        )
        report = FinancialReport(
            agent_id=agent.id,
            step=self.current_step,
            cash=current_balance,
            assets=0,
            breach_prob=len([_ for _ in self._breaches_of[agent.id] if _])
            / len(self._breaches_of[agent.id]),
            breach_level=sum(self._breach_levels[agent.id])
            / len(self._breach_levels[agent.id]),
            is_bankrupt=self.is_bankrupt[agent.id],
            agent_name=agent.name,
        )
        repstr = str(report).replace("\n", " ")
        self.logdebug(f"{agent.name}: {repstr}")
        if reports_agent.get(agent.id, None) is None:
            reports_agent[agent.id] = {}
        reports_agent[agent.id][self.current_step] = report
        if reports_time.get(str(self.current_step), None) is None:
            reports_time[str(self.current_step)] = {}
        reports_time[str(self.current_step)][agent.id] = report

    def simulation_step(self, stage):
        s = self.current_step
        self.exogenous_qout = defaultdict(int)
        self.exogenous_qin = defaultdict(int)
        self.exogenous_pout = defaultdict(int)
        self.exogenous_pin = defaultdict(int)

        if stage == 0:

            # Register exogenous contracts as concluded
            # -----------------------------------------
            for contract in self.exogenous_contracts[s]:
                seller = contract.annotation["seller"]
                buyer = contract.annotation["buyer"]
                quantity = contract.agreement["quantity"]
                unit_price = contract.agreement["unit_price"]
                self.exogenous_qout[seller] += quantity
                self.exogenous_pout[seller] += quantity * unit_price
                self.exogenous_qin[buyer] += quantity
                self.exogenous_pin[buyer] += quantity * unit_price
                self.on_contract_concluded(
                    contract, to_be_signed_at=contract.to_be_signed_at
                )
                if self.exogenous_force_max:
                    contract.signatures = dict(
                        zip(contract.partners, contract.partners)
                    )
                else:
                    if SYSTEM_SELLER_ID in contract.partners:
                        contract.signatures[SYSTEM_SELLER_ID] = SYSTEM_SELLER_ID
                    else:
                        contract.signatures[SYSTEM_BUYER_ID] = SYSTEM_BUYER_ID
            if self.exogenous_dynamic:
                raise NotImplementedError("Exogenous-dynamic is not yet implemented")

            # publish public information
            # --------------------------
            if self.publish_trading_prices:
                self.bulletin_board.record(
                    "trading_prices", value=self.trading_prices, key=self.current_step,
                )
            if self.publish_exogenous_summary:
                q, p = np.zeros(self.n_products), np.zeros(self.n_products)
                for contract in self.exogenous_contracts[s]:
                    product = contract.annotation["product"]
                    quantity, unit_price = (
                        contract.agreement["quantity"],
                        contract.agreement["unit_price"],
                    )
                    q[product] += quantity
                    p[product] += quantity * unit_price
                self.exogenous_contracts_summary = [(a, b) for a, b in zip(q, p)]
                self.bulletin_board.record(
                    "exogenous_contracts_summary",
                    value=self.exogenous_contracts_summary,
                    key=self.current_step,
                )

            # zero quantities and prices
            # ==========================
            self._input_quantity = defaultdict(int)
            self._input_price = defaultdict(int)
            self._output_quantity = defaultdict(int)
            self._output_price = defaultdict(int)

            # request all negotiations
            # ========================
            self._make_negotiations()
            return

        # update trading price information
        # --------------------------------
        has_trade = self._sold_quantity[:, s + 1] > 0

        self._trading_price[~has_trade, s + 1] = self._trading_price[~has_trade, s]
        self._betas_sum[~has_trade, s + 1] = self._betas_sum[~has_trade, s]
        assert not np.any(
            np.isnan(self._real_price[has_trade, s + 1])
        ), f"Nans in _real_price at step {self.current_step}\n{self._real_price}"
        self._trading_price[has_trade, s + 1] = (
            self._trading_price[has_trade, s]
            * self._betas_sum[has_trade, s]
            * self.trading_price_discount
            + self._real_price[has_trade, s + 1] * self._sold_quantity[has_trade, s + 1]
        )
        self._betas_sum[has_trade, s + 1] = (
            self._betas_sum[has_trade, s] * self.trading_price_discount
            + self._sold_quantity[has_trade, s + 1]
        )
        self._trading_price[has_trade, s + 1] /= self._betas_sum[has_trade, s + 1]
        self._traded_quantity += self._sold_quantity[:, s + 1]
        # self._trading_price[has_trade, s] = (
        #         np.sum(self._betas[:s+1] * self._real_price[has_trade, s::-1])
        #     ) / self._betas_sum[s+1]

        # update agent profits
        # ---------------------
        for aid, agent in self.agents.items():
            if is_system_agent(aid):
                continue
            profile = agent.profile
            qin, pin, qout, pout = (
                self._input_quantity[aid],
                self._input_price[aid],
                self._output_quantity[aid],
                self._output_price[aid],
            )
            self._profits[aid].append(
                OneShotUFun.eval(
                    qin,
                    qout,
                    pin,
                    pout,
                    profile.cost,
                    self.agent_storage_cost[aid][self.current_step],
                    self.agent_delivery_penalty[aid][self.current_step],
                )
            )
            self._breach_levels[aid].append(OneShotUFun.breach_level(qin, qout,))
            self._breaches_of[aid].append(OneShotUFun.is_breach(qin, qout,))
            if self._breaches_of[aid][-1]:
                self.bulletin_board.record(
                    section="breaches",
                    key=unique_name("", add_time=False),
                    value=self._breach_record(
                        aid, self._breach_levels[aid][-1], "product"
                    ),
                )

        # publish financial reports
        # -------------------------
        if self.current_step % self.financial_reports_period == 0:
            reports_agent = self.bulletin_board.data["reports_agent"]
            reports_time = self.bulletin_board.data["reports_time"]
            for aid, agent in self.agents.items():
                if is_system_agent(agent.id):
                    continue
                self.add_financial_report(agent, reports_agent, reports_time)

    def _breach_record(self, perpetrator, level, type_,) -> Dict[str, Any]:
        return {
            "perpetrator": perpetrator,
            "perpetrator_name": perpetrator,
            "level": level,
            "type": type_,
            "time": self.current_step,
        }

    def on_contract_signed(self, contract: Contract) -> bool:
        product = contract.annotation["product"]
        bought = contract.agreement["quantity"]
        buy_cost = bought * contract.agreement["unit_price"]
        oldq = self._sold_quantity[product, self.current_step + 1]
        oldp = self._real_price[product, self.current_step + 1]
        totalp = 0.0
        if oldq > 0:
            totalp = oldp * oldq
        self._sold_quantity[product, self.current_step + 1] += bought
        self._real_price[product, self.current_step + 1] = (totalp + buy_cost) / (
            oldq + bought
        )
        self._input_quantity[contract.annotation["buyer"]] += bought
        self._input_price[contract.annotation["buyer"]] += buy_cost
        self._output_quantity[contract.annotation["seller"]] += bought
        self._output_price[contract.annotation["seller"]] += buy_cost
        contract.executed_at = self.current_step
        return super().on_contract_signed(contract)

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
            "signed_at": contract.signed_at,
            "nullified_at": contract.nullified_at,
            "concluded_at": contract.concluded_at,
            "signatures": "|".join(str(_) for _ in contract.signatures),
            "issues": contract.issues if not self.compact else None,
            "seller": contract.annotation["seller"],
            "buyer": contract.annotation["buyer"],
            "product_name": "p" + str(contract.annotation["product"]),
        }
        if not self.compact:
            c.update(contract.annotation)
        c["n_neg_steps"] = (
            contract.mechanism_state.step if contract.mechanism_state else 0
        )
        return c

    def breach_record(self, breach: Breach) -> Dict[str, Any]:
        return {
            "perpetrator": breach.perpetrator,
            "perpetrator_name": breach.perpetrator,
            "level": breach.level,
            "type": breach.type,
            "time": breach.step,
        }

    def execute_action(self, action, agent, callback: Callable = None) -> bool:
        pass

    def post_step_stats(self):
        self._stats["n_contracts_nullified_now"].append(0)
        scores = self.scores()
        for p in range(self.n_products):
            self._stats[f"trading_price_{p}"].append(
                self._trading_price[p, self.current_step + 1]
            )
            self._stats[f"sold_quantity_{p}"].append(
                self._sold_quantity[p, self.current_step + 1]
            )
            self._stats[f"unit_price_{p}"].append(
                self._real_price[p, self.current_step + 1]
            )
        for aid, a in self.agents.items():
            if is_system_agent(aid):
                continue
            self._stats[f"score_{aid}"].append(scores[aid])

    def pre_step_stats(self):
        pass

    def welfare(self, include_bankrupt: bool = False) -> float:
        """Total welfare of all agents"""
        scores = self.scores()
        return sum(
            scores[a.id] for a in self.agents.values() if not is_system_agent(a.id)
        )

    def relative_welfare(self, include_bankrupt: bool = False) -> Optional[float]:
        """Total welfare relative to expected value. Returns None if no expectation is found in self.info"""
        if "expected_income" not in self.info.keys():
            return None
        return self.welfare(include_bankrupt) / np.sum(self.info["expected_income"])

    def is_valid_contact(self, contract: Contract) -> bool:
        """Checks whether a signed contract is valid"""

        return (
            contract.agreement["time"] >= self.current_step
            and contract.agreement["time"] < self.n_steps
            and contract.agreement["unit_price"] > 0
            and contract.agreement["quantity"] > 0
        )

    def scores(self, assets_multiplier: float = 0.5) -> Dict[str, float]:
        """Scores of all agents given the asset multiplier.

        Args:
            assets_multiplier: A multiplier to multiply the assets with.
        """
        scores = dict()
        for aid, agent in self.agents.items():
            if is_system_agent(aid):
                continue
            scores[aid] = sum(self._profits[aid])
        return scores

    @property
    def winners(self):
        """The winners of this world (factory managers with maximum wallet balance"""
        if len(self.agents) < 1:
            return []
        scores = self.scores()
        sa = sorted(zip(scores.values(), scores.keys()), reverse=True)
        max_score = sa[0][0]
        winners = []
        for s, a in sa:
            if s < max_score:
                break
            winners.append(self.agents[a])
        return winners

    def trading_prices_for(
        self, discount: float = 1.0, condition="executed"
    ) -> np.ndarray:
        """
        Calculates the prices at which all products traded using an optional discount factor

        Args:
            discount: A discount factor to treat older prices less importantly (exponential discounting).
            condition: The condition for contracts to consider. Possible values are executed, signed, concluded,
                       nullified

        Returns:
            an n_products vector of trading prices
        """
        prices = np.nan * np.ones((self.n_products, self.n_steps), dtype=float)
        quantities = np.zeros((self.n_products, self.n_steps), dtype=int)
        for contract in self.saved_contracts:
            if contract["condition" + "_at"] < 0:
                continue
            p, t, q, u = (
                contract["product"],
                contract["delivery_time"],
                contract["quantity"],
                contract["unit_price"],
            )
            prices[p, t] = (prices[p, t] * quantities[p, t] + u * q) / (
                quantities[p, t] + q
            )
            quantities[p, t] += q
        discount = np.cumprod(discount * np.ones(self.n_steps))
        discount /= sum(discount)
        return np.nansum(np.nanprod(prices, discount), axis=-1)

    @property
    def trading_prices(self):
        if self.current_step == self.n_steps:
            return self._trading_price[:, -1]
        return self._trading_price[:, self.current_step + 1]

    @property
    def stats_df(self) -> pd.DataFrame:
        """Returns a pandas data frame with the stats"""
        return pd.DataFrame(super().stats)

    @property
    def contracts_df(self) -> pd.DataFrame:
        """Returns a pandas data frame with the contracts"""
        contracts = pd.DataFrame(self.saved_contracts)
        contracts["product_index"] = contracts.product_name.str.replace("p", "").astype(
            int
        )
        contracts["breached"] = contracts.breaches.str.len() > 0
        contracts["executed"] = contracts.executed_at >= 0
        contracts["erred"] = contracts.erred_at >= 0
        contracts["nullified"] = contracts.nullified_at >= 0
        return contracts

    @property
    def system_agents(self) -> List[_SystemAgent]:
        """Returns the two system agents"""
        return [_ for _ in self.agents.values() if is_system_agent(_.id)]

    @property
    def system_agent_names(self) -> List[str]:
        """Returns the names two system agents"""
        return [_ for _ in self.agents.keys() if is_system_agent(_)]

    @property
    def non_system_agents(self) -> List[_OneShotBaseAgent]:
        """Returns all agents except system agents"""
        return [_ for _ in self.agents.values() if not is_system_agent(_.id)]

    @property
    def non_system_agent_names(self) -> List[str]:
        """Returns names of all agents except system agents"""
        return [_ for _ in self.agents.keys() if not is_system_agent(_)]

    @property
    def agreement_fraction(self) -> float:
        """Fraction of negotiations ending in agreement and leading to signed contracts"""
        n_negs = sum(self.stats["n_negotiations"])
        n_contracts = self.n_saved_contracts(True)
        return n_contracts / n_negs if n_negs != 0 else np.nan

    system_agent_ids = system_agent_names
    non_system_agent_ids = non_system_agent_names

    def draw(
        self,
        steps: Optional[Union[Tuple[int, int], int]] = None,
        what: Collection[str] = DEFAULT_EDGE_TYPES,
        who: Callable[[Agent], bool] = None,
        where: Callable[[Agent], Union[int, Tuple[float, float]]] = None,
        together: bool = True,
        axs: Collection[Axis] = None,
        ncols: int = 4,
        figsize: Tuple[int, int] = (15, 15),
        **kwargs,
    ) -> Union[Tuple[Axis, nx.Graph], Tuple[List[Axis], List[nx.Graph]]]:
        if where is None:
            where = (
                lambda x: self.n_processes + 1
                if x == SYSTEM_BUYER_ID
                else 0
                if x == SYSTEM_SELLER_ID
                else int(self.agents[x].awi.profile.level + 1)
            )
        return super().draw(
            steps,
            what,
            who,
            where,
            together=together,
            axs=axs,
            ncols=ncols,
            figsize=figsize,
            **kwargs,
        )

    def _request_negotiations(
        self,
        agent_id: str,
        product: int,
        quantity: Union[int, Tuple[int, int]],
        unit_price: Union[int, Tuple[int, int]],
        time: Union[int, Tuple[int, int]],
        controller: Optional[SAOController] = None,
        negotiators: List[SAONegotiator] = None,
        extra: Dict[str, Any] = None,
    ) -> bool:
        """
        Requests negotiations (used internally)

        Args:

            agent_id: the agent requesting
            product: The product to negotiate about
            quantity: The minimum and maximum quantities. Passing a single value q is equivalent to passing (q,q)
            unit_price: The minimum and maximum unit prices. Passing a single value u is equivalent to passing (u,u)
            time: The minimum and maximum delivery step. Passing a single value t is equivalent to passing (t,t)
            controller: The controller to manage the complete set of negotiations
            negotiators: An optional list of negotiators to use for negotiating with the given partners (in the same
                         order).
            extra: Extra information accessible through the negotiation annotation to the caller

        Returns:

            `True` if the partner accepted and the negotiation is ready to start

        """
        is_buy = True
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
        partners = [_ for _ in self.suppliers[product] if not self.is_bankrupt[_]]
        if not partners:
            return False
        if negotiators is None:
            negotiators = [
                controller.create_negotiator(PassThroughSAONegotiator, name=_)
                for _ in partners
            ]
        results = [
            self._request_negotiation(
                agent_id,
                product,
                quantity,
                unit_price,
                time,
                partner,
                negotiator,
                extra,
            )
            for partner, negotiator in zip(partners, negotiators)
        ]
        # for p, r in zip(partners, results):
        #     if r:
        #         self._world._registered_negs.add(tuple(sorted([p, self.agent.id])))
        return all(results)

    def _request_negotiation(
        self,
        agent_id: str,
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

            product: The product to negotiate about
            quantity: The minimum and maximum quantities. Passing a single value q is equivalent to passing (q,q)
            unit_price: The minimum and maximum unit prices. Passing a single value u is equivalent to passing (u,u)
            time: The minimum and maximum delivery step. Passing a single value t is equivalent to passing (t,t)
            partner: ID of the partner to negotiate with.
            negotiator: The negotiator to use for this negotiation (if the partner accepted to negotiate)
            extra: Extra information accessible through the negotiation annotation to the caller

        Returns:

            `True` if the partner accepted and the negotiation is ready to start

        """
        agent = self.agents[agent_id]
        is_buy = True
        if extra is None:
            extra = dict()

        def values(x: Union[int, Tuple[int, int]]):
            if not isinstance(x, Iterable):
                return int(x), int(x)
            return int(x[0]), int(x[1])

        self.logdebug(
            f"{agent.name} requested to {'buy' if is_buy else 'sell'} {product} to {partner}"
            f" q: {quantity}, u: {unit_price}, t: {time}"
        )

        annotation = {
            "product": product,
            "is_buy": is_buy,
            "buyer": agent_id if is_buy else partner,
            "seller": partner if is_buy else agent_id,
            "caller": agent_id,
        }
        issues = [
            Issue(values(quantity), name="quantity", value_type=int),
            Issue(values(time), name="time", value_type=int),
            Issue(values(unit_price), name="unit_price", value_type=int),
        ]
        partners = [agent_id, partner]
        extra["negotiator_id"] = negotiator.id
        req_id = agent.create_negotiation_request(
            issues=issues,
            partners=partners,
            negotiator=negotiator,
            annotation=annotation,
            extra=dict(**extra),
        )
        result = self.request_negotiation_about(
            caller=agent,
            issues=issues,
            partners=[self.agents[_] for _ in partners],
            req_id=req_id,
            annotation=annotation,
        )
        # if result:
        #     self._registered_negs.add(tuple(sorted([partner, agent_id])))
        return result

    def _make_negotiations(self):
        controllers = dict()
        for aid, a in self.agents.items():
            if is_system_agent(aid):
                continue
            controllers[aid] = a.make_controller()
        for product in range(1, self.n_products):
            for aid in self.consumers[product]:
                if is_system_agent(aid):
                    continue
                success = self._request_negotiations(
                    agent_id=aid,
                    product=product,
                    quantity=(1, self.agents[aid].profile.n_lines),
                    unit_price=(
                        1,
                        int(
                            self.price_multiplier
                            * (
                                self.trading_prices[product]
                                if self.publish_trading_prices
                                else self.catalog_prices[product]
                            )
                        ),
                    ),
                    time=(self.current_step, self.current_step),
                    controller=controllers[aid],
                    negotiators=None,
                    extra=None,
                )
            if not success:
                raise ValueError(
                    f"Failed to start negotiations for product " f"{product}"
                )

    def order_contracts_for_execution(
        self, contracts: Collection[Contract]
    ) -> Collection[Contract]:
        for contract in contracts:
            contract.executed_at = self.current_step
        return contracts

    def get_private_state(self, agent: "Agent") -> dict:
        return agent.awi.state
