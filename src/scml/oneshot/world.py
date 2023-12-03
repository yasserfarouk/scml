from __future__ import annotations

from negmas.outcomes.cardinal_issue import DiscreteCardinalIssue

"""
Implements the one shot version of SCML
"""
import copy
import itertools
import logging
import math
import random
import sys
import warnings
from collections import defaultdict
from typing import Any, Callable, Collection, Iterable, Literal

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.axis import Axis
from negmas import (
    DEFAULT_EDGE_TYPES,
    Agent,
    Breach,
    BreachProcessing,
    ContiguousIssue,
    Contract,
    Operations,
    SAOResponse,
    TimeInAgreementMixin,
    World,
    make_issue,
)
from negmas.helpers import get_class, get_full_type_name, instantiate, unique_name
from negmas.sao import ControlledSAONegotiator, SAOController, SAONegotiator
from negmas.situated import NegotiationInfo

from ..common import (
    distribute_quantities,
    integer_cut,
    intin,
    make_array,
    realin,
    strin,
)
from .adapter import OneShotSCML2020Adapter
from .agent import OneShotAgent
from .common import (
    INFINITE_COST,
    SYSTEM_BUYER_ID,
    SYSTEM_SELLER_ID,
    FinancialReport,
    NegotiationDetails,
    OneShotExogenousContract,
    OneShotProfile,
    is_system_agent,
)
from .sysagents import DefaultOneShotAdapter, _StdSystemAgent

__all__ = [
    "SCML2020OneShotWorld",
    "SCML2021OneShotWorld",
    "SCML2022OneShotWorld",
    "SCML2023OneShotWorld",
]


class OneShotWorld(TimeInAgreementMixin, World):
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
        profiles: list[OneShotProfile],
        agent_types: list[type[OneShotAgent]],
        agent_params: list[dict[str, Any]],
        catalog_quantities: int | np.ndarray = 50,
        # breach processing parameters
        financial_report_period=5,
        bankruptcy_limit=0.0,
        penalize_bankrupt_for_future_contracts=True,
        penalties_scale: Literal["trading", "catalog", "unit", "none"] = "trading",
        # external contracts parameters
        exogenous_contracts: Collection[OneShotExogenousContract] = tuple(),
        exogenous_dynamic: bool = False,
        exogenous_force_max: bool = False,
        # factory parameters
        initial_balance: np.ndarray | tuple[int, int] | int = 1000,
        # General SCML2020World Parameters
        compact=False,
        no_logs=False,
        n_steps=1000,
        time_limit=60 * 90,
        sync_calls=False,
        # mechanism params
        neg_n_steps=20,
        neg_time_limit=None,
        neg_hidden_time_limit=3 * 60,
        neg_step_time_limit=60,
        negotiation_speed=None,
        shuffle_negotiations=False,
        one_offer_per_step=False,
        # public information
        publish_exogenous_summary=True,
        publish_trading_prices=True,
        # negotiation params,
        price_multiplier=0.0,
        wide_price_range=False,
        # trading price parameters
        trading_price_discount=0.9,
        # simulation parameters
        signing_delay=0,
        force_signing=False,
        batch_signing=True,
        name: str | None = None,
        # debugging parameters
        agent_name_reveals_position: bool = True,
        agent_name_reveals_type: bool = True,
        # evaluation paramters (for compatibility with SCML2020World)
        inventory_valuation_catalog=0,
        inventory_valuation_trading=0,
        # set to True to add more assertions during debuging
        **kwargs,
    ):
        self._debug = False
        # neg_n_steps is ALWAYS the number of rounds. We multiply it by 2 if mechanisms are stepped one offer at a time
        if one_offer_per_step and neg_n_steps is not None:
            neg_n_steps *= 2

        self._profits: dict[str, list[float]] = defaultdict(list)
        self._breach_levels: dict[str, list[float]] = defaultdict(list)
        self._breaches_of: dict[str, list[bool]] = defaultdict(list)
        self.trading_price_discount = trading_price_discount
        self.catalog_quantities = catalog_quantities
        self.publish_exogenous_summary = publish_exogenous_summary
        self.price_multiplier = price_multiplier
        self.wide_price_range = wide_price_range
        self.publish_trading_prices = publish_trading_prices
        self.penalize_bankrupt_for_future_contracts = (
            penalize_bankrupt_for_future_contracts
        )
        self.agent_disposal_cost: dict[str, list[float]] = dict()
        self.agent_shortfall_penalty: dict[str, list[float]] = dict()
        kwargs["log_to_file"] = not no_logs
        if compact:
            kwargs["event_file_name"] = None
            kwargs["event_types"] = []
            kwargs["log_screen_level"] = logging.CRITICAL
            kwargs["log_file_level"] = logging.ERROR
            kwargs["log_negotiations"] = False
            kwargs["log_ufuns"] = False
            # kwargs["save_mechanism_state_in_contract"] = False
            kwargs["save_cancelled_contracts"] = False
            kwargs["save_resolved_breaches"] = False
            kwargs["save_negotiations"] = True
        else:
            kwargs["save_negotiations"] = True

        self.compact = compact
        # if negotiation_speed == 0:
        #     negotiation_speed = neg_n_steps + 1
        mechanisms = kwargs.pop("mechanisms", {})
        super().__init__(
            bulletin_board=None,
            breach_processing=BreachProcessing.NONE,
            awi_type="scml.oneshot.OneShotAWI",
            shuffle_negotiations=shuffle_negotiations,
            mechanisms={
                "negmas.sao.SAOMechanism": mechanisms.get(
                    "negmas.sao.SAOMechanism",
                    dict(
                        end_on_no_response=True,
                        dynamic_entry=False,
                        max_wait=len(agent_types),
                        check_offers=True,
                        enforce_issue_types=True,
                        cast_offers=True,
                        hidden_time_limit=neg_hidden_time_limit,
                        sync_calls=sync_calls,
                        one_offer_per_step=one_offer_per_step,
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
        if not self.bulletin_board:
            raise ValueError(f"Cannot find the bulletin-board")
        self.bulletin_board.record("settings", 1, "horizon")
        self.bulletin_board.record(
            "settings", publish_trading_prices, "public_trading_prices"
        )
        self.bulletin_board.record(
            "settings", publish_exogenous_summary, "public_exogenous_summary"
        )
        self.bulletin_board.record(
            "settings", financial_report_period, "financial_report_period"
        )
        self.bulletin_board.record(
            "settings",
            penalize_bankrupt_for_future_contracts,
            "penalize_bankrupt_for_future_contracts",
        )
        self.bulletin_board.record(
            "settings", exogenous_force_max, "exogenous_force_max"
        )
        # self.bulletin_board.record("settings", disposal_cost, "ufun_disposal_cost")
        # self.bulletin_board.record(
        #     "settings", shortfall_penalty, "ufun_shortfall_penalty"
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
        agent_params = (
            [copy.deepcopy(_) for _ in agent_params] if agent_params else agent_params
        )
        self.penalties_scale = penalties_scale
        TimeInAgreementMixin.init(self, time_field="time")
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
        self.info.update(
            shuffle_negotiations=shuffle_negotiations,
            process_inputs=process_inputs,
            process_outputs=process_outputs,
            catalog_prices=catalog_prices,
            agent_types_final=[get_full_type_name(_) for _ in agent_types],
            agent_params_final=[copy.deepcopy(_) for _ in agent_params]
            if agent_params is not None
            else agent_params,
            initial_balance_final=initial_balance,
            penalties_scale_final=penalties_scale,
            penalize_bankrupt_for_future_contracts=penalize_bankrupt_for_future_contracts,
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
            # disposal_cost=disposal_cost,
            # shortfall_penalty=shortfall_penalty,
            exogenous_dynamic=exogenous_dynamic,
            publish_exogenous_summary=publish_exogenous_summary,
            publish_trading_prices=publish_trading_prices,
            selected_price_multiplier=price_multiplier,
            wide_price_range=wide_price_range,
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
        agent_types = [get_class(_) for _ in agent_types]
        for p in agent_params:
            p["obj"] = get_class(p["controller_type"])(
                **p.get("controller_params", dict())
            )
            del p["controller_type"]
            if "controller_params" in p.keys():
                del p["controller_params"]

        self.controller_types = [
            get_class(_["obj"])._type_name() if _["obj"] else "system_agent"
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
            for i, at in enumerate(agent_params):
                s2 = get_class(at["obj"]).__class__.__name__
                s = s2.replace("Agent", "").replace("OneShot", "")
                s = "".join([c for c in s if c.isupper()])[:3]
                try:
                    if len(s) < 3:
                        if len(s2) > 3:
                            s = s2[:2]
                        elif len(s2) >= 2:
                            s = s2[0] + s2[1 : 1 + (3 - len(s))] + s2[1:]
                        elif len(s2) > 0:
                            s = s2[0] * 3
                        else:
                            s = "Agt"
                except:
                    pass
                default_names[i] += f"{s}"
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
                b["name"] = name  # type: ignore
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
        agent_types += [_StdSystemAgent, _StdSystemAgent]  # type: ignore
        agent_params += [
            {"role": SYSTEM_SELLER_ID, "obj": None},
            {"role": SYSTEM_BUYER_ID, "obj": None},
        ]
        profiles.append(
            OneShotProfile(
                cost=INFINITE_COST,
                input_product=-1,
                n_lines=0,
                disposal_cost_mean=0.0,
                shortfall_penalty_mean=0.0,
                disposal_cost_dev=0.0,
                shortfall_penalty_dev=0.0,
            )
        )
        profiles.append(
            OneShotProfile(
                cost=INFINITE_COST,
                input_product=n_processes,
                n_lines=0,
                disposal_cost_mean=0.0,
                shortfall_penalty_mean=0.0,
                disposal_cost_dev=0.0,
                shortfall_penalty_dev=0.0,
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
            if a.adapted_object:
                a.adapted_object.id = a.id
                a.adapted_object.name = a.name
                if isinstance(a.adapted_object, OneShotAgent):
                    a.adapted_object.connect_to_oneshot_adapter(a)
                else:
                    a.adapted_object._owner = a
            self.join(a, i)
            agents.append(a)
        self.agent_types = [_.type_name for _ in agents]
        self.agent_params = [
            {k: v for k, v in _.items() if k != "name" and k != "obj"}
            for _ in agent_params
        ]
        self.agent_unique_types = [
            f"{t}{hash(str(p))}" if len(p) > 0 else t
            for t, p in zip(self.agent_types, self.agent_params)
        ]

        self.agent_n_contracts = dict(zip((_.id for _ in agents), itertools.repeat(0)))

        self.suppliers: list[list[str]] = [[] for _ in range(n_products)]
        self.consumers: list[list[str]] = [[] for _ in range(n_products)]
        self.agent_processes: dict[str, list[int]] = defaultdict(list)
        self.agent_inputs: dict[str, list[int]] = defaultdict(list)
        self.agent_outputs: dict[str, list[int]] = defaultdict(list)
        self.agent_consumers: dict[str, list[str]] = defaultdict(list)
        self.agent_suppliers: dict[str, list[str]] = defaultdict(list)
        self.agent_profiles: dict[str, OneShotProfile] = defaultdict(list)

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
            self.agent_disposal_cost[aid] = np.abs(
                np.random.randn(self.n_steps) * profile.disposal_cost_dev
                + profile.disposal_cost_mean
            )
            self.agent_shortfall_penalty[aid] = np.abs(
                np.random.randn(self.n_steps) * profile.shortfall_penalty_dev
                + profile.shortfall_penalty_mean
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
        self.is_bankrupt: dict[str, bool] = dict(
            zip(self.agents.keys(), itertools.repeat(False))
        )
        self.exogenous_contracts: dict[int : list[Contract]] = defaultdict(list)
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

        self.initial_balances = dict(zip(self.agents.keys(), initial_balance))
        self._max_n_lines = max(_.n_lines for _ in self.profiles)
        self.a2i = dict(zip((_.id for _ in agents), range(n_agents)))
        self._current_issues: list[list[ContiguousIssue]] = []
        self.__contracts: dict[str, list[Contract]] = defaultdict(list)

        def values(x: int | tuple[int, int]):
            if not isinstance(x, Iterable):
                return int(x), int(x)
            return int(x[0]), int(x[1])

        for product in range(self.n_products):
            unit_price, time, quantity = self._make_issues(product)
            _issues = [
                make_issue(values(quantity), name="quantity"),
                make_issue(values(time), name="time"),
                make_issue(values(unit_price), name="unit_price"),
            ]
            if self._debug:
                assert all(isinstance(_, ContiguousIssue) for _ in _issues)
            self._current_issues.append(_issues)

        def to_lists(d):
            return {
                k: v.tolist() if isinstance(v, np.ndarray) else list(v)
                for k, v in d.items()
            }

        self.info.update(
            dict(
                agent_profiles={
                    k: dict(
                        cost=v.cost,
                        n_lines=v.n_lines,
                        input_product=v.input_product,
                        shortfall_penalty_mean=v.shortfall_penalty_mean,
                        shortfall_penalty_dev=v.shortfall_penalty_dev,
                        disposal_cost_mean=v.disposal_cost_mean,
                        disposal_cost_dev=v.disposal_cost_dev,
                    )
                    for k, v in self.agent_profiles.items()
                }
            )
        )
        self.info.update(dict(agent_inputs=to_lists(self.agent_inputs)))
        self.info.update(dict(agent_outputs=to_lists(self.agent_outputs)))
        self.info.update(dict(agent_processes=to_lists(self.agent_processes)))
        self.info.update(dict(agent_initial_balances=self.initial_balances))
        self._update_exogenous(0)
        # self._current_negotiations: list[NegotiationDetails] = []
        self._agent_negotiations: dict[
            str, dict[str, dict[str, NegotiationDetails]]
        ] = dict()

    @classmethod
    def generate(
        cls,
        agent_types: list[str | type[OneShotAgent]],
        agent_params: list[dict[str, Any]] | None = None,
        agent_processes: list[int] | None = None,
        n_steps: tuple[int, int] | int = (50, 200),
        n_processes: tuple[int, int] | int = 2,
        n_lines: np.ndarray | tuple[int, int] | int = 10,
        n_agents_per_process: np.ndarray | tuple[int, int] | int = (4, 8),
        process_inputs: np.ndarray | tuple[int, int] | int = 1,
        process_outputs: np.ndarray | tuple[int, int] | int = 1,
        production_costs: np.ndarray | tuple[int, int] | int = (1, 4),
        profit_means: np.ndarray | tuple[float, float] | float = (0.1, 0.2),
        profit_stddevs: np.ndarray | tuple[float, float] | float = 0.05,
        max_productivity: np.ndarray | tuple[float, float] | float = (0.8, 1.0),
        initial_balance: np.ndarray | tuple[int, int] | int | None = None,
        cost_increases_with_level=True,
        equal_exogenous_supply=False,
        equal_exogenous_sales=False,
        exogenous_supply_predictability: tuple[float, float] | float = (0.6, 0.9),
        exogenous_sales_predictability: tuple[float, float] | float = (0.6, 0.9),
        exogenous_control: tuple[float, float] | float = -1,
        cash_availability: tuple[float, float] | float = (1.5, 2.5),
        force_signing=True,
        profit_basis=np.max,
        disposal_cost: np.ndarray | tuple[float, float] | float = (0.0, 0.2),
        shortfall_penalty: np.ndarray | tuple[float, float] | float = (0.2, 1.0),
        disposal_cost_dev: np.ndarray | tuple[float, float] | float = (0.0, 0.02),
        shortfall_penalty_dev: np.ndarray
        | tuple[float, float]
        | float = (
            0.0,
            0.1,
        ),
        exogenous_price_dev: np.ndarray | tuple[float, float] | float = (0.1, 0.2),
        price_multiplier: np.ndarray | tuple[float, float] | float = (1.5, 2.0),
        random_agent_types: bool = False,
        penalties_scale: str | list[str] = "trading",
        cap_exogenous_quantities: bool = True,
        method="profitable",
        **kwargs,
    ) -> dict[str, Any]:
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
            agent_processes: The process for each agent. If not `None` , it will override `n_agents_per_process` and must be a list/tuple
                             of the same length as `agent_types` . Morevoer, `random_agent_types` must be False in this case
            cost_increases_with_level: If true, production cost will be higher for processes nearer to the final
                                       product.
            profit_basis: The statistic used when controlling catalog prices by profit arguments. It can be np.mean,
                          np.median, np.min, np.max or any Callable[[list[float]], float] and is used to summarize
                          production costs at every level.
            equal_exogenous_supply: If true, external supply will be distributed equally among all agents in the first
                                   layer
            equal_exogenous_sales: If true, external sales will be distributed equally among all agents in the last
                                   layer
            exogenous_supply_predictability: How predictable are exogenous supplies of each agent over time. 1.0 means
                                             that every agent will have the same quantity for all of its contracts over
                                             time. 0.0 means quantities per agent are completely random
            exogenous_sales_predictability: How predictable are exogenous supplies of each agent over time. 1.0 means
                                             that every agent will have the same quantity for all of its contracts over
                                             time. 0.0 means quantities per agent are completely random
            force_signing: Whether to force contract signatures (exogenous contracts are treated in the same way).
            exogenous_control: How much control does the agent have over exogenous contract signing. Only effective if
                               force_signing is False and use_exogenous_contracts is True
            cap_exogenous_quantities: If True, all exogenous quantities in all contracts are capped to be no more than the number of lines
            cash_availability: The fraction of the total money needs of the agent to work at maximum capacity that
                               is available as `initial_balance` . This is only effective if `initial_balance` is set
                               to `None` .
            exogenous_control: How much control does the agent have over exogenous contract signing. Only effective if
                               force_signing is False and use_exogenous_contracts is True
            disposal_cost: A range to sample mean-disposal costs for all factories from
            shortfall_penalty: A range to sample mean-shortfall penalty for all factories from
            disposal_cost_dev: A range to sample std. dev of disposal costs for all factories from
            shortfall_penalty_dev: A range to sample std. dev of shortfall penalty for all factories from
            exogenous_price_dev: The standard deviation of exogenous contract prices relative to the mean price
            price_multiplier: A value to multiply with trading/catalog price to get the upper limit on prices for all negotiations
            random_agent_types: If True, the final agent types used by the generato wil always be sampled from the given types.
                                If False, this random sampling will only happin if len(agent_types) != n_agents.
            penalties_scale: What are `disposal_cost` and `shortfall_penalty` relative to.
                            There are four options: `trading`, `catalog` mean trading
                            and catalog prices of the product. `unit` means the unit
                            price in the contract and `none` means the `storage-cost`
                            and `shortfall_penalty` are absolute values (in money unit).
                            If not given will be read through the AWI
            method: the generation method. This is only for compatibility with SCML2020World and is not used.
            **kwargs:

        Returns:

            world configuration as a Dict[str, Any]. A world can be generated from this dict by calling SCML2020World(**d)

        Remarks:

            - There are two general ways to use this generator:
                1. Pass `random_agent_types = False`, and pass `agent_types`, `agent_processes` to control placement of each
                   agent in each level of the production graph.
                2. Pass `random_agent_types = True` and pass `agent_types`, `n_agents_per_process` to make the system randomly
                   place the specified number of agents in each production level
            - Most parameters (i.e. `process_inputs` , `process_outputs` , `n_agents_per_process` , `costs` ) can
              take a single value, a tuple of two values, or a list of values.
              If it has a single value, it is repeated for all processes/factories as appropriate. If it is a
              tuple of two numbers $(i, j)$, each process will take a number sampled from a uniform distribution
              supported on $[i, j]$ inclusive. If it is a list of values, of the length `n_processes` , it is used as
              it is otherwise, it is used to sample values for each process.

        """
        if agent_processes is not None and random_agent_types:
            raise ValueError(
                "You cannot pass `agent_processes` and use `random_agent_types`. The first is only "
                "used when you want to fix the assignment of all agents to specific processes"
            )
        if agent_processes is not None and len(agent_processes) != len(agent_types):
            raise ValueError(
                f"Length of `agent_processes` ({len(agent_processes)}) must equal the length of `agent_types` ({len(agent_types)})"
            )
        info = dict(
            n_steps=n_steps,
            n_processes=n_processes,
            n_lines=n_lines,
            force_signing=force_signing,
            agent_processes=agent_processes,
            n_agents_per_process=n_agents_per_process,
            process_inputs=process_inputs,
            process_outputs=process_outputs,
            production_costs=production_costs,
            profit_means=profit_means,
            profit_stddevs=profit_stddevs,
            max_productivity=max_productivity,
            initial_balance=initial_balance,
            cost_increases_with_level=cost_increases_with_level,
            equal_exogenous_sales=equal_exogenous_sales,
            equal_exogenous_supply=equal_exogenous_supply,
            exogenous_supply_predictability=exogenous_supply_predictability,
            exogenous_sales_predictability=exogenous_sales_predictability,
            cash_availability=cash_availability,
            price_multiplier=price_multiplier,
            exogenous_price_dev=exogenous_price_dev,
            penalties_scale=penalties_scale,
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
        penalties_scale = strin(penalties_scale)
        price_multiplier = realin(price_multiplier)
        n_processes = intin(n_processes)
        n_steps = intin(n_steps)
        exogenous_control = realin(exogenous_control)
        exogenous_sales_predictability = realin(exogenous_sales_predictability)
        exogenous_supply_predictability = realin(exogenous_supply_predictability)
        np.errstate(divide="ignore")
        # n_startup = n_processes
        # if n_steps <= n_startup:
        #     raise ValueError(
        #         f"Cannot generate a world with n_steps <= n_processes: {n_steps} <= {n_startup}"
        #     )

        process_inputs = make_array(process_inputs, n_processes, dtype=int)
        process_outputs = make_array(process_outputs, n_processes, dtype=int)
        fixed_assignment = agent_processes is not None and not random_agent_types
        if agent_processes is not None:
            pcount = defaultdict(int)
            for i in agent_processes:
                pcount[i] += 1
            pnums = list(pcount.keys())
            assert (
                min(pnums) == 0 and max(pnums) == len(pnums) - 1
            ), f"`agent_processes` is invalid: {agent_processes} as it leads to the following `n_agents_per_process`: {dict(pcount)}"
            n_agents_per_process = np.asarray([pcount[i] for i in range(len(pnums))])
            assert not any(
                _ <= 0 for _ in n_agents_per_process
            ), f"We have some levels with no processes"
        else:
            n_agents_per_process = make_array(
                n_agents_per_process, n_processes, dtype=int
            )
        profit_means = make_array(profit_means, n_processes, dtype=float)
        profit_stddevs = make_array(profit_stddevs, n_processes, dtype=float)
        max_productivity = make_array(
            max_productivity, n_processes * n_steps, dtype=float
        ).reshape((n_processes, n_steps))
        n_agents = n_agents_per_process.sum()
        assert n_agents >= n_processes
        n_products = n_processes + 1
        production_costs = make_array(production_costs, n_agents, dtype=int)
        if initial_balance is not None:
            initial_balance = make_array(initial_balance, n_agents, dtype=int)
        if not isinstance(agent_types, Iterable):
            agent_types = [agent_types] * n_agents
            if agent_params is None:
                agent_params = dict()
            if isinstance(agent_params, dict):
                agent_params = [copy.copy(agent_params) for _ in range(n_agents)]
            else:
                assert len(agent_params) == 1
                agent_params = [copy.copy(agent_params[0]) for _ in range(n_agents)]
        elif not fixed_assignment:
            if agent_params is None:
                agent_params = [dict() for _ in range(len(agent_types))]
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
                agent_params = [dict() for _ in range(len(agent_types))]
            if isinstance(agent_params, dict):
                agent_params = [
                    copy.copy(agent_params) for _ in range(len(agent_types))
                ]
            agent_types = list(agent_types)
            agent_params = list(agent_params)
            assert len(agent_types) == len(agent_params)
        agent_params = [_ if not _ is None else dict() for _ in agent_params]
        for t, p in zip(agent_types, agent_params):
            p["controller_type"] = t
        agent_types = [
            DefaultOneShotAdapter
            if at and issubclass(get_class(at), OneShotAgent)
            else OneShotSCML2020Adapter
            if at
            else None
            for at in agent_types
        ]
        # generate production costs making sure that every agent can do exactly one process
        n_agents_cumsum = n_agents_per_process.cumsum().tolist()
        first_agent = [0] + n_agents_cumsum[:-1]
        last_agent = n_agents_cumsum[:-1] + [n_agents]
        process_of_agent = np.empty(n_agents, dtype=int)
        for i, (f, l) in enumerate(zip(first_agent, last_agent)):
            process_of_agent[f:l] = i
            if cost_increases_with_level:
                production_costs[f:l] = np.round(
                    production_costs[f:l] * (i + 1)  # math.sqrt(i + 1)
                ).astype(int)

        costs = INFINITE_COST * np.ones((n_agents, n_lines, n_processes), dtype=int)
        for p, (f, l) in enumerate(zip(first_agent, last_agent)):
            costs[f:l, :, p] = production_costs[f:l].reshape((l - f), 1)

        # generate external contract amounts (controlled by productivity):

        # - generate total amount of input to the market
        #   (it will end up being an n_products list of n_steps vectors)
        quantities = [
            np.round(n_lines * n_agents_per_process[0] * max_productivity[0, :]).astype(
                int
            )
        ]
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

        # - divide the quantity at every level between factories
        exogenous_supplies = distribute_quantities(
            equal_exogenous_supply,
            exogenous_supply_predictability,
            quantities[0],
            n_agents_per_process[0],
            n_steps,
            n_lines if cap_exogenous_quantities else None,
        )
        quantities[0] = [sum(_) for _ in exogenous_supplies]
        exogenous_sales = distribute_quantities(
            equal_exogenous_sales,
            exogenous_sales_predictability,
            quantities[-1],
            n_agents_per_process[-1],
            n_steps,
            n_lines if cap_exogenous_quantities else None,
        )
        quantities[-1] = [sum(_) for _ in exogenous_sales]

        # - now exogenous_supplies and exogenous_sales are both n_steps lists

        #   of n_agents_per_process[p] vectors (jagged)

        # assign prices to the quantities given the profits
        catalog_prices = np.zeros(n_products, dtype=int)
        catalog_prices[0] = 10
        supply_prices = catalog_prices[0] * np.ones(
            (n_agents_per_process[0], n_steps), dtype=int
        )
        # We will calculate these later
        sale_prices = np.zeros((n_agents_per_process[-1], n_steps), dtype=int)

        # calculate manufacturing cost per process per step (this is per line)
        # we will multiply this by the number of active lines later
        manufacturing_costs = np.zeros((n_processes, n_steps), dtype=int)
        for p in range(n_processes):
            manufacturing_costs[p, :] = profit_basis(
                costs[first_agent[p] : last_agent[p], :, p]
            )

        # calculate an "average" profit per process per step
        profits = np.zeros((n_processes, n_steps))
        for p in range(n_processes):
            profits[p, :] = np.random.randn() * profit_stddevs[p] + profit_means[p]

        # total input costs come from buying exogenous supplies (quantity * unit price)
        input_costs = np.zeros((n_processes, n_steps), dtype=int)
        for step in range(n_steps):
            input_costs[0, step] = np.sum(
                exogenous_supplies[step] * supply_prices[:, step][:]
            )

        # total input quantities per process are simply inputs of the corresponding
        # product type in quantities.
        input_quantity = np.zeros((n_processes, n_steps), dtype=int)
        input_quantity[0, :] = quantities[0]

        # the number of active lines come from dividing input by n. inputs consumed
        # by each line
        active_lines = np.hstack(
            [(n_lines * n_agents_per_process).reshape((n_processes, 1))] * n_steps
        )
        assert active_lines.shape == (n_processes, n_steps)
        active_lines[0, :] = input_quantity[0, :] // process_inputs[0]

        # output_quantity = np.zeros((n_processes, n_steps), dtype=int)
        output_quantity = np.zeros((n_processes, n_steps), dtype=int)
        output_quantity[0, :] = active_lines[0, :] * process_outputs[0]

        # find the total manufacturing_costs per process
        manufacturing_costs[0, :] *= active_lines[0, :]

        # cost = cost of input + cost of manufacturing
        total_costs = input_costs + manufacturing_costs

        # should sell at the cost plus profit
        output_total_prices = np.ceil(total_costs * (1 + profits)).astype(int)

        for p in range(1, n_processes):
            input_costs[p, :] = output_total_prices[p - 1, :]
            input_quantity[p, :] = output_quantity[p - 1, :]
            active_lines[p, :] = input_quantity[p, :] // process_inputs[p]
            output_quantity[p, :] = active_lines[p, :] * process_outputs[p]
            manufacturing_costs[p, :] *= active_lines[p - 1, :]
            total_costs[p, :] = input_costs[p, :] + manufacturing_costs[p, :]
            output_total_prices[p, :] = np.ceil(
                total_costs[p, :] * (1 + profits[p, :])
            ).astype(int)

        sale_prices[:, :] = np.ceil(
            output_total_prices[-1, :] / output_quantity[-1, :]
        ).astype(int)

        product_prices = np.zeros((n_products, n_steps))
        product_prices[0, :] = catalog_prices[0]
        product_prices[1:, :] = np.ceil(
            np.divide(
                output_total_prices.astype(float),
                output_quantity.astype(float),
                out=np.zeros_like(output_total_prices, dtype=float),
                where=output_quantity != 0,
            )
        ).astype(int)
        catalog_prices = np.ceil(
            [
                profit_basis(product_prices[p, p : p + n_steps])
                for p in range(n_products)
            ]
        ).astype(int)
        profile_info: list[
            tuple[OneShotProfile, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = []

        nxt = 0
        for l in range(n_processes):
            for a in range(n_agents_per_process[l]):
                esales = np.zeros((n_steps, n_products), dtype=int)
                esupplies = np.zeros((n_steps, n_products), dtype=int)
                esale_prices = np.zeros((n_steps, n_products), dtype=int)
                esupply_prices = np.zeros((n_steps, n_products), dtype=int)
                # TODO make sale_prices vary around a mean
                if l == 0:
                    esupplies[:, 0] = [exogenous_supplies[s][a] for s in range(n_steps)]
                    esupply_prices[:, 0] = supply_prices[a, :]
                if l == n_processes - 1:
                    esales[:, -1] = [exogenous_sales[s][a] for s in range(n_steps)]
                    esale_prices[:, -1] = sale_prices[a, :]
                dp = realin(shortfall_penalty)
                sc = realin(disposal_cost)
                profile_info.append(
                    (
                        OneShotProfile(
                            cost=[_ for _ in costs[nxt][0] if _ != INFINITE_COST][0],
                            input_product=l,
                            n_lines=n_lines,
                            disposal_cost_mean=sc,
                            shortfall_penalty_mean=dp,
                            disposal_cost_dev=realin(disposal_cost_dev) * sc,
                            shortfall_penalty_dev=realin(shortfall_penalty_dev) * dp,
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
        if initial_balance is None:
            # every agent at every level will have just enough to do all the needed to do cash_availability fraction of
            # production (even though it may not have enough lines to do so)
            cash_availability = realin(cash_availability)
            balance = np.ceil(
                np.sum(total_costs, axis=1) / n_agents_per_process
            ).astype(int)
            initial_balance = []
            for b, a in zip(balance, n_agents_per_process):
                initial_balance += [int(math.ceil(b * cash_availability))] * a
        b = np.sum(initial_balance)

        info.update(
            dict(
                product_prices=product_prices,
                active_lines=active_lines,
                input_quantities=input_quantity,
                output_quantities=output_quantity,
                exogenous_supplies=exogenous_supplies,
                exogenous_sales=exogenous_sales,
                expected_productivity=float(np.sum(active_lines))
                / np.sum(n_lines * n_steps * n_agents_per_process),
                expected_n_products=np.sum(active_lines, axis=-1),
                expected_income=max_income,
                expected_welfare=float(np.sum(max_income)),
                expected_income_per_step=max_income.sum(axis=0),
                expected_income_per_process=max_income.sum(axis=-1),
                expected_mean_profit=float(np.sum(max_income))
                if b != 0
                else np.sum(max_income),
                expected_profit_sum=float(n_agents * np.sum(max_income) / b)
                if b != 0
                else n_agents * np.sum(max_income),
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
            initial_balance=initial_balance,
            n_steps=n_steps,
            info=info,
            force_signing=force_signing,
            price_multiplier=price_multiplier,
            inventory_valuation_trading=0,
            inventory_valuation_catalog=0,
            penalties_scale=penalties_scale,
            **kwargs,
        )

    def current_balance(self, agent_id: str):
        return sum(self._profits[agent_id]) + self.initial_balances[agent_id]

    def add_financial_report(
        self, agent: DefaultOneShotAdapter, reports_agent, reports_time
    ) -> None:
        """
        Records a financial report for the given agent in the agent indexed
        reports and time indexed reports

        Args:
            agent: The agent
            reports_agent: A dictionary of financial reports indexed by agent id
            reports_time: A dictionary of financial reports indexed by time

        Returns:

        """
        current_balance = sum(self._profits[agent.id]) + self.initial_balances[agent.id]
        self.is_bankrupt[agent.id] = (
            current_balance < self.bankruptcy_limit
        ) or self.is_bankrupt[agent.id]
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

    def complete_contract_execution(self, *args, **kwargs):
        pass

    def start_contract_execution(self, contract: Contract) -> set[Breach] | None:
        return set()

    def _update_exogenous(self, s):
        self.exogenous_qout = defaultdict(int)
        self.exogenous_qin = defaultdict(int)
        self.exogenous_pout = defaultdict(int)
        self.exogenous_pin = defaultdict(int)
        self.__contracts = defaultdict(list)
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
            self.on_contract_concluded(contract, to_be_signed_at=self.current_step)
            if self.exogenous_force_max:
                contract.signatures = dict(zip(contract.partners, contract.partners))
            else:
                if SYSTEM_SELLER_ID in contract.partners:
                    contract.signatures[SYSTEM_SELLER_ID] = SYSTEM_SELLER_ID
                else:
                    contract.signatures[SYSTEM_BUYER_ID] = SYSTEM_BUYER_ID
        if self.exogenous_dynamic:
            raise NotImplementedError("Exogenous-dynamic is not yet implemented")

    def step_with(self, actions: dict[str, dict[str, SAOResponse]], init=False) -> bool:
        """
        Runs a simulation step for the agents given in keys passing the corresponding values as counter offers.

        Returns:
            False if this is the last negotiation.

        Remarks:
            - You must call this with `init=True` once at the beginning of every simulation to
              make sure that `init()` and other initialization code is called correctly.
            - Every step advances all negotiations one step.
            - Negotiators belonging to the given agents are never called as long as a corresponding
              action (response) is given in the agents dict.
            - The world MUST be created with `one_offer_per_step` passed as `True` (default is `False`).
        """
        from scml.oneshot.awi import OneShotAWI

        neg_actions = dict()
        existing = set(self._negotiations.keys())
        # existing = set(_.nmi.id for _ in self._current_negotiations)
        for agent, responses in actions.items():
            awi: OneShotAWI = self.agents[agent].awi  # type: ignore
            negotiations = awi.current_negotiation_details["buy"].copy()
            negotiations.update(awi.current_negotiation_details["sell"])
            for partner, neg in negotiations.items():
                neg: NegotiationDetails
                mynegs = [
                    _ for _ in neg.nmi.mechanism.negotiators if _.owner.id == agent
                ]
                assert len(mynegs) == 1
                assert neg.nmi.mechanism._one_offer_per_step  # type: ignore
                response = responses.get(partner, None)
                mid = neg.nmi.mechanism.id
                # if mid not in existing:
                #     continue

                if self._debug:
                    assert (
                        mid in existing
                    ), f"{mid} mechanism (with {partner}) does not exist for {agent}"
                if response is not None:
                    neg_actions[mid] = {mynegs[0].id: response}
                else:
                    warnings.warn(f"{agent=} has no response for partner {partner}")
        return self.step(n_neg_steps=int(not init), neg_actions=neg_actions)

    def simulation_step(self, stage):
        s = self.current_step

        if stage == 0:
            self._update_exogenous(s)

            # publish public information
            # --------------------------
            if self.publish_trading_prices:
                self.bulletin_board.record(
                    "trading_prices",
                    value=self.trading_prices,
                    key=self.current_step,
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

            # make agent ufuns
            # ================
            for aid, a in self.agents.items():
                if is_system_agent(aid):
                    continue
                a.make_ufun(add_exogenous=True)

            # zero quantities and prices
            # ==========================
            self._input_quantity = defaultdict(int)
            self._input_price = defaultdict(int)
            self._output_quantity = defaultdict(int)
            self._output_price = defaultdict(int)

            # Reset all agents
            # ================
            for aid, a in self.agents.items():
                if hasattr(a, "reset"):
                    a.reset()

            # request all negotiations
            # ========================
            self._make_negotiations()

            # initialize all agents for this step
            # ===================================
            for aid, a in self.agents.items():
                if hasattr(a, "before_step"):
                    a.before_step()

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
        self._trading_price[:, s + 1 :] = self._trading_price[:, s + 1].reshape(
            (self.n_products, 1)
        )
        self._traded_quantity += self._sold_quantity[:, s + 1]
        # self._trading_price[has_trade, s] = (
        #         np.sum(self._betas[:s+1] * self._real_price[has_trade, s::-1])
        #     ) / self._betas_sum[s+1]

        # update agent profits
        # ---------------------
        for aid, agent in self.agents.items():
            if is_system_agent(aid):
                continue
            if (
                not self.penalize_bankrupt_for_future_contracts
                and self.is_bankrupt[aid]
            ):
                continue
            # agent.profile
            # todo: I am accessing the ufun of the agent directly to avoid running
            # unnecessary optimizations to find best and worst utility. This is
            # dangerous because the agent can change its own ufun. May be I should
            # directly create the ufun here using a global ufun method defined in
            # the ufun.py module that takes a world and an agent (or just an AWI)
            qin, pin, qout, pout = (
                self._input_quantity[aid],
                self._input_price[aid],
                self._output_quantity[aid],
                self._output_price[aid],
            )
            ufun = agent.ufun
            ucon = ufun.from_contracts(self.__contracts[aid])
            self._profits[aid].append(ucon)
            self._breach_levels[aid].append(
                ufun.breach_level(
                    qin,
                    qout,
                )
            )
            self._breaches_of[aid].append(
                ufun.is_breach(
                    qin,
                    qout,
                )
            )
            current_balance = self.current_balance(aid)
            self.is_bankrupt[aid] = (
                current_balance < self.bankruptcy_limit or self.is_bankrupt[aid]
            )
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

        # Clean negotiation details
        # -------------------------
        # self._current_negotiations = []
        self._agent_negotiations = dict(
            zip(
                [_ for _ in self.agents.keys()],
                [dict(buy=dict(), sell=dict()) for _ in self.agents.keys()],
            )
        )

    def _breach_record(
        self,
        perpetrator,
        level,
        type_,
    ) -> dict[str, Any]:
        return {
            "perpetrator": perpetrator,
            "perpetrator_name": perpetrator,
            "level": level,
            "type": type_,
            "time": self.current_step,
        }

    def _adjust_contract_types(self, contract):
        for k in ("unit_price", "quantity"):
            if not isinstance(contract.agreement[k], int):
                contract.agreement[k] = int(contract.agreement[k] + 0.5)
        return contract

    def on_contract_signed(self, contract: Contract) -> bool:
        contract = self._adjust_contract_types(contract)

        product = contract.annotation["product"]
        bought = contract.agreement["quantity"]
        total_price = bought * contract.agreement["unit_price"]
        oldq = self._sold_quantity[product, self.current_step + 1]
        oldp = self._real_price[product, self.current_step + 1]
        totalp = 0.0
        if oldq > 0:
            totalp = oldp * oldq
        self._sold_quantity[product, self.current_step + 1] += bought
        self._real_price[product, self.current_step + 1] = (totalp + total_price) / (
            oldq + bought
        )
        self._input_quantity[contract.annotation["buyer"]] += bought
        self._input_price[contract.annotation["buyer"]] += total_price
        self._output_quantity[contract.annotation["seller"]] += bought
        self._output_price[contract.annotation["seller"]] += total_price
        contract.executed_at = self.current_step
        self.__contracts[contract.annotation["buyer"]].append(contract)
        self.__contracts[contract.annotation["seller"]].append(contract)
        return super().on_contract_signed(contract)

    def contract_size(self, contract: Contract) -> float:
        contract = self._adjust_contract_types(contract)
        return contract.agreement["quantity"] * contract.agreement["unit_price"]

    def contract_record(self, contract: Contract) -> dict[str, Any]:
        contract = self._adjust_contract_types(contract)
        c = {
            "id": contract.id,
            "seller_name": self.agents[contract.annotation["seller"]].name,
            "buyer_name": self.agents[contract.annotation["buyer"]].name,
            "seller_type": self.agents[contract.annotation["seller"]].type_name,
            "buyer_type": self.agents[contract.annotation["buyer"]].type_name,
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

    def breach_record(self, breach: Breach) -> dict[str, Any]:
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
            self._stats[f"balance_{aid}"].append(self.current_balance(aid))
            self._stats[f"bankrupt_{aid}"].append(self.is_bankrupt.get(aid, False))
            self._stats[f"productivity_{aid}"].append(1.0)
            self._stats[f"spot_market_quantity_{aid}"].append(0)
            self._stats[f"spot_market_loss_{aid}"].append(0)
            self._stats[f"inventory_{aid}_input"].append(0)
            self._stats[f"inventory_{aid}_output"].append(0)

    def pre_step_stats(self):
        pass

    def welfare(self, include_bankrupt: bool = False) -> float:
        """Total welfare of all agents"""
        scores = self.scores()
        return sum(
            scores[a.id] for a in self.agents.values() if not is_system_agent(a.id)
        )

    def relative_welfare(self, include_bankrupt: bool = False) -> float | None:
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

    def scores(self, assets_multiplier: float = 0.0) -> dict[str, float]:
        """
        Scores of all agents given the asset multiplier.

        Args:
            assets_multiplier: A multiplier to multiply the assets with.
        """
        scores = dict()
        for aid, agent in self.agents.items():
            if is_system_agent(aid):
                continue
            if not self.initial_balances[aid]:
                scores[aid] = self.initial_balances[aid] + sum(self._profits[aid])
                continue
            scores[aid] = (
                self.initial_balances[aid] + sum(self._profits[aid])
            ) / self.initial_balances[aid]
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
    def system_agents(self) -> list[_StdSystemAgent]:
        """Returns the two system agents"""
        return [_ for _ in self.agents.values() if is_system_agent(_.id)]

    @property
    def system_agent_names(self) -> list[str]:
        """Returns the names two system agents"""
        return [_ for _ in self.agents.keys() if is_system_agent(_)]

    @property
    def non_system_agents(self) -> list[DefaultOneShotAdapter]:
        """Returns all agents except system agents"""
        return [_ for _ in self.agents.values() if not is_system_agent(_.id)]

    @property
    def non_system_agent_names(self) -> list[str]:
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
        steps: tuple[int, int] | int | None = None,
        what: Collection[str] = DEFAULT_EDGE_TYPES,
        who: Callable[[Agent], bool] = None,
        where: Callable[[Agent], int | tuple[float, float]] = None,
        together: bool = True,
        axs: Collection[Axis] = None,
        ncols: int = 4,
        figsize: tuple[int, int] = (15, 15),
        **kwargs,
    ) -> tuple[Axis, nx.Graph] | tuple[list[Axis], list[nx.Graph]]:
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
        quantity: int | tuple[int, int],
        unit_price: int | tuple[int, int],
        time: int | tuple[int, int],
        controller: SAOController | None = None,
        negotiators: list[SAONegotiator] | None = None,
        extra: dict[str, Any] | None = None,
        consumer_starts: bool = True,
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
            consumer_starts: Whether the consumer or supplier sends the first offer in the negotiation

        Returns:

            `True` if the partner accepted and the negotiation is ready to start

        """
        if self.is_bankrupt[agent_id] or is_system_agent(agent_id):
            return True
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

        responding_agents = (
            self.suppliers[product] if consumer_starts else self.consumers[product]
        )
        partners = [
            _
            for _ in responding_agents
            if not self.is_bankrupt[_] and not is_system_agent(_)
        ]
        if not partners:
            return True
        if negotiators is None:
            negotiators = [
                controller.create_negotiator(ControlledSAONegotiator, name=_, id=_)
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
                consumer_starts=consumer_starts,
            )
            for partner, negotiator in zip(partners, negotiators)
        ]
        # for p, r in zip(partners, results):
        #     if r:
        #         self._world._registered_negs.add(tuple(sorted([P, self.agent.id])))
        if self._debug and not all(results):
            failed = set()
            for r, p in zip(results, partners):
                if not r:
                    failed.add(p)
            assert failed
            raise AssertionError(
                f"Partners {failed} failed to accept negotiation request from {agent_id}"
            )
        return all(results)

    def _request_negotiation(
        self,
        agent_id: str,
        product: int,
        quantity: int | tuple[int, int],
        unit_price: int | tuple[int, int],
        time: int | tuple[int, int],
        partner: str,
        negotiator: SAONegotiator,
        extra: dict[str, Any] | None = None,
        consumer_starts: bool = True,
    ) -> NegotiationInfo:
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
            consumer_starts: whether the consumer starts the negotiation

        Returns:

            `True` if the partner accepted and the negotiation is ready to start

        """
        agent = self.agents[agent_id]
        is_buy = consumer_starts
        if extra is None:
            extra = dict()
        # if is_buy:
        #     buyer, seller = agent, self.agents[partner]
        # else:
        #     seller, buyer = agent, self.agents[partner]
        # if product != self.agent_profiles[buyer.id].input_product:
        #     self.logerror(
        #         f"Buyer {buyer.id} wants to buy {product} but its input is different"
        #     )
        # if product != self.agent_profiles[seller.id].output_product:
        #     self.logerror(
        #         f"Seller {seller.id} wants to sell {product} but its output is different"
        #     )
        # if (
        #     self.agent_profiles[buyer.id].input_product
        #     == self.agent_profiles[seller.id].input_product
        # ):
        #     self.logerror(
        #         f"Seller {seller.id} and buyer {buyer.id} are in the same level ({self.agent_profiles[buyer.id].input_product})"
        #     )
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
        issues = self._current_issues[product]
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
        if result:
            if is_buy:
                buyer, seller = agent.id, [_ for _ in partners if _ != agent.id][0]
            else:
                seller, buyer = agent.id, [_ for _ in partners if _ != agent.id][0]
            info = NegotiationDetails(
                seller=seller,
                buyer=buyer,
                nmi=result.mechanism.nmi,  # type: ignore
                product=product,
            )
            if self._debug:
                assert result.mechanism is not None
                assert (
                    buyer in self.agents[seller].awi.my_consumers
                ), f"{seller=}, {buyer=}"
                assert (
                    seller in self.agents[buyer].awi.my_suppliers
                ), f"{seller=}, {buyer=}"
            # self._current_negotiations.append(info)
            self._agent_negotiations[seller]["sell"][buyer] = info
            self._agent_negotiations[buyer]["buy"][seller] = info
        return result

    def _make_issues(
        self, product
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        """
        Creates the negotiation agendas

        Args:
            product (int): The product to be negotiated about

        Returns:
            A tuple of minimum and maximum values for unit-price, time, and quantity in that order
        """
        price_of_product = (
            self.trading_prices[product]
            if self.publish_trading_prices
            else self.catalog_prices[product]
        )
        if self.wide_price_range:
            if product:
                p = (
                    self.trading_prices[product - 1]
                    if self.publish_trading_prices
                    else self.catalog_prices[product - 1]
                )
            else:
                p = 0
        else:
            p = price_of_product
        if self.price_multiplier > 1e-6:
            unit_price = (
                max(
                    1,
                    int(p // self.price_multiplier),
                ),
                int(self.price_multiplier * price_of_product),
            )
        else:
            ceil = int(math.ceil(price_of_product))
            unit_price = (
                max(1, ceil - 1),
                max(1, ceil),
            )
            assert unit_price[0] + 1 == unit_price[1] or unit_price[1] == 1
        time = (self.current_step, self.current_step)
        quantity = (1, self._max_n_lines)
        return unit_price, time, quantity

    def _make_negotiations(self):
        # consumer_starts = random.random() > 0.5
        consumer_starts = True

        def values(x: int | tuple[int, int]):
            if not isinstance(x, Iterable):
                return int(x), int(x)
            return int(min(x)), int(max(x))

        controllers = dict()

        for aid, a in self.agents.items():
            if is_system_agent(aid) or isinstance(a, OneShotSCML2020Adapter):
                continue
            controllers[aid] = a.adapted_object
            a.adapted_object.make_ufun(add_exogenous=True)

        # initialize negotiation details
        # self._current_negotiations = []
        self._agent_negotiations = dict(
            zip(
                [_ for _ in self.agents.keys()],
                [dict(buy=dict(), sell=dict()) for _ in self.agents.keys()],
            )
        )

        expected_negs = set()
        if self._debug:
            assert (
                len(self._negotiations) == 0
            ), f"Found unexpected negotiations at step {self.current_step}\n{self._negotiations}"
        for product in range(1, self.n_products):
            if self._debug:
                for c in self.consumers[product]:
                    if is_system_agent(c) or self.is_bankrupt[c]:
                        continue
                    for s in self.suppliers[product]:
                        if is_system_agent(s) or self.is_bankrupt[s]:
                            continue
                        expected_negs.add(tuple(sorted((c, s))))
            unit_price, time, quantity = self._make_issues(product)
            self._current_issues[product] = [
                make_issue(values(quantity), name="quantity"),
                make_issue(values(time), name="time"),
                make_issue(values(unit_price), name="unit_price"),
            ]
            requesting_agents = (
                self.consumers[product] if consumer_starts else self.suppliers[product]
            )
            for aid in requesting_agents:
                if is_system_agent(aid) or isinstance(
                    self.agents[aid], OneShotSCML2020Adapter
                ):
                    continue
                self._request_negotiations(
                    agent_id=aid,
                    product=product,
                    time=time,
                    quantity=quantity,
                    unit_price=unit_price,
                    controller=controllers[aid],
                    negotiators=None,
                    extra=None,
                    consumer_starts=consumer_starts,
                )
        if self._debug:
            found_negs = set()
            for n in self._negotiations.values():
                found_negs.add(tuple(sorted(_.id for _ in n.partners)))
            assert (
                found_negs == expected_negs
            ), f"{expected_negs=}\n{found_negs=}\n{found_negs.difference(expected_negs)=}\n{expected_negs.difference(found_negs)=}"
        # if not success:
        #     raise ValueError(
        #         f"Failed to start negotiations for product " f"{product}"
        #     )

    def order_contracts_for_execution(
        self, contracts: Collection[Contract]
    ) -> Collection[Contract]:
        for contract in contracts:
            contract.executed_at = self.current_step
        return contracts

    def get_private_state(self, agent: Agent) -> dict:
        return agent.awi.state

    def _contract_record(self, contract):
        record = super()._contract_record(contract)
        record["executed_at"] = self.current_step
        return record


class SCML2020OneShotWorld(OneShotWorld):
    pass


class SCML2021OneShotWorld(SCML2020OneShotWorld):
    def __init__(self, *args, **kwargs):
        kwargs["price_multiplier"] = 2.0
        kwargs["wide_price_range"] = True
        super().__init__(*args, **kwargs)


class SCML2022OneShotWorld(SCML2021OneShotWorld):
    pass


class SCML2023OneShotWorld(SCML2020OneShotWorld):
    def __init__(self, *args, **kwargs):
        kwargs["price_multiplier"] = 0.0
        kwargs["wide_price_range"] = False
        super().__init__(*args, **kwargs)


class SCML2024OneShotWorld(SCML2023OneShotWorld):
    pass
