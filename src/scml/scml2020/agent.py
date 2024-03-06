"""Implements the world class for the SCML2020 world """
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from negmas import (
    Agent,
    Breach,
    Contract,
    DiscreteCartesianOutcomeSpace,
    Issue,
    MechanismState,
    Negotiator,
    NegotiatorMechanismInterface,
    RenegotiationRequest,
    make_issue,
    make_os,
)
from negmas.helpers import get_full_type_name, instantiate
from negmas.situated import Adapter

from .common import Failure
from ..oneshot.agent import OneShotAgent
from ..oneshot.common import OneShotProfile, OneShotState
from ..oneshot.mixins import OneShotUFunCreatorMixin
from .components.production import DemandDrivenProductionStrategy
from .components.signing import SignAll
from .components.trading import MarketAwareTradePredictionStrategy

__all__ = [
    "SCML2020Agent",
    "OneShotAdapter",
]


class SCML2020Agent(Agent):
    """Base class for all SCML2020 agents (factory managers)"""

    def init(self):
        pass

    def before_step(self):
        pass

    def step_(self):
        if self.awi.is_bankrupt():
            return
        super().step_()

    def step(self):
        pass

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type_name,
            "level": self.awi.my_input_product if self.awi else None,
            "levels": self.awi.my_input_products if self.awi else None,
        }

    def _respond_to_negotiation_request(
        self,
        initiator: str,
        partners: List[str],
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: NegotiatorMechanismInterface,
        role: Optional[str],
        req_id: Optional[str],
    ) -> Optional[Negotiator]:
        if self.awi.is_bankrupt():
            return None
        return self.respond_to_negotiation_request(
            initiator, issues, annotation, mechanism
        )

    def on_contract_breached(
        self, contract: Contract, breaches: List[Breach], resolution: Optional[Contract]
    ) -> None:
        pass

    def on_contract_executed(self, contract: Contract) -> None:
        pass

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

    def on_neg_request_accepted(
        self, req_id: str, mechanism: NegotiatorMechanismInterface
    ):
        pass

    @property
    def internal_state(self) -> Dict[str, Any]:
        """Returns the internal state of the agent for debugging purposes"""
        return {}

    def on_negotiation_failure(
        self,
        partners: List[str],
        annotation: Dict[str, Any],
        mechanism: NegotiatorMechanismInterface,
        state: MechanismState,
    ) -> None:
        """Called whenever a negotiation ends without agreement"""

    def on_negotiation_success(
        self, contract: Contract, mechanism: NegotiatorMechanismInterface
    ) -> None:
        """Called whenever a negotiation ends with agreement"""

    def on_agent_bankrupt(
        self,
        agent: str,
        contracts: List[Contract],
        quantities: List[int],
        compensation_money: int,
    ) -> None:
        """
        Called whenever a contract is nullified (because the partner is bankrupt)

        Args:

            agent: The ID of the agent that went bankrupt.
            contracts: All future contracts between this agent and the bankrupt agent.
            quantities: The actual quantities that these contracts will be executed at.
            compensation_money: The compensation money that is already added to the agent's wallet (if ANY).


        Remarks:

            - compensation_money will be nonzero iff immediate_compensation is enabled for this world


        """

    def on_failures(self, failures: List["Failure"]) -> None:
        """
        Called whenever there are failures either in production or in execution of guaranteed transactions

        Args:

            failures: A list of `Failure` s.
        """

    def respond_to_negotiation_request(
        self,
        initiator: str,
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: NegotiatorMechanismInterface,
    ) -> Optional[Negotiator]:
        """
        Called whenever another agent requests a negotiation with this agent.

        Args:
            initiator: The ID of the agent that requested this negotiation
            issues: Negotiation issues
            annotation: Annotation attached with this negotiation
            mechanism: The `NegotiatorMechanismInterface` interface to the mechanism to be used for this negotiation.

        Returns:
            None to reject the negotiation, otherwise a negotiator
        """

    def confirm_production(
        self, commands: np.ndarray, balance: int, inventory
    ) -> np.ndarray:
        """
        Called just before production starts at every time-step allowing the agent to change what is to be
        produced in its factory

        Args:

            commands: an n_lines vector giving the process to be run at every line (NO_COMMAND indicates nothing to be
                      processed
            balance: The current balance of the factory
            inventory: an n_products vector giving the number of items available in the inventory of every product type.

        Returns:

            an n_lines vector giving the process to be run at every line (NO_COMMAND indicates nothing to be
            processed

        Remarks:

            - Not called in SCML2020 competition.
            - The inventory will contain zero items of all products that the factory does not buy or sell
            - The default behavior is to just retrun commands confirming production of everything.
        """
        return commands

    def sign_all_contracts(self, contracts: List[Contract]) -> List[Optional[str]]:
        """Signs all contracts"""
        return [self.id] * len(contracts)


class _SystemAgent(SCML2020Agent):
    """Implements an agent for handling system operations"""

    def __init__(self, *args, role, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = role
        self.name = role

    def on_agent_bankrupt(
        self,
        agent: str,
        contracts: List[Contract],
        quantities: List[int],
        compensation_money: int,
    ) -> None:
        pass

    def on_failures(self, failures: List["Failure"]) -> None:
        pass

    def respond_to_negotiation_request(
        self,
        initiator: str,
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: NegotiatorMechanismInterface,
    ) -> Optional[Negotiator]:
        pass

    def step(self):
        pass

    def before_step(self):
        pass

    def init(self):
        pass

    def on_negotiation_failure(
        self,
        partners: List[str],
        annotation: Dict[str, Any],
        mechanism: NegotiatorMechanismInterface,
        state: MechanismState,
    ) -> None:
        pass

    def on_negotiation_success(
        self, contract: Contract, mechanism: NegotiatorMechanismInterface
    ) -> None:
        pass

    def on_contract_executed(self, contract: Contract) -> None:
        pass

    def on_contract_breached(
        self, contract: Contract, breaches: List[Breach], resolution: Optional[Contract]
    ) -> None:
        pass

    def sign_all_contracts(self, contracts: List[Contract]) -> List[Optional[str]]:
        """Signs all contracts"""
        return [self.id] * len(contracts)


class AWIHelper:
    """An AWI to use with the embedded OneShotAgent"""

    def __init__(self, owner: "OneShotAdapter"):
        self._owner = owner
        self._world = owner.awi._world

    @property
    def sales(self) -> dict[str, int]:
        """Sales per customer so far (this day)"""
        return defaultdict(int)

    @property
    def supplies(self) -> dict[str, int]:
        """Supplies per supplier so far (this day)"""
        return defaultdict(int)

    @property
    def total_sales(self) -> int:
        """Total sales so far (this day)"""
        return 0

    @property
    def total_supplies(self) -> int:
        """Total supplies so far (this day)"""
        return 0

    @property
    def needed_sales(self) -> int:
        """Sales that need to be secured (exogenous input + total supplies - exogenous output - total sales so far)"""
        return 0

    @property
    def needed_supplies(self) -> int:
        """Supplies that need to be secured (exogenous output + total sales - exogenous input - total supplies so far)"""
        return 0

    @property
    def current_inventory(self) -> tuple[int, int]:
        """Current input and output inventory quantity"""
        return (0, 0)

    @property
    def current_inventory_input(self) -> int:
        """Current input inventory quantity"""
        return 0

    @property
    def current_inventory_output(self) -> int:
        """Current output inventory quantity"""
        return 0

    # private information
    # ===================
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
    def is_exogenous_forced(self):
        """
        Are exogenous contracts forced in the sense that the agent cannot decide
        not to sign them?
        """
        return self.bb_read("settings", "force_signing") or self.bb_read(
            "settings", "force_exogenous"
        )

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
        return self.my_input_product == 0

    @property
    def is_last_level(self):
        return self.my_output_product == self.n_products - 1

    @property
    def is_middle_level(self):
        return 0 < self.my_input_product < self.n_products - 2

    @property
    def level(self):
        return self.my_input_product

    @property
    def profile(self) -> OneShotProfile:
        """Gets the profile (static private information) associated with the agent"""
        return self._owner.get_profile()

    def state(self) -> OneShotState:
        q, p = self._owner.get_exogenous_input()
        qo, po = self._owner.get_exogenous_output()
        # TODO add missing components
        all_agents = [_ for _ in self._world.agents.keys() if self.is_system(_)]
        return OneShotState(
            production_capacities=[],
            perishable=True,
            exogenous_input_quantity=q,
            exogenous_input_price=p,
            exogenous_output_quantity=qo,
            exogenous_output_price=po,
            disposal_cost=self._owner.get_disposal_cost(),
            shortfall_penalty=self._owner.get_shortfall_penalty(),
            current_balance=self._owner.get_current_balance(),
            total_sales=0,
            total_supplies=0,
            total_future_sales=0,
            total_future_supplies=0,
            n_products=self.n_products,
            n_processes=self.n_processes,
            n_competitors=self.n_competitors,
            all_suppliers=self.all_suppliers,
            all_consumers=self.all_consumers,
            catalog_prices=self.catalog_prices.tolist(),
            price_multiplier=1.0,
            is_exogenous_forced=False,
            current_step=self._owner.awi.current_step,
            n_steps=self._owner.awi.n_steps,
            relative_simulation_time=self._owner.awi.relative_time,
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
            my_partners=self.my_partners,
            penalties_scale=self.penalties_scale,  # type: ignore
            n_input_negotiations=self.n_input_negotiations,
            n_output_negotiations=self.n_output_negotiations,
            trading_prices=self.trading_prices.tolist(),
            exogenous_contract_summary=self.exogenous_contract_summary,
            current_input_outcome_space=self.current_input_outcome_space,
            current_output_outcome_space=self.current_output_outcome_space,
            current_negotiation_details=dict(),
            sales=dict(),
            supplies=dict(),
            needed_sales=0,
            needed_supplies=0,
            bankrupt_agents=[_ for _ in all_agents if self.is_bankrupt(_)],
            reports_of_agents=dict(
                zip(
                    all_agents,
                    [self.reports_of_agent(_) for _ in all_agents],
                )
            ),
            # running_negotiations=dict(),
        )

    @property
    def is_perishable(self) -> bool:
        """Are all products perishable (original design of OneShot)"""
        return False

    @property
    def current_storage_cost(self) -> float:
        """Cost of storing one unit (penalizes buying too much/ selling too little)"""
        return 0.0

    @property
    def price_multiplier(self):
        return self._owner.price_multiplier

    @property
    def current_exogenous_input_quantity(self) -> int:
        """
        The exogenous contracts for the input (this step)
        """
        return self._owner.get_exogenous_input()[0]

    @property
    def current_exogenous_input_price(self) -> int:
        """
        The exogenous contracts for the input (this step)
        """
        return self._owner.get_exogenous_input()[1]

    @property
    def current_exogenous_output_quantity(self) -> int:
        """
        The exogenous contracts for the input (this step)
        """
        return self._owner.get_exogenous_output()[0]

    @property
    def current_exogenous_output_price(self) -> int:
        """
        The exogenous contracts for the input (this step)
        """
        return self._owner.get_exogenous_output()[1]

    @property
    def current_disposal_cost(self) -> float:
        """Cost of storing one unit (penalizes buying too much/ selling too little)"""
        return self._owner.get_disposal_cost()

    @property
    def current_shortfall_penalty(self) -> float:
        """Cost of failure to deliver one unit (penalizes buying too little / selling too much)"""
        return self._owner.get_shortfall_penalty()

    @property
    def current_input_issues(self) -> List[Issue]:
        if self.my_input_product == 0:
            issues = []
        else:
            u, t, q = self._owner._make_issues(self.my_input_product)
            issues = [
                make_issue(values=q, name="quantity"),
                make_issue(values=t, name="time"),
                make_issue(values=u, name="unit_price"),
            ]
        return issues  # type: ignore

    @property
    def current_output_issues(self) -> List[Issue]:
        if self.my_output_product == 0:
            issues = []
        else:
            u, t, q = self._owner._make_issues(self.my_output_product)
            issues = [
                make_issue(values=q, name="quantity"),
                make_issue(values=t, name="time"),
                make_issue(values=u, name="unit_price"),
            ]
        return issues  # type: ignore

    @property
    def current_input_outcome_space(self) -> DiscreteCartesianOutcomeSpace:
        return make_os(self.current_input_issues) if self.current_input_issues else None  # type: ignore

    @property
    def current_output_outcome_space(self) -> DiscreteCartesianOutcomeSpace:
        return (  # type: ignore
            make_os(self.current_output_issues) if self.current_output_issues else None
        )

    @property
    def penalties_scale(self) -> Literal["trading", "catalog", "input", "unit", "none"]:
        return "unit"

    # Public Information
    # ==================
    @property
    def exogenous_contract_summary(self) -> List[Tuple[int, int]]:
        """
        The exogenous contracts in the current step for all products

        Returns:
            A list of tuples giving the total quantity and total price of
            all revealed exogenous contracts of all products at the current
            step.
        """
        if not self._world.publish_exogenous_summary:
            return []
        n_steps = self._owner.awi.n_steps
        n_products = self._owner.awi.n_products
        y = self._world.exogenous_contracts_summary
        # we shift each product down by its number
        x = np.zeros_like(self._world.exogenous_contracts_summary)
        for i in range(0, n_products):
            x[i, 0 : n_steps - i, :] = y[i, i:n_steps, :]
        summary = [
            (int(x[i, self.current_step, 0]), int(x[i, self.current_step, 1]))
            for i in range(n_products)
        ]
        return summary

    @property
    def allow_zero_quantity(self) -> bool:
        """
        Does negotiations allow zero quantity?
        """
        return False

    # Everything else
    # ===============
    def __getattr__(self, attr):
        return getattr(self._owner.awi, attr)


class OneShotAdapter(
    SignAll,
    DemandDrivenProductionStrategy,
    MarketAwareTradePredictionStrategy,
    SCML2020Agent,
    Adapter,
    OneShotUFunCreatorMixin,
):
    """
    An adapter allowing agents developed for SCML-OneShot to run in
    `SCML2020World` simulations.
    """

    def __init__(
        self,
        oneshot_type: Union[str, OneShotAgent],
        oneshot_params: Dict[str, Any],
        obj: Optional[OneShotAgent] = None,
        name=None,
        type_postfix="",
        ufun=None,
    ):
        if obj:
            self.oneshot_type = get_full_type_name(obj)  # type: ignore
            # note that this may not be true and we cannot guarantee that
            # we can instantiate an agent of the same type
            self.oneshot_params = dict()
        else:
            if not oneshot_params:
                oneshot_params = dict()
            self.oneshot_type, self.oneshot_params = (
                get_full_type_name(oneshot_type),  # type: ignore
                oneshot_params,
            )
            obj = instantiate(oneshot_type, **oneshot_params)  # type: ignore
        super().__init__(obj=obj, name=name, type_postfix=type_postfix, ufun=ufun)
        obj.connect_to_2021_adapter(self)  # type: ignore
        self._obj: SCML2020Agent

    def init(self):
        self._oneshot_awi = AWIHelper(self)
        self._obj._awi = self._oneshot_awi  # type: ignore
        self._obj.ufun = self.make_ufun(add_exogenous=True)  # type: ignore
        super().init()
        self._obj.init()

    @property
    def price_multiplier(self):
        return 1.2

    def _make_issues(self, product):
        unit_price = (
            max(
                1,
                int(
                    1.0
                    / self.price_multiplier
                    * (
                        self.awi.trading_prices[product - 1]
                        if self.awi._world.publish_trading_prices
                        else self.awi.catalog_prices[product - 1]
                    )
                    if product
                    else 0
                ),
            ),
            int(
                self.price_multiplier
                * (
                    self.awi.trading_prices[product]
                    if self.awi._world.publish_trading_prices
                    else self.awi.catalog_prices[product]
                )
            ),
        )
        unit_price = (int(unit_price[0]), int(max(unit_price)))
        t = self.awi.current_step + (self.awi.my_output_product == product)
        time = (t, t)
        quantity = (1, self.awi.n_lines)
        return unit_price, time, quantity

    def before_step(self):
        if self.awi.my_input_product == 0:
            pass
        else:
            u, t, q = self._make_issues(self.awi.my_input_product)
            self.awi.request_negotiations(
                is_buy=True,
                product=self.awi.my_input_product,
                quantity=q,
                time=t,
                unit_price=u,
                controller=self._obj,
            )
        if self.awi.my_output_product == self.awi.n_products - 1:
            pass
        else:
            u, t, q = self._make_issues(self.awi.my_output_product)
            self.awi.request_negotiations(
                is_buy=False,
                product=self.awi.my_output_product,
                quantity=q,
                time=t,
                unit_price=u,
                controller=self._obj,
            )
        self.utility_function = self.make_ufun(add_exogenous=True)
        super().before_step()
        if hasattr(self._obj, "before_step"):
            self._obj.before_step()

    def step(self):
        super().step()
        if hasattr(self._obj, "before_step"):
            self._obj.before_step()

    def make_ufun(self, add_exogenous: bool):  # type: ignore
        return super().make_ufun(add_exogenous, in_adapter=True)  # type: ignore

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type_name,
            "adapted_type": self._obj.type_name,
            "level": self.awi.my_input_product if self.awi else None,
            "levels": self.awi.my_input_products if self.awi else None,
        }

    def respond_to_negotiation_request(
        self,
        initiator,
        issues,
        annotation,
        mechanism,
    ):
        # reject all negotiations that do not match my agenda in time
        # if (
        #     issues[TIME].min_value != self.awi.current_step
        #     or issues[TIME].max_value != self.awi.current_step + 1
        # ):
        #     return None
        return self._obj.create_negotiator()  # type: ignore

    def get_disposal_cost(self) -> float:
        # penalty for buying too much
        return 0.0

    def get_shortfall_penalty_mean(self):
        return self.get_shortfall_penalty()

    def get_disposal_cost_mean(self):
        return self.get_disposal_cost()

    def get_shortfall_penalty_dev(self):
        return 0.0

    def get_disposal_cost_dev(self):
        return 0.0

    def get_storage_cost_mean(self):
        return 0.0

    def get_storage_cost_dev(self):
        return 0.0

    def get_profile(self):
        return OneShotProfile(
            cost=float(self.awi.profile.costs[:, self.awi.my_input_product].mean()),
            n_lines=self.awi.profile.n_lines,
            input_product=self.awi.my_input_product,
            shortfall_penalty_mean=self.get_shortfall_penalty_mean(),
            disposal_cost_mean=self.get_disposal_cost_mean(),
            shortfall_penalty_dev=self.get_shortfall_penalty_dev(),
            disposal_cost_dev=self.get_disposal_cost_dev(),
            storage_cost_mean=self.get_storage_cost_mean(),
            storage_cost_dev=self.get_storage_cost_dev(),
        )

    def get_shortfall_penalty(self):
        return self.awi.trading_prices[self.awi.my_output_product]

    def get_current_balance(self):
        return self.awi.current_balance

    # todo: correct this
    def get_exogenous_output(self) -> Tuple[int, int]:
        return 0, 0

    def get_exogenous_input(self) -> Tuple[int, int]:
        return 0, 0

    @property
    def is_perishable(self) -> bool:
        """Are all products perishable (original design of OneShot)"""
        return False

    @property
    def current_disposal_cost(self) -> float:
        """Cost of storing one unit (penalizes buying too much/ selling too little)"""
        return 0.0

    @property
    def current_storage_cost(self) -> float:
        """Cost of storing one unit (penalizes buying too much/ selling too little)"""
        return 0.0

    @property
    def current_shortfall_penalty(self) -> float:
        """Cost of failure to deliver one unit (penalizes buying too little / selling too much)"""
        return 0.0

    @property
    def allow_zero_quantity(self) -> bool:
        """
        Does negotiations allow zero quantity?
        """
        return False
