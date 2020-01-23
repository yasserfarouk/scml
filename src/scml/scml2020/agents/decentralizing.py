"""
Implements the `DecentralizingAgent` which creates ony buy and one sell controller for each time-step and relinquishes
control of negotiations to buy/sell the required number of items of its input/output product.
"""

import copy
import functools
import itertools
import math
import random
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from typing import Union, Optional, Dict, Any, List, Iterable, Tuple, Type

import numpy as np
from negmas import (
    UtilityFunction,
    Outcome,
    outcome_as_dict,
    LinearUtilityFunction,
    SAONegotiator,
    Issue,
    Negotiator,
    AgentMechanismInterface,
    Contract,
    SAOController,
    MechanismState,
    ResponseType,
    AspirationMixin,
    AspirationNegotiator,
    PassThroughNegotiator,
)
from negmas.events import Notifier, Notification
from negmas.helpers import instantiate, get_class
from pprint import pformat
from scml.scml2020 import AWI, NO_COMMAND, ANY_LINE
from scml.scml2020.components import SupplyDrivenProductionStrategy
from scml.scml2020.components import StepController

from .do_nothing import DoNothingAgent
from ..world import is_system_agent

QUANTITY = 0
TIME = 1
UNIT_PRICE = 2

__all__ = ["DecentralizingAgent"]


@dataclass
class ControllerInfo:
    """Keeps a record of information about one of the controllers used by the agent"""

    controller: StepController
    time_step: int
    is_seller: bool
    time_range: Tuple[int, int]
    target: int
    expected: int
    done: bool = False


class DecentralizingAgent(SupplyDrivenProductionStrategy, DoNothingAgent):
    """An agent that keeps schedules of what it needs to buy and sell and tries to satisfy them.

    It assumes that the agent can run a single process.

    Args:

        negotiator_type: The negotiator type to use for all negotiations
        negotiator_params: The parameters used to initialize all negotiators
        horizon: The number of steps in the future to consider for selling outputs.
        predicted_demand: A prediction of the number of units needed by the market of the output
                          product at each timestep
        predicted_supply: A prediction of the nubmer of units available within the market of the input
                          product at each timestep
        agreement_fraction: A prediction about the fraction of the quantity negotiated about that will
                            be secured
        *args: Position arguments to pass the the base `SCML2020Agent` constructor
        **kwargs: Keyword arguments to pass to the base `SCML2020Agent` constructor

    """

    def __init__(
        self,
        *args,
        negotiator_type: Union[SAONegotiator, str] = AspirationNegotiator,
        negotiator_params: Optional[Dict[str, Any]] = None,
        horizon=5,
        predicted_demand: Union[int, np.ndarray] = None,
        predicted_supply: Union[int, np.ndarray] = None,
        agreement_fraction: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # save construction parameters
        self.predicted_demand = predicted_demand
        self.predicted_supply = predicted_supply
        self.negotiator_type = get_class(negotiator_type)
        self.negotiator_params = (
            negotiator_params if negotiator_params is not None else dict()
        )
        self.agreement_fraction = agreement_fraction
        self.horizon = horizon

        # attributes that will be read during init() from the AWI
        # -------------------------------------------------------
        self.exogenous_horizon = None
        """The number of steps between the revelation of an exogenous contract and its delivery time"""
        self.input_product: int = -1
        """My input product index"""
        self.output_product: int = -1
        """My output product index"""
        self.process: int = -1
        """The process I can execute"""
        self.production_cost: int = -1
        """My production cost to convert a unit of the input to a unit of the output"""
        self.inputs_needed: np.ndarray = None
        """How many items of the input product do I need at every time step"""
        self.outputs_needed: np.ndarray = None
        """How many items of the output product do I need at every time step"""
        self.input_cost: np.ndarray = None
        """Expected unit price of the input"""
        self.output_price: np.ndarray = None
        """Expected unit price of the output"""
        self.inputs_secured: np.ndarray = None
        """How many units of the input product I have already secured per step"""
        self.outputs_secured: np.ndarray = None
        """How many units of the output product I have already secured per step"""
        self.buyers = self.sellers = None
        """Buyer controllers and seller controllers. Each of them is responsible of covering the
        needs for one step (either buying or selling)."""

    def init(self):
        awi: AWI
        awi = self.awi  # type: ignore

        # read my basic parameters from the AWI
        self.exogenous_horizon = awi.bb_read("settings", "exogenous_horizon")
        self.input_product = int(awi.my_input_product)
        self.output_product = self.input_product + 1
        self.process = self.input_product
        self.production_cost = np.max(awi.profile.costs[:, self.process])
        self.input_cost = self.awi.catalog_prices[self.input_product]
        self.output_price = self.awi.catalog_prices[self.output_product]

        # initialize one controller for buying and another for selling for each time-step
        self.buyers: List[ControllerInfo] = [
            ControllerInfo(None, i, False, tuple(), 0, 0, False)
            for i in range(self.awi.n_steps)
        ]
        self.sellers: List[ControllerInfo] = [
            ControllerInfo(None, i, True, tuple(), 0, 0, False)
            for i in range(self.awi.n_steps)
        ]

        def adjust(x, demand):
            """Adjust the predicted demand/supply filling it with a default value or repeating as needed"""
            if x is None:
                x = max(1, awi.n_lines // 2)
            elif isinstance(x, Iterable):
                return np.array(x)
            predicted = int(x) * np.ones(awi.n_steps, dtype=int)
            if demand:
                predicted[: self.input_product + 1] = 0
            else:
                predicted[self.input_product - awi.n_processes: ] = 0
            return predicted

        # adjust predicted demand and supply
        self.predicted_demand = adjust(self.predicted_demand, True)
        self.predicted_supply = adjust(self.predicted_supply, False)

        # initialize needed/secured for inputs and outputs to all zeros
        self.inputs_secured = np.zeros(awi.n_steps, dtype=int)
        self.outputs_secured = np.zeros(awi.n_steps, dtype=int)
        self.inputs_needed = np.zeros(awi.n_steps, dtype=int)
        self.outputs_needed = np.zeros(awi.n_steps, dtype=int)

        # If I expect to sell x outputs at step t, I should buy  x inputs at t-1
        self.inputs_needed[:-1] = self.predicted_demand[1:]
        # If I expect to buy x inputs at step t, I should sell x inputs at t+1
        self.outputs_needed[1:] = self.predicted_supply[:-1]
        self.awi.logdebug_agent(f"Initialized\n{pformat(self._debug_state())}")

    def _debug_state(self):
        return {
            "inputs_secured": self.inputs_secured.tolist(),
            "inputs_needed": self.inputs_needed.tolist(),
            "outputs_secured": self.outputs_secured.tolist(),
            "outputs_needed": self.outputs_needed.tolist(),
            "buyers": [
                f"step: {_.controller.step} secured {_.controller.secured} of {_.controller.target} units "
                f"[Completed {len([_ for _ in _.controller.completed.values() if _])} "
                f"of {len(_.controller.completed)}]"
                for _ in self.buyers
                if _ is not None and _.controller is not None
            ],
            "sellers": [
                f"step: {_.controller.step} secured {_.controller.secured} of {_.controller.target} units "
                f"[Completed {len([_ for _ in _.controller.completed.values() if _])} "
                f"of {len(_.controller.completed)}]"
                for _ in self.sellers
                if _ is not None and _.controller is not None
            ],
            "buy_negotiations": [
                _.annotation["seller"]
                for _ in self.running_negotiations
                if _.annotation["buyer"] == self.id
            ],
            "sell_negotiations": [
                _.annotation["buyer"]
                for _ in self.running_negotiations
                if _.annotation["seller"] == self.id
            ],
            "_balance": self.awi.state.balance,
            "_input_inventory": self.awi.state.inventory[self.awi.my_input_product],
            "_output_inventory": self.awi.state.inventory[self.awi.my_output_product],
        }

    def step(self):
        """Generates buy and sell negotiations as needed"""
        self.awi.logdebug_agent(f"Enter step:\n{pformat(self._debug_state())}")
        s = self.awi.current_step
        if s == 0:
            # in the first step, generate buy/sell negotiations for horizon steps in the future
            last = min(self.awi.n_steps - 1, self.horizon + 2)
            for step in range(1, last):
                self.generate_buy_negotiations(step)
                self.generate_sell_negotiations(step)
        else:
            # generate buy and sell negotiations to secure inputs/outputs the step after horizon steps
            nxt = s + self.horizon + 1
            if nxt > self.awi.n_steps - 1:
                return
            self.generate_buy_negotiations(nxt)
            self.generate_sell_negotiations(nxt)
        self.awi.logdebug_agent(f"End step:\n{pformat(self._debug_state())}")

    def respond_to_negotiation_request(
        self,
        initiator: str,
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
    ) -> Optional[Negotiator]:

        # find negotiation parameters
        is_seller = annotation["seller"] == self.id
        tmin, tmax = issues[TIME].min_value, issues[TIME].max_value + 1
        # find the time-step for which this negotiation should be added
        step = max(0, tmin - 1) if is_seller else min(self.awi.n_steps - 1, tmax + 1)
        # find the corresponding controller.
        controller_info: ControllerInfo
        controller_info = self.sellers[step] if is_seller else self.buyers[step]
        # check if we need to negotiate and indicate that we are negotiating some amount if we need
        if is_seller:
            assert annotation["product"] == self.output_product
            target = (
                self.outputs_needed[tmin : tmax + 1].sum()
                - self.outputs_secured[tmin : tmax + 1].sum()
            )
            if target <= 0:
                return None
        else:
            assert annotation["product"] == self.input_product
            target = (
                self.inputs_needed[tmin : tmax + 1].sum()
                - self.inputs_secured[tmin : tmax + 1].sum()
            )
            if target <= 0:
                return None

        self.awi.loginfo(
            f"Accepting request from {initiator}: {[str(_) for _ in mechanism.issues]} "
            f"({Issue.num_outcomes(mechanism.issues)})"
        )
        # create a controller for the time-step if one does not exist or use the one already running
        if controller_info.controller is None:
            controller = self.add_controller(
                is_seller,
                target,
                self._urange(step, is_seller, (tmin, tmax)),
                int(self.agreement_fraction * target),
                step,
            )
        else:
            controller = controller_info.controller

        # create a new negotiator, add it to the controller and return it
        return controller.create_negotiator()

    def on_negotiation_failure(
        self,
        partners: List[str],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
        state: MechanismState,
    ) -> None:
        # inform the corresponding controller about the event
        controller = self._get_controller(mechanism)
        neg = self._running_negotiations[mechanism.id]
        negotiator_id = neg.negotiator.id
        controller.negotiation_concluded(negotiator_id, None)

    def on_negotiation_success(
        self, contract: Contract, mechanism: AgentMechanismInterface
    ) -> None:
        # inform the corresponding controller about the event
        controller = self._get_controller(mechanism)
        neg = self._running_negotiations[mechanism.id]
        negotiator_id = neg.negotiator.id
        controller.negotiation_concluded(negotiator_id, contract.agreement)

    def on_contracts_finalized(
        self,
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        self.awi.logdebug_agent(
            f"Enter Contracts Finalized:\n"
            f"Signed {pformat([self._format(_) for _ in signed])}\n"
            f"Cancelled {pformat([self._format(_) for _ in cancelled])}\n"
            f"{pformat(self._debug_state())}"
        )
        consumed = 0
        for contract in signed:
            is_seller = contract.annotation["seller"] == self.id
            q, u, t = (
                contract.agreement["quantity"],
                contract.agreement["unit_price"],
                contract.agreement["time"],
            )
            if is_seller:
                # if I am a seller, I will buy my needs to produce
                output_product = contract.annotation["product"]
                input_product = output_product - 1
                self.outputs_secured[t] += q
                if input_product >= 0 and t > 0:
                    # find the maximum possible production I can do and saturate to it
                    steps, lines = self.awi.available_for_production(
                        repeats=q, step=(self.awi.current_step, t - 1)
                    )
                    q = min(len(steps) - consumed, q)
                    consumed += q
                    if contract.annotation["caller"] != self.id:
                        # this is a sell contract that I did not expect yet. Update needs accordingly
                        self.inputs_needed[t - 1] += max(1, q)
                continue

            # I am a buyer. I need not produce anything but I need to negotiate to sell the production of what I bought
            input_product = contract.annotation["product"]
            output_product = input_product + 1
            self.inputs_secured[t] += q
            if output_product < self.awi.n_products and t < self.awi.n_steps - 1:
                if contract.annotation["caller"] != self.id:
                    # this is a buy contract that I did not expect yet. Update needs accordingly
                    self.outputs_needed[t + 1] += max(1, q)
        self.awi.logdebug_agent(
            f"Exit Contracts Finalized:\n{pformat(self._debug_state())}"
        )

    def _format(self, c: Contract):
        return (
            f"{f'>' if c.annotation['seller'] == self.id else '<'}"
            f"{c.annotation['buyer'] if c.annotation['seller'] == self.id else c.annotation['seller']}: "
            f"{c.agreement['quantity']} of {c.annotation['product']} @ {c.agreement['unit_price']} on {c.agreement['time']}"
        )

    def sign_all_contracts(self, contracts: List[Contract]) -> List[Optional[str]]:
        self.awi.logdebug_agent(
            f"Enter Sign Contracts {pformat([self._format(_) for _ in contracts])}:\n{pformat(self._debug_state())}"
        )
        # sort contracts by time and then put system contracts first within each time-step
        contracts = sorted(
            contracts,
            key=lambda x: (
                x.agreement["time"],
                0
                if is_system_agent(x.annotation["seller"])
                or is_system_agent(x.annotation["buyer"])
                else 1,
            ),
        )
        signatures = [None] * len(contracts)
        taken = 0
        s = self.awi.current_step
        for i, contract in enumerate(contracts):
            q, u, t = (
                contract.agreement["quantity"],
                contract.agreement["unit_price"],
                contract.agreement["time"],
            )
            # check that the contract is executable in principle
            if t <= s and len(contract.issues) == 3:
                continue
            if contract.annotation["seller"] == self.id:
                trange = (s, t)
                secured, needed = (
                    self.outputs_secured,
                    self.outputs_needed,
                )
            else:
                trange = (t + 1, self.awi.n_steps - 1)
                secured, needed = (self.inputs_secured, self.inputs_needed)

            # check that I can produce the required quantities even in principle
            steps, lines = self.awi.available_for_production(
                q, trange, ANY_LINE, override=False, method="all"
            )
            if len(steps) - taken < q:
                continue
            taken += q

            if (
                secured[trange[0] : trange[1] + 1].sum()
                + q
                + taken * self.agreement_fraction
                <= needed[trange[0] : trange[1] + 1].sum()
            ):
                signatures[i] = self.id
        self.awi.logdebug_agent(f"Exit Sign Contracts:\n{pformat(self._debug_state())}")
        return signatures

    def confirm_production(
        self, commands: np.ndarray, balance: int, inventory: np.ndarray
    ) -> np.ndarray:
        return super().confirm_production(commands, balance, inventory)

    def on_agent_bankrupt(
        self,
        agent: str,
        contracts: List[Contract],
        quantities: List[int],
        compensation_money: int,
    ) -> None:
        for contract, new_quantity in zip(contracts, quantities):
            q = contract.agreement["quantity"]
            if new_quantity < q:
                t = contract.agreement["time"]
                missing = q - new_quantity
                if t < self.awi.current_step:
                    return
                if contract.annotation["seller"] == self.id:
                    self.outputs_secured[t] -= missing
                    if t > 0:
                        self.inputs_needed[t - 1] -= missing
                    if (
                        self.sellers[t] is not None
                        and self.sellers[t].controller is not None
                    ):
                        self.sellers[t].controller.target -= missing
                else:
                    self.inputs_secured[t] += missing
                    if t < self.awi.n_steps - 1:
                        self.outputs_needed[t + 1] -= missing
                    if (
                        self.buyers[t] is not None
                        and self.buyers[t].controller is not None
                    ):
                        self.buyers[t].controller.target += missing

    def all_negotiations_concluded(
        self, controller_index: int, is_seller: bool
    ) -> None:
        """Called by the `StepController` to affirm that it is done negotiating for some time-step"""
        info = (
            self.sellers[controller_index]
            if is_seller
            else self.buyers[controller_index]
        )
        info.done = True
        c = info.controller
        quantity = c.secured
        target = c.target
        expected = info.expected
        time_range = info.time_range
        if is_seller:
            secured, needed = (
                self.outputs_secured,
                self.outputs_needed,
            )
            controllers, generator = self.sellers, self.generate_sell_negotiations
        else:
            secured, needed = (self.inputs_secured, self.inputs_needed)
            controllers, generator = self.buyers, self.generate_buy_negotiations

        self.awi.loginfo(
            f"Killing Controller {str(controllers[controller_index].controller)}"
        )
        controllers[controller_index].controller = None
        if quantity <= target:
            secured[time_range[0]] += quantity
            generator(step=controller_index)
            return

    def generate_buy_negotiations(self, step):
        """Creates the controller and starts negotiations to acquire all required inputs (supplies) at the given step"""
        quantity = (
            self.inputs_needed[step]
            - self.inputs_secured[step]
            # - self.inputs_negotiating[step]
        )
        if quantity <= 0:
            return
        self.start_negotiations(
            product=self.input_product,
            step=step,
            quantity=max(
                1, min(self.awi.n_lines * (step - self.awi.current_step), quantity)
            ),
            unit_price=self.output_price - self.production_cost,
        )

    def generate_sell_negotiations(self, step):
        """Creates the controller and starts negotiations to sell all required outputs (sales) at the given step"""

        # find out if I need to sell any output products
        quantity = (
            self.outputs_needed[step]
            - self.outputs_secured[step]
            # - self.outputs_negotiating[step]
        )
        if quantity <= 0:
            return
        # if so, negotiate to sell as much of them as possible
        self.start_negotiations(
            product=self.output_product,
            step=step,
            quantity=max(
                1, min(self.awi.n_lines * (step - self.awi.current_step), quantity)
            ),
            unit_price=self.production_cost + self.input_cost,
        )

    def start_negotiations(
        self, product: int, quantity: int, unit_price: int, step: int
    ) -> None:
        """
        Starts a set of negotiations to by/sell the product with the given limits

        Args:
            product: product type. If it is an input product, negotiations to buy it will be started otherweise to sell.
            quantity: The maximum quantity to negotiate about
            unit_price: The maximum/minimum unit price for buy/sell
            step: The maximum/minimum time for buy/sell

        Remarks:

            - This method assumes that products cannot be in my_input_products and my_output_products

        """
        awi: AWI
        awi = self.awi  # type: ignore
        is_seller = product == self.output_product
        if quantity < 1 or unit_price < 1 or step < awi.current_step + 1:
            awi.logdebug(
                f"Less than 2 valid issues (q:{quantity}, u:{unit_price}, t:{step})"
            )
            return
        # choose ranges for the negotiation agenda.
        qvalues = (1, quantity)
        tvalues = self._trange(step, is_seller)
        uvalues = self._urange(step, is_seller, tvalues)
        if tvalues[0] > tvalues[1]:
            return
        if is_seller:
            partners = awi.my_consumers
            expected_quantity = int(math.floor(qvalues[1] * self.agreement_fraction))
        else:
            partners = awi.my_suppliers
            expected_quantity = int(math.floor(qvalues[1] * self.agreement_fraction))

        # negotiate with everyone
        controller = self.add_controller(
            is_seller, qvalues[1], uvalues, expected_quantity, step
        )
        awi.loginfo(
            f"Requesting {'selling' if is_seller else 'buying'} negotiation "
            f"on u={uvalues}, q={qvalues}, t={tvalues}"
            f" with {str(partners)} using {str(controller)}"
        )
        self.awi.request_negotiations(
            is_buy=not is_seller,
            product=product,
            quantity=qvalues,
            unit_price=uvalues,
            time=tvalues,
            partners=partners,
            controller=controller,
            extra=dict(controller_index=step, is_seller=is_seller),
        )

    def max_production_till(self, step) -> int:
        """Returns the maximum number of units that can be produced until the given step given current production
        schedule"""
        n = self.awi.n_lines * (step - self.awi.current_step + 1)
        steps, lines = self.awi.available_for_production(
            repeats=n, step=(self.awi.current_step, step - 1)
        )
        return len(steps)

    def max_consumption_till(self, step) -> int:
        """Returns the maximum number of units that can be consumed until the given step given current production
        schedule"""
        n = self.awi.n_lines * (step - self.awi.current_step + 1)
        steps, lines = self.awi.available_for_production(
            repeats=n, step=(self.awi.current_step, step - 1)
        )
        return len(steps)

    def add_controller(
        self,
        is_seller: bool,
        target,
        urange: Tuple[int, int],
        expected_quantity: int,
        step: int,
    ) -> StepController:
        if is_seller and self.sellers[step].controller is not None:
            return self.sellers[step].controller
        if not is_seller and self.buyers[step].controller is not None:
            return self.buyers[step].controller
        controller = StepController(
            is_seller=is_seller,
            target_quantity=target,
            negotiator_type=self.negotiator_type,
            negotiator_params=self.negotiator_params,
            step=step,
            urange=urange,
            product=self.output_product if is_seller else self.input_product,
            partners=self.awi.my_consumers if is_seller else self.awi.my_suppliers,
            horizon=self.horizon,
            negotiations_concluded_callback=functools.partial(
                DecentralizingAgent.all_negotiations_concluded, self
            ),
            parent_name=self.name,
            awi=self.awi,
        )
        if is_seller:
            assert self.sellers[step].controller is None
            self.sellers[step] = ControllerInfo(
                controller,
                step,
                is_seller,
                self._trange(step, is_seller),
                target,
                expected_quantity,
                False,
            )
        else:
            assert self.buyers[step].controller is None
            self.buyers[step] = ControllerInfo(
                controller,
                step,
                is_seller,
                self._trange(step, is_seller),
                target,
                expected_quantity,
                False,
            )
        return controller

    def _urange(self, step, is_seller, time_range):
        if is_seller:
            cprice = self.awi.catalog_prices[self.output_product]
            return cprice, 2 * cprice

        cprice = self.awi.catalog_prices[self.input_product]
        return 1, cprice

    def _trange(self, step, is_seller):
        if is_seller:
            return (
                max(step, self.awi.current_step + 1),
                min(step + self.horizon, self.awi.n_steps - 1),
            )
        return self.awi.current_step + 1, step - 1

    def _get_controller(self, mechanism) -> StepController:
        neg = self._running_negotiations[mechanism.id]
        return neg.negotiator.parent
        # if neg.my_request:
        #     controller_index = neg.extra["controller_index"]
        #     if neg.extra["is_seller"]:
        #         return self.sellers[controller_index].controller
        #     else:
        #         return self.buyers[controller_index].controller
        # else:
