"""
Implements the `DecentralizingAgent` which creates ony buy and one sell controller for each time-step and relinquishes
control of negotiations to buy/sell the required number of items of its input/output product.
"""

import copy
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
from scml.scml2020 import AWI

from .do_nothing import DoNothingAgent

QUANTITY = 1
TIME = 2
UNIT_PRICE = 3


class StepController(SAOController, AspirationMixin, Notifier):
    """A controller for managing a set of negotiations about selling/buying the a product starting/ending at some
    specific time-step. It works in conjunction with the `DecentralizingAgent` .

    Remarks:

        - It uses whatever negotiator type on all of its negotiations and it assumes that the ufun will never change
        - Once it accumulates the required quantity, it ends all remaining negotiations
        - It assumes that all ufuns are identical so there is no need to keep a separate negotiator for each one and it
          instantiates a single negotiator that dynamically changes the AMI but always uses the same ufun.

    """

    def __init__(
        self,
        *args,
        target_quantity: int,
        is_seller: bool,
        parent: "DecentralizingAgent",
        step: int,
        urange: Tuple[int, int],
        product: int,
        partners: List[str],
        negotiator_type: SAONegotiator,
        negotiator_params: Dict[str, Any] = None,
        max_retries: int = 2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.__parent = parent
        self.is_seller = is_seller
        self.target = target_quantity
        self.urange = urange
        self.partners = partners
        self.product = product
        negotiator_params = (
            negotiator_params if negotiator_params is not None else dict()
        )
        self.secured = 0
        if is_seller:
            self.ufun = LinearUtilityFunction((1, 1, 10))
        else:
            self.ufun = LinearUtilityFunction((1, -1, -10))
        negotiator_params["ufun"] = self.ufun
        self.__negotiator = instantiate(negotiator_type, **negotiator_params)
        self.completed = defaultdict(bool)
        self.step = step
        self.retries: Dict[str, int] = defaultdict(int)
        self.max_retries = max_retries

    def join(
        self,
        negotiator_id: str,
        ami: AgentMechanismInterface,
        state: MechanismState,
        *,
        ufun: Optional["UtilityFunction"] = None,
        role: str = "agent",
    ) -> bool:
        joined = super().join(negotiator_id, ami, state, ufun=ufun, role=role)
        if joined:
            self.completed[negotiator_id] = False
        return joined

    def propose(self, negotiator_id: str, state: MechanismState) -> Optional["Outcome"]:
        self.__negotiator._ami = self.negotiators[negotiator_id][0]._ami
        return self.__negotiator.propose(state)

    def respond(
        self, negotiator_id: str, state: MechanismState, offer: "Outcome"
    ) -> ResponseType:
        if self.secured >= self.target:
            return ResponseType.END_NEGOTIATION
        self.__negotiator._ami = self.negotiators[negotiator_id][0]._ami
        return self.__negotiator.respond(offer=offer, state=state)

    def create_negotiator(
        self,
        negotiator_type: Union[str, Type[PassThroughNegotiator]] = None,
        name: str = None,
        cntxt: Any = None,
        **kwargs,
    ) -> PassThroughNegotiator:
        neg = super().create_negotiator(negotiator_type, name, cntxt, **kwargs)
        self.completed[neg.id] = False
        return neg

    def negotiation_concluded(
        self, negotiator_id: str, agreement: Dict[str, Any]
    ) -> None:
        awi: AWI
        awi = self.__parent.awi  # type: ignore
        # mark this negotiation as completed
        self.completed[negotiator_id] = True
        # if there is an agreement increase the secured amount and check if we are done.
        if agreement is not None:
            self.secured += agreement["quantity"]
            if self.secured >= self.target:
                # If we are done, end all other negotiations
                for k in self.negotiators.keys():
                    if self.completed[k]:
                        continue
                    self.notify(
                        self.negotiators[k][0], Notification("end_negotiation", None)
                    )

        if all(self.completed.values()):
            # If we secured everything, just return control to the agent
            if self.secured >= self.target:
                self.__parent.all_negotiations_concluded(self.step, self.is_seller)
                return
            # If we did not secure everything we need yet and time allows it, create new negotiations
            tmin, tmax = self.__parent._trange(self.step, self.is_seller)

            if awi.current_step < tmax and tmin <= tmax:
                # get a good partner: one that was not retired too much
                random.shuffle(self.partners)
                for other in self.partners:
                    if self.retries[other] <= self.max_retries:
                        partner = other
                        break
                else:
                    return
                neg = self.create_negotiator()
                self.completed[neg.id] = False
                awi.request_negotiation(
                    not self.is_seller,
                    product=self.product,
                    quantity=(1, self.target - self.secured),
                    unit_price=self.urange,
                    time=(tmin, tmax),
                    partner=partner,
                    negotiator=neg,
                    extra=dict(
                        controller_index=self.step,
                        is_seller=self.is_seller,
                    ),
                )


@dataclass
class ControllerInfo:
    controller: StepController
    time_step: int
    is_seller: bool
    time_range: Tuple[int, int]
    target: int
    expected: int
    done: bool = False


class DecentralizingAgent(DoNothingAgent):
    """An agent that keeps schedules of what it needs to buy and sell and tries to satisfy them.

    It assumes that the agent can run a single process

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
        self.predicted_demand = predicted_demand
        self.predicted_supply = predicted_supply
        self.negotiator_type = get_class(negotiator_type)
        self.negotiator_params = (
            negotiator_params if negotiator_params is not None else dict()
        )
        self.horizon = horizon
        self.input_product: int = -1
        self.output_product: int = -1
        self.process: int = -1
        self.pcost: int = -1
        self.n_inputs: int = -1
        self.n_outputs: int = -1
        self.supplies_needed: np.ndarray = None
        self.sales_needed: np.ndarray = None
        self.input_cost: np.ndarray = None
        self.output_price: np.ndarray = None
        self.supplies_secured: np.ndarray = None
        self.sales_secured: np.ndarray = None
        self.production_factor = 1
        self.buyers = self.sellers = None
        self.catalog_n_equivalent = 0
        self.supplies_negotiating = None
        self.sales_negotiating = None
        self.agreement_fraction = agreement_fraction

    def init(self):
        awi: AWI
        awi = self.awi  # type: ignore
        self.buyers: List[ControllerInfo] = [
            ControllerInfo(None, i, False, tuple(), 0, 0, False)
            for i in range(self.awi.n_steps)
        ]
        self.sellers: List[ControllerInfo] = [
            ControllerInfo(None, i, True, tuple(), 0, 0, False)
            for i in range(self.awi.n_steps)
        ]
        self.catalog_n_equivalent = self.awi.n_steps * 2

        def adjust(x):
            if x is None:
                x = max(1, awi.n_lines // 2)
            elif isinstance(x, Iterable):
                return np.array(x)
            return np.array([int(x)] * awi.n_steps)

        self.input_product = int(awi.my_input_product)
        self.output_product = self.input_product + 1
        self.predicted_demand = adjust(self.predicted_demand)
        self.predicted_supply = adjust(self.predicted_supply)
        if self.input_product == 0:
            self.predicted_supply = np.zeros(awi.n_steps, dtype=int)
            self.predicted_demand = np.zeros(awi.n_steps, dtype=int)
        if self.output_product == awi.n_products - 1:
            self.predicted_supply = np.zeros(awi.n_steps, dtype=int)
            self.predicted_demand = np.zeros(awi.n_steps, dtype=int)
        self.process = self.input_product
        self.pcost = int(np.ceil(np.mean(awi.profile.costs[:, self.process])))
        self.n_inputs = awi.inputs[self.process]
        self.n_outputs = awi.outputs[self.process]
        self.production_factor = self.n_outputs / self.n_inputs
        self.supplies_secured = awi.profile.external_supplies[:, self.input_product]
        self.sales_secured = awi.profile.external_sales[:, self.output_product]
        self.supplies_needed = np.zeros(awi.n_steps, dtype=int)
        self.supplies_needed[:-1] = np.floor(
            self.predicted_demand[1:] / self.production_factor
        ).astype(int)
        self.sales_needed = np.zeros(awi.n_steps, dtype=int)
        self.sales_needed[1:] = np.floor(
            self.predicted_supply[:-1] * self.production_factor
        ).astype(int)
        self.supplies_needed[:-1] += np.ceil(
            self.sales_secured[1:] / self.production_factor
        ).astype(int)
        self.sales_needed[1:] += np.floor(
            self.supplies_secured[:-1] * self.production_factor
        ).astype(int)

        self.supplies_negotiating = np.zeros_like(self.supplies_needed)
        self.sales_negotiating = np.zeros_like(self.sales_needed)

        # self.supplies_needed = self.supplies_needed.cumsum()
        # self.supplies_secured = self.supplies_secured.cumsum()
        # self.sales_needed = self.sales_needed.cumsum()
        # self.sales_secured = self.sales_secured.cumsum()

        inprices = awi.profile.external_supply_prices[:, self.input_product]
        inprices[self.supplies_secured == 0] = 0
        outprices = awi.profile.external_sale_prices[:, self.output_product]
        outprices[self.sales_secured == 0] = 0

        self.input_cost = np.maximum(
            inprices, self.awi.catalog_prices[self.input_product]
        )
        self.output_price = np.maximum(
            outprices, self.awi.catalog_prices[self.output_product]
        )

    def step(self):
        """Generates buy and sell negotiations as needed"""
        s = self.awi.current_step
        if s == 0:
            for step in range(1, self.horizon + 2):
                self.generate_buy_negotiations(step)
                self.generate_sell_negotiations(step)
        else:
            nxt = s + self.horizon + 1
            if nxt > self.awi.n_steps - 1:
                return
            self.generate_buy_negotiations(nxt)
            self.generate_sell_negotiations(nxt)

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
                self.sales_needed[tmin: tmax + 1].sum()
                - self.sales_secured[tmin: tmax + 1].sum()
            )
            if target <= 0:
                return None
            self.sales_negotiating[tmin : tmax + 1] += int(
                math.ceil(self.agreement_fraction * issues[QUANTITY].max_value)
            )
        else:
            assert annotation["product"] == self.input_product
            target = (
                self.supplies_needed[tmin: tmax + 1].sum()
                - self.supplies_secured[tmin: tmax + 1].sum()
            )
            if target <= 0:
                return None
            self.supplies_negotiating[
                issues[TIME].min_value : issues[TIME].max_value + 1
            ] += int(math.ceil(self.agreement_fraction * issues[QUANTITY].max_value))

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
        controller = self._get_controller(mechanism)
        neg = self._running_negotiations[mechanism.id]
        negotiator_id = neg.negotiator.id
        controller.negotiation_concluded(negotiator_id, None)

    def on_negotiation_success(
        self, contract: Contract, mechanism: AgentMechanismInterface
    ) -> None:
        controller = self._get_controller(mechanism)
        neg = self._running_negotiations[mechanism.id]
        negotiator_id = neg.negotiator.id
        controller.negotiation_concluded(negotiator_id, contract.agreement)

    def on_contract_signed(self, contract: Contract) -> None:
        is_seller = contract.annotation["seller"] == self.id
        q, u, t = (
            contract.agreement["quantity"],
            contract.agreement["unit_price"],
            contract.agreement["time"],
        )
        if is_seller:
            # if I am a seller, I will schedule production then buy my needs to produce
            output_product = contract.annotation["product"]
            input_product = output_product - 1
            self.output_price[t] = (
                self.output_price[t]
                * (self.catalog_n_equivalent + self.sales_secured[t])
                + u * q
            ) / (self.sales_secured[t] + q)
            self.sales_secured[t] += q
            if input_product >= 0 and t > 0:
                steps, lines = self.awi.available_for_production(
                    repeats=q, step=(self.awi.current_step + 1, t - 1)
                )
                if len(steps) < q:
                    return
                self.awi.order_production(input_product, steps, lines)
                if contract.annotation["caller"] != self.id:
                    self.supplies_needed[t - 1] += max(
                        1, int(math.ceil(q / self.production_factor))
                    )
            return

        # I am a buyer. I need not produce anything but I need to negotiate to sell the production of what I bought
        input_product = contract.annotation["product"]
        output_product = input_product + 1
        self.input_cost[t] = (
            self.input_cost[t] * (self.catalog_n_equivalent + self.supplies_secured[t])
            + u * q
        ) / (self.supplies_secured[t] + q)
        self.supplies_secured[t] += q
        if output_product < self.awi.n_products and t < self.awi.n_steps - 1:
            if contract.annotation["caller"] != self.id:
                self.sales_needed[t + 1] += max(1, int(q * self.production_factor))

    def sign_contract(self, contract: Contract) -> Optional[str]:
        """Only signs contracts that are needed"""
        s = self.awi.current_step
        q, u, t = (
            contract.agreement["quantity"],
            contract.agreement["unit_price"],
            contract.agreement["time"],
        )

        # check that I can produce the required quantities even in principle
        if contract.annotation["seller"] == self.id:
            q = contract.agreement["quantity"]
            steps, lines = self.awi.available_for_production(
                q, (s, t), -1, override=False, method="all"
            )
            if len(steps) < q:
                return None

        # check that I need this contract
        if contract.annotation["seller"] == self.id:
            if self.sales_secured[s:t].sum() + q <= self.sales_needed[s:t].sum():
                return self.id
            return None
        if self.supplies_secured[s:t].sum() + q <= self.supplies_needed[s:t].sum():
            return self.id
        return None

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
            negotiating, secured, needed = (
                self.sales_negotiating,
                self.sales_secured,
                self.sales_needed,
            )
            controllers, generator = self.sellers, self.generate_sell_negotiations
        else:
            negotiating, secured, needed = (
                self.supplies_negotiating,
                self.supplies_secured,
                self.supplies_needed,
            )
            controllers, generator = self.buyers, self.generate_buy_negotiations

        negotiating[time_range[0] : time_range[1] + 1] -= expected
        controllers[controller_index].controller = None
        if quantity <= target:
            secured[time_range[0]] += quantity
            generator(step=controller_index)
            return

    def confirm_external_sales(
        self, quantities: np.ndarray, unit_prices: np.ndarray
    ) -> np.ndarray:
        p, s = self.output_product, self.awi.current_step
        quantities[p] = max(
            0, min(quantities[p], self.sales_needed[s] - self.sales_secured[s])
        )
        return quantities

    def confirm_external_supplies(
        self, quantities: np.ndarray, unit_prices: np.ndarray
    ) -> np.ndarray:
        p, s = self.input_product, self.awi.current_step
        quantities[p] = max(
            0, min(quantities[p], self.supplies_needed[s] - self.supplies_secured[s])
        )
        return quantities

    def generate_buy_negotiations(self, step):
        """Creates the controller and starts negotiations to acquire all required inputs (supplies) at the given step"""
        quantity = (
            self.supplies_needed[step]
            - self.supplies_secured[step]
            - self.supplies_negotiating[step]
        )
        if quantity <= 0:
            return
        self.start_negotiations(
            product=self.input_product,
            step=step,
            quantity=max(
                1,
                min(
                    self.awi.n_lines * (step - self.awi.current_step),
                    int(quantity * self.production_factor),
                ),
            ),
            unit_price=int(
                math.ceil(
                    (self.n_outputs * self.output_price[step] - self.pcost)
                    / self.n_inputs
                )
            ),
        )

    def generate_sell_negotiations(self, step):
        """Creates the controller and starts negotiations to sell all required outputs (sales) at the given step"""

        # find out if I need to sell any output products
        quantity = (
            self.sales_needed[step]
            - self.sales_secured[step]
            - self.sales_negotiating[step]
        )
        if quantity <= 0:
            return
        # if so, negotiate to sell as much of them as possible
        self.start_negotiations(
            product=self.output_product,
            step=step,
            quantity=max(
                1,
                min(
                    self.awi.n_lines * (step - self.awi.current_step),
                    int(math.floor(quantity / self.production_factor)),
                ),
            ),
            unit_price=(self.pcost + self.input_cost[step] * self.n_inputs)
            // self.n_outputs,
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
            self.sales_negotiating[tvalues[0] : tvalues[1] + 1] += expected_quantity
        else:
            partners = awi.my_suppliers
            expected_quantity = int(math.floor(qvalues[1] * self.agreement_fraction))
            self.supplies_negotiating[tvalues[0] : tvalues[1] + 1] += expected_quantity

        # negotiate with everyone
        controller = self.add_controller(
            is_seller, qvalues[1], uvalues, expected_quantity, step
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
        return int(len(steps) * self.production_factor)

    def max_consumption_till(self, step) -> int:
        """Returns the maximum number of units that can be consumed until the given step given current production
        schedule"""
        n = self.awi.n_lines * (step - self.awi.current_step + 1)
        steps, lines = self.awi.available_for_production(
            repeats=n, step=(self.awi.current_step, step - 1)
        )
        return int(math.ceil(len(steps) / self.production_factor))

    def add_controller(
        self,
        is_seller: bool,
        target,
        urange: Tuple[int, int],
        expected_quantity: int,
        step: int,
    ) -> StepController:
        controller = StepController(
            is_seller=is_seller,
            target_quantity=target,
            negotiator_type=self.negotiator_type,
            negotiator_params=self.negotiator_params,
            parent=self,
            step=step,
            urange=urange,
            product=self.output_product if is_seller else self.input_product,
            partners=self.awi.my_consumers if is_seller else self.awi.my_suppliers,
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
            cprice = self.output_price[time_range[0] : time_range[1]]
            if len(cprice):
                cprice = int(cprice.mean())
            else:
                cprice = self.awi.catalog_prices[self.input_product]
            return cprice, 2 * cprice

        cprice = self.input_cost[time_range[0] : time_range[1]]
        if len(cprice):
            cprice = int(cprice.mean())
        else:
            cprice = self.awi.catalog_prices[self.output_product]
        return 1, cprice

    def _trange(self, step, is_seller):
        if is_seller:
            return step + 1, self.awi.n_steps - 1
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

