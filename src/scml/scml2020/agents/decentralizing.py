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
from pprint import pformat
from scml.scml2020 import AWI, NO_COMMAND, ANY_LINE

from .do_nothing import DoNothingAgent

QUANTITY = 0
TIME = 1
UNIT_PRICE = 2

__all__ = ["DecentralizingAgent"]


class StepController(SAOController, AspirationMixin, Notifier):
    """A controller for managing a set of negotiations about selling/buying the a product starting/ending at some
    specific time-step. It works in conjunction with the `DecentralizingAgent` .

    Args:

        target_quantity: The quantity to be secured
        is_seller:  Is this a seller or a buyer
        parent: The parent `DecedntralizingAgent`
        step:  The simulation step that this controller is responsible about
        urange: The range of unit prices used for negotiation
        product: The product that this controller negotiates about
        partners: A list of partners to negotiate with
        negotiator_type: The type of the negotiator used for all negotiations.
        negotiator_params: The parameters of the negotiator used for all negotiations
        max_retries: How many times can the controller try negotiating with each partner.
        *args: Position arguments passed to the base Controller constructor
        **kwargs: Keyword arguments passed to the base Controller constructor


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

    def __str__(self):
        return (
            f"{'selling' if self.is_seller else 'buying'} p{self.product} [{self.step}] "
            f"secured {self.secured} of {self.target} for {self.__parent.name} "
            f"({len([_ for _ in self.completed.values() if _])} completed of {len(self.completed)} negotiators)"
        )

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
                awi.loginfo(f"Ending all negotiations on controller {str(self)}")
                # If we are done, end all other negotiations
                for k in self.negotiators.keys():
                    if self.completed[k]:
                        continue
                    self.notify(
                        self.negotiators[k][0], Notification("end_negotiation", None)
                    )
        self.kill_negotiator(negotiator_id, force=True)
        if all(self.completed.values()):
            # If we secured everything, just return control to the agent
            if self.secured >= self.target:
                awi.loginfo(f"Secured Everything: {str(self)}")
                self.__parent.all_negotiations_concluded(self.step, self.is_seller)
                return
            # If we did not secure everything we need yet and time allows it, create new negotiations
            tmin, tmax = self.__parent._trange(self.step, self.is_seller)

            if awi.current_step < tmax + 1 and tmin <= tmax:
                # get a good partner: one that was not retired too much
                random.shuffle(self.partners)
                for other in self.partners:
                    if self.retries[other] <= self.max_retries:
                        partner = other
                        break
                else:
                    return
                self.retries[partner] += 1
                neg = self.create_negotiator()
                self.completed[neg.id] = False
                awi.loginfo(
                    f"{str(self)} negotiating with {partner} on u={self.urange}"
                    f", q=(1,{self.target-self.secured}), u=({tmin}, {tmax})"
                )
                awi.request_negotiation(
                    not self.is_seller,
                    product=self.product,
                    quantity=(1, self.target - self.secured),
                    unit_price=self.urange,
                    time=(tmin, tmax),
                    partner=partner,
                    negotiator=neg,
                    extra=dict(controller_index=self.step, is_seller=self.is_seller),
                )


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


class DecentralizingAgent(DoNothingAgent):
    """An agent that keeps schedules of what it needs to buy and sell and tries to satisfy them.

    It assumes that the agent can run a single process.

    Args:

        negotiator_type: The negotiator type to use for all negotiations
        negotiator_params: The parameters used to initialize all negotiators
        horizon: The number of steps in the future to consider for securing inputs and
                 outputs.
        predicted_demand: A prediction of the number of units needed by the market of the output
                          product at each timestep
        predicted_supply: A prediction of the nubmer of units available within the market of the input
                          product at each timestep
        agreement_fraction: A prediction about the fraction of the quantity negotiated about that will
                            be secured
        adapt_prices: If true, the agent tries to adapt the unit price range it negotites about to market
                      conditions (i.e. previous trade). If false, catalog prices will be used to constrain
                      the unit price ranges to (1, catalog price) for buying and (catalog price, 2* catalog price)
                      for selling
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
        adapt_prices: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.adapt_prices = adapt_prices
        self.predicted_demand = predicted_demand
        self.predicted_supply = predicted_supply
        self.negotiator_type = get_class(negotiator_type)
        self.negotiator_params = (
            negotiator_params if negotiator_params is not None else dict()
        )
        self.horizon = horizon
        self.exogenous_horizon = None
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
        self.production_needed = None
        self.production_secured = None
        self.production_factor = 1
        self.buyers = self.sellers = None
        self.catalog_n_equivalent = 0
        self.supplies_negotiating = None
        self.sales_negotiating = None
        self.agreement_fraction = agreement_fraction
        self.use_exogenous_contracts = True

    def init(self):
        awi: AWI
        awi = self.awi  # type: ignore
        self.exogenous_horizon = awi.bb_read("settings", "exogenous_horizon")
        self.buyers: List[ControllerInfo] = [
            ControllerInfo(None, i, False, tuple(), 0, 0, False)
            for i in range(self.awi.n_steps)
        ]
        self.sellers: List[ControllerInfo] = [
            ControllerInfo(None, i, True, tuple(), 0, 0, False)
            for i in range(self.awi.n_steps)
        ]
        self.catalog_n_equivalent = self.awi.n_steps * 2

        self.input_product = int(awi.my_input_product)
        self.output_product = self.input_product + 1

        def adjust(x, demand):
            if x is None:
                x = max(1, awi.n_lines // 2)
            elif isinstance(x, Iterable):
                return np.array(x)
            predicted = int(x) * np.ones(awi.n_steps, dtype=int)
            if demand:
                predicted[: self.input_product + 1] = 0
            else:
                predicted[self.input_product - awi.n_processes :] = 0
            return predicted

        self.predicted_demand = adjust(self.predicted_demand, True)
        self.predicted_supply = adjust(self.predicted_supply, False)
        self.use_exogenous_contracts = awi.bb_read(
            "settings", "has_exogenous_contracts"
        )
        if (
            not self.use_exogenous_contracts
            and (self.input_product == 0
            or self.output_product == awi.n_products - 1)
        ):
            self.predicted_supply = np.zeros(awi.n_steps, dtype=int)
            self.predicted_demand = np.zeros(awi.n_steps, dtype=int)
        self.process = self.input_product
        self.pcost = int(np.ceil(np.mean(awi.profile.costs[:, self.process])))
        self.n_inputs = awi.inputs[self.process]
        self.n_outputs = awi.outputs[self.process]
        self.production_factor = self.n_outputs / self.n_inputs
        self.supplies_secured = np.zeros(awi.n_steps, dtype=int)
        self.sales_secured = np.zeros(awi.n_steps, dtype=int)
        self.supplies_secured[
            : self.exogenous_horizon
        ] = awi.profile.exogenous_supplies[: self.exogenous_horizon, self.input_product]
        self.sales_secured[: self.exogenous_horizon] = awi.profile.exogenous_sales[
            : self.exogenous_horizon, self.output_product
        ]
        self.supplies_needed = np.zeros(awi.n_steps, dtype=int)
        self.sales_needed = np.zeros(awi.n_steps, dtype=int)
        self.production_needed = np.zeros(awi.n_steps, dtype=int)
        self.production_secured = np.zeros(awi.n_steps, dtype=int)
        if awi.my_input_product != 0 or self.use_exogenous_contracts:
            self.supplies_needed[:-1] = np.floor(
                self.predicted_demand[1:] / self.production_factor
            ).astype(int)
        if awi.my_output_product != awi.n_products - 1 or self.use_exogenous_contracts:
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
        self.production_needed[:-1] = np.minimum(
            self.supplies_needed[:-1], self.sales_needed[1:]
        )
        inprices = awi.profile.exogenous_supply_prices[:, self.input_product]
        inprices[self.supplies_secured == 0] = 0
        outprices = awi.profile.exogenous_sale_prices[:, self.output_product]
        outprices[self.sales_secured == 0] = 0

        self.input_cost = np.maximum(
            inprices, self.awi.catalog_prices[self.input_product]
        )
        self.output_price = np.maximum(
            outprices, self.awi.catalog_prices[self.output_product]
        )
        self.awi.logdebug_agent(f"Initialized\n{pformat(self._debug_state())}")

    def _debug_state(self):
        return {
            "supplies_secured": self.supplies_secured.tolist(),
            "supplies_needed": self.supplies_needed.tolist(),
            "supplies_negotiating": self.supplies_negotiating.tolist(),
            "sales_secured": self.sales_secured.tolist(),
            "sales_needed": self.sales_needed.tolist(),
            "sales_negotiating": self.sales_negotiating.tolist(),
            "production_secured": self.production_secured.tolist(),
            "production_needed": self.production_needed.tolist(),
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
        if self.exogenous_horizon != self.awi.n_steps and self.use_exogenous_contracts:
            nxt = s + self.exogenous_horizon
            if nxt < self.awi.n_steps:
                self.supplies_secured[nxt] += self.awi.profile.exogenous_supplies[nxt]
                if nxt + 1 < self.awi.n_steps:
                    self.sales_needed[nxt + 1] += self.awi.profile.exogenous_supplies[
                        nxt
                    ]

                self.sales_secured[nxt] += self.awi.profile.exogenous_sales[nxt]
                if nxt - 1 >= 0:
                    self.supplies_needed[nxt - 1] += self.awi.profile.exogenous_sales[
                        nxt
                    ]
        if s == 0:
            last = min(self.awi.n_steps - 1, self.horizon + 2)
            for step in range(1, last):
                self.generate_buy_negotiations(step)
                self.generate_sell_negotiations(step)
        else:
            nxt = s + self.horizon + 1
            if nxt > self.awi.n_steps - 1:
                self.awi.logdebug_agent(f"End step:\n{pformat(self._debug_state())}")
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
                self.sales_needed[tmin : tmax + 1].sum()
                - self.sales_secured[tmin : tmax + 1].sum()
            )
            if target <= 0:
                return None
            self.sales_negotiating[tmin : tmax + 1] += int(
                math.ceil(self.agreement_fraction * issues[QUANTITY].max_value)
                // (tmax + 1 - tmin)
            )
        else:
            assert annotation["product"] == self.input_product
            target = (
                self.supplies_needed[tmin : tmax + 1].sum()
                - self.supplies_secured[tmin : tmax + 1].sum()
            )
            if target <= 0:
                return None
            self.supplies_negotiating[
                issues[TIME].min_value : issues[TIME].max_value + 1
            ] += int(
                math.ceil(self.agreement_fraction * issues[QUANTITY].max_value)
            ) // (
                issues[TIME].max_value + 1 - issues[TIME].min_value
            )

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
                        repeats=q, step=(self.awi.current_step, t - 1)
                    )
                    q = min(len(steps) - consumed, q)
                    consumed += q
                    self.production_needed[t - 1] += q
                    if contract.annotation["caller"] != self.id:
                        self.supplies_needed[t - 1] += max(
                            1, int(math.ceil(q / self.production_factor))
                        )
                continue

            # I am a buyer. I need not produce anything but I need to negotiate to sell the production of what I bought
            input_product = contract.annotation["product"]
            output_product = input_product + 1
            self.input_cost[t] = (
                self.input_cost[t]
                * (self.catalog_n_equivalent + self.supplies_secured[t])
                + u * q
            ) / (self.supplies_secured[t] + q)
            self.supplies_secured[t] += q
            if output_product < self.awi.n_products and t < self.awi.n_steps - 1:
                if contract.annotation["caller"] != self.id:
                    self.sales_needed[t + 1] += max(1, int(q * self.production_factor))
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
        contracts = sorted(contracts, key=lambda x: x.agreement["time"])
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
            # check that I can produce the required quantities even in principle
            if contract.annotation["seller"] == self.id:
                trange = (s, t)
                secured, needed, negotiating = (
                    self.sales_secured,
                    self.sales_needed,
                    self.sales_negotiating,
                )
            else:
                trange = (t + 1, self.awi.n_steps - 1)
                secured, needed, negotiating = (
                    self.supplies_secured,
                    self.supplies_needed,
                    self.supplies_negotiating,
                )

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
        self, commands: np.ndarray, balance: int, inventory
    ) -> np.ndarray:
        commands = np.ones_like(commands) * NO_COMMAND
        awi: AWI
        awi = self.awi
        s = awi.current_step
        inputs = awi.state.inventory[self.input_product]
        n_needed = max(0, min(awi.n_lines, self.production_needed[: s + 1].sum()))
        if inputs < n_needed:
            if s < awi.n_steps - 1:
                self.production_needed[s + 1] += inputs - n_needed
            n_needed = inputs
        commands[:n_needed] = self.input_product
        self.production_needed[s] -= n_needed
        return commands

    def on_contract_nullified(
        self, contract: Contract, compensation_money: int, new_quantity: int
    ) -> None:
        q = contract.agreement["quantity"]
        if new_quantity < q:
            t = contract.agreement["time"]
            missing = q - new_quantity
            if t < self.awi.current_step:
                return
            if contract.annotation["seller"] == self.id:
                self.sales_secured[t] -= missing
                if t > 0:
                    self.production_needed[t - 1] -= missing
                    self.supplies_needed[t - 1] -= missing
                if (
                    self.sellers[t] is not None
                    and self.sellers[t].controller is not None
                ):
                    self.sellers[t].controller.target -= missing
            else:
                self.supplies_secured[t] += missing
                if t < self.awi.n_steps - 1:
                    self.sales_needed[t + 1] -= missing
                if self.buyers[t] is not None and self.buyers[t].controller is not None:
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

        if time_range[1] + 1 - time_range[0] > 0:
            negotiating[time_range[0] : time_range[1] + 1] -= expected // (
                time_range[1] + 1 - time_range[0]
            )
        self.awi.loginfo(
            f"Killing Controller {str(controllers[controller_index].controller)}"
        )
        controllers[controller_index].controller = None
        if quantity <= target:
            secured[time_range[0]] += quantity
            generator(step=controller_index)
            return

    def confirm_exogenous_sales(
        self, quantities: np.ndarray, unit_prices: np.ndarray
    ) -> np.ndarray:
        p, s = self.output_product, self.awi.current_step
        quantities[p] = max(
            0, min(quantities[p], self.sales_needed[s] - self.sales_secured[s])
        )
        return quantities

    def confirm_exogenous_supplies(
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
        if is_seller:
            self.sales_negotiating[
                tvalues[0] : tvalues[1] + 1
            ] += expected_quantity // (tvalues[1] + 1 - tvalues[0])
        else:
            self.supplies_negotiating[
                tvalues[0] : tvalues[1] + 1
            ] += expected_quantity // (tvalues[1] + 1 - tvalues[0])

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
        if is_seller and self.sellers[step].controller is not None:
            return self.sellers[step].controller
        if not is_seller and self.buyers[step].controller is not None:
            return self.buyers[step].controller
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
            cprice = self.awi.catalog_prices[self.output_product]
            if self.adapt_prices:
                cprice = self.output_price[time_range[0] : time_range[1]]
                if len(cprice):
                    cprice = int(cprice.mean())
                else:
                    cprice = self.awi.catalog_prices[self.output_product]
            return cprice, 2 * cprice

        cprice = self.awi.catalog_prices[self.input_product]
        if self.adapt_prices:
            cprice = self.input_cost[time_range[0] : time_range[1]]
            if len(cprice):
                cprice = int(cprice.mean())
            else:
                cprice = self.awi.catalog_prices[self.input_product]
        return 1, cprice

    def _trange(self, step, is_seller):
        if is_seller:
            return max(step, self.awi.current_step + 1), self.awi.n_steps - 1
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
