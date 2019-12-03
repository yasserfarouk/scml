import copy
import math
from typing import Union, Optional, Dict, Any, List

import numpy as np
from negmas import UtilityFunction, Outcome, outcome_as_dict, LinearUtilityFunction, SAONegotiator, Issue, Negotiator, \
    AgentMechanismInterface, Contract, SAOController, MechanismState, ResponseType, AspirationMixin, \
    AspirationNegotiator
from negmas.helpers import instantiate, get_class

from .do_nothing import DoNothingAgent


class SatisfiserController(SAOController, AspirationMixin):
    """A controller for managing a set of negotiations about selling/buying the same product.

    Remarks:

        - It uses whatever negotiator type on all of its negotiations and it assumes that the ufun will never change
        - Once it accumulates the required quantity, it ends all remaining negotiations
        - It assumes that all ufuns are identical so there is no need to keep a separate negotiator for each one and it
          instantiates a single negotiator that dynamically changes the AMI but always uses the same ufun.

    """

    def __init__(self, *args, target_quantity: int, is_seller: bool,
                 negotiator_type: SAONegotiator, negotiator_params: Dict[str, Any] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.is_seller = is_seller
        self.target = target_quantity
        negotiator_params = negotiator_params if negotiator_params is not None else dict()
        self.secured = 0
        if is_seller:
            self.ufun = LinearUtilityFunction((1, 1, 10))
        else:
            self.ufun = LinearUtilityFunction((1, -1, -10))
        negotiator_params["ufun"] = self.ufun
        self.__negotiator = instantiate(negotiator_type, **negotiator_params)

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

    def on_negotiation_end(self, negotiator_id: str, state: MechanismState) -> None:
        if state.agreement is not None:
            self.secured += state.agreement["quantity"]


class SatisfiserAgent(DoNothingAgent):
    """An agent that keeps schedules of what it needs to buy and sell and tries to satisfy them.

    It assumes that the agent can run a single process

    """

    def __init__(self, *args,
                 negotiator_type: Union[SAONegotiator, str] = AspirationNegotiator,
                 negotiator_params: Optional[Dict[str, Any]] = None,
                 horizon=5,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.negotiator_type = get_class(negotiator_type)
        self.negotiator_params = negotiator_params if negotiator_params is not None else dict()
        self.horizon = horizon
        self.input_product: int = -1
        self.output_product: int = -1
        self.process: int = -1
        self.production_cost: int = -1
        self.production_inputs: int = -1
        self.production_outputs: int = -1
        self.inputs_needed: np.ndarray = None
        self.outputs_needed: np.ndarray = None
        self.input_cost: np.ndarray = None
        self.output_cost: np.ndarray = None
        self.inputs_available: np.ndarray = None
        self.outputs_available: np.ndarray = None
        self.production_factor = 1

    def init(self):
        self.input_product = int(self.awi.my_input_product)
        self.output_product = self.input_product + 1
        self.process = self.input_product
        self.production_cost = int(np.ceil(np.mean(self.awi.profile.costs[:, self.process])))
        self.production_inputs = self.awi.inputs[self.process]
        self.production_outputs = self.awi.outputs[self.process]
        self.production_factor = self.production_outputs / self.production_inputs
        self.inputs_available = self.awi.profile.external_supplies[:, self.input_product]
        self.outputs_available = self.awi.profile.external_sales[:, self.input_product]
        self.inputs_needed = np.zeros(self.awi.n_steps, dtype=int)
        self.outputs_needed = np.zeros(self.awi.n_steps, dtype=int)
        self.inputs_needed[:-1] += np.ceil(self.outputs_available[1:] / self.production_factor).astype(int)
        self.outputs_needed[1:] = np.floor(self.inputs_available[:-1] * self.production_factor).astype(int)

        self.input_cost = self.awi.profile.external_supply_prices[:, self.input_product]
        self.output_cost = self.awi.profile.external_sale_prices[:, self.output_product]

    def step(self):

        # I wait two steps, one for running the negotiation and another for signing
        nxt_step = self.awi.current_step + self.input_product + 2

        if nxt_step > self.awi.n_steps - 1:
            return

        # negotiate to sell extra quantities
        input_product = self.input_product

        # find out if I need to sell any output products
        quantity = self.outputs_needed[nxt_step] - self.outputs_available[nxt_step]
        if quantity > 0:
            # if so, negotiate to sell as much of them as possible
            output_product = self.output_product
            n_inputs = self.production_inputs
            cost = self.production_cost
            n_outputs = self.production_outputs
            self.start_negotiations(
                product=output_product,
                quantity=max(1, int(math.ceil(quantity / self.production_factor))),
                unit_price=(cost + quantity * n_inputs) // n_outputs,
                time=nxt_step
            )

        # if I have guaranteed outputs, negotiate to buy corresponding inputs
        nxt_step = self.awi.n_processes - self.output_product + 2
        quantity = self.inputs_needed[nxt_step] - self.outputs_needed[nxt_step]
        if quantity > 0:
            n_inputs = self.production_inputs
            cost = self.production_cost
            n_outputs = self.production_outputs
            self.start_negotiations(product=input_product,
                                    quantity=max(1, quantity * n_inputs // n_outputs),
                                    unit_price=(n_outputs * quantity - cost) // n_inputs,
                                    time=nxt_step
                                    )

    def create_ufun(self, is_seller: bool):
        if is_seller:
            return LinearUtilityFunction((1, 1, 10))
        return LinearUtilityFunction((1, -1, -10))

    def negotiator(self, is_seller: bool, issues=None, outcomes=None) -> SAONegotiator:
        """Creates a negotiator"""
        params = copy.deepcopy(self.negotiator_params)
        params["ufun"] = self.create_ufun(is_seller=is_seller)
        return instantiate(self.negotiator_type, **params)

    def respond_to_negotiation_request(self, initiator: str, issues: List[Issue], annotation: Dict[str, Any],
                                       mechanism: AgentMechanismInterface) -> Optional[Negotiator]:
        return self.negotiator(annotation["seller"] == self.id, issues=issues)

    def confirm_external_sales(self, quantities: np.ndarray, unit_prices: np.ndarray) -> np.ndarray:
        p, s = self.output_product, self.awi.current_step
        quantities[p] = max(0, min(quantities[p], self.outputs_needed[s] - self.outputs_available[s]))
        return quantities

    def confirm_external_supplies(self, quantities: np.ndarray, unit_prices: np.ndarray) -> np.ndarray:
        p, s = self.input_product, self.awi.current_step
        quantities[p] = max(0, min(quantities[p], self.inputs_needed[s] - self.inputs_available[s]))
        return quantities

    def start_negotiations(self, product: int, quantity: int, unit_price: int, time: int) -> None:
        """
        Starts a set of negotiations to by/sell the product with the given limits

        Args:
            product: product type. If it is an input product, negotiations to buy it will be started otherweise to sell.
            quantity: The maximum quantity to negotiate about
            unit_price: The maximum/minimum unit price for buy/sell
            time: The maximum/minimum time for buy/sell

        Remarks:

            - This method assumes that products cannot be in my_input_products and my_output_products

        """
        to_buy = product == self.input_product
        if quantity < 1 or unit_price < 1 or time < self.awi.current_step + 1:
            self.awi.logdebug(f"Less than 2 valid issues (q:{quantity}, u:{unit_price}, t:{time})")
            return
        # choose ranges for the negotiation agenda.
        cprice = self.awi.catalog_prices[product]
        qvalues = (1, quantity)
        if to_buy:
            uvalues = (1, cprice)
            tvalues = (self.awi.current_step + 1, time - 1)
            partners = self.awi.all_suppliers[product]
        else:
            uvalues = (cprice, 2 * cprice)
            tvalues = (time + 1, self.awi.n_steps - 1)
            partners = self.awi.all_consumers[product]

        # negotiate with everyone
        self.awi.request_negotiations(
            is_buy=to_buy,
            product=product,
            quantity=qvalues,
            unit_price=uvalues,
            time=tvalues,
            partners=partners,
            controller=SatisfiserController(is_seller=not to_buy
                                            , target_quantity=qvalues[1]
                                            , negotiator_type=self.negotiator_type
                                            , negotiator_params=self.negotiator_params),
        )

    def on_contract_signed(self, contract: Contract) -> None:
        is_seller = contract.annotation["seller"] == self.id
        s = self.awi.current_step
        cost = self.production_cost
        current_input = self.awi.inputs[self.input_product]
        current_output = self.awi.outputs[self.input_product]
        q, u, t = contract.agreement["quantity"], contract.agreement["unit_price"], contract.agreement["time"]
        if is_seller:
            # if I am a seller, I will schedule production then buy my needs to produce
            output_product = contract.annotation["product"]
            input_product = output_product - 1
            self.output_cost[t] = (self.output_cost[t] * self.outputs_available[t] + u * q) / (
                self.outputs_available[t] + q)
            self.outputs_available[t] += q
            if input_product >= 0 and t > 0:
                steps, lines = self.awi.available_for_production(repeats=q, step=(self.awi.current_step+1, t - 1))
                if len(steps) < q:
                    return
                self.awi.order_production(input_product, steps, lines)
                n_needs = self.production_inputs
                n_outputs = self.production_outputs
                self.inputs_needed[t - 1] += max(1, math.ceil(q * n_needs // n_outputs))
                if self.inputs_needed[s: t - 1].sum() > self.inputs_available[s: t - 1].sum() + current_input:
                    self.start_negotiations(product=input_product,
                                            quantity=max(1, math.ceil(q * n_needs // n_outputs)),
                                            unit_price=(n_outputs * u - cost) // n_needs,
                                            time=t
                                            )
            return

        # I am a buyer. I need not produce anything but I need to negotiate to sell the production of what I bought
        input_product = contract.annotation["product"]
        output_product = input_product + 1
        self.input_cost[t] = (self.input_cost[t] * self.inputs_available[t] + u * q) / (
            self.inputs_available[t] + q)
        self.inputs_available[t] += q
        if output_product < self.awi.n_products and t < self.awi.n_steps - 1:
            n_inputs = self.production_inputs
            cost = self.production_cost
            n_produced = self.production_outputs
            self.outputs_needed[t + 1] = max(1, q * n_produced // n_inputs)
            if self.outputs_needed[s: t + 1].sum() > self.outputs_available[s: t + 1].sum() + current_output:
                self.start_negotiations(product=output_product,
                                        quantity=max(1, q * n_produced // n_inputs),
                                        unit_price=(cost + u * n_inputs) // n_produced,
                                        time=t)

    def sign_contract(self, contract: Contract) -> Optional[str]:
        """Only signs contracts that are needed"""
        s = self.awi.current_step
        q, u, t = contract.agreement["quantity"], contract.agreement["unit_price"], contract.agreement["time"]
        # if contract.annotation["seller"] == self.id:
        #
        #     if self.outputs_available[s:t].sum() + q <= self.outputs_needed[s: t].sum():
        #         return self.id
        # else:
        #     if self.inputs_available[s:t].sum() + q <= self.inputs_needed[s: t].sum():
        #         return self.id
        if contract.annotation["seller"] == self.id:
            q = contract.agreement["quantity"]
            steps, lines = self.awi.available_for_production(
                q,
                (self.awi.current_step+1, s),
                -1,
                override=False,
                method="all",
            )
            if len(steps) < q:
                return None
        return None
