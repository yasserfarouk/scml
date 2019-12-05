"""Implements a randomly behaving agent"""
import copy
from abc import abstractmethod
from typing import List, Optional, Dict, Any, Union

import numpy as np
from negmas import (
    Contract,
    Breach,
    AgentMechanismInterface,
    MechanismState,
    Issue,
    Negotiator,
    SAONegotiator,
)
from negmas import AspirationNegotiator
from negmas.helpers import get_class, instantiate

from .do_nothing import DoNothingAgent

__all__ = ["IndependentNegotiationsAgent"]


class IndependentNegotiationsAgent(DoNothingAgent):
    """An agent that negotiates independently with everyone"""

    def __init__(
        self,
        *args,
        negotiator_type: Union[SAONegotiator, str] = AspirationNegotiator,
        negotiator_params: Optional[Dict[str, Any]] = None,
        horizon=5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.negotiator_type = get_class(negotiator_type)
        self.negotiator_params = (
            negotiator_params if negotiator_params is not None else dict()
        )
        self.costs: np.ndarray = None
        self.horizon = horizon

    def init(self):
        self.costs = np.ceil(
            np.sum(self.awi.profile.costs, axis=0) / self.awi.profile.n_lines
        ).astype(int)

    def step(self):
        np.seterr(divide="ignore")
        if self.awi.current_step % self.horizon != 0:
            return
        input_products = self.awi.my_input_products
        output_products = self.awi.my_output_products
        earliest = self.awi.current_step + 2
        final = min(earliest + self.horizon - 1, self.awi.n_steps)
        # if I have external inputs, negotiate to sell them (after production)
        supplies = self.awi.profile.external_supplies[earliest:final, input_products]
        quantity = np.sum(supplies, axis=0)
        prices = (
            np.sum(
                self.awi.profile.external_supply_prices[earliest:final, input_products]
                * supplies
            )
            // quantity
        )
        nonzero = np.transpose(np.nonzero(supplies))
        for step, i in nonzero:
            input_product = input_products[i]
            price = prices[i]
            output_product = input_product + 1
            n_inputs = self.awi.inputs[input_product]
            cost = self.costs[input_product]
            n_outputs = self.awi.outputs[input_product]
            self.start_negotiations(
                product=output_product,
                quantity=max(1, quantity[i] * n_outputs // n_inputs),
                unit_price=(cost + price * n_inputs) // n_outputs,
                time=step + final + 1,
                to_buy=False,
            )

        # if I have guaranteed outputs, negotiate to buy corresponding inputs
        sales = self.awi.profile.external_sales[earliest:final, output_products]
        quantity = np.sum(sales, axis=0)
        prices = (
            np.sum(
                sales
                * self.awi.profile.external_sale_prices[earliest:final, output_products]
            )
            // quantity
        )
        nonzero = np.transpose(np.nonzero(sales))
        for step, o in nonzero:
            output_product = output_products[o]
            input_product = output_product - 1
            price = prices[o]
            n_inputs = self.awi.inputs[input_product]
            cost = self.costs[input_product]
            n_outputs = self.awi.outputs[input_product]
            self.start_negotiations(
                product=input_product,
                quantity=max(1, quantity[o] * n_inputs // n_outputs),
                unit_price=(n_outputs * price - cost) // n_inputs,
                time=step + earliest - 1,
                to_buy=True,
            )

    @abstractmethod
    def create_ufun(self, is_seller: bool, issues=None, outcomes=None):
        """Creates a utility function"""
        pass

    def negotiator(self, is_seller: bool, issues=None, outcomes=None) -> SAONegotiator:
        """Creates a negotiator"""
        params = self.negotiator_params
        params["ufun"] = self.create_ufun(
            is_seller=is_seller, outcomes=outcomes, issues=issues
        )
        return instantiate(self.negotiator_type, **params)

    def respond_to_negotiation_request(
        self,
        initiator: str,
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
    ) -> Optional[Negotiator]:
        return self.negotiator(annotation["seller"] == self.id, issues=issues)

    def confirm_external_sales(
        self, quantities: np.ndarray, unit_prices: np.ndarray
    ) -> np.ndarray:
        return quantities

    def confirm_external_supplies(
        self, quantities: np.ndarray, unit_prices: np.ndarray
    ) -> np.ndarray:
        return quantities

    def start_negotiations(
        self, product: int, quantity: int, unit_price: int, time: int, to_buy: bool
    ) -> None:
        """
        Starts a set of negotiations to by/sell the product with the given limits

        Args:
            product: product type. If it is an input product, negotiations to buy it will be started otherweise to sell.
            quantity: The maximum quantity to negotiate about
            unit_price: The maximum/minimum unit price for buy/sell
            time: The maximum/minimum time for buy/sell
            to_buy: Is the negotiation to buy or to sell

        Remarks:

            - This method assumes that products cannot be in my_input_products and my_output_products

        """
        if quantity < 1 or unit_price < 1 or time < self.awi.current_step + 1:
            self.awi.logdebug(
                f"Less than 2 valid issues (q:{quantity}, u:{unit_price}, t:{time})"
            )
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
        issues = [
            Issue(qvalues, name="quantity", value_type=int),
            Issue(tvalues, name="time", value_type=int),
            Issue(uvalues, name="unit_price", value_type=int),
        ]
        if Issue.num_outcomes(issues) < 2:
            self.awi.logdebug(
                f"Less than 2 issues for product {product}: {[str(_) for _ in issues]}"
            )
            return

        # negotiate with all suppliers of the input product I need to produce
        for partner in partners:
            self.awi.request_negotiation(
                is_buy=to_buy,
                product=product,
                quantity=qvalues,
                unit_price=uvalues,
                time=tvalues,
                partner=partner,
                negotiator=self.negotiator(not to_buy, issues=issues),
            )

    def sign_contract(self, contract: Contract) -> Optional[str]:
        step = contract.agreement["time"]
        if step > self.awi.n_steps - 1 or step < self.awi.current_step + 1:
            return None
        if contract.annotation["seller"] == self.id:
            q = contract.agreement["quantity"]
            steps, lines = self.awi.available_for_production(
                q, (self.awi.current_step + 1, step), -1, override=False, method="all"
            )
            if len(steps) < q:
                return None
        return self.id

    def on_contract_signed(self, contract: Contract) -> None:
        is_seller = contract.annotation["seller"] == self.id
        step = contract.agreement["time"]
        # find the earliest time I can do anything about this contract
        earliest_production = self.awi.current_step + 1
        if step > self.awi.n_steps - 1 or step < earliest_production:
            return
        if is_seller:
            # if I am a seller, I will schedule production then buy my needs to produce
            output_product = contract.annotation["product"]
            input_product = output_product - 1
            if input_product >= 0:
                steps, _ = self.awi.schedule_production(
                    process=input_product,
                    repeats=contract.agreement["quantity"],
                    step=(earliest_production, step - 1),
                    line=-1,
                )
                if len(steps) < 1:
                    return
                scheduled_at = steps.min()
                n_inputs = self.awi.inputs[input_product]
                cost = self.costs[input_product]
                n_outputs = self.awi.outputs[input_product]
                self.start_negotiations(
                    product=input_product,
                    quantity=max(
                        1, contract.agreement["quantity"] * n_inputs // n_outputs
                    ),
                    unit_price=(n_outputs * contract.agreement["unit_price"] - cost)
                    // n_inputs,
                    time=scheduled_at - 1,
                    to_buy=True,
                )
            return

        # I am a buyer. I need not produce anything but I need to negotiate to sell the production of what I bought
        input_product = contract.annotation["product"]
        output_product = input_product + 1
        if output_product < self.awi.n_products:
            n_inputs = self.awi.inputs[input_product]
            cost = self.costs[input_product]
            n_outputs = self.awi.outputs[input_product]
            self.start_negotiations(
                product=output_product,
                quantity=max(1, contract.agreement["quantity"] * n_inputs / n_outputs),
                unit_price=(cost + contract.agreement["unit_price"] * n_inputs)
                // n_outputs,
                time=step - 1,
                to_buy=False,
            )
