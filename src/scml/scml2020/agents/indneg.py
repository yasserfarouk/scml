"""Implements a randomly behaving agent"""
import copy
from abc import abstractmethod
from typing import List, Optional, Dict, Any, Union

import numpy as np
from negmas import Contract, Breach, AgentMechanismInterface, MechanismState, Issue, Negotiator, SAONegotiator
from negmas import AspirationNegotiator
from negmas.helpers import get_class, instantiate

from .do_nothing import DoNothingAgent

__all__ = ["IndependentNegotiationsAgent"]


class IndependentNegotiationsAgent(DoNothingAgent):
    """An agent that negotiates independently with everyone"""

    def __init__(self, *args,
                 negotiator_type: Union[SAONegotiator, str] = AspirationNegotiator,
                 negotiator_params: Optional[Dict[str, Any]] = None,
                 horizon=5,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.negotiator_type = get_class(negotiator_type)
        self.negotiator_params = negotiator_params if negotiator_params is not None else dict()
        self.costs: np.ndarray = None
        self.horizon = horizon

    def init(self):
        self.costs = self.awi.profile.costs

    def step(self):
        if self.awi.current_step % self.horizon != 0:
            return
        input_products = self.awi.my_input_products
        output_products = self.awi.my_output_products
        nxt = self.awi.current_step + 3  # (1 for the negotiation to end and another for transfer of product and 1 extra)

        # if I have guaranteed inputs, negotiate to sell them
        supplies = self.awi.profile.guaranteed_supplies[nxt: nxt + self.horizon, input_products]
        nonzero = np.transpose(np.nonzero(supplies))
        for step, input_product in nonzero:
            quantity = supplies[step, input_product]
            output_product = input_product + 1
            n_inputs = self.awi.inputs[input_product]
            cost = self.costs[step, output_product]
            produce = self.awi.outputs[input_product]
            self.start_negotiations(
                product=output_product,
                quantity=max(1, quantity * n_inputs // produce),
                unit_price=(cost + self.awi.profile.guaranteed_supply_prices[step, input_product]
                            * n_inputs) // produce,
                time=step, to_buy=False
            )

        # if I have guaranteed outputs, negotiate to buy corresponding inputs
        sales = self.awi.profile.guaranteed_sales[nxt: nxt + self.horizon, output_products]
        nonzero = np.transpose(np.nonzero(sales))
        for step, output_product in nonzero:
            input_product = output_product - 1
            quantity = sales[step, output_product]
            needs = self.awi.inputs[input_product]
            cost = self.costs[step, input_product]
            n_outputs = self.awi.outputs[input_product]
            self.start_negotiations(product=input_product,
                                    quantity=max(1, quantity * needs // n_outputs),
                                    unit_price=(n_outputs *
                                                self.awi.profile.guaranteed_sale_prices[step, output_product]
                                                - cost) // needs,
                                    time=step, to_buy=True
                                    )

    @abstractmethod
    def create_ufun(self, is_seller: bool, issues=None, outcomes=None):
        """Creates a utility function"""
        pass

    def negotiator(self, is_seller: bool, issues=None, outcomes=None) -> SAONegotiator:
        """Creates a negotiator"""
        params = copy.deepcopy(self.negotiator_params)
        params["ufun"] = self.create_ufun(is_seller=is_seller, outcomes=outcomes, issues=issues)
        return instantiate(self.negotiator_type, **params)

    def respond_to_negotiation_request(self, initiator: str, issues: List[Issue], annotation: Dict[str, Any],
                                       mechanism: AgentMechanismInterface) -> Optional[Negotiator]:
        return self.negotiator(annotation["seller"] == self.id, issues=issues)

    def confirm_guaranteed_sales(self, quantities: np.ndarray, unit_prices: np.ndarray) -> np.ndarray:
        return quantities

    def confirm_guaranteed_supplies(self, quantities: np.ndarray, unit_prices: np.ndarray) -> np.ndarray:
        return quantities

    def start_negotiations(self, product: int, quantity: int, unit_price: int, time: int, to_buy: bool) -> None:
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
        # choose ranges for the negotiation agenda.
        cprice = self.awi.catalog_prices[product]
        cprice = min(cprice, unit_price) if to_buy else max(cprice, unit_price)
        qvalues = (1, quantity)
        uvalues = (1, int(1.2 * cprice)) if to_buy else (int(0.8 * cprice), 2 * cprice)
        tvalues = (self.awi.current_step + 1, time - 1) if to_buy else (time + 1, self.awi.n_steps - 1)
        issues = [
            Issue(qvalues, name="quantity"),
            Issue(uvalues, name="unit_price"),
            Issue(tvalues, name="time"),
        ]
        if Issue.num_outcomes(issues) < 2:
            self.awi.logdebug(f"Less than 2 issues for product {product}: {[str(_) for _ in issues]}")
            return

        # negotiate with all suppliers of the input product I need to produce
        partners = self.awi.all_suppliers[product] if to_buy else self.awi.all_consumers[product]
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

    def on_contract_signed(self, contract: Contract) -> None:
        is_seller = contract.annotation["seller"] == self.id
        step = contract.agreement["time"]
        if is_seller:
            # if I am a seller, I will schedule production then buy my needs to produce
            output_product = contract.annotation["product"]
            input_product = output_product - 1
            if input_product >= 0:
                self.awi.schedule_production(process=input_product, step=-1, line=-1)
                needs = self.awi.inputs[input_product]
                cost = self.costs[step, input_product]
                n_outputs = self.awi.outputs[input_product]
                self.start_negotiations(product=input_product,
                                        quantity=max(1, contract.agreement["quantity"] * needs // n_outputs),
                                        unit_price=(n_outputs * contract.agreement["unit_price"] - cost) // needs,
                                        time=contract.agreement["time"], to_buy=True
                                        )
            return

        # I am a buyer. I need not produce anything but I need to negotiate to sell the production of what I bought
        input_product = contract.annotation["product"]
        output_product = input_product + 1
        if output_product < self.awi.n_products:
            n_inputs = self.awi.inputs[input_product]
            cost = self.costs[step, output_product]
            produce = self.awi.outputs[input_product]
            self.start_negotiations(product=output_product,
                                    quantity=max(1, contract.agreement["quantity"] * n_inputs / produce),
                                    unit_price=(cost + contract.agreement["unit_price"] * n_inputs) // produce,
                                    time=contract.agreement["time"], to_buy=False)
