"""
Implements the base class for agents that negotiate independently with different partners.

These agents do not take production capacity, availability of materials or any other aspects of the simulation into
account. They are to serve only as baselines.

Assumptions
-----------

The main assumptions of the agents based on `IndependentNegotiationsAgent` are:

1. All production processes take one input time and generate one output type.

"""

from abc import abstractmethod
from typing import List, Optional, Dict, Any, Union

import numpy as np
from negmas import Contract, AgentMechanismInterface, Issue, Negotiator, SAONegotiator
from negmas import AspirationNegotiator
from negmas.helpers import get_class, instantiate

from .do_nothing import DoNothingAgent

__all__ = ["IndependentNegotiationsAgent"]

from ..components.production import DemandDrivenProductionStrategy
from ..world import is_system_agent


class IndependentNegotiationsAgent(DemandDrivenProductionStrategy, DoNothingAgent):
    """
    Implements the base class for agents that negotiate independently with different partners.

    These agents do not take production capacity, availability of materials or any other aspects of the simulation into
    account. They are to serve only as baselines.

    Args:

        negotiator_type: The type of the negotiator to use. The default is `AspirationNegotiator`
        negotiator_params: key-value pairs to pass to the constructor of negotiators
        horizon: The number of production step to handle together using a single set of negotiations. See `step` for
                 details


    Remarks:

        - `IndependentNegotiationsAgent` agents assume that each production process has one input type with the same
           index as itself and one output type with one added to the index (i.e. process $i$ takes product $i$ as input
           and creates product $i+1$ as output.
        - It does not assume that all lines have the same production cost (it uses the average cost though).
        - It does not assume that the agent has a single production process.

    """

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
        self.expected_outputs = None
        self.expected_inputs = None

    def init(self):
        """Initializes the agent by finding the average production cost."""
        self.costs = np.ceil(
            np.sum(self.awi.profile.costs, axis=0) / self.awi.profile.n_lines
        ).astype(int)

        self.expected_outputs = (self.awi.n_lines // 4) * np.ones(self.awi.n_steps)
        self.expected_inputs = (self.awi.n_lines // 4) * np.ones(self.awi.n_steps)

    def step(self):
        """Every `horizon` steps, create new negotiations based on external supplies and sales."""

        # avoid division by zero error in numpy
        np.seterr(divide="ignore")

        # only run this process once every `horizon` steps
        if self.awi.current_step % self.horizon != 0:
            return
        earliest = self.awi.current_step
        final = min(earliest + self.horizon - 1, self.awi.n_steps)

        # if I have expected inputs, negotiate to sell them (after production)
        expected_inputs = self.expected_inputs[earliest:final]
        quantity = np.sum(expected_inputs)

        prices = self.awi.catalog_prices
        i, o = self.awi.my_input_product, self.awi.my_output_product

        # for every step and product, start negotiations to sell the output of this external supply
        for step in np.nonzero(expected_inputs)[0]:
            price = prices[i]
            self.start_negotiations(
                product=i + 1,
                quantity=max(1, quantity),
                unit_price=self.costs[i] + price,
                time=step + final + 1,
                to_buy=False,
            )

        # if I have external outputs, negotiate to buy corresponding inputs
        expected_outputs = self.expected_outputs[earliest:final]
        quantity = np.sum(expected_outputs)
        for step in np.nonzero(expected_outputs)[0]:
            process = o - 1
            cost = self.costs[process]
            self.start_negotiations(
                product=process,
                quantity=max(1, quantity),
                unit_price=prices[o] - cost,
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

    def sign_all_contracts(self, contracts: List[Contract]) -> List[Optional[str]]:
        results = [None] * len(contracts)
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
        for i, contract in enumerate(contracts):
            step = contract.agreement["time"]
            q = contract.agreement["quantity"]
            if step > self.awi.n_steps - 1 or step < self.awi.current_step:
                continue
            if contract.annotation["seller"] == self.id:
                steps, lines = self.awi.available_for_production(
                    q, (self.awi.current_step, step), -1, override=False, method="all"
                )
                if len(steps) < q:
                    continue
            results[i] = self.id
            if contract.annotation["seller"] == self.id:
                self.expected_outputs[step] += q
            else:
                self.expected_inputs[step] += q
        return results

    def on_contracts_finalized(
        self,
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        # call the production strategy
        super().on_contracts_finalized(signed, cancelled, rejectors)

        for contract in signed:
            is_seller = contract.annotation["seller"] == self.id
            step = contract.agreement["time"]
            # find the earliest time I can do anything about this contract
            earliest_production = self.awi.current_step
            if step > self.awi.n_steps - 1 or step < earliest_production:
                continue
            if is_seller:
                # if I am a seller, I will schedule production then buy my needs to produce
                if not self.can_be_produced(contract.id):
                    del self.earliest_schedule[contract.id]
                    continue
                earliest = self.earliest_schedule.pop(contract.id)
                output_product = contract.annotation["product"]
                input_product = output_product - 1
                if input_product < 0:
                    continue
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
                    time=earliest - 1,
                    to_buy=True,
                )
                continue

            # I am a buyer. I need not produce anything but I need to negotiate to sell the production of what I bought
            input_product = contract.annotation["product"]
            output_product = input_product + 1
            if output_product < self.awi.n_products:
                n_inputs = self.awi.inputs[input_product]
                cost = self.costs[input_product]
                n_outputs = self.awi.outputs[input_product]
                self.start_negotiations(
                    product=output_product,
                    quantity=max(
                        1, contract.agreement["quantity"] * n_inputs / n_outputs
                    ),
                    unit_price=(cost + contract.agreement["unit_price"] * n_inputs)
                    // n_outputs,
                    time=step - 1,
                    to_buy=False,
                )
