from typing import List
from typing import Optional

import numpy as np
from negmas import Contract

from scml.scml2020.common import ANY_LINE
from scml.scml2020.common import is_system_agent
from scml.scml2020.components import SignAllPossible
from scml.scml2020.components.prediction import FixedTradePredictionStrategy
from scml.scml2020.components.prediction import MarketAwareTradePredictionStrategy
from scml.scml2020.components.prediction import MeanERPStrategy

__all__ = [
    "TradingStrategy",
    "ReactiveTradingStrategy",
    "PredictionBasedTradingStrategy",
    "MarketAwarePredictionBasedTradingStrategy",
]


class TradingStrategy:
    """Base class for all trading strategies.

    Provides:
        - `inputs_needed` (np.ndarray):  How many items of the input product do
          I need to buy at every time step (n_steps vector).
          This should be read **but not updated** by the `NegotiationManager`.
        - `outputs_needed` (np.ndarray):  How many items of the output product
          do I need to sell at every time step (n_steps vector).
          This should be read **but not updated** by the `NegotiationManager`.
        - `inputs_secured` (np.ndarray):  How many items of the input product I
          already contracted to buy (n_steps vector) [out of `input_needed`].
          This can be read **but not updated** by the `NegotiationManager`.
        - `outputs_secured` (np.ndarray):  How many units of the output product
          I already contracted to sell (n_steps vector) [out of `outputs_secured`]
          This can be read **but not updated** by the `NegotiationManager`.

    Hooks Into:
        - `init`
        - `internal_state`

    Remarks:
        - `Attributes` section describes the attributes that can be used to construct the component (passed to its
          `__init__` method).
        - `Provides` section describes the attributes (methods, properties, data-members) made available by this
          component directly. Note that everything provided by the bases of this components are also available to the
          agent (Check the `Bases` section above for all the bases of this component).
        - `Requires` section describes any requirements from the agent using this component. It defines a set of methods
          or properties/data-members that must exist in the agent that uses this component. These requirement are
          usually implemented as abstract methods in the component
        - `Abstract` section describes abstract methods that MUST be implemented by any descendant of this component.
        - `Hooks Into` section describes the methods this component overrides calling `super` () which allows other
          components to hook into the same method (by overriding it). Usually callbacks starting with `on_` are
          hooked into this way.
        - `Overrides` section describes the methods this component overrides without calling `super` effectively
          disallowing any other components after it in the MRO to call this method. Usually methods that do some
          action (i.e. not starting with `on_`) are overridden this way.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inputs_needed: np.ndarray = None
        """How many items of the input product do I need at every time step"""
        self.outputs_needed: np.ndarray = None
        """How many items of the output product do I need at every time step"""
        self.inputs_secured: np.ndarray = None
        """How many units of the input product I have already secured per step"""
        self.outputs_secured: np.ndarray = None
        """How many units of the output product I have already secured per step"""

    def init(self):
        super().init()
        awi = self.awi
        # initialize needed/secured for inputs and outputs to all zeros
        self.inputs_secured = np.zeros(awi.n_steps, dtype=int)
        self.outputs_secured = np.zeros(awi.n_steps, dtype=int)
        self.inputs_needed = np.zeros(awi.n_steps, dtype=int)
        self.outputs_needed = np.zeros(awi.n_steps, dtype=int)

    @property
    def internal_state(self):
        state = super().internal_state
        state.update(
            {
                "inputs_secured": self.inputs_secured
                if self.inputs_secured is not None
                else None,
                "inputs_needed": self.inputs_needed
                if self.inputs_needed is not None
                else None,
                "outputs_secured": self.outputs_secured
                if self.outputs_secured is not None
                else None,
                "outputs_needed": self.outputs_needed
                if self.outputs_needed is not None
                else None,
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
                "_output_inventory": self.awi.state.inventory[
                    self.awi.my_output_product
                ],
            }
        )
        return state


class NoTradingStrategy(SignAllPossible, TradingStrategy):
    """A null trading strategy that just uses a signing strategy but no predictions.

    Remarks:
        - `Attributes` section describes the attributes that can be used to construct the component (passed to its
          `__init__` method).
        - `Provides` section describes the attributes (methods, properties, data-members) made available by this
          component directly. Note that everything provided by the bases of this components are also available to the
          agent (Check the `Bases` section above for all the bases of this component).
        - `Requires` section describes any requirements from the agent using this component. It defines a set of methods
          or properties/data-members that must exist in the agent that uses this component. These requirement are
          usually implemented as abstract methods in the component
        - `Abstract` section describes abstract methods that MUST be implemented by any descendant of this component.
        - `Hooks Into` section describes the methods this component overrides calling `super` () which allows other
          components to hook into the same method (by overriding it). Usually callbacks starting with `on_` are
          hooked into this way.
        - `Overrides` section describes the methods this component overrides without calling `super` effectively
          disallowing any other components after it in the MRO to call this method. Usually methods that do some
          action (i.e. not starting with `on_`) are overridden this way.

    """


class ReactiveTradingStrategy(SignAllPossible, TradingStrategy):
    """The agent reactively responds to contracts for selling by buying and vice versa.

    Hooks Into:
        - `on_contracts_finalized`

    Remarks:
        - `Attributes` section describes the attributes that can be used to construct the component (passed to its
          `__init__` method).
        - `Provides` section describes the attributes (methods, properties, data-members) made available by this
          component directly. Note that everything provided by the bases of this components are also available to the
          agent (Check the `Bases` section above for all the bases of this component).
        - `Requires` section describes any requirements from the agent using this component. It defines a set of methods
          or properties/data-members that must exist in the agent that uses this component. These requirement are
          usually implemented as abstract methods in the component
        - `Abstract` section describes abstract methods that MUST be implemented by any descendant of this component.
        - `Hooks Into` section describes the methods this component overrides calling `super` () which allows other
          components to hook into the same method (by overriding it). Usually callbacks starting with `on_` are
          hooked into this way.
        - `Overrides` section describes the methods this component overrides without calling `super` effectively
          disallowing any other components after it in the MRO to call this method. Usually methods that do some
          action (i.e. not starting with `on_`) are overridden this way.

    """

    def on_contracts_finalized(
        self,
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        # call the production strategy
        super().on_contracts_finalized(signed, cancelled, rejectors)
        this_step = self.awi.current_step
        inp, outp = self.awi.my_input_product, self.awi.my_output_product

        for contract in signed:
            t, q = contract.agreement["time"], contract.agreement["quantity"]
            # If I started this negotiation, I must have had a reason to do so.
            # This implies that I need not plan anything about it
            if contract.annotation["caller"] == self.id:
                continue
            # If I am buying something, I do not need to plan anything regarding
            # it. I only react to selling contracts
            is_seller = contract.annotation["seller"] == self.id
            # If this contract is too late or too early, I can do nothing.
            if t > self.awi.n_steps - 1 or t < this_step:
                continue
            if is_seller:
                # if I am a seller, try to find a way to schedule production
                # to have the required items
                steps, _ = self.awi.available_for_production(
                    repeats=q, step=(this_step + 1, t - 1)
                )
                # If I cannot produce the required items, ignore the contract
                if len(steps) < 1:
                    continue
                # registers needs for inputs
                self.inputs_needed[min(steps)] += q
                continue
            # I am a buyer. I need not produce anything but I need to negotiate to sell the production of what I bought
            if inp != contract.annotation["product"]:
                continue
            self.outputs_needed[t] += q


class PredictionBasedTradingStrategy(
    FixedTradePredictionStrategy, MeanERPStrategy, TradingStrategy
):
    """A trading strategy that uses prediction strategies to manage inputs/outputs needed

    Hooks Into:
        - `init`
        - `on_contracts_finalized`
        - `sign_all_contracts`
        - `on_agent_bankrupt`

    Requires:
        - `expected_inputs` (np.ndarray):  How many items of the input product do
          I expect to have every day. Should be adjusted by the `TradePredictionStrategy` .
        - `expected_outputs` (np.ndarray):  How many items of the output product do
          I expect to have every day. Should be adjusted by the `TradePredictionStrategy` .

    Remarks:
        - `Attributes` section describes the attributes that can be used to construct the component (passed to its
          `__init__` method).
        - `Provides` section describes the attributes (methods, properties, data-members) made available by this
          component directly. Note that everything provided by the bases of this components are also available to the
          agent (Check the `Bases` section above for all the bases of this component).
        - `Requires` section describes any requirements from the agent using this component. It defines a set of methods
          or properties/data-members that must exist in the agent that uses this component. These requirement are
          usually implemented as abstract methods in the component
        - `Abstract` section describes abstract methods that MUST be implemented by any descendant of this component.
        - `Hooks Into` section describes the methods this component overrides calling `super` () which allows other
          components to hook into the same method (by overriding it). Usually callbacks starting with `on_` are
          hooked into this way.
        - `Overrides` section describes the methods this component overrides without calling `super` effectively
          disallowing any other components after it in the MRO to call this method. Usually methods that do some
          action (i.e. not starting with `on_`) are overridden this way.


    """

    def init(self):
        super().init()
        # If I expect to sell x outputs at step t, I should buy  x inputs at t-1
        self.inputs_needed[:-1] = self.expected_outputs[1:]
        # If I expect to buy x inputs at step t, I should sell x inputs at t+1
        self.outputs_needed[1:] = self.expected_inputs[:-1]

    def on_contracts_finalized(
        self,
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        super().on_contracts_finalized(signed, cancelled, rejectors)
        # keeps track of the procution slots consumed by signed contracts processed
        consumed = 0
        for contract in signed:
            # If I intiated the negotiation for this contract, ignore it.
            if contract.annotation["caller"] == self.id:
                continue
            is_seller = contract.annotation["seller"] == self.id
            q, t = (
                contract.agreement["quantity"],
                contract.agreement["time"],
            )
            if is_seller:
                # if I am a seller, I will buy my needs to produce
                output_product = contract.annotation["product"]
                input_product = output_product - 1
                # register that I secued the given outputs.
                self.outputs_secured[t] += q
                # If I need to produce, do production
                if input_product >= 0 and t > 0:
                    # find the maximum possible production I can do and saturate to it
                    steps, _ = self.awi.available_for_production(
                        repeats=q, step=(self.awi.current_step, t - 1)
                    )
                    # register the number of production slots consumed for this contract
                    q = min(len(steps) - consumed, q)
                    consumed += q
                    # this is a sell contract that I did not expect yet. Update needs accordingly
                    # I must buy all my needs one day earlier at most
                    self.inputs_needed[t - 1] += max(1, q)
                continue

            # I am a buyer. I need not produce anything but I need to negotiate to sell the production of what I bought
            input_product = contract.annotation["product"]
            output_product = input_product + 1
            # register that I secured the given outputs
            self.inputs_secured[t] += q
            if output_product < self.awi.n_products and t < self.awi.n_steps - 1:
                # this is a buy contract that I did not expect yet. Update needs accordingly
                # I must sell these inputs after production one day later at least
                self.outputs_needed[t + 1] += max(1, q)

    def sign_all_contracts(self, contracts: List[Contract]) -> List[Optional[str]]:
        signatures = [None] * len(contracts)
        # sort contracts by time and then put system contracts first within each time-step
        contracts = sorted(
            zip(contracts, range(len(contracts))),
            key=lambda x: (
                x[0].agreement["unit_price"],
                x[0].agreement["time"],
                0
                if is_system_agent(x[0].annotation["seller"])
                or is_system_agent(x[0].annotation["buyer"])
                else 1,
                x[0].agreement["unit_price"],
            ),
        )
        sold, bought = 0, 0
        s = self.awi.current_step
        for contract, indx in contracts:
            is_seller = contract.annotation["seller"] == self.id
            q, u, t = (
                contract.agreement["quantity"],
                contract.agreement["unit_price"],
                contract.agreement["time"],
            )
            # check that the contract is executable in principle. The second
            # condition checkes that the contract is negotiated and not exogenous
            if t < s and len(contract.issues) == 3:
                continue
            # catalog_buy = self.input_cost[t]
            # catalog_sell = self.output_price[t]
            # # check that the gontract has a good price
            # if (is_seller and u < 0.5 * catalog_sell) or (
            #     not is_seller and u > 1.5 * catalog_buy
            # ):
            #     continue
            if is_seller:
                trange = (s, t - 1)
                secured, needed = (self.outputs_secured, self.outputs_needed)
                taken = sold
            else:
                trange = (t + 1, self.awi.n_steps - 1)
                secured, needed = (self.inputs_secured, self.inputs_needed)
                taken = bought

            # check that I can produce the required quantities even in principle
            steps, _ = self.awi.available_for_production(
                q, trange, ANY_LINE, override=False, method="all"
            )
            if len(steps) - taken < q:
                continue

            if (
                secured[trange[0] : trange[1] + 1].sum() + q + taken
                <= needed[trange[0] : trange[1] + 1].sum()
            ):
                signatures[indx] = self.id
                if is_seller:
                    sold += q
                else:
                    bought += q
        return signatures

    def _format(self, c: Contract):
        return (
            f"{f'>' if c.annotation['seller'] == self.id else '<'}"
            f"{c.annotation['buyer'] if c.annotation['seller'] == self.id else c.annotation['seller']}: "
            f"{c.agreement['quantity']} of {c.annotation['product']} @ {c.agreement['unit_price']} on {c.agreement['time']}"
        )

    def on_agent_bankrupt(
        self,
        agent: str,
        contracts: List[Contract],
        quantities: List[int],
        compensation_money: int,
    ) -> None:
        super().on_agent_bankrupt(agent, contracts, quantities, compensation_money)
        for contract, new_quantity in zip(contracts, quantities):
            q = contract.agreement["quantity"]
            if new_quantity == q:
                continue
            t = contract.agreement["time"]
            missing = q - new_quantity
            s = self.awi.current_step
            if t < self.awi.current_step:
                continue
            # distribute the missing quantity over time
            if contract.annotation["seller"] == self.id:
                # self.outputs_secured[t] -= missing
                if t > s:
                    for tau in range(t - 1, s - 1, -1):
                        if self.inputs_needed[tau] <= 0:
                            continue
                        if self.inputs_needed[tau] >= missing:
                            self.inputs_needed[tau] -= missing
                            missing = 0
                            break
                        self.inputs_needed[tau] = 0
                        missing -= self.inputs_needed[tau]
                        if missing <=0:
                            break
                if missing > 0:
                    if t < self.awi.n_steps - 1:
                        for tau in range(t + 1, self.awi.n_steps):
                            if self.outputs_secured[tau] <= 0:
                                continue
                            if self.outputs_secured[tau] >= missing:
                                self.outputs_secured[tau] -= missing
                                missing = 0
                                break
                            self.outputs_secured[tau] = 0
                            missing -= self.outputs_secured[tau]
                            if missing <=0:
                                break

            else:
                if t < self.awi.n_steps - 1:
                    for tau in range(t + 1, self.awi.n_steps):
                        if self.outputs_needed[tau] <= 0:
                            continue
                        if self.outputs_needed[tau] >= missing:
                            self.outputs_needed[tau] -= missing
                            missing = 0
                            break
                        self.outputs_needed[tau] = 0
                        missing -= self.outputs_needed[tau]
                        if missing <=0:
                            break
                if missing > 0:
                    if t > s:
                        for tau in range(t - 1, s-1, -1):
                            if self.inputs_secured[tau] <= 0:
                                continue
                            if self.inputs_secured[tau] >= missing:
                                self.inputs_secured[tau] -= missing
                                missing = 0
                                break
                            self.inputs_secured[tau] = 0
                            missing -= self.inputs_secured[tau]
                            if missing <=0:
                                break


class MarketAwarePredictionBasedTradingStrategy(
    MarketAwareTradePredictionStrategy, PredictionBasedTradingStrategy
):
    pass
