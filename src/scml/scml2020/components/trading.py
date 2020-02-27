import numpy as np
from pprint import pformat
from typing import List, Optional

from negmas import Contract

from scml.scml2020.components import FixedTradePredictionStrategy, SignAllPossible
from scml.scml2020.common import is_system_agent
from scml.scml2020.common import ANY_LINE
from scml.scml2020.components.prediction import MeanERPStrategy

__all__ = [
    "TradingStrategy",
    "ReactiveTradingStrategy",
    "PredictionBasedTradingStrategy",
]


class TradingStrategy:
    """Base class for all trading strategies.

    Provides:
        - `inputs_needed` (np.ndarray):  How many items of the input product do I need at every time step
          (n_steps vector)
        - `outputs_needed` (np.ndarray):  How many items of the output product do I need at every time step
          (n_steps vector)
        - `inputs_secured` (np.ndarray):  How many items of the output product do I need at every time step
          (n_steps vector)
        - `outputs_secured` (np.ndarray):  How many units of the output product I have already secured per step
              (n_steps vector)

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
            if contract.annotation["caller"] == self.id:
                continue
            if contract.annotation["product"] != outp:
                continue
            is_seller = contract.annotation["seller"] == self.id
            # find the earliest time I can do anything about this contract
            if t > self.awi.n_steps - 1 or t < this_step:
                continue
            if is_seller:
                # if I am a seller, I will schedule production then buy my needs to produce
                steps, _ = self.awi.available_for_production(
                    repeats=q, step=(this_step + 1, t - 1)
                )
                if len(steps) < 1:
                    continue
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
        self.awi.logdebug_agent(
            f"Enter Contracts Finalized:\n"
            f"Signed {pformat([self._format(_) for _ in signed])}\n"
            f"Cancelled {pformat([self._format(_) for _ in cancelled])}\n"
            f"{pformat(self.internal_state)}"
        )
        super().on_contracts_finalized(signed, cancelled, rejectors)
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
            f"Exit Contracts Finalized:\n{pformat(self.internal_state)}"
        )

    def sign_all_contracts(self, contracts: List[Contract]) -> List[Optional[str]]:
        # sort contracts by time and then put system contracts first within each time-step
        signatures = [None] * len(contracts)
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
        taken = 0
        s = self.awi.current_step
        for contract, indx in contracts:
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
                secured, needed = (self.outputs_secured, self.outputs_needed)
            else:
                trange = (t + 1, self.awi.n_steps - 1)
                secured, needed = (self.inputs_secured, self.inputs_needed)

            # check that I can produce the required quantities even in principle
            steps, lines = self.awi.available_for_production(
                q, trange, ANY_LINE, override=False, method="all"
            )
            if len(steps) - taken < q:
                continue

            if (
                secured[trange[0] : trange[1] + 1].sum() + q + taken
                <= needed[trange[0] : trange[1] + 1].sum()
            ):
                signatures[indx] = self.id
                taken += self.predict_quantity(contract)
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
            if t < self.awi.current_step:
                continue
            if contract.annotation["seller"] == self.id:
                self.outputs_secured[t] -= missing
                if t > 0:
                    self.inputs_needed[t - 1] -= missing
            else:
                self.inputs_secured[t] += missing
                if t < self.awi.n_steps - 1:
                    self.outputs_needed[t + 1] -= missing
