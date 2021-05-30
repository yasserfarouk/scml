__all__ = ["SignAll", "SignAllPossible", "KeepOnlyGoodPrices"]

from typing import List
from typing import Optional

from negmas import Contract

from scml.scml2020.world import is_system_agent


class SignAll:
    """Signs all contracts no matter what.

    Overrides:
        - `sign_all_contracts`

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

    def sign_all_contracts(self, contracts: List[Contract]) -> List[Optional[str]]:
        # calls the super class to allow it to do any book-keeping.
        return [self.id] * len(contracts)


class SignAllPossible:
    """
    Signs all contracts that can in principle be honored.
    The only check made by this strategy is that for sell contracts there is enough production capacity

    Overrides:
        - `sign_all_contracts`

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

    def sign_all_contracts(self, contracts: List[Contract]) -> List[Optional[str]]:
        results = [None] * len(contracts)
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
            ),
        )
        consumed = 0
        for contract, i in contracts:
            step = contract.agreement["time"]
            q = contract.agreement["quantity"]
            if step > self.awi.n_steps - 1 or step < self.awi.current_step:
                continue
            if contract.annotation["seller"] == self.id:
                steps, _ = self.awi.available_for_production(
                    q, (self.awi.current_step, step), -1, override=False, method="all"
                )
                if len(steps) - consumed < q:
                    continue
                consumed += q
            results[i] = self.id
        return results


class KeepOnlyGoodPrices:
    """Signs all contracts that have good prices

    Overrides:
        - `sign_all_contracts`

    Attributes:
        - buying_margin: The margin from the catalog price to allow for buying. The agent will never buy at a price
          higher than the catalog price by more than this margin (relative to catalog price).
        - selling_margin: The margin from the catalog price to allow for selling. The agent will never sell at a price
          lower than the catalog price by more than this margin (relative to catalog price).

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

    def __init__(self, *args, buying_margin=0.5, selling_margin=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        if buying_margin is None:
            buying_margin = 0.5
        if selling_margin is None:
            selling_margin = 0.5
        self._buying_factor, self._selling_factor = (
            1.0 + buying_margin,
            1.0 - selling_margin,
        )

    def sign_all_contracts(self, contracts: List[Contract]) -> List[Optional[str]]:
        # calls the super class to allow it to do any book-keeping.
        signatures = super().sign_all_contracts(contracts)
        # calculate current trading prices (approximate them with catalog price
        # if not available)
        tp = self.awi.trading_prices
        if tp is None:
            in_price = self.awi.catalog_prices[self.awi.my_input_product]
            out_price = self.awi.catalog_prices[self.awi.my_output_product]
        else:
            in_price = self.awi.trading_prices[self.awi.my_input_product]
            out_price = self.awi.trading_prices[self.awi.my_output_product]

        for i, c in enumerate(contracts):
            if signatures[i] is None:
                continue
            if (
                c.annotation["buyer"] == self.id
                and c.agreement["unit_price"] > self._buying_factor * in_price
            ) or (
                c.annotation["seller"] == self.id
                and c.agreement["unit_price"] < self._selling_factor * out_price
            ):
                signatures[i] = None
        return signatures
