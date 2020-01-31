
__all__ = [
    "SignAll",
    "SignAllPossible",
]

from typing import List, Optional

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
        for contract, i in contracts:
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
        return results
