from abc import abstractmethod
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union

import numpy as np
from negmas import Breach
from negmas import Contract

__all__ = [
    "TradePredictionStrategy",
    "FixedTradePredictionStrategy",
    "ExecutionRatePredictionStrategy",
    "FixedERPStrategy",
    "MeanERPStrategy",
    "MarketAwareTradePredictionStrategy",
]


class TradePredictionStrategy:
    """A prediction strategy for expected inputs and outputs at every step

    Args:
        - `predicted_inputs`: None for default, a number of an n_steps numbers giving predicted inputs
        - `predicted_outputs`: None for default, a number of an n_steps numbers giving predicted outputs
        - `add_trade`: If true, actual contracts will be just added to the expectations

    Provides:
        - `expected_inputs` : n_steps vector giving the predicted inputs at every time-step
        - `expected_outputs` : n_steps vector giving the predicted outputs at every time-step
        - `input_cost` : n_steps vector giving the predicted input cost at every time-step
        - `output_price` : n_steps vector giving the predicted output price at every time-step

    Hooks Into:
        - `init`
        - `before_step`

    Abstract:
        - `trade_prediction_init`: Called during init() to initialize the trade prediction.
        - `trade_prediction_step`: Called during step() to update the trade prediction.

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

    def __init__(
        self,
        *args,
        predicted_outputs: Union[int, np.ndarray] = None,
        predicted_inputs: Union[int, np.ndarray] = None,
        add_trade=True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.expected_outputs = predicted_outputs
        """Expected output quantity every step"""
        self.expected_inputs = predicted_inputs
        """Expected input quantity every step"""
        self.input_cost: np.ndarray = None
        """Expected unit price of the input"""
        self.output_price: np.ndarray = None
        """Expected unit price of the output"""
        self._add_trade = add_trade

    @abstractmethod
    def trade_prediction_init(self) -> None:
        """Will be called to update expected_outputs, expected_inputs,
        input_cost, output_cost during init()"""

    def trade_prediction_step(self) -> None:
        """Will be called at every step to update the prediction"""

    def init(self):
        self.input_cost = self.awi.catalog_prices[self.awi.my_input_product] * np.ones(
            self.awi.n_steps, dtype=int
        )
        self.output_price = self.awi.catalog_prices[
            self.awi.my_output_product
        ] * np.ones(self.awi.n_steps, dtype=int)

        inp = self.awi.my_input_product

        def adjust(x):
            return max(1, self.awi.n_lines // 2) * np.ones(self.awi.n_steps, dtype=int)

        # adjust predicted demand and supply
        self.expected_outputs = adjust(self.expected_outputs)
        self.expected_inputs = adjust(self.expected_inputs)
        self.trade_prediction_init()
        super().init()

    def before_step(self):
        self.trade_prediction_step()
        super().before_step()


class ExecutionRatePredictionStrategy:
    """
    A prediction strategy for expected inputs and outputs at every step

    Provides:
        - `predict_quantity` : A method for predicting the quantity that will actually be executed from a contract

    Abstract:
        - `predict_quantity` : A method for predicting the quantity that will actually be executed from a contract

    Hooks Into:
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

    @abstractmethod
    def predict_quantity(self, contract: Contract):
        raise NotImplementedError(
            "predict_quantity should be implemented by some other component."
        )


class FixedTradePredictionStrategy(TradePredictionStrategy):
    """
    Predicts a fixed amount of trade both for the input and output products.

    Hooks Into:
        - `internal_state`
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

    def trade_prediction_init(self):
        inp = self.awi.my_input_product

        def adjust(x, demand):
            """Adjust the predicted demand/supply filling it with a default value or repeating as needed"""
            if x is None:
                x = max(1, int(self.awi.n_lines * 0.75))
            elif isinstance(x, Iterable):
                return np.array(x)
            predicted = int(x) * np.ones(self.awi.n_steps, dtype=int)
            if demand:
                predicted[: inp + 1] = 0
            else:
                predicted[inp - self.awi.n_processes :] = 0
            return predicted

        # adjust predicted demand and supply
        self.expected_outputs = adjust(self.expected_outputs, True)
        self.expected_inputs = adjust(self.expected_inputs, False)

    def trade_prediction_step(self):
        pass

    @property
    def internal_state(self):
        state = super().internal_state
        state.update(
            {
                "expected_inputs": self.expected_inputs,
                "expected_outputs": self.expected_outputs,
                "input_cost": self.input_cost,
                "output_price": self.output_price,
            }
        )
        return state

    def on_contracts_finalized(
        self,
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        super().on_contracts_finalized(signed, cancelled, rejectors)
        if not self._add_trade:
            return
        for contract in signed:
            t, q = contract.agreement["time"], contract.agreement["quantity"]
            if contract.annotation["seller"] == self.id:
                self.expected_outputs[t] += q
            else:
                self.expected_inputs[t] += q


class MarketAwareTradePredictionStrategy(TradePredictionStrategy):
    """
    Predicts an amount based on publicly available market information. Falls
    back to fixed prediction if no information is available

    Hooks Into:
        - `internal_state`
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

    def trade_prediction_init(self):
        inp = self.awi.my_input_product

        def adjust(x, demand):
            """Adjust the predicted demand/supply filling it with a default value or repeating as needed"""
            if x is None:
                x = max(1, self.awi.n_lines // 2)
            elif isinstance(x, Iterable):
                return np.array(x)
            predicted = int(x) * np.ones(self.awi.n_steps, dtype=int)
            if demand:
                predicted[: inp + 1] = 0
            else:
                predicted[inp - self.awi.n_processes :] = 0
            return predicted

        # adjust predicted demand and supply
        self.expected_outputs = adjust(self.expected_outputs, True)
        self.expected_inputs = adjust(self.expected_inputs, False)

    def trade_prediction_step(self):
        exogenous = self.awi.exogenous_contract_summary
        horizon = self.awi.settings.get("horizon", 1)
        a, b = self.awi.current_step, self.awi.current_step + horizon
        # breakpoint()
        self.expected_inputs[a:b] = exogenous[self.awi.my_input_product, a:b, 0]
        self.expected_outputs[a:b] = exogenous[self.awi.my_output_product, a:b, 0]

    @property
    def internal_state(self):
        state = super().internal_state
        state.update(
            {
                "expected_inputs": self.expected_inputs,
                "expected_outputs": self.expected_outputs,
                "input_cost": self.input_cost,
                "output_price": self.output_price,
            }
        )
        return state


class FixedERPStrategy(ExecutionRatePredictionStrategy):
    """Predicts that the there is a fixed execution rate that does not change for all partners

    Args:
        execution_fraction: The expected fraction of any contract's quantity to be executed

    Provides:
        - `predict_quantity` : A method for predicting the quantity that will actually be executed from a contract

    Hooks Into:
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

    def __init__(self, *args, execution_fraction=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self._execution_fraction = execution_fraction

    def predict_quantity(self, contract: Contract):
        return contract.agreement["quantity"] * self._execution_fraction


class MeanERPStrategy(ExecutionRatePredictionStrategy):
    """
    Predicts that the there is a fixed execution rate that does not change for all partners

    Args:
        execution_fraction: The expected fraction of any contract's quantity to be executed

    Provides:
        - `predict_quantity` : A method for predicting the quantity that will actually be executed from a contract

    Hooks Into:
        - `internal_state`
        - `init`
        - `on_contract_executed`
        - `on_contract_breached`

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

    def __init__(self, *args, execution_fraction=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self._execution_fraction = execution_fraction
        self._total_quantity = None

    def predict_quantity(self, contract: Contract):
        return contract.agreement["quantity"] * self._execution_fraction

    def init(self):
        super().init()
        self._total_quantity = max(1, self.awi.n_steps * self.awi.n_lines // 10)

    @property
    def internal_state(self):
        state = super().internal_state
        state.update({"execution_fraction": self._execution_fraction})
        return state

    def on_contract_executed(self, contract: Contract) -> None:
        super().on_contract_executed(contract)
        old_total = self._total_quantity
        q = contract.agreement["quantity"]
        self._total_quantity += q
        self._execution_fraction = (
            self._execution_fraction * old_total + q
        ) / self._total_quantity

    def on_contract_breached(
        self, contract: Contract, breaches: List[Breach], resolution: Optional[Contract]
    ) -> None:
        super().on_contract_breached(contract, breaches, resolution)
        old_total = self._total_quantity
        q = int(contract.agreement["quantity"] * (1.0 - max(b.level for b in breaches)))
        self._total_quantity += contract.agreement["quantity"]
        self._execution_fraction = (
            self._execution_fraction * old_total + q
        ) / self._total_quantity
