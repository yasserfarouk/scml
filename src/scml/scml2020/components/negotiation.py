import functools
import math
from abc import abstractmethod
from dataclasses import dataclass
from pprint import pformat
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from negmas import AgentMechanismInterface
from negmas import AspirationNegotiator
from negmas import Contract
from negmas import Issue
from negmas import Negotiator
from negmas import SAONegotiator
from negmas import UtilityFunction
from negmas.helpers import get_class
from negmas.helpers import instantiate

from scml.scml2020 import AWI
from scml.scml2020.common import TIME
from scml.scml2020.components.prediction import MeanERPStrategy
from scml.scml2020.services.controllers import StepController
from scml.scml2020.services.controllers import SyncController

__all__ = [
    "NegotiationManager",
    "StepNegotiationManager",
    "IndependentNegotiationsManager",
    "MovingRangeNegotiationManager",
]


class NegotiationManager:
    """A negotiation manager is a component that provides negotiation control functionality to an agent

    Args:
        horizon: The number of steps in the future to consider for selling outputs.

    Provides:
        - `start_negotiations` An easy to use method to start a set of buy/sell negotiations

    Requires:
        - `acceptable_unit_price`
        - `target_quantity`
        - OPTIONALLY `target_quantities`

    Abstract:
        - `respond_to_negotiation_request`

    Hooks Into:
        - `init`
        - `step`
        - `on_contracts_finalized`
        - `respond_to_negotiation_request`

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
        horizon=5,
        negotiate_on_signing=True,
        logdebug=False,
        use_trading_prices=True,
        min_price_margin=0.5,
        max_price_margin=0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._horizon = horizon
        self._negotiate_on_signing = negotiate_on_signing
        self._log = logdebug
        self._use_trading = use_trading_prices
        self._min_margin = 1 - min_price_margin
        self._max_margin = 1 + max_price_margin

    @property
    def use_trading(self):
        return self._use_trading

    @use_trading.setter
    def use_trading(self, v):
        self._use_trading = v

    def init(self):
        # set horizon to exogenous horizon
        if self._horizon is None:
            self._horizon = self.awi.bb_read("settings", "exogenous_horizon")
            if self._horizon is None:
                self._horizon = 5

        # let other component work. We init last here as we do not depend on any other components for this method.
        super().init()

    def start_negotiations(
        self,
        product: int,
        quantity: int,
        unit_price: int,
        step: int,
        partners: List[str] = None,
    ) -> None:
        """
        Starts a set of negotiations to buy/sell the product with the given limits

        Args:
            product: product type. If it is an input product, negotiations to buy it will be started otherweise to sell.
            quantity: The maximum quantity to negotiate about
            unit_price: The maximum/minimum unit price for buy/sell
            step: The maximum/minimum time for buy/sell
            partners: A list of partners to negotiate with

        Remarks:

            - This method assumes that product is either my_input_product or my_output_product

        """
        awi: AWI
        awi = self.awi  # type: ignore
        is_seller = product == self.awi.my_output_product
        if quantity < 1 or unit_price < 1 or step < awi.current_step + 1:
            # awi.logdebug_agent(
            #     f"Less than 2 valid issues (q:{quantity}, u:{unit_price}, t:{step})"
            # )
            return
        # choose ranges for the negotiation agenda.
        qvalues = (1, quantity)
        tvalues = self._trange(step, is_seller)
        uvalues = self._urange(step, is_seller, tvalues)
        if tvalues[0] > tvalues[1]:
            return
        if partners is None:
            if is_seller:
                partners = awi.my_consumers
            else:
                partners = awi.my_suppliers
        self._start_negotiations(
            product, is_seller, step, qvalues, uvalues, tvalues, partners
        )

    def step(self):
        super().step()
        """Generates buy and sell negotiations as needed"""
        s = self.awi.current_step
        if s == 0:
            # in the first step, generate buy/sell negotiations for horizon steps in the future
            last = min(self.awi.n_steps - 1, self._horizon + 2)
            for step in range(1, last):
                self._generate_negotiations(step, False)
                self._generate_negotiations(step, True)
        else:
            # generate buy and sell negotiations to secure inputs/outputs the step after horizon steps
            nxt = s + self._horizon + 1
            if nxt > self.awi.n_steps - 1:
                return
            self._generate_negotiations(nxt, False)
            self._generate_negotiations(nxt, True)
        if self._log:
            self.awi.logdebug_agent(f"End step:\n{pformat(self.internal_state)}")

    def on_contracts_finalized(
        self,
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        if self._negotiate_on_signing:
            steps = (self.awi.current_step + 1, self.awi.n_steps)
            in_pre = self.target_quantities(steps, False)
            out_pre = self.target_quantities(steps, True)
        super().on_contracts_finalized(signed, cancelled, rejectors)
        if (not self._negotiate_on_signing) or (in_pre is None) or (out_pre is None):
            return
        inputs_needed = self.target_quantities(steps, False)
        outputs_needed = self.target_quantities(steps, True)
        if (inputs_needed is None) or (outputs_needed is None):
            return
        inputs_needed -= in_pre
        outputs_needed -= out_pre
        for s in np.nonzero(inputs_needed)[0]:
            if inputs_needed[s] < 0:
                continue
            self.start_negotiations(
                self.awi.my_input_product,
                inputs_needed[s],
                self.acceptable_unit_price(s, False),
                s,
            )
        for s in np.nonzero(outputs_needed)[0]:
            if outputs_needed[s] < 0:
                continue
            self.start_negotiations(
                self.awi.my_output_product,
                outputs_needed[s],
                self.acceptable_unit_price(s, True),
                s,
            )

    def _generate_negotiations(self, step: int, sell: bool) -> None:
        """Generates all the required negotiations for selling/buying for the given step"""
        product = self.awi.my_output_product if sell else self.awi.my_input_product
        quantity = self.target_quantity(step, sell)
        unit_price = self.acceptable_unit_price(step, sell)

        if quantity <= 0 or unit_price <= 0:
            return
        self.start_negotiations(
            product=product,
            step=step,
            quantity=min(self.awi.n_lines * (step - self.awi.current_step), quantity),
            unit_price=unit_price,
        )

    def _urange(self, step, is_seller, time_range):
        prices = (
            self.awi.catalog_prices
            if not self._use_trading
            or not self.awi.settings.get("public_trading_prices", False)
            else self.awi.trading_prices
        )
        if is_seller:
            cprice = prices[self.awi.my_output_product]
            return int(cprice * self._min_margin), int(self._max_margin * cprice + 0.5)

        cprice = prices[self.awi.my_input_product]
        return int(cprice * self._min_margin), int(self._max_margin * cprice + 0.5)

    def _trange(self, step, is_seller):
        if is_seller:
            return (
                max(step, self.awi.current_step + 1),
                min(step + self._horizon, self.awi.n_steps - 1),
            )
        return self.awi.current_step + 1, step - 1

    def target_quantities(self, steps: Tuple[int, int], sell: bool) -> np.ndarray:
        """
        Returns the target quantity to negotiate about for each step in the range given (beginning included and ending
        excluded) for buying/selling

        Args:
            steps: Simulation step
            sell: Sell or buy
        """
        return np.array([self.target_quantity(s, sell) for s in range(*steps)])

    @abstractmethod
    def _start_negotiations(
        self,
        product: int,
        sell: bool,
        step: int,
        qvalues: Tuple[int, int],
        uvalues: Tuple[int, int],
        tvalues: Tuple[int, int],
        partners: List[str],
    ) -> None:
        """
        Actually start negotiations with the given agenda

        Args:
            product: The product to negotiate about.
            sell: If true, this is a sell negotiation
            step: The step
            qvalues: the range of quantities
            uvalues: the range of unit prices
            tvalues: the range of times
            partners: partners
        """
        pass

    @abstractmethod
    def target_quantity(self, step: int, sell: bool) -> int:
        """
        Returns the target quantity to sell/buy at a given time-step

        Args:
            step: Simulation step
            sell: Sell or buy
        """
        raise ValueError("You must implement target_quantity")

    @abstractmethod
    def acceptable_unit_price(self, step: int, sell: bool) -> int:
        """
        Returns the maximum/minimum acceptable unit price for buying/selling at the given time-step

        Args:
            step: Simulation step
            sell: Sell or buy
        """
        raise ValueError("You must implement acceptable_unit_price")

    @abstractmethod
    def respond_to_negotiation_request(
        self,
        initiator: str,
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
    ) -> Optional[Negotiator]:
        raise ValueError("You must implement respond_to_negotiation_request")


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


class StepNegotiationManager(MeanERPStrategy, NegotiationManager):
    """
    A negotiation manager that controls a controller and another for selling for every timestep

    Args:
        negotiator_type: The negotiator type to use to manage all negotiations
        negotiator_params: Paramters of the negotiator

    Provides:
        - `all_negotiations_concluded`

    Requires:
        - `acceptable_unit_price`
        - `target_quantity`
        - OPTIONALLY `target_quantities`

    Hooks Into:
        - `init`
        - `respond_to_negotiation_request`

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
        negotiator_type: Union[SAONegotiator, str] = AspirationNegotiator,
        negotiator_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # save construction parameters
        self.negotiator_type = get_class(negotiator_type)
        self.negotiator_params = (
            negotiator_params if negotiator_params is not None else dict()
        )

        # attributes that will be read during init() from the AWI
        # -------------------------------------------------------
        self.buyers = self.sellers = None
        """Buyer controllers and seller controllers. Each of them is responsible of covering the
        needs for one step (either buying or selling)."""

    def init(self):
        super().init()

        # initialize one controller for buying and another for selling for each time-step
        self.buyers: List[ControllerInfo] = [
            ControllerInfo(None, i, False, tuple(), 0, 0, False)
            for i in range(self.awi.n_steps)
        ]
        self.sellers: List[ControllerInfo] = [
            ControllerInfo(None, i, True, tuple(), 0, 0, False)
            for i in range(self.awi.n_steps)
        ]
        # self.awi.logdebug_agent(f"Initialized\n{pformat(self.internal_state)}")

    def _start_negotiations(
        self,
        product: int,
        sell: bool,
        step: int,
        qvalues: Tuple[int, int],
        uvalues: Tuple[int, int],
        tvalues: Tuple[int, int],
        partners: List[str],
    ) -> None:
        if sell:
            expected_quantity = int(math.floor(qvalues[1] * self._execution_fraction))
        else:
            expected_quantity = int(math.floor(qvalues[1] * self._execution_fraction))

        # negotiate with everyone
        controller = self.create_controller(
            sell, qvalues[1], uvalues, expected_quantity, step
        )
        # self.awi.loginfo_agent(
        #     f"Requesting {'selling' if sell else 'buying'} negotiation "
        #     f"on u={uvalues}, q={qvalues}, t={tvalues}"
        #     f" with {str(partners)} using {str(controller)}"
        # )
        if self.awi.request_negotiations(
            is_buy=not sell,
            product=product,
            quantity=qvalues,
            unit_price=uvalues,
            time=tvalues,
            partners=partners,
            controller=controller,
            extra=dict(controller_index=step, is_seller=sell),
        ):
            self.add_controller(
                controller, sell, qvalues[1], uvalues, expected_quantity, step
            )

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
        target = self.target_quantities((tmin, tmax + 1), is_seller).sum()
        if target <= 0:
            return None
        # self.awi.loginfo_agent(
        #     f"Accepting request from {initiator}: {[str(_) for _ in mechanism.issues]} "
        #     f"({Issue.num_outcomes(mechanism.issues)})"
        # )
        # create a controller for the time-step if one does not exist or use the one already running
        if controller_info.controller is None:
            controller = self.create_controller(
                is_seller,
                target,
                self._urange(step, is_seller, (tmin, tmax)),
                int(target),
                step,
            )
            controller = self.add_controller(
                controller,
                is_seller,
                target,
                self._urange(step, is_seller, (tmin, tmax)),
                int(target),
                step,
            )
        else:
            controller = controller_info.controller

        # create a new negotiator, add it to the controller and return it
        return controller.create_negotiator(id=initiator)

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
        if c is None:
            return
        quantity = c.secured
        target = c.target
        time_range = info.time_range
        if is_seller:
            controllers = self.sellers
        else:
            controllers = self.buyers

        # self.awi.logdebug_agent(
        #     f"Killing Controller {str(controllers[controller_index].controller)}"
        # )
        controllers[controller_index].controller = None
        if quantity <= target:
            self._generate_negotiations(step=controller_index, sell=is_seller)
            return

    def add_controller(
        self,
        controller: StepController,
        is_seller: bool,
        target,
        urange: Tuple[int, int],
        expected_quantity: int,
        step: int,
    ) -> StepController:
        if is_seller:
            # assert self.sellers[step].controller is None
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
            # assert self.buyers[step].controller is None
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

    def create_controller(
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
        return StepController(
            is_seller=is_seller,
            target_quantity=target,
            negotiator_type=self.negotiator_type,
            negotiator_params=self.negotiator_params,
            step=step,
            urange=urange,
            product=self.awi.my_output_product
            if is_seller
            else self.awi.my_input_product,
            partners=self.awi.my_consumers if is_seller else self.awi.my_suppliers,
            horizon=self._horizon,
            negotiations_concluded_callback=functools.partial(
                self.__class__.all_negotiations_concluded, self
            ),
            parent_name=self.name,
            awi=self.awi,
        )

    def _get_controller(self, mechanism) -> StepController:
        neg = self._running_negotiations[mechanism.id]
        return neg.negotiator.parent


class IndependentNegotiationsManager(NegotiationManager):
    """
    A negotiation manager that manages independent negotiators that do not share any information once created

    Args:
        negotiator_type: The negotiator type to use to manage all negotiations
        negotiator_params: Parameters of the negotiator

    Requires:
        - `create_ufun`
        - `acceptable_unit_price`
        - `target_quantity`
        - OPTIONALLY `target_quantities`

    Hooks Into:
        - `respond_to_negotiation_request`

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
        negotiator_type: Union[SAONegotiator, str] = AspirationNegotiator,
        negotiator_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.negotiator_type = get_class(negotiator_type)
        self.negotiator_params = (
            negotiator_params if negotiator_params is not None else dict()
        )

    def _start_negotiations(
        self,
        product: int,
        sell: bool,
        step: int,
        qvalues: Tuple[int, int],
        uvalues: Tuple[int, int],
        tvalues: Tuple[int, int],
        partners: List[str],
    ) -> None:
        # negotiate with all suppliers of the input product I need to produce

        issues = [
            Issue((int(qvalues[0]), int(qvalues[1])), name="quantity"),
            Issue((int(tvalues[0]), int(tvalues[1])), name="time"),
            Issue((int(uvalues[0]), int(uvalues[1])), name="uvalues"),
        ]

        for partner in partners:
            self.awi.request_negotiation(
                is_buy=not sell,
                product=product,
                quantity=qvalues,
                unit_price=uvalues,
                time=tvalues,
                partner=partner,
                negotiator=self.negotiator(sell, issues=issues, partner=partner),
            )

    def respond_to_negotiation_request(
        self,
        initiator: str,
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
    ) -> Optional[Negotiator]:
        return self.negotiator(
            annotation["seller"] == self.id, issues=issues, partner=initiator
        )

    @abstractmethod
    def create_ufun(
        self, is_seller: bool, issues=None, outcomes=None
    ) -> UtilityFunction:
        """Creates a utility function"""

    def negotiator(
        self, is_seller: bool, issues=None, outcomes=None, partner=None
    ) -> SAONegotiator:
        """Creates a negotiator"""
        if outcomes is None and (
            issues is None or not Issue.enumerate(issues, astype=tuple)
        ):
            return None
        params = self.negotiator_params
        params["ufun"] = self.create_ufun(
            is_seller=is_seller, outcomes=outcomes, issues=issues
        )
        return instantiate(self.negotiator_type, id=partner, **params)


class MovingRangeNegotiationManager:
    """My negotiation strategy

    Args:
        price_weight: The relative importance of price in the utility calculation.
        utility_threshold: The fraction of maximum utility above which all offers will be accepted.
        time_threshold: The fraction of the negotiation time after which any valid offers will be accepted.
        time_range: The time-range for each controller as a fraction of the number of simulation steps

    Hooks Into:
        - `init`
        - `step`

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
        price_weight=0.7,
        utility_threshold=0.9,
        time_threshold=0.9,
        time_horizon=0.1,
        min_price_margin=0.5,
        max_price_margin=0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.index: List[int] = None
        self.time_horizon = time_horizon
        self._time_threshold = time_threshold
        self._price_weight = price_weight
        self._utility_threshold = utility_threshold
        self._min_margin = 1 - min_price_margin
        self._max_margin = 1 + max_price_margin
        self.controllers: Dict[bool, SyncController] = {
            False: SyncController(
                is_seller=False,
                parent=self,
                price_weight=self._price_weight,
                time_threshold=self._time_threshold,
                utility_threshold=self._utility_threshold,
            ),
            True: SyncController(
                is_seller=True,
                parent=self,
                price_weight=self._price_weight,
                time_threshold=self._time_threshold,
                utility_threshold=self._utility_threshold,
            ),
        }
        self._current_end = -1
        self._current_start = -1

    def step(self):
        super().step()
        step = self.awi.current_step
        self._current_start = step + 1
        self._current_end = min(
            self.awi.n_steps - 1,
            self._current_start + max(1, int(self.time_horizon * self.awi.n_steps)),
        )
        if self._current_start >= self._current_end:
            return
        prices = (
            self.awi.catalog_prices
            if not self.awi.settings.get("public_trading_prices", False)
            else self.awi.trading_prices
        )
        for seller, needed, secured, product in [
            (False, self.inputs_needed, self.inputs_secured, self.awi.my_input_product),
            (
                True,
                self.outputs_needed,
                self.outputs_secured,
                self.awi.my_output_product,
            ),
        ]:
            needs = np.max(
                needed[self._current_start : self._current_end]
                - secured[self._current_start : self._current_end]
            )
            if needs < 1:
                continue
            if seller:
                price = prices[self.awi.my_input_product]
            else:
                price = prices[self.awi.my_output_product]
            price_range = (self._min_margin * price, self._max_margin * price)
            if price_range[0] >= price_range[1]:
                continue
            if (
                seller
                and price_range[-1]
                < prices[self.awi.my_input_product]
                + self.awi.profile.costs[self.awi.my_input_product].min()
            ):
                continue
            if (
                not seller
                and price_range[0]
                > prices[self.awi.my_output_product]
                - self.awi.profile.costs[self.awi.my_input_product].min()
            ):
                continue
            self.awi.request_negotiations(
                not seller,
                product,
                (1, needs),
                price_range,
                time=(self._current_start, self._current_end),
                controller=self.controllers[seller],
            )

    def respond_to_negotiation_request(
        self,
        initiator: str,
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
    ) -> Optional[Negotiator]:
        if not (
            issues[TIME].min_value < self._current_end
            or issues[TIME].max_value > self._current_start
        ):
            return None
        controller = self.controllers[not annotation["is_buy"]]
        if controller is None:
            return None
        return controller.create_negotiator()
