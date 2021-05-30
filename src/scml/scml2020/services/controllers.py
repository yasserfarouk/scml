import random
from collections import defaultdict
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import numpy as np
from negmas import AgentWorldInterface
from negmas import AspirationMixin
from negmas import Issue
from negmas import LinearUtilityFunction
from negmas import MechanismState
from negmas import Outcome
from negmas import PassThroughNegotiator
from negmas import ResponseType
from negmas import UtilityFunction
from negmas import outcome_is_valid
from negmas.common import AgentMechanismInterface
from negmas.events import Notification
from negmas.events import Notifier
from negmas.helpers import instantiate
from negmas.sao import SAOController
from negmas.sao import SAONegotiator
from negmas.sao import SAOResponse
from negmas.sao import SAOState
from negmas.sao import SAOSyncController

from scml.scml2020.common import QUANTITY
from scml.scml2020.common import TIME
from scml.scml2020.common import UNIT_PRICE

__all__ = ["StepController", "SyncController"]


class StepController(SAOController, AspirationMixin, Notifier):
    """A controller for managing a set of negotiations about selling or buying (but not both)  starting/ending at some
    specific time-step.

    Args:

        target_quantity: The quantity to be secured
        is_seller:  Is this a seller or a buyer
        parent_name: Name of the parent
        horizon: How many steps in the future to allow negotiations for selling to go for.
        step:  The simulation step that this controller is responsible about
        urange: The range of unit prices used for negotiation
        product: The product that this controller negotiates about
        partners: A list of partners to negotiate with
        negotiator_type: The type of the single negotiator used for all negotiations.
        negotiator_params: The parameters of the negotiator used for all negotiations
        max_retries: How many times can the controller try negotiating with each partner.
        negotiations_concluded_callback: A method to be called with the step of this controller and whether is it a
                                         seller when all negotiations are concluded
        *args: Position arguments passed to the base Controller constructor
        **kwargs: Keyword arguments passed to the base Controller constructor


    Remarks:

        - It uses whatever negotiator type on all of its negotiations and it assumes that the ufun will never change
        - Once it accumulates the required quantity, it ends all remaining negotiations
        - It assumes that all ufuns are identical so there is no need to keep a separate negotiator for each one and it
          instantiates a single negotiator that dynamically changes the AMI but always uses the same ufun.

    """

    def __init__(
        self,
        *args,
        target_quantity: int,
        is_seller: bool,
        step: int,
        urange: Tuple[int, int],
        product: int,
        partners: List[str],
        negotiator_type: SAONegotiator,
        horizon: int,
        awi: AgentWorldInterface,
        parent_name: str,
        negotiations_concluded_callback: Callable[[int, bool], None],
        negotiator_params: Dict[str, Any] = None,
        max_retries: int = 2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.parent_name = parent_name
        self.awi = awi
        self.horizon = horizon
        self.negotiations_concluded_callback = negotiations_concluded_callback
        self.is_seller = is_seller
        self.target = target_quantity
        self.urange = urange
        self.partners = partners
        self.product = product
        negotiator_params = (
            negotiator_params if negotiator_params is not None else dict()
        )
        self.secured = 0
        if is_seller:
            self.ufun = LinearUtilityFunction((1, 1, 10))
        else:
            self.ufun = LinearUtilityFunction((1, -1, -10))
        negotiator_params["ufun"] = self.ufun
        self.__negotiator = instantiate(negotiator_type, **negotiator_params)
        self.completed = defaultdict(bool)
        self.step = step
        self.retries: Dict[str, int] = defaultdict(int)
        self.max_retries = max_retries

    def join(
        self,
        negotiator_id: str,
        ami: AgentMechanismInterface,
        state: MechanismState,
        *,
        ufun: Optional["UtilityFunction"] = None,
        role: str = "agent",
    ) -> bool:
        joined = super().join(negotiator_id, ami, state, ufun=ufun, role=role)
        if joined:
            self.completed[negotiator_id] = False
        return joined

    def propose(self, negotiator_id: str, state: MechanismState) -> Optional["Outcome"]:
        if negotiator_id not in self.negotiators.keys():
            return None
        self.__negotiator._ami = self.negotiators[negotiator_id][0]._ami
        return self.__negotiator.propose(state)

    def respond(
        self, negotiator_id: str, state: MechanismState, offer: "Outcome"
    ) -> ResponseType:
        if negotiator_id not in self.negotiators.keys():
            return ResponseType.END_NEGOTIATION
        if self.secured >= self.target:
            return ResponseType.END_NEGOTIATION
        # if negotiator_id not in self.negotiators:
        #     breakpoint()
        self.__negotiator._ami = self.negotiators[negotiator_id][0]._ami
        return self.__negotiator.respond(offer=offer, state=state)

    def __str__(self):
        return (
            f"{'selling' if self.is_seller else 'buying'} p{self.product} [{self.step}] "
            f"secured {self.secured} of {self.target} for {self.parent_name} "
            f"({len([_ for _ in self.completed.values() if _])} completed of {len(self.completed)} negotiators)"
        )

    def create_negotiator(
        self,
        negotiator_type: Union[str, Type[PassThroughNegotiator]] = None,
        name: str = None,
        cntxt: Any = None,
        **kwargs,
    ) -> PassThroughNegotiator:
        neg = super().create_negotiator(negotiator_type, name, cntxt, **kwargs)
        self.completed[neg.id] = False
        return neg

    def time_range(self, step, is_seller):
        if is_seller:
            return (
                max(step, self.awi.current_step + 1),
                min(step + self.horizon, self.awi.n_steps - 1),
            )
        return self.awi.current_step + 1, step - 1

    def on_negotiation_end(self, negotiator_id: str, state: MechanismState) -> None:
        super().on_negotiation_end(negotiator_id, state)
        agreement = state.agreement
        # mark this negotiation as completed
        self.completed[negotiator_id] = True
        # if there is an agreement increase the secured amount and check if we are done.
        if agreement is not None:
            self.secured += agreement[QUANTITY]
            if self.secured >= self.target:
                self.awi.loginfo(f"Ending all negotiations on controller {str(self)}")
                # If we are done, end all other negotiations
                for k in self.negotiators.keys():
                    if self.completed[k]:
                        continue
                    self.notify(
                        self.negotiators[k][0], Notification("end_negotiation", None)
                    )
        self.kill_negotiator(negotiator_id, force=True)
        if all(self.completed.values()):
            # If we secured everything, just return control to the agent
            if self.secured >= self.target:
                self.awi.loginfo(f"Secured Everything: {str(self)}")
                self.negotiations_concluded_callback(self.step, self.is_seller)
                return
            # If we did not secure everything we need yet and time allows it, create new negotiations
            tmin, tmax = self.time_range(self.step, self.is_seller)

            if self.awi.current_step < tmax + 1 and tmin <= tmax:
                # get a good partner: one that was not retired too much
                random.shuffle(self.partners)
                for other in self.partners:
                    if self.retries[other] <= self.max_retries:
                        partner = other
                        break
                else:
                    return
                self.retries[partner] += 1
                neg = self.create_negotiator()
                self.completed[neg.id] = False
                self.awi.loginfo(
                    f"{str(self)} negotiating with {partner} on u={self.urange}"
                    f", q=(1,{self.target-self.secured}), u=({tmin}, {tmax})"
                )
                self.awi.request_negotiation(
                    not self.is_seller,
                    product=self.product,
                    quantity=(1, self.target - self.secured),
                    unit_price=self.urange,
                    time=(tmin, tmax),
                    partner=partner,
                    negotiator=neg,
                    extra=dict(controller_index=self.step, is_seller=self.is_seller),
                )


# our controller
class SyncController(SAOSyncController):
    """
    Will try to get the best deal which is defined as being nearest to the agent needs and with lowest price
    """

    def __init__(
        self,
        *args,
        is_seller: bool,
        parent: "PredictionBasedTradingStrategy",
        price_weight=0.7,
        utility_threshold=0.9,
        time_threshold=0.9,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._is_seller = is_seller
        self.__parent = parent
        self._time_threshold = time_threshold
        self._price_weight = price_weight
        self._utility_threshold = utility_threshold
        self._best_utils: Dict[str, float] = {}
        # find out my needs and the amount secured lists

    def utility(self, offer: Tuple[int, int, int], max_price: int) -> float:
        """A simple utility function

        Remarks:
             - If the time is invalid or there is no need to get any more agreements
               at the given time, return -1000
             - Otherwise use the price-weight to calculate a linear combination of
               the price and the how much of the needs is satisfied by this contract

        """
        if self._is_seller:
            _needed, _secured = (
                self.__parent.outputs_needed,
                self.__parent.outputs_secured,
            )
        else:
            _needed, _secured = (
                self.__parent.inputs_needed,
                self.__parent.inputs_secured,
            )
        if offer is None:
            return -1000
        t = offer[TIME]
        if t < self.__parent.awi.current_step or t > self.__parent.awi.n_steps - 1:
            return -1000.0
        q = _needed[offer[TIME]] - (offer[QUANTITY] + _secured[t])
        if q < 0:
            return -1000.0
        if self._is_seller:
            price = offer[UNIT_PRICE]
        else:
            price = max_price - offer[UNIT_PRICE]
        return self._price_weight * price + (1 - self._price_weight) * q

    def is_valid(self, negotiator_id: str, offer: "Outcome") -> bool:
        issues = self.negotiators[negotiator_id][0].ami.issues
        return outcome_is_valid(offer, issues)

    def counter_all(
        self, offers: Dict[str, "Outcome"], states: Dict[str, SAOState]
    ) -> Dict[str, SAOResponse]:
        """Calculate a response to all offers from all negotiators (negotiator ID is the key).

        Args:
            offers: Maps negotiator IDs to offers
            states: Maps negotiator IDs to offers AT the time the offers were made.

        Remarks:
            - The response type CANNOT be WAIT.

        """

        # find the best offer
        negotiator_ids = list(offers.keys())
        utils = np.array(
            [
                self.utility(
                    tuple(o), self.negotiators[nid][0].ami.issues[UNIT_PRICE].max_value
                )
                for nid, o in offers.items()
            ]
        )

        best_index = int(np.argmax(utils))
        best_utility = utils[best_index]
        best_partner = negotiator_ids[best_index]
        best_offer = offers[best_partner]

        # find my best proposal for each negotiation
        best_proposals = self.first_proposals()

        # if the best offer is still so bad just reject everything
        if best_utility < 0:
            return {
                k: SAOResponse(ResponseType.REJECT_OFFER, best_proposals[k])
                for k in offers.keys()
                if k in self.negotiators.keys()
            }

        if best_partner not in self._best_utils.keys():
            breakpoint()
        relative_time = min(_.relative_time for _ in states.values())

        # if this is good enough or the negotiation is about to end accept the best offer
        if (
            best_utility >= self._utility_threshold * self._best_utils[best_partner]
            or relative_time > self._time_threshold
        ):
            responses = {
                k: SAOResponse(
                    ResponseType.REJECT_OFFER,
                    best_offer if self.is_valid(k, best_offer) else best_proposals[k],
                )
                for k in offers.keys()
            }
            responses[best_partner] = SAOResponse(ResponseType.ACCEPT_OFFER, None)
            return responses

        # send the best offer to everyone else and try to improve it
        responses = {
            k: SAOResponse(
                ResponseType.REJECT_OFFER,
                best_offer if self.is_valid(k, best_offer) else best_proposals[k],
            )
            for k in offers.keys()
        }
        responses[best_partner] = SAOResponse(
            ResponseType.REJECT_OFFER, best_proposals[best_partner]
        )
        return responses

#     def on_negotiation_end(self, negotiator_id: str, state: MechanismState) -> None:
#         """Update the secured quantities whenever a negotiation ends"""
#         if state.agreement is None:
#             return
# 
#         q, t = state.agreement[QUANTITY], state.agreement[TIME]
#         if self._is_seller:
#             self.__parent.outputs_secured[t] += q
#         else:
#             self.__parent.inputs_secured[t] += q
# 
    def best_proposal(self, nid: str) -> Tuple[Optional[Outcome], float]:
        """
        Finds the best proposal for the given negotiation

        Args:
            nid: Negotiator ID

        Returns:
            The outcome with highest utility and the corresponding utility
        """
        negotiator = self.negotiators[nid][0]
        if negotiator.ami is None:
            return None, -1000
        utils = np.array(
            [
                self.utility(_, negotiator.ami.issues[UNIT_PRICE].max_value)
                for _ in negotiator.ami.outcomes
            ]
        )
        best_indx = np.argmax(utils)
        self._best_utils[nid] = utils[best_indx]
        if utils[best_indx] < 0:
            return None, utils[best_indx]
        return negotiator.ami.outcomes[best_indx], utils[best_indx]

    def first_proposals(self) -> Dict[str, "Outcome"]:
        """Gets a set of proposals to use for initializing the negotiation."""
        return {nid: self.best_proposal(nid)[0] for nid in self.negotiators.keys()}
