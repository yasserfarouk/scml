from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from negmas import AgentMechanismInterface
from negmas import Contract
from negmas import Entity
from negmas import MechanismState
from negmas import Outcome
from negmas import PassThroughSAONegotiator
from negmas import ResponseType
from negmas import SAOController
from negmas import SAOResponse
from negmas import SAOState
from negmas import SAOSyncController, SAOSingleAgreementController
from .ufun import OneShotUFun

__all__ = ["OneShotAgent", "OneShotSyncAgent", "OneShotSingleAgreementAgent"]


class OneShotAgent(SAOController, Entity, ABC):
    """Base class for all agents in the One-Shot game."""

    def __init__(self, owner=None, ufun=None, name=None):
        super().__init__(
            default_negotiator_type=PassThroughSAONegotiator,
            default_negotiator_params=None,
            auto_kill=False,
            name=name,
            ufun=ufun,
        )
        self._awi = owner._awi if owner else None

    @property
    def awi(self):
        return self._awi

    def init(self):
        pass

    def make_ufun(self, add_exogenous=False):
        self.ufun = OneShotUFun(
            owner=self,
            qin=self.awi.current_exogenous_input_quantity if add_exogenous else 0,
            pin=self.awi.current_exogenous_input_price if add_exogenous else 0,
            qout=self.awi.current_exogenous_output_quantity if add_exogenous else 0,
            pout=self.awi.current_exogenous_output_price if add_exogenous else 0,
            production_cost=self.awi.profile.cost,
            storage_cost=self.awi.current_storage_cost,
            delivery_penalty=self.awi.current_delivery_penalty,
            input_agent=self.awi.my_input_product == 0,
            output_agent=self.awi.my_output_product == self.awi.n_products - 1,
        )
        return self.ufun

    def step(self):
        pass

    def connect_to_oneshot_adapter(self, owner, ufun):
        """Connects the agent to its adapter (used internally)"""
        self._awi = owner._awi
        self.utility_function = ufun

    def connect_to_2021_adapter(self, owner, ufun):
        """Connects the agent to its adapter (used internally)"""
        self._awi = owner.awi
        self.utility_function = ufun

    @abstractmethod
    def propose(self, negotiator_id: str, state: MechanismState) -> "Outcome":
        """
        Proposes an offer to one of the partners.

        Args:
            negotiator_id: ID of the negotiator (and partner)
            state: Mechanism state including current step

        Returns:
            an outcome to offer.
        """

    def respond(
        self, negotiator_id: str, state: MechanismState, offer: "Outcome"
    ) -> "ResponseType":
        """
        Responds to an offer from one of the partners.

        Args:
            negotiator_id: ID of the negotiator (and partner)
            state: Mechanism state including current step
            offer: The offer received.

        Returns:
            A response type which can either be reject, accept, or end negotiation.

        Remarks:
            default behavior is to accept only if the current offer is the same
            or has a higher utility compared with what the agent would have
            proposed in the given state and reject otherwise

        """
        myoffer = self.propose(negotiator_id, state)
        if myoffer == offer:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    @property
    def internal_state(self) -> Dict[str, Any]:
        """Returns the internal state of the agent for debugging purposes"""
        return {}

    def on_negotiation_failure(
        self,
        partners: List[str],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
        state: MechanismState,
    ) -> None:
        """Called whenever a negotiation ends without agreement"""

    def on_negotiation_success(
        self, contract: Contract, mechanism: AgentMechanismInterface
    ) -> None:
        """Called whenever a negotiation ends with agreement"""

    def sign_all_contracts(self, contracts: List[Contract]) -> List[Optional[str]]:
        """Signs all contracts"""
        return [self.id] * len(contracts)


class OneShotSyncAgent(SAOSyncController, OneShotAgent, ABC):
    def __init__(self, *args, **kwargs):
        kwargs["global_ufun"] = True
        super().__init__(*args, **kwargs)

    @abstractmethod
    def counter_all(
        self, offers: Dict[str, "Outcome"], states: Dict[str, SAOState]
    ) -> Dict[str, SAOResponse]:
        """Calculate a response to all offers from all negotiators
        (negotiator ID is the key).

        Args:
            offers: Maps negotiator IDs to offers
            states: Maps negotiator IDs to offers AT the time the offers were made.

        Remarks:
            - The response type CANNOT be WAIT.
            - If the system determines that a loop is formed, the agent may
            receive this call for a subset of negotiations not all of them.

        """

    @abstractmethod
    def first_proposals(self) -> Dict[str, "Outcome"]:
        """Gets a set of proposals to use for initializing the negotiation."""
        return super().first_proposals()

    def sign_all_contracts(self, contracts: List[Contract]) -> List[Optional[str]]:
        """Signs all contracts"""
        return [self.id] * len(contracts)


class OneShotSingleAgreementAgent(SAOSingleAgreementController, OneShotSyncAgent):
    """
    A synchronized controller that tries to get no more than one agreement.

    This controller manages a set of negotiations from which only a single one
    -- at most -- is likely to result in an agreement. An example of a case in which
    it is useful is an agent negotiating to buy a car from multiple suppliers.
    It needs a single car at most but it wants the best one from all of those
    suppliers. To guarentee a single agreement, pass strict=True

    The general algorithm for this controller is something like this:

        - Receive offers from all partners.
        - Find the best offer among them by calling the abstract `best_offer`
          method.
        - Check if this best offer is acceptable using the abstract `is_acceptable`
          method.

            - If the best offer is acceptable, accept it and end all other negotiations.
            - If the best offer is still not acceptable, then all offers are rejected
              and with the partner who sent it receiving the result of `best_outcome`
              while the rest of the partners receive the result of `make_outcome`.

        - The default behavior of `best_outcome` is to return the outcome with
          maximum utility.
        - The default behavior of `make_outcome` is to return the best offer
          received in this round if it is valid for the respective negotiation
          and the result of `best_outcome` otherwise.

    Args:
        strict: If True the controller is **guaranteed** to get a single
                agreement but it will have to send no-response repeatedly so
                there is a higher chance of never getting an agreement when
                two of those controllers negotiate with each other
    """

    def __init__(self, *args, strict: bool = False, **kwargs):
        super().__init__(*args, strict=strict, **kwargs)

    @abstractmethod
    def is_acceptable(self, offer: "Outcome", source: str, state: SAOState) -> bool:
        """Should decide if the given offer is acceptable

        Args:
            offer: The offer being tested
            source: The ID of the negotiator that received this offer
            state: The state of the negotiation handled by that negotiator

        Remarks:
            - If True is returned, this offer will be accepted and all other
              negotiations will be ended.
        """

    @abstractmethod
    def best_offer(self, offers: Dict[str, "Outcome"]) -> Optional[str]:
        """
        Return the ID of the negotiator with the best offer

        Args:
            offers: A mapping from negotiator ID to the offer it received

        Returns:
            The ID of the negotiator with best offer. Ties should be broken.
            Return None only if there is no way to calculate the best offer.
        """

    @abstractmethod
    def is_better(self, a: "Outcome", b: "Outcome", negotiator: str, state: SAOState):
        """Compares two outcomes of the same negotiation

        Args:
            a: "Outcome"
            b: "Outcome"
            negotiator: The negotiator for which the comparison is to be made
            state: Current state of the negotiation

        Returns:
            True if utility(a) > utility(b)
        """
