from abc import ABC, abstractmethod
from typing import List, Dict, Any
from negmas import (
    AgentMechanismInterface,
    ResponseType,
    MechanismState,
    PassThroughSAONegotiator,
    SAOController,
    SAOSyncController,
    SAOState,
    SAOResponse,
    Outcome,
    Contract,
    Entity,
)


__all__ = ["OneShotAgent", "OneShotSyncAgent"]


class OneShotAgent(SAOController, Entity, ABC):
    """Base class for all agents in the One-Shot game."""

    def __init__(self, owner, ufun, name=None):
        super().__init__(
            default_negotiator_type=PassThroughSAONegotiator,
            default_negotiator_params=None,
            auto_kill=True,
            name=name,
            ufun=ufun,
        )
        self.awi = owner.awi

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

    def init(self):
        pass

    def step(self):
        pass


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
