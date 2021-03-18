import random
from typing import Dict, Optional

from negmas import MechanismState
from negmas import ResponseType
from negmas.sao import SAOResponse
from negmas.sao import SAOState
from negmas.sao import SAOSingleAgreementRandomController, SAOSingleAgreementController
from negmas.outcomes import Outcome

from scml.oneshot.agent import OneShotAgent, OneShotSingleAgreementAgent
from scml.oneshot.agent import OneShotSyncAgent

__all__ = ["RandomOneShotAgent", "SyncRandomOneShotAgent", "SingleAgreementRandomAgent"]

PROB_ACCEPTANCE = 0.1
PROB_END = 0.005


class RandomOneShotAgent(OneShotAgent):
    def _random_offer(self, negotiator_id: str):
        ami = self.get_ami(negotiator_id)
        if not ami:
            return None
        return ami.random_outcomes(1)[0]

    def propose(self, negotiator_id: str, state: MechanismState) -> "Outcome":
        return self._random_offer(negotiator_id)

    def respond(self, negotiator_id, state, offer):
        if random.random() < PROB_END:
            return ResponseType.END_NEGOTIATION
        if random.random() < PROB_ACCEPTANCE:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER


class SyncRandomOneShotAgent(OneShotSyncAgent):
    def _random_offer(self, negotiator_id: str):
        ami = self.get_ami(negotiator_id)
        if not ami:
            return None
        return ami.random_outcomes(1)[0]

    def counter_all(
        self, offers: Dict[str, "Outcome"], states: Dict[str, SAOState]
    ) -> Dict[str, SAOResponse]:
        proposals = dict()
        for id in self.negotiators.keys():

            proposals[id] = (
                SAOResponse(ResponseType.ACCEPT_OFFER, None)
                if random.random() < PROB_ACCEPTANCE
                else SAOResponse(ResponseType.REJECT_OFFER, self._random_offer(id))
            )
        return proposals

    def first_proposals(self) -> Dict[str, "Outcome"]:
        proposals = dict()
        for id, (neg, cntxt) in self.negotiators.items():
            proposals[id] = self._random_offer(id)
        return proposals


class SingleAgreementRandomAgent(OneShotSingleAgreementAgent):
    """A controller that agrees randomly to one offer"""

    def __init__(self, *args, p_accept: float = PROB_ACCEPTANCE, **kwargs):
        super().__init__(*args, **kwargs)
        self._p_accept = p_accept

    def is_acceptable(self, offer: "Outcome", source: str, state: SAOState) -> bool:
        return random.random() < self._p_accept

    def best_offer(self, offers: Dict[str, "Outcome"]) -> Optional[str]:
        return random.choice(list(offers.keys()))

    def is_better(self, a: "Outcome", b: "Outcome", negotiator: str, state: SAOState):
        return random.random() < 0.5
