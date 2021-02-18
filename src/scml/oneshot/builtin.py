from typing import Dict
import random
from negmas import MechanismState, SAOState, SAOResponse, ResponseType

from .agent import OneShotAgent, OneShotSyncAgent

__all__ = [
    "RandomOneShotAgent",
    "SyncRandomOneShotAgent",
]

PROB_ACCEPTANCE = 0.1
PROB_END = 0.005


class RandomOneShotAgent(OneShotAgent):
    def _random_offer(self, negotiator_id: str):
        return self.negotiators[negotiator_id][0].ami.random_outcomes(1)[0]

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
        return self.negotiators[negotiator_id][0].ami.random_outcomes(1)[0]

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
