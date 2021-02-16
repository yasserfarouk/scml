from typing import Dict

from negmas import MechanismState, SAOState, SAOResponse, ResponseType

from .agent import OneShotAgent, OneShotSyncAgent

__all__ = [
    "RandomOneShotAgent",
    "SyncRandomOneShotAgent",
]


class RandomOneShotAgent(OneShotAgent):
    def _random_offer(self, negotiator_id: str):
        return self.negotiators[negotiator_id][0].ami.random_outcomes(1)[0]

    def propose(self, negotiator_id: str, state: MechanismState) -> "Outcome":
        return self._random_offer(negotiator_id)


class SyncRandomOneShotAgent(OneShotSyncAgent):
    def _random_offer(self, negotiator_id: str):
        return self.negotiators[negotiator_id][0].ami.random_outcomes(1)[0]

    def counter_all(
        self, offers: Dict[str, "Outcome"], states: Dict[str, SAOState]
    ) -> Dict[str, SAOResponse]:
        proposals = dict()
        for id in self.negotiators.keys():
            proposals[id] = SAOResponse(
                ResponseType.REJECT_OFFER, self._random_offer(id)
            )
        return proposals

    def first_proposals(self) -> Dict[str, "Outcome"]:
        proposals = dict()
        for id, (neg, cntxt) in self.negotiators.items():
            proposals[id] = self._random_offer(id)
        return proposals
