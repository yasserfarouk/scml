import random
from itertools import chain, combinations

from negmas import Outcome, ResponseType
from negmas.negotiators.modular import itertools
from negmas.sao import SAOResponse, SAOState

from scml.oneshot.agents import SyncRandomOneShotAgent
from scml.oneshot.common import TIME, UNIT_PRICE
from scml.std.agent import StdAgent, StdSyncAgent

__all__ = ["SyncRandomStdAgent", "SyncRandomOneShotAgent", "RandomStdAgent"]

PROB_ACCEPTANCE = 0.05
PROB_END = 0.005


class RandomStdAgent(StdAgent):
    def __init__(
        self, owner=None, ufun=None, name=None, p_accept=PROB_ACCEPTANCE, p_end=PROB_END
    ):
        self.p_accept, self.p_end = p_accept, p_end
        super().__init__(owner, ufun, name)

    def propose(self, negotiator_id: str, state: SAOState) -> Outcome | None:  # type: ignore
        nmi = self.get_nmi(negotiator_id)
        if not nmi:
            return None
        return nmi.random_outcome()

    def respond(self, negotiator_id, state, source=None):
        if random.random() < self.p_end:
            return ResponseType.END_NEGOTIATION
        if random.random() < self.p_accept:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER


def distribute(q: int, n: int) -> list[int]:
    """Distributes n values over m bins with at least one item per bin assuming q > n"""
    from collections import Counter

    from numpy.random import choice

    if n <= 0:
        return []
    if q <= 0:
        return [0] * n
    if q == n:
        return [1] * n
    r = Counter(choice(n, q - n))
    return [r.get(_, 0) + 1 for _ in range(n)]


class SyncRandomStdAgent(StdSyncAgent):
    """An agent that distributes its needs over its partners randomly."""

    def __init__(self, *args, threshold=1, paccept=0.05, pfuture=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self._threshold = threshold
        self._paccept = paccept
        self._ptoday = 1.0 - pfuture

    def best_price(self, partner_id):
        issue = self.get_nmi(partner_id).issues[UNIT_PRICE]
        return issue.max_value if self.is_consumer(partner_id) else issue.min_value

    def rand_price(self, partner_id):
        return self.get_nmi(partner_id).issues[UNIT_PRICE].rand()

    def first_proposals(self):  # type: ignore
        # just randomly distribute my needs over my partners (with best price for me).
        # remaining partners get random future offers
        distribution = self.distribute_todays_needs()
        offers = {
            k: (q, self.awi.current_step, self.best_price(k))
            if q > 0
            else self.sample_future_offer(k)
            for k, q in distribution.items()
        }
        return offers

    def counter_all(self, offers, states):
        # start by assigning future random offers to everyone
        distribution = self.distribute_todays_needs(
            list(offers.keys()) if offers else None
        )
        myoffers = {
            k: (q, self.awi.current_step, self.rand_price(k))
            if q > 0
            else self.sample_future_offer(k)
            for k, q in distribution.items()
        }
        response = dict(
            zip(
                offers.keys(),
                (
                    SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                    if random.random() < self._paccept
                    else SAOResponse(ResponseType.REJECT_OFFER, myoffers[k])
                    for k in offers.keys()
                ),
            )
        )
        return response

    def distribute_todays_needs(self, partners=None) -> dict[str, int]:
        """Distributes my needs randomly over all my partners"""
        if partners is None:
            partners = self.negotiators.keys()

        response = dict(zip(partners, itertools.repeat(0)))
        for is_partner, edge_needs in (
            (self.is_supplier, self.awi.needed_supplies),
            (self.is_consumer, self.awi.needed_sales),
        ):
            needs = self.awi.n_lines if self.awi.is_middle_level else edge_needs
            # find my partners and the quantity I need
            active_partners = [_ for _ in partners if is_partner(_)]
            if not active_partners or needs < 1:
                continue
            random.shuffle(active_partners)
            active_partners = active_partners[
                : max(1, int(self._ptoday * len(active_partners)))
            ]
            n_partners = len(active_partners)

            # if I need nothing, end all negotiations
            if needs <= 0 or n_partners <= 0:
                continue

            # If my needs are small, use a subset of negotiators
            if needs < n_partners:
                active_partners = random.sample(
                    active_partners, random.randint(1, needs)
                )
                n_partners = len(active_partners)

            # distribute my needs over my (remaining) partners.
            response |= dict(zip(active_partners, distribute(needs, n_partners)))

        return response

    def sample_future_offer(self, partner_id) -> Outcome | None:
        nmi = self.get_nmi(partner_id)
        outcome = nmi.random_outcome()
        if outcome[TIME] == self.awi.current_step:
            outcome = list(outcome)
            tissue = nmi.issues[TIME]
            if tissue.cardinality == 1:
                return None
            outcome[TIME] = random.randint(self.awi.current_step + 1, tissue.max_value)
            return tuple(outcome)
        return outcome

    def is_supplier(self, negotiator_id):
        return negotiator_id in self.awi.my_suppliers

    def is_consumer(self, negotiator_id):
        return negotiator_id in self.awi.my_consumers
