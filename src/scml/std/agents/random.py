import random
from itertools import chain, combinations

from negmas import ResponseType
from negmas.sao import SAOResponse

from scml.oneshot.agents import SyncRandomOneShotAgent
from scml.oneshot.common import QUANTITY, UNIT_PRICE
from scml.std.agent import OneShotSyncAgent

__all__ = ["SyncRandomStdAgent", "SyncRandomOneShotAgent"]

PROB_ACCEPTANCE = 0.1
PROB_END = 0.005


def distribute(q: int, n: int) -> list[int]:
    """Distributes n values over m bins with at least one item per bin assuming q > n"""
    from collections import Counter

    from numpy.random import choice

    if q == n:
        return [1] * n
    r = Counter(choice(n, q - n))
    return [r.get(_, 0) + 1 for _ in range(n)]


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


class SyncRandomStdAgent(OneShotSyncAgent):
    """An agent that distributes its needs over its partners randomly."""

    def __init__(self, *args, threshold=1, **kwargs):
        super().__init__(*args, **kwargs)
        self._threshold = threshold

    def distribute_needs(self) -> dict[str, int]:
        """Distributes my needs randomly over all my partners"""

        # find my partners and the quantity I need
        partner_ids = list(self.negotiators.keys())
        partners = len(partner_ids)
        needs = self._needs()

        # if I need nothing, end all negotiations
        if needs <= 0:
            return dict(zip(partner_ids, [0] * partners))

        # If my needs are small, end some of the negotiations
        response = dict()
        if needs < partners:
            to_end = random.sample(partner_ids, (partners - needs))
            response = dict(zip(to_end, [0] * len(to_end)))
            partner_ids = [_ for _ in partner_ids if _ not in to_end]
            partners = len(partner_ids)

        # distribute my needs over my (remaining) partners.
        response.update(dict(zip(partner_ids, distribute(needs, partners))))
        return response

    def first_proposals(self):
        # just randomly distribute my needs over my partners (with best price for me).
        s, p = self._step_and_price(best_price=True)
        distribution = self.distribute_needs()
        return {k: (q, s, p) if q > 0 else None for k, q in distribution.items()}

    def counter_all(self, offers, states):
        _ = states
        # get current step, some valid price, the quantity I need, and my partners
        s, p = self._step_and_price()
        needs = self._needs()
        partners = set(offers.keys())

        # find the set of partners that gave me the best offer set
        # (i.e. total quantity nearest to my needs)
        plist = list(powerset(partners))
        best_diff, best_indx = float("inf"), -1
        for i, partner_ids in enumerate(plist):
            others = partners.difference(partner_ids)
            offered = sum(offers[p][QUANTITY] for p in partner_ids)
            diff = abs(offered - needs)
            if diff < best_diff:
                best_diff, best_indx = diff, i
            if diff == 0:
                break

        # If the best combination of offers is good enough, accept them and end all
        # other negotiations
        if best_diff <= self._threshold:
            partner_ids = plist[best_indx]
            others = list(partners.difference(partner_ids))
            return {
                k: SAOResponse(ResponseType.ACCEPT_OFFER, None) for k in partner_ids
            } | {k: SAOResponse(ResponseType.END_NEGOTIATION, None) for k in others}

        # If I still do not have a good enough offer, distribute my current needs
        # randomly over my partners.
        distribution = self.distribute_needs()
        return {
            k: SAOResponse(ResponseType.END_NEGOTIATION, None)
            if q == 0
            else SAOResponse(ResponseType.REJECT_OFFER, (q, s, p))
            for k, q in distribution.items()
        }

    def _needs(self):
        """How many items do I need?"""
        return self.awi.needed_sales + self.awi.needed_supplies

    def _step_and_price(self, best_price=False):
        """Returns current step and a random (or max) price"""
        s = self.awi.current_step
        seller = self.awi.is_first_level
        issues = (
            self.awi.current_output_issues if seller else self.awi.current_input_issues
        )
        pmin = issues[UNIT_PRICE].min_value
        pmax = issues[UNIT_PRICE].max_value
        if best_price:
            return s, pmax if seller else pmin
        return s, random.randint(pmin, pmax)
