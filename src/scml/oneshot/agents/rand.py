import random
from itertools import chain, combinations

from negmas import MechanismState, ResponseType
from negmas.outcomes import Outcome
from negmas.sao import SAOResponse, SAOState

from scml.oneshot.agent import (
    OneShotAgent,
    OneShotSingleAgreementAgent,
    OneShotSyncAgent,
)
from scml.oneshot.common import QUANTITY, UNIT_PRICE

__all__ = [
    "RandomOneShotAgent",
    "SyncRandomOneShotAgent",
    "SingleAgreementRandomAgent",
]

PROB_ACCEPTANCE = 0.1
PROB_END = 0.005


def distribute(q: int, n: int) -> list[int]:
    """Distributes n values over m bins with at least one item per bin assuming q > n"""
    from collections import Counter

    from numpy.random import choice

    if q < n:
        lst = [0] * (n - q) + [1] * q
        random.shuffle(lst)
        return lst

    if q == n:
        return [1] * n
    r = Counter(choice(n, q - n))
    return [r.get(_, 0) + 1 for _ in range(n)]


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


class RandomOneShotAgent(OneShotAgent):
    """An agent that randomly leaves the negotiation, accepts or counters with random outcomes"""

    def __init__(
        self, owner=None, ufun=None, name=None, p_accept=PROB_ACCEPTANCE, p_end=PROB_END
    ):
        self.p_accept, self.p_end = p_accept, p_end
        super().__init__(owner, ufun, name)

    def _random_offer(self, negotiator_id: str):
        nmi = self.get_nmi(negotiator_id)
        if not nmi:
            return None  # will end the negotiation
        return nmi.random_outcome()

    def propose(self, negotiator_id, state) -> Outcome | None:
        return self._random_offer(negotiator_id)

    def respond(self, negotiator_id, state, source=None) -> ResponseType:
        if random.random() < self.p_end:
            return ResponseType.END_NEGOTIATION
        if random.random() < self.p_accept:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER


class SyncRandomOneShotAgent(OneShotSyncAgent):
    """An agent that distributes its needs over its partners randomly."""

    def distribute_needs(self) -> dict[str, int]:
        """Distributes my needs randomly over all my partners"""

        dist = dict()
        for needs, all_partners in [
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ]:
            # find suppliers and consumers still negotiating with me
            partner_ids = [_ for _ in all_partners if _ in self.negotiators.keys()]
            partners = len(partner_ids)

            # if I need nothing, end all negotiations
            if needs <= 0:
                dist.update(dict(zip(partner_ids, [0] * partners)))
                continue

            # distribute my needs over my (remaining) partners.
            dist.update(dict(zip(partner_ids, distribute(needs, partners))))
        return dist

    def first_proposals(self):
        # just randomly distribute my needs over my partners (with best price for me).
        s, p = self._step_and_price(best_price=True)
        distribution = self.distribute_needs()
        d = {k: (q, s, p) if q > 0 else None for k, q in distribution.items()}
        return d

    def counter_all(self, offers, states):
        response = dict()
        # process for sales and supplies independently
        for needs, all_partners, issues in [
            (
                self.awi.needed_supplies,
                self.awi.my_suppliers,
                self.awi.current_input_issues,
            ),
            (
                self.awi.needed_sales,
                self.awi.my_consumers,
                self.awi.current_output_issues,
            ),
        ]:
            # get a random price
            price = issues[UNIT_PRICE].rand()
            # find active partners
            partners = {_ for _ in all_partners if _ in offers.keys()}

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
            th = self._current_threshold(min(_.relative_time for _ in states.values()))
            if best_diff <= th:
                partner_ids = plist[best_indx]
                others = list(partners.difference(partner_ids))
                response |= {
                    k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                    for k in partner_ids
                } | {k: SAOResponse(ResponseType.END_NEGOTIATION, None) for k in others}
                continue

            # If I still do not have a good enough offer, distribute my current needs
            # randomly over my partners.
            distribution = self.distribute_needs()
            response.update(
                {
                    k: SAOResponse(ResponseType.END_NEGOTIATION, None)
                    if q == 0
                    else SAOResponse(
                        ResponseType.REJECT_OFFER, (q, self.awi.current_step, price)
                    )
                    for k, q in distribution.items()
                }
            )
        return response

    def _current_threshold(self, r: float):
        mn, mx = 0, self.awi.n_lines // 2
        return mn + (mx - mn) * (r**4.0)

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


class SingleAgreementRandomAgent(OneShotSingleAgreementAgent):
    """A controller that agrees randomly to one offer"""

    def __init__(self, *args, p_accept: float = PROB_ACCEPTANCE, **kwargs):
        super().__init__(*args, **kwargs)
        self._p_accept = p_accept

    def is_acceptable(self, offer: Outcome, source: str, state: SAOState) -> bool:
        return random.random() < self._p_accept

    def best_offer(self, offers: dict[str, Outcome]) -> str | None:
        return random.choice(list(offers.keys()))

    def is_better(
        self, a: Outcome | None, b: Outcome | None, negotiator: str, state: SAOState
    ) -> bool:
        return random.random() < 0.5
