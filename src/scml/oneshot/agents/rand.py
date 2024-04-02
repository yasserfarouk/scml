import random
from itertools import chain, combinations

from negmas import ResponseType
from negmas.outcomes import Outcome
from negmas.sao import SAOResponse, SAOState

from scml.common import distribute
from scml.oneshot.agent import (
    OneShotAgent,
    OneShotSingleAgreementAgent,
    OneShotSyncAgent,
)
from scml.oneshot.common import QUANTITY, TIME, UNIT_PRICE

__all__ = [
    "RandomOneShotAgent",
    "RandDistOneShotAgent",
    "EqualDistOneShotAgent",
    "SyncRandomOneShotAgent",
    "SingleAgreementRandomAgent",
]

PROB_ACCEPTANCE = 0.1
PROB_END = 0.005


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


class RandomOneShotAgent(OneShotAgent):
    """An agent that randomly leaves the negotiation, accepts or counters with random outcomes"""

    def __init__(
        self,
        *args,
        p_accept=PROB_ACCEPTANCE,
        p_end=PROB_END,
        **kwargs,
    ):
        self.p_accept, self.p_end = p_accept + p_end, p_end
        super().__init__(*args, **kwargs)

    def _random_offer(self, negotiator_id: str):
        nmi = self.get_nmi(negotiator_id)
        if not nmi:
            return None  # will end the negotiation
        return nmi.random_outcome()

    def propose(self, negotiator_id, state) -> Outcome | None:
        return self._random_offer(negotiator_id)

    def respond(self, negotiator_id, state, source=None) -> ResponseType:
        r = random.random()
        if r < self.p_end:
            return ResponseType.END_NEGOTIATION
        if r < self.p_accept:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER


class NiceAgent(RandomOneShotAgent):
    """An agent that offers randomly and accepts anything"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, p_accept=0, p_end=0, **kwargs)

    def respond(self, negotiator_id, state, source=None) -> ResponseType:
        return ResponseType.ACCEPT_OFFER


class SyncRandomOneShotAgent(OneShotSyncAgent):
    """
    An agent that distributes its needs over its partners randomly.

    Args:
        equal: If given, it tries to equally distribute its needs over as many of its
               suppliers/consumers as possible
        overordering_max: Maximum fraction of needs to over-order. For example, it the
                          agent needs 5 items and this is 0.2, it will order 6 in the first
                          negotiation step.
        overordering_min: Minimum fraction of needs to over-order. Used in the last negotiation
                          step.
        overordering_exp: Controls how fast does the over-ordering quantity go from max to min.
        concession_exp: Controls how fast does the agent concedes on matching its needs exactly.
        mismatch_max: Maximum mismtach in quantity allowed between needs and accepted offers. If
                      a fraction, it is will be this fraction of the production capacity (n_lines).
    """

    def __init__(
        self,
        *args,
        equal: bool = False,
        overordering_max: float = 0.2,
        overordering_min: float = 0.0,
        overordering_exp: float = 0.4,
        mismatch_exp: float = 4.0,
        mismatch_max: float = 0.3,
        **kwargs,
    ):
        self.equal_distribution = equal
        self.overordering_max = overordering_max
        self.overordering_min = overordering_min
        self.overordering_exp = overordering_exp
        self.mismatch_exp = mismatch_exp
        self.mismatch_max = mismatch_max
        super().__init__(*args, **kwargs)

    def init(self):
        if 0 < self.mismatch_max < 1:
            self.mismatch_max *= self.awi.n_lines
        return super().init()

    def distribute_needs(self, t: float) -> dict[str, int]:
        """Distributes my needs randomly over all my partners"""

        dist = dict()
        for needs, all_partners in [
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ]:
            # find suppliers and consumers still negotiating with me
            partners = [_ for _ in all_partners if _ in self.negotiators.keys()]
            n_partners = len(partners)

            # if I need nothing, end all negotiations
            if needs <= 0:
                dist.update(dict(zip(partners, [0] * n_partners)))
                continue

            # distribute my needs over my (remaining) partners.
            dist.update(
                dict(
                    zip(
                        partners,
                        distribute(
                            int(needs * (1 + self._overordering_fraction(t))),
                            n_partners,
                            equal=self.equal_distribution,
                            allow_zero=self.awi.allow_zero_quantity,
                        ),
                    )
                )
            )
        return dist

    def first_proposals(self):
        # just randomly distribute my needs over my partners (with best price for me).
        s, p = self._step_and_price(best_price=True)
        distribution = self.distribute_needs(t=0)
        d = {
            k: (q, s, p) if q > 0 or self.awi.allow_zero_quantity else None
            for k, q in distribution.items()
        }
        return d

    def counter_all(self, offers, states):
        response = dict()
        future_partners = {
            k for k, v in offers.items() if v[TIME] != self.awi.current_step
        }
        offers = {k: v for k, v in offers.items() if v[TIME] == self.awi.current_step}
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
            # find active partners in some random order
            partners = [_ for _ in all_partners if _ in offers.keys()]
            random.shuffle(partners)
            partners = set(partners)

            # find the set of partners that gave me the best offer set
            # (i.e. total quantity nearest to my needs)
            plist = list(powerset(partners))[::-1]
            best_diff, best_indx = float("inf"), -1
            for i, partner_ids in enumerate(plist):
                offered = sum(offers[p][QUANTITY] for p in partner_ids)
                diff = abs(offered - needs)
                if diff < best_diff:
                    best_diff, best_indx = diff, i
                if diff == 0:
                    break
            unneeded_response = (
                SAOResponse(ResponseType.END_NEGOTIATION, None)
                if not self.awi.allow_zero_quantity
                else SAOResponse(
                    ResponseType.REJECT_OFFER, (0, self.awi.current_step, 0)
                )
            )

            # If the best combination of offers is good enough, accept them and end all
            # other negotiations
            th = self._allowed_mismatch(min(_.relative_time for _ in states.values()))
            if best_diff <= th:
                partner_ids = plist[best_indx]
                others = list(partners.difference(partner_ids).union(future_partners))
                response |= {
                    k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                    for k in partner_ids
                } | {k: unneeded_response for k in others}
                continue

            # If I still do not have a good enough offer, distribute my current needs
            # randomly over my partners.
            t = min(_.relative_time for _ in states.values())
            distribution = self.distribute_needs(t)
            response.update(
                {
                    k: (
                        unneeded_response
                        if q == 0
                        else SAOResponse(
                            ResponseType.REJECT_OFFER, (q, self.awi.current_step, price)
                        )
                    )
                    for k, q in distribution.items()
                }
            )
        return response

    def _allowed_mismatch(self, r: float):
        mn, mx = 0, self.mismatch_max
        return mn + (mx - mn) * (r**self.mismatch_exp)

    def _overordering_fraction(self, t: float):
        mn, mx = self.overordering_min, self.overordering_max
        return mx - (mx - mn) * (t**self.overordering_exp)

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


class RandDistOneShotAgent(SyncRandomOneShotAgent):
    """
    An agent that distributes its needs over its partners randomly.

    Args:
        equal: If given, it tries to equally distribute its needs over as many of its
               suppliers/consumers as possible
        overordering_max: Maximum fraction of needs to over-order. For example, it the
                          agent needs 5 items and this is 0.2, it will order 6 in the first
                          negotiation step.
        overordering_min: Minimum fraction of needs to over-order. Used in the last negotiation
                          step.
        overordering_exp: Controls how fast does the over-ordering quantity go from max to min.
        concession_exp: Controls how fast does the agent concedes on matching its needs exactly.
        mismatch_max: Maximum mismtach in quantity allowed between needs and accepted offers. If
                      a fraction, it is will be this fraction of the production capacity (n_lines).
    """

    def __init__(self, *args, **kwargs):
        kwargs["equal"] = False
        super().__init__(*args, **kwargs)


class EqualDistOneShotAgent(SyncRandomOneShotAgent):
    """Same as RandDistOneShotAgent but defaulting to equal distribution of needs

    Args:
        equal: If given, it tries to equally distribute its needs over as many of its
               suppliers/consumers as possible
        overordering_max: Maximum fraction of needs to over-order. For example, it the
                          agent needs 5 items and this is 0.2, it will order 6 in the first
                          negotiation step.
        overordering_min: Minimum fraction of needs to over-order. Used in the last negotiation
                          step.
        overordering_exp: Controls how fast does the over-ordering quantity go from max to min.
        concession_exp: Controls how fast does the agent concedes on matching its needs exactly.
        mismatch_max: Maximum mismtach in quantity allowed between needs and accepted offers. If
                      a fraction, it is will be this fraction of the production capacity (n_lines).
    """

    def __init__(self, *args, **kwargs):
        kwargs["equal"] = True
        super().__init__(*args, **kwargs)


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
