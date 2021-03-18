import random
from typing import Dict, Optional

from negmas import MechanismState
from negmas import ResponseType
from negmas import SAOResponse
from negmas import SAOState, Outcome
from negmas import SAOSingleAgreementRandomController, SAOSingleAgreementController

from scml.oneshot.agent import (
    OneShotAgent,
    OneShotSingleAgreementAgent,
    OneShotSyncAgent,
)
from scml.oneshot.agent import OneShotSyncAgent
from scml.scml2020.common import QUANTITY, TIME, UNIT_PRICE

__all__ = ["GreedyOneShotAgent", "GreedySyncAgent", "GreedySingleAgreementAgent"]


class GreedyOneShotAgent(OneShotAgent):
    """A greedy agent based on OneShotAgent"""

    def init(self):
        self._sales = self._supplies = 0

    def step(self):
        self._sales = self._supplies = 0

    def on_negotiation_success(self, contract, mechanism):
        if contract.annotation["product"] == self.awi.my_input_product:
            self._sales += contract.agreement["quantity"]
        else:
            self._supplies += contract.agreement["quantity"]

    def propose(self, negotiator_id: str, state) -> "Outcome":
        return self.best_offer(negotiator_id)

    def respond(self, negotiator_id, state, offer):
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            return ResponseType.END_NEGOTIATION
        return (
            ResponseType.ACCEPT_OFFER
            if offer[QUANTITY] <= my_needs
            else ResponseType.REJECT_OFFER
        )

    def best_offer(self, negotiator_id):
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            return None
        ami = self.get_ami(negotiator_id)
        if not ami:
            return None
        quantity_issue = ami.issues[QUANTITY]
        unit_price_issue = ami.issues[UNIT_PRICE]
        offer = [-1] * 3
        offer[QUANTITY] = max(
            min(my_needs, quantity_issue.max_value), quantity_issue.min_value
        )
        offer[TIME] = self.awi.current_step
        if self._is_selling(ami):
            offer[UNIT_PRICE] = unit_price_issue.max_value
        else:
            offer[UNIT_PRICE] = unit_price_issue.min_value
        return tuple(offer)

    def _needed(self, negotiator_id):
        ami = self.get_ami(negotiator_id)
        if not ami:
            return 0
        summary = self.awi.exogenous_contract_summary
        secured = self._sales if self._is_selling(ami) else self._supplies
        demand = min(summary[0][0], summary[-1][0]) / (self.awi.n_competitors + 1)
        return demand - secured

    def _is_selling(self, ami):
        if not ami:
            return None
        return ami.annotation["product"] == self.awi.my_output_product


class GreedySyncAgent(OneShotSyncAgent, GreedyOneShotAgent):
    """A greedy agent based on OneShotSyncAgent"""

    def __init__(self, *args, threshold=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self._threshold = threshold

    def init(self):
        super().init()
        self.ufun.best = self.ufun.find_limit(True)
        self.ufun.worst = self.ufun.find_limit(False)

    def first_proposals(self):
        """Decide a first proposal on every negotiation.
        Returning None for a negotiation means ending it."""
        return dict(
            zip(
                self.negotiators.keys(),
                (self.best_offer(_) for _ in self.negotiators.keys()),
            )
        )

    def counter_all(self, offers, states):
        """Respond to a set of offers given the negotiation state of each."""
        responses = {
            k: SAOResponse(ResponseType.REJECT_OFFER, None)
            for k, v in self.first_proposals().items()
        }
        my_input_needs, my_output_needs = self._needs()
        input_offers = {
            k: v for k, v in offers.items() if self._is_selling(self.get_ami(k))
        }
        output_offers = {
            k: v for k, v in offers.items() if not self._is_selling(self.get_ami(k))
        }

        def calc_responses(my_needs, offers, is_selling):
            nonlocal responses
            if len(offers) == 0:
                return
            sorted_offers = sorted(
                offers.values(),
                key=lambda x: -x[UNIT_PRICE] if is_selling else x[UNIT_PRICE],
            )
            secured, outputs, chosen = 0, [], dict()
            for i, k in enumerate(offers.keys()):
                offer = sorted_offers[i]
                secured += offer[QUANTITY]
                if secured >= my_needs:
                    break
                chosen[k] = offer
                outputs.append(is_selling)

            u = self.ufun.from_offers(list(chosen.values()), outputs)
            if u > self._threshold * self.ufun.max_utility:
                for k in chosen.keys():
                    responses[k] = SAOResponse(ResponseType.ACCEPT_OFFER, None)

        calc_responses(my_input_needs, input_offers, False),
        calc_responses(my_output_needs, output_offers, True)
        return responses

    def _needs(self):
        """
        Returns both input and output needs
        """
        if self.awi.is_middle_level:
            summary = self.awi.exogenous_contract_summary
            n = min(summary[0][0], summary[-1][0])
            return n - self._supplies, n - self._sales
        if self.awi.is_first_level:
            return 0, self.awi.current_exogenous_input_quantity - self._sales
        return self.awi.current_exogenous_output_quantity - self._supplies, 0


class GreedySingleAgreementAgent(OneShotSingleAgreementAgent):
    """ A greedy agent based on `OneShotSingleAgreementAgent`"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init(self):
        super().init()
        self.ufun.best = self.ufun.find_limit(True)
        self.ufun.worst = self.ufun.find_limit(False)

    def is_acceptable(self, offer, source, state) -> bool:
        mx, mn = self.ufun.max_utility, self.ufun.min_utility
        u = (self.ufun(offer) - mn) / (mx - mn)
        return u >= (1 - state.relative_time)

    def best_offer(self, offers):
        ufuns = [(self.ufun(_), i) for i, _ in enumerate(offers.values())]
        keys = list(offers.keys())
        return keys[max(ufuns)[1]]

    def is_better(self, a, b, negotiator, state):
        return self.ufun(a) > self.ufun(b)
