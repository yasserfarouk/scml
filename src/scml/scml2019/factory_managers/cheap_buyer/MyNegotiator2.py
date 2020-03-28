import math
import random

from negmas import SAONegotiator, MechanismState, ResponseType
from typing import Optional


class MyNegotiator2(SAONegotiator):
    STRATEGY_ONLY_THE_BEST = "only_the_best"
    STRATEGY_TIME_BASED_CONCESSION = "time_based_concession"
    STRATEGY_INTERVAL = "interval_strategy"

    def __init__(
        self,
        name,
        ufun,
        concession_coefficient=10,
        strategy=STRATEGY_ONLY_THE_BEST,
        interval=0.05,
        reserved_value=1,
    ):
        super(MyNegotiator2, self).__init__(name=name, ufun=ufun)
        self.ordered_outcomes = []
        self.ufun = ufun
        self.utility_values_of_offers = {}
        self.offers_to_us = []
        self.our_offers = []
        self.concession_coefficient = concession_coefficient
        self.strategy = strategy
        self.interval = interval
        self.reserved_valuee = reserved_value

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        if self.strategy == self.STRATEGY_ONLY_THE_BEST:
            our_offer = self.propose_only_the_best(state)
        elif self.strategy == self.STRATEGY_TIME_BASED_CONCESSION:
            our_offer = self.propose_time_based_concession(state=state)
        elif self.strategy == self.STRATEGY_INTERVAL:
            our_offer = self.propose_interval()
        else:
            our_offer = self.propose_only_the_best(state)
        self.our_offers.append(
            "STEP : " + str(state.step) + " OFFER : " + str(our_offer)
        )
        return our_offer

    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        self.offers_to_us.append("STEP : " + str(state.step) + " OFFER : " + str(offer))
        if self.strategy == self.STRATEGY_ONLY_THE_BEST:
            return self.respond_only_the_best(offer, state)
        elif self.strategy == self.STRATEGY_TIME_BASED_CONCESSION:
            return self.respond_time_based_concession(state=state, offer=offer)
        elif self.strategy == self.STRATEGY_INTERVAL:
            return self.respond_interval(offer=offer)
        else:
            return self.respond_only_the_best(offer, state)

    def on_ufun_changed(self):
        super().on_ufun_changed()
        if self._ami is None:
            return
        outcomes = self._ami.discrete_outcomes()
        self.ordered_outcomes = sorted(
            [(self._utility_function(outcome), outcome) for outcome in outcomes],
            key=lambda x: float(x[0]) if x[0] is not None else float("-inf"),
            reverse=True,
        )
        biggest_utility = self.ordered_outcomes[0][0]
        index = 0
        self.utility_values_of_offers = {}
        if biggest_utility == 0:
            a = 0
        for outcome in self.ordered_outcomes:
            self.ordered_outcomes[index] = outcome[0] / biggest_utility, outcome[1]
            self.utility_values_of_offers[
                str(self.ordered_outcomes[index][1])
            ] = self.ordered_outcomes[index][0]
            index = index + 1

    def propose_(self, state: MechanismState) -> Optional["Outcome"]:
        if self._ufun_modified:
            self.on_ufun_changed()
        return self.propose(state)

    def respond_(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        if self._ufun_modified:
            self.on_ufun_changed()
        return self.respond(state=state, offer=offer)

    def get_utility_value(self, offer):
        return self.utility_values_of_offers.get(str(offer), 0.0)

    def get_our_offers(self):
        return self.our_offers

    def get_offers_to_us(self):
        return self.offers_to_us

    def propose_only_the_best(self, state):
        if self.ordered_outcomes is None or len(self.ordered_outcomes) < 1:
            self.on_ufun_changed()
        our_offer = self.ordered_outcomes[0][1]
        return our_offer

    def respond_only_the_best(self, offer, state):
        if self.get_utility_value(offer) == 1:
            return ResponseType.ACCEPT_OFFER
        else:
            return ResponseType.REJECT_OFFER

    def propose_time_based_concession(self, state):
        if self.ordered_outcomes is None or len(self.ordered_outcomes) < 1:
            self.on_ufun_changed()
        our_offer = self.ordered_outcomes[0][1]
        concession_score = self.get_concession_score(state)
        for ordered_outcome in self.ordered_outcomes:
            if ordered_outcome[0] < concession_score:
                our_offer = ordered_outcome[1]
                break
        return our_offer

    def respond_time_based_concession(self, offer, state):
        if self.get_utility_value(offer) >= self.get_concession_score(state=state):
            return ResponseType.ACCEPT_OFFER
        else:
            return ResponseType.REJECT_OFFER

    def get_concession_score(self, state):
        return 1 + (self.reserved_valuee - 1) * math.pow(
            state.relative_time, self.concession_coefficient
        )

    def propose_interval(self):
        candidate_offers = []
        for outcome in self.ordered_outcomes:
            if outcome[0] >= 1 - self.interval:
                candidate_offers.append(outcome[1])
            else:
                break
        if len(candidate_offers) > 0:
            return random.choice(candidate_offers)
        else:
            return self.ordered_outcomes[0][1]

    def respond_interval(self, offer):
        if self.get_utility_value(offer) >= 1 - self.interval:
            return ResponseType.ACCEPT_OFFER
        else:
            return ResponseType.REJECT_OFFER
