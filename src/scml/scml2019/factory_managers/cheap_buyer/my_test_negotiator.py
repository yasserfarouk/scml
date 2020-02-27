import random

from negmas import ResponseType
from negmas.common import MechanismState
from negmas.sao import AspirationNegotiator
from typing import Optional

from scml.scml2019.common import INVALID_UTILITY


class MyTestnegotiator(AspirationNegotiator):
    def __init__(self, name, ufun):
        super(MyTestnegotiator, self).__init__(name=name, ufun=ufun)

    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        # print("RESPONSE CALLED")
        if self._utility_function is None:
            return ResponseType.REJECT_OFFER
        u = self._utility_function(offer)
        if u is None:
            return ResponseType.REJECT_OFFER

        asp = (
            self.aspiration(state.relative_time) * (self.ufun_max - self.ufun_min)
            + self.ufun_min
        )
        if u >= asp and u > self.reserved_value:
            return ResponseType.ACCEPT_OFFER
        if asp < self.reserved_value:
            # print("END")
            return ResponseType.END_NEGOTIATION
        return ResponseType.REJECT_OFFER

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        # print("PROPOSE CALLED")
        result = None
        aspiration = self.aspiration(state.relative_time)
        asp = aspiration * (self.ufun_max - self.ufun_min) + self.ufun_min
        if asp < self.reserved_value:
            # print("Offer test : "+str(result))
            return None
        for i, (u, o) in enumerate(self.ordered_outcomes):
            if u is None:
                continue
            if u < asp:
                if u < self.reserved_value:
                    # print("Offer test : " + str(result))
                    return None
                if i == 0:
                    result = self.ordered_outcomes[i][1]
                    # print("Offer test : " + str(result))
                    return result
                if self.randomize_offer:
                    result = random.sample(self.ordered_outcomes[:i], 1)[0][1]
                    # print("Offer test : " + str(result))
                    return result
                result = self.ordered_outcomes[i - 1][1]
                # print("Offer test : " + str(result))
                return result
        if self.randomize_offer:
            result = random.sample(self.ordered_outcomes, 1)[0][1]
            # print("Offer test : " + str(result))
            return result
        result = self.ordered_outcomes[-1][1]
        # print("Offer : test" + str(result))
        return result

    def normalize(self):
        """"""
        isBiggestNegative = False
        if self.ordered_outcomes[0][0] < 0:
            isBiggestNegative = True
            max_utility = self.ordered_outcomes[0][0]
        else:
            max_utility = self.ordered_outcomes[0][0]
        for index in range(len(self.ordered_outcomes)):
            outcome = self.ordered_outcomes[index]
            if outcome[0] <= INVALID_UTILITY:
                self.ordered_outcomes[index] = (0, outcome[1])
            else:
                if isBiggestNegative:
                    self.ordered_outcomes[index] = (
                        (max_utility / outcome[0]),
                        outcome[1],
                    )
                else:
                    self.ordered_outcomes[index] = (
                        max(0, (outcome[0] / max_utility)),
                        outcome[1],
                    )
        a = 0

    def on_ufun_changed(self):
        super().on_ufun_changed()
        outcomes = self._ami.discrete_outcomes()
        self.ordered_outcomes = sorted(
            [(self._utility_function(outcome), outcome) for outcome in outcomes],
            key=lambda x: x[0],
            reverse=True,
        )
        if not self.assume_normalized:
            self.normalize()
            self.ufun_max = self.ordered_outcomes[0][0]
            self.ufun_min = self.ordered_outcomes[-1][0]
            if self.reserved_value is not None and self.ufun_min < self.reserved_value:
                self.ufun_min = self.reserved_value
