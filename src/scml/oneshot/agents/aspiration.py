from typing import Optional, Dict
import itertools
from negmas import AspirationMixin, Outcome, ResponseType
from negmas.sao import SAOState, SAOResponse
from negmas.utilities import utility_range
from ..agent import OneShotSingleAgreementAgent

__all__ = ["SingleAgreementAspirationAgent"]


class SingleAgreementAspirationAgent(AspirationMixin, OneShotSingleAgreementAgent):
    """
    Uses a time-based strategy to accept a single agreement from the set
    it is considering.
    """

    def init(self):
        self.__endall = (
            self.awi.my_input_product != 0
            and self.awi.my_output_product < self.awi.n_products - 1
        )
        super().init()
        if not self.ufun:
            self.make_ufun(add_exogenous=True)

    def counter_all(self, offers, states):
        if self.__endall:
            return dict(
                zip(
                    offers.keys(),
                    itertools.repeat(SAOResponse(ResponseType.END_NEGOTIATION, None)),
                )
            )
        issues = (
            self.awi.current_input_issues
            if self.awi.is_last_level
            else self.awi.current_output_issues
        )
        self._min_utility, self._max_utility = utility_range(self.ufun, issues=issues)
        self.ufun.reserved_value = self._min_utility
        AspirationMixin.aspiration_init(
            self, max_aspiration=self._max_utility, aspiration_type="boulware"
        )
        return super().counter_all(offers, states)

    def is_acceptable(self, offer: "Outcome", source: str, state: SAOState) -> bool:
        if self.__endall:
            return False
        u = (self.ufun(offer) - self._min_utility) / (
            self._max_utility - self._min_utility
        )
        return u > self.aspiration(state.relative_time)

    def best_offer(self, offers: Dict[str, "Outcome"]) -> Optional[str]:
        if self.__endall:
            return None
        ufuns = [(self.ufun(_), i) for i, _ in enumerate(offers.values())]
        keys = list(offers.keys())
        return keys[max(ufuns)[1]]

    def is_better(self, a: "Outcome", b: "Outcome", negotiator: str, state: SAOState):
        return self.ufun(a) > self.ufun(b) if self.ufun else False
