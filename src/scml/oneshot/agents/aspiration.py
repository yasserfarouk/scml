import itertools
import random
from typing import Dict
from typing import Optional

from negmas import AspirationMixin
from negmas import Issue
from negmas import Outcome
from negmas import ResponseType
from negmas.sao import SAOResponse
from negmas.sao import SAOState

from ..agent import OneShotSyncAgent

__all__ = ["SingleAgreementAspirationAgent"]


class SingleAgreementAspirationAgent(AspirationMixin, OneShotSyncAgent):
    """
    Uses a time-based strategy to accept a single agreement from the set
    it is considering.
    """

    def before_step(self):
        self.__endall = not self.awi.is_first_level and not self.awi.is_last_level
        if self.__endall:
            return
        # we assume that we are either in the first or the latest layer
        # and calculate our ufun limits and reserved value
        self.ufun.reserved_value = self.ufun.from_contracts([])
        self._reserved_value = self.ufun.reserved_value
        AspirationMixin.aspiration_init(
            self,
            max_aspiration=1.0,
            aspiration_type=float(random.randint(1, 4)) if random.random() < 0.7 else random.random(),
            above_reserved_value=False,
        )
        # if self.awi.current_exogenous_input_quantity or self.awi.current_exogenous_output_quantity:
        #     breakpoint()
        self._limit = self.ufun.find_limit(
            True, int(self.awi.is_last_level), int(self.awi.is_first_level)
        )
        self._max_utility = self._limit.utility
        urange = self._max_utility - self._reserved_value
        if urange <= 1e-5:
            urange = 1e-5
        self._urange = urange

        if self.awi.is_last_level:
            self._best = (self._limit.input_quantity, self._limit.input_price)
        else:
            self._best = (self._limit.output_quantity, self._limit.output_price)

        # compile a list of all outcomes with their utilities and sort it
        # descendigly by utility
        issues = (
            self.awi.current_output_issues
            if self.awi.is_first_level
            else self.awi.current_output_issues
        )
        outcomes = list(Issue.enumerate(issues, astype=tuple))
        self._outcomes = sorted(
            zip(
                (
                    (
                        self.ufun.from_offers([_], [self.awi.is_first_level])
                        - self._reserved_value
                    )
                    / (self._urange)
                    for _ in outcomes
                ),
                outcomes,
            ),
            key=lambda x: -x[0],
        )
        self._last_index = 0


    def counter_all(self, offers, states):

        if self.__endall:
            return dict(
                zip(
                    offers.keys(),
                    itertools.repeat(SAOResponse(ResponseType.END_NEGOTIATION, None)),
                )
            )
        # find current aspiration level between zero and one
        asp = max(self.aspiration(state.relative_time) for state in states.values())

        # acceptance strategy
        partner_utils = sorted(
            zip(
                offers.keys(),
                (
                    (self.ufun(_) - self._reserved_value) / self._urange
                    for _ in offers.values()
                ),
            ),
            key=lambda x: -x[1],
        )
        if partner_utils[0][1] >= asp:
            response = dict(
                zip(
                    offers.keys(),
                    itertools.repeat(SAOResponse(ResponseType.END_NEGOTIATION, None)),
                )
            )
            response[partner_utils[0][0]] = SAOResponse(ResponseType.ACCEPT_OFFER, None)
            self.__endall = True
            return response

        # offering strategy
        i = self._last_index
        for i, (u, _) in enumerate(self._outcomes[self._last_index :]):
            if u >= asp:
                continue
            if i > 0:
                outcome, self._last_index = self._outcomes[i - 1][1], i - 1
            else:
                outcome, self._last_index = self._outcomes[0][1], 0
            break
        else:
            outcome, self._last_index = self._outcomes[i][1], i
        return self.choose_agents(offers, outcome)

    def choose_agents(self, offers, outcome):
        """Selects an appropriate way to distribute this outcome to agents with
        given IDs."""
        if len(offers) == 0:
            return dict()
        # fidn the partner which gave me the offer most similar to my best
        dists = sorted(
            (
                (sum((a - b) * (a - b) for a, b in zip(outcome, v)), k)
                for k, v in offers.items()
            ),
            key=lambda x: x[0],
        )
        # offer everyone nothing excdpt the one agent that gave me the offer most
        # similar to my preferred outcome
        result = dict(
            zip(
                offers.keys(),
                itertools.repeat(SAOResponse(ResponseType.REJECT_OFFER, None)),
            )
        )
        result[dists[0][1]] = SAOResponse(ResponseType.REJECT_OFFER, outcome)
        return result

    def first_proposals(self) -> Dict[str, "Outcome"]:
        """
        Gets a set of proposals to use for initializing the negotiation.

        Returns:
            A dictionary mapping each negotiator (in self.negotiators dict) to
            an outcome to be used as the first proposal if the agent is to start
            a negotiation.

        """
        if self.__endall:
            return dict(
                zip(
                    self.negotiators.keys(),
                    itertools.repeat(None),
                )
            )
        # that is a risk. The agent will send its best offer to everyone risking
        # two of them accepting it which is suboptimal.
        return dict(
            zip(
                self.negotiators.keys(),
                itertools.repeat(self._outcomes[0][1]),
            )
        )
