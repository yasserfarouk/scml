import random
from collections import defaultdict

from negmas import Outcome, ResponseType

from scml.oneshot.agents.greedy import GreedyOneShotAgent, GreedySyncAgent
from scml.std.agent import StdAgent
from scml.std.common import QUANTITY, TIME, UNIT_PRICE

__all__ = [
    "GreedyStdAgent",
    "GreedySyncAgent",
    "GreedyOneShotAgent",
]


class GreedyStdAgent(StdAgent):
    """
    A greedy agent based on StdAgent

    Args:
        concession_exponent: A real number controlling how fast does the agent
                             concede on price.
        acc_price_slack: The allowed slack in price limits compared with best
                         prices I got so far
        step_price_slack: The allowed slack in price limits compared with best
                         prices I got this step
        opp_price_slack: The allowed slack in price limits compared with best
                         prices I got so far from a given opponent in this step
        opp_acc_price_slack: The allowed slack in price limits compared with best
                         prices I got so far from a given opponent so far
        range_slack: Always consider prices above (1-`range_slack`) of the best
                     possible prices *good enough*.
        production_target: Fraction of production capacity to be secured in advance
    Remarks:
        - A `concession_exponent` greater than one makes the agent concede
          super linearly and vice versa

    """

    def __init__(
        self,
        *args,
        concession_exponent=None,
        acc_price_slack=float("inf"),
        step_price_slack=None,
        opp_price_slack=None,
        opp_acc_price_slack=None,
        range_slack=None,
        future_threshold=0.9,
        production_target=0.75,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if concession_exponent is None:
            concession_exponent = 0.2 + random.random() * 0.8
        if step_price_slack is None:
            step_price_slack = random.random() * 0.1 + 0.05
        if opp_price_slack is None:
            opp_price_slack = random.random() * 0.1 + 0.05
        if opp_acc_price_slack is None:
            opp_acc_price_slack = random.random() * 0.1 + 0.05
        if range_slack is None:
            range_slack = random.random() * 0.2 + 0.05

        self._e = concession_exponent
        self._acc_price_slack = acc_price_slack
        self._step_price_slack = step_price_slack
        self._opp_price_slack = opp_price_slack
        self._opp_acc_price_slack = opp_acc_price_slack
        self._range_slack = range_slack
        self._sales = self._supplies = 0
        self._production_target = production_target
        self._future_threshold = future_threshold

    def init(self):
        """Initialize the quantities and best prices received so far"""
        self._best_acc_selling, self._best_acc_buying = 0.0, float("inf")
        self._best_opp_acc_selling = defaultdict(float)
        self._best_opp_acc_buying = defaultdict(lambda: float("inf"))

    def before_step(self):
        """Initialize the quantities and best prices received for next step"""
        self._best_selling, self._best_buying = 0.0, float("inf")
        self._best_opp_selling = defaultdict(float)
        self._best_opp_buying = defaultdict(lambda: float("inf"))
        self._sales = self._supplies = 0

    def on_negotiation_success(self, contract, mechanism):
        """Record sales/supplies secured"""
        super().on_negotiation_success(contract, mechanism)

        # update my current best price to use for limiting concession in other
        # negotiations
        up = contract.agreement["unit_price"]
        if contract.annotation["product"] == self.awi.my_output_product:
            partner = contract.annotation["buyer"]
            self._best_acc_selling = max(up, self._best_acc_selling)
            self._best_opp_acc_selling[partner] = max(
                up, self._best_opp_acc_selling[partner]
            )
        else:
            partner = contract.annotation["seller"]
            self._best_acc_buying = min(up, self._best_acc_buying)
            self._best_opp_acc_buying[partner] = min(
                up, self._best_opp_acc_buying[partner]
            )

    def propose(self, negotiator_id: str, state, source=None) -> Outcome | None:
        # find the absolute best offer for me. This will most likely has an
        # unrealistic price
        offer = self.best_offer(negotiator_id)

        # if there are no best offers, just return None to end the negotiation
        if not offer:
            return None

        # over-write the unit price in the best offer with a good-enough price
        offer = list(offer)
        offer[UNIT_PRICE] = self._find_good_price(
            self.get_nmi(negotiator_id), state, offer
        )
        return tuple(offer)

    def respond(self, negotiator_id, state, source=None) -> ResponseType:
        offer = state.current_offer  # type: ignore
        assert offer is not None
        # find the quantity I still need and end negotiation if I need nothing more
        my_needs = self._needed(negotiator_id)
        # reject any offers with quantities above my needs
        response = (
            ResponseType.ACCEPT_OFFER
            if (offer[QUANTITY] <= my_needs and offer[TIME] == self.awi.current_step)
            or (
                offer[QUANTITY] < self._future_needs(negotiator_id, offer[TIME])
                and offer[TIME] > self.awi.current_step
            )
            else ResponseType.REJECT_OFFER
        )
        if response != ResponseType.ACCEPT_OFFER:
            return response

        # reject offers with prices that are deemed NOT good-enough
        nmi = self.get_nmi(negotiator_id)
        response = (
            response
            if self._is_good_price(nmi, state, offer)
            else ResponseType.REJECT_OFFER
        )
        # If this response is about today, do not update internal stats
        if offer[TIME] != self.awi.current_step:
            return response

        # update my current best price to use for limiting concession in other
        # negotiations
        up = offer[UNIT_PRICE]
        if self._is_selling(nmi):
            self._best_selling = max(up, self._best_selling)
            partner = nmi.annotation["buyer"]
            self._best_opp_selling[partner] = self._best_selling
        else:
            self._best_buying = min(up, self._best_buying)
            partner = nmi.annotation["seller"]
            self._best_opp_buying[partner] = self._best_buying
        return response

    def best_offer(self, negotiator_id):
        nmi = self.get_nmi(negotiator_id)
        if not nmi:
            return None
        my_needs = int(self._needed(negotiator_id))
        if my_needs <= 0:
            # see if I can get something in the future
            time_issue = nmi.issues[TIME]
            times = list(time_issue.all)
            random.shuffle(times)
            for t in times:
                my_needs = self._future_needs(negotiator_id, t)
                if my_needs <= 0:
                    continue
                offer = [-1] * 3
                quantity_issue = nmi.issues[QUANTITY]
                unit_price_issue = nmi.issues[UNIT_PRICE]
                mx = max(
                    min(my_needs, quantity_issue.max_value), quantity_issue.min_value
                )
                # never contract offer more than production capacity
                mx = max(0, min(mx, self.awi.n_lines * (t - self.awi.current_step)))
                if mx < 1:
                    continue
                mn_ = max(1, int(0.5 + mx * self.awi.current_step / self.awi.n_steps))
                mx_ = int(mx)
                offer[QUANTITY] = random.randint(mn_, mx_) if mn_ < mx_ else mn_
                offer[TIME] = t
                if self._is_selling(nmi):
                    offer[UNIT_PRICE] = unit_price_issue.max_value
                else:
                    offer[UNIT_PRICE] = unit_price_issue.min_value
                return tuple(offer)

        quantity_issue = nmi.issues[QUANTITY]
        unit_price_issue = nmi.issues[UNIT_PRICE]
        offer = [-1] * 3
        mx = max(min(my_needs, quantity_issue.max_value), quantity_issue.min_value)
        # never contract offer more than production capacity
        mx = min(mx, self.awi.n_lines)
        offer[QUANTITY] = random.randint(
            max(1, int(0.5 + mx * self.awi.current_step / self.awi.n_steps)), int(mx)
        )
        offer[TIME] = self.awi.current_step
        if self._is_selling(nmi):
            offer[UNIT_PRICE] = unit_price_issue.max_value
        else:
            offer[UNIT_PRICE] = unit_price_issue.min_value
        return tuple(offer)

    def _future_needs(self, negotiator_id, t):
        return self._production_target * (
            self.awi.n_lines
            - sum(
                (
                    self.awi.future_sales
                    if negotiator_id in self.awi.my_consumers
                    else self.awi.future_supplies
                )
                .get(t, dict())
                .values()
            )
        )

    def _needed(self, negotiator_id):
        if self.awi.is_middle_level:
            return self._production_target * self.awi.n_lines
        return (
            self.awi.needed_sales
            if negotiator_id in self.awi.my_consumers
            else self.awi.needed_supplies
        )

    def _is_selling(self, nmi):
        if not nmi:
            return None
        return nmi.annotation["product"] == self.awi.my_output_product

    def _is_good_price(self, nmi, state, offer):
        """Checks if a given price is good enough at this stage"""
        price = offer[UNIT_PRICE]
        mn, mx = self._price_range(nmi, offer)
        th = (
            self._th(state.step, nmi.n_steps)
            if offer[TIME] == self.awi.current_step
            else self._future_threshold
        )
        # a good price is one better than the threshold
        if self._is_selling(nmi):
            return (price - mn) >= th * (mx - mn)
        else:
            return (mx - price) >= th * (mx - mn)

    def _find_good_price(self, nmi, state, offer):
        """Finds a good-enough price conceding linearly over time"""
        mn, mx = self._price_range(nmi, offer)
        th = self._th(state.step, nmi.n_steps)
        # offer a price that is around th of your best possible price
        if self._is_selling(nmi):
            return int(mn + th * (mx - mn))
        else:
            return int(mx - th * (mx - mn))

    def _price_range(self, nmi, offer):
        """Limits the price by the best price received"""
        mn = nmi.issues[UNIT_PRICE].min_value
        mx = nmi.issues[UNIT_PRICE].max_value
        if offer[TIME] != self.awi.current_step:
            mn, mx = int(mx * self._future_threshold), mx
        if self._is_selling(nmi):
            partner = nmi.annotation["buyer"]
            mn = min(
                mx * (1 - self._range_slack),
                max(
                    [mn]
                    + [
                        p * (1 - slack)
                        for p, slack in (
                            (self._best_selling, self._step_price_slack),
                            (self._best_acc_selling, self._acc_price_slack),
                            (self._best_opp_selling[partner], self._opp_price_slack),
                            (
                                self._best_opp_acc_selling[partner],
                                self._opp_acc_price_slack,
                            ),
                        )
                    ]
                ),
            )
        else:
            partner = nmi.annotation["seller"]
            mx = max(
                mn * (1 + self._range_slack),
                min(
                    [mx]
                    + [
                        p * (1 + slack)
                        for p, slack in (
                            (self._best_buying, self._step_price_slack),
                            (self._best_acc_buying, self._acc_price_slack),
                            (self._best_opp_buying[partner], self._opp_price_slack),
                            (
                                self._best_opp_acc_buying[partner],
                                self._opp_acc_price_slack,
                            ),
                        )
                    ]
                ),
            )
        return int(mn), int(mx)

    def _th(self, step, n_steps):
        """calculates a descending threshold (0 <= th <= 1)"""
        return ((n_steps - step - 1) / (n_steps - 1)) ** self._e
