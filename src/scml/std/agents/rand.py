import random

from negmas import Outcome, ResponseType
from negmas.negotiators.modular import itertools
from negmas.sao import SAOResponse, SAOState

from scml.common import distribute
from scml.oneshot.agents import SyncRandomOneShotAgent
from scml.oneshot.common import QUANTITY, TIME, UNIT_PRICE
from scml.std.agent import StdAgent, StdSyncAgent

__all__ = ["SyncRandomStdAgent", "SyncRandomOneShotAgent", "RandomStdAgent"]

PROB_ACCEPTANCE = 0.1
PROB_END = 0.005


class RandomStdAgent(StdAgent):
    """A naive random agent"""

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


class SyncRandomStdAgent(StdSyncAgent):
    """An agent that distributes its needs over its partners randomly."""

    def __init__(
        self,
        *args,
        today_target_productivity=0.3,
        future_target_productivity=0.3,
        today_concentration=0.25,
        future_concentration=0.75,
        today_concession_exp=2.0,
        future_concession_exp=4.0,
        future_min_price=0.25,
        prioritize_near_future: bool = False,
        prioritize_far_future: bool = False,
        pfuture=0.15,
        **kwargs,
    ):
        """A simply agent that distributes its offers between today's needs and some future needs

        Args:
            today_target_productivity: Estimated productivity today used to set needed supply and demand
                                       for agents in the middle of the production graph
            future_target_productivity: Estimated productivity in the future used to limit estimates of
                                        needed future supplies and sales.
            future_concentration: How concentrated should our offers for future supplies/sales. This is
                                  the fraction of future supply/sale distributions that will use the minimum
                                  possible number of partners.
            today_concentration: How concentrated should our offers for today's supplies/sales. This is
                                  the fraction of today's supply/sale distributions that will use the minimum
                                  possible number of partners.
            today_concession_exp: The concession exponent to use for prices today
            future_concession_exp:The concession exponent to use for prices in offers regarding the future
            pfuture: Fraction of available offers to always use for future supplies/sales.
            future_min_price: Fraction of the price range not to go under/over for future sales/supplies
            prioritize_near_future: Prioritize near-future when distributing future needs
            prioritize_far_future: Prioritize far-future when distributing future needs
        """
        super().__init__(*args, **kwargs)
        self.ptoday = 1.0 - pfuture
        self.today_exp = today_concession_exp
        self.future_exp = future_concession_exp
        self.fmin = future_min_price
        self.today_productivity = today_target_productivity
        self.future_productivity = future_target_productivity
        self.near = prioritize_near_future
        self.far = prioritize_far_future
        self.future_concentration = future_concentration
        self.today_concentration = today_concentration

    def first_proposals(self):  # type: ignore
        # just randomly distribute my needs over my partners (with best price for me).
        # remaining partners get random future offers
        distribution = self.distribute_todays_needs()
        future_suppliers = {k for k, v in distribution.items() if v <= 0}
        unneeded = (
            None if not self.awi.allow_zero_quantity else (0, self.awi.current_step, 0)
        )

        offers = {
            k: ((q, self.awi.current_step, self.best_price(k)) if q > 0 else unneeded)
            for k, q in distribution.items()
        } | self.distribute_future_offers(list(future_suppliers))

        return offers

    def counter_all(self, offers, states):
        max_sell = self.awi.current_output_issues[UNIT_PRICE].max_value
        min_sell = max(
            self.awi.current_output_issues[UNIT_PRICE].min_value,
            self.awi.current_input_issues[UNIT_PRICE].max_value,
        )
        min_buy = self.awi.current_input_issues[UNIT_PRICE].min_value
        max_buy = min(
            self.awi.current_input_issues[UNIT_PRICE].max_value,
            self.awi.current_output_issues[UNIT_PRICE].min_value,
        )
        # find everything I need from now to the end of time
        needed_supplies, needed_sales = self.estimate_future_needs()
        needed_sales[self.awi.current_step] = self.awi.needed_sales
        needed_supplies[self.awi.current_step] = self.awi.needed_supplies
        if self.awi.is_middle_level:
            needed_sales[self.awi.current_step] = max(
                needed_sales[self.awi.current_step],
                int(self.awi.n_lines * self.today_productivity),
            )
            needed_supplies[self.awi.current_step] = max(
                needed_supplies[self.awi.current_step],
                int(self.awi.n_lines * self.today_productivity),
            )

        # accept all offers I seem to need if they have good price
        responses = dict()
        c = self.awi.current_step
        n = max(self.awi.n_steps - c, 1)
        for is_partner, needs, is_good_price, mn, mx in (
            (self.is_supplier, needed_supplies, self.good2buy, min_buy, max_buy),
            (self.is_consumer, needed_sales, self.good2sell, min_sell, max_sell),
        ):
            if mn > mx:
                continue
            for partner, offer in offers.items():
                if not is_partner(partner):
                    continue
                if offer is None:
                    continue
                q, t = offer[QUANTITY], offer[TIME]
                today = t == c
                r = states[partner].relative_time if today else (t - c) / n
                if not is_good_price(offer[UNIT_PRICE], r, mn, mx, today):
                    continue
                if 0 < q < needs.get(t, 0):
                    responses[partner] = SAOResponse(ResponseType.ACCEPT_OFFER, offer)
                    needs[t] -= q
        remaining = {k for k in offers.keys() if k not in responses.keys()}

        # distribute today's needs over the partners with rejected offers
        distribution = self.distribute_todays_needs(partners=remaining)
        future_partners = {k for k, v in distribution.items() if v <= 0}
        unneeded = (
            None if not self.awi.allow_zero_quantity else (0, self.awi.current_step, 0)
        )

        # distribute my future needs over people I did not use today
        myoffers = {
            k: (
                (q, self.awi.current_step, self.good_price(k, today=False))
                if q > 0
                else unneeded
            )
            for k, q in distribution.items()
        } | self.distribute_future_offers(list(future_partners))
        responses |= {
            k: SAOResponse(ResponseType.REJECT_OFFER, offer)
            for k, offer in myoffers.items()
        }

        return responses

    def distribute_todays_needs(self, partners=None) -> dict[str, int]:
        """Distributes my needs randomly over all my partners"""
        ignored = []
        if partners is None:
            partners = self.negotiators.keys()
        partners = list(partners)
        random.shuffle(partners)
        n = min(len(partners), max(1, int(self.ptoday * len(partners))))
        ignored = partners[n:]
        partners = partners[:n]

        response = dict(zip(partners, itertools.repeat(0))) | dict(
            zip(ignored, itertools.repeat(0))
        )

        mxin = self.awi.current_input_issues[QUANTITY].max_value
        mxout = self.awi.current_output_issues[QUANTITY].max_value
        for is_partner, edge_needs, mxq in (
            (self.is_supplier, self.awi.needed_supplies, mxin),
            (self.is_consumer, self.awi.needed_sales, mxout),
        ):
            needs = self.awi.n_lines if self.awi.is_middle_level else edge_needs
            # find my partners and the quantity I need
            active_partners = [_ for _ in partners if is_partner(_)]
            if not active_partners or needs < 1:
                continue
            random.shuffle(active_partners)
            n_partners = len(active_partners)

            # distribute my needs over my (remaining) partners.
            # we always allow zero quantity because these will be overriden
            # by future offers later
            response |= dict(
                zip(
                    active_partners,
                    distribute(
                        needs,
                        n_partners,
                        allow_zero=True,
                        concentrated=random.random() < self.today_concentration,
                        equal=random.random() > 0.5,
                        mx=mxq,
                    ),
                )
            )

        return response

    def estimate_future_needs(self):
        """Estimates how much I need to buy and sell for each future step"""
        current_step, n_steps = self.awi.current_step, self.awi.n_steps
        trange = (
            max(
                self.awi.current_input_issues[TIME].min_value, self.awi.current_step + 1
            ),
            min(self.awi.current_input_issues[TIME].max_value, self.awi.n_steps - 1),
        )
        trange = (
            min(
                trange[0],
                max(
                    self.awi.current_input_issues[TIME].min_value,
                    self.awi.current_step + 1,
                ),
            ),
            max(
                trange[1],
                min(
                    self.awi.current_input_issues[TIME].max_value, self.awi.n_steps - 1
                ),
            ),
        )
        target_supplies, target_sales = dict(), dict()
        for t in range(trange[0], trange[1] + 1):
            secured_supplies = (
                self.awi.total_supplies_until(t)
                + self.awi.current_inventory_input
                + self.awi.current_inventory_output
            )
            secured_sales = self.awi.total_sales_from(t)
            secured_supplies += (
                self.awi.current_exogenous_input_quantity * (t - current_step)
                if self.awi.is_first_level
                else 0
            )
            secured_sales += (
                self.awi.current_exogenous_output_quantity * (n_steps - t)
                if self.awi.is_last_level
                else 0
            )
            secured_supplies = max(
                self.future_productivity * (t - current_step), secured_supplies
            )
            secured_sales = max(self.future_productivity * (n_steps - t), secured_sales)
            target_supplies[t] = secured_sales - secured_supplies
            target_sales[t] = secured_supplies - secured_sales
            if self.awi.is_first_level:
                target_supplies[t] = 0
            elif self.awi.is_last_level:
                target_sales[t] = 0

        target_supplies = {k: int(v) for k, v in target_supplies.items() if v > 0}
        target_sales = {k: int(v) for k, v in target_sales.items() if v > 0}
        return target_supplies, target_sales

    def distribute_future_offers(
        self, partners: list[str]
    ) -> dict[str, Outcome | None]:
        """Distribute future offers over the given partners"""
        if not partners:
            return dict()

        c = self.awi.current_step
        n = max((self.awi.n_steps - c), 1)
        # get minimum and maximum price and quantity according to current
        # negotiations.
        # - For prices make sure that the limits do not lead to loss
        # - We assume here that trading prices are not going to change much
        # - We know according to the rules that the range of quantities is the
        #   same every day
        mxoutp = self.awi.current_output_issues[UNIT_PRICE].max_value
        mnoutp = max(
            self.awi.current_output_issues[UNIT_PRICE].min_value,
            self.awi.current_input_issues[UNIT_PRICE].max_value,
        )
        mninp = self.awi.current_input_issues[UNIT_PRICE].min_value
        mxinp = min(
            self.awi.current_input_issues[UNIT_PRICE].max_value,
            self.awi.current_output_issues[UNIT_PRICE].min_value,
        )
        mxinq = self.awi.current_input_issues[QUANTITY].max_value
        mxoutq = self.awi.current_output_issues[QUANTITY].max_value

        # estimate needed supplies up  to and sales starting from each
        # time-step in the future
        needed_supplies, needed_sales = self.estimate_future_needs()
        # Separate suppliers and consumers
        suppliers = [_ for _ in partners if self.is_supplier(_)]
        consumers = [_ for _ in partners if self.is_consumer(_)]
        # prioritize which time to try to satisfy first
        if self.near or self.far:
            if needed_supplies:
                shffl = sorted(needed_supplies.keys(), reverse=self.far)
                needed_supplies = {k: needed_supplies[k] for k in shffl}
            if needed_sales:
                shffl = sorted(needed_sales.keys(), reverse=self.far)
                needed_sales = {k: needed_sales[k] for k in shffl}
        else:
            if needed_supplies:
                shffl = list(needed_supplies.keys())
                random.shuffle(shffl)
                needed_supplies = {k: needed_supplies[k] for k in shffl}
            if needed_sales:
                shffl = list(needed_sales.keys())
                random.shuffle(shffl)
                needed_sales = {k: needed_sales[k] for k in shffl}
        # initialize indicating that I do not need anything
        unneeded = (
            None if not self.awi.allow_zero_quantity else (0, self.awi.current_step, 0)
        )
        offers = dict(zip(partners, itertools.repeat(unneeded)))
        # loop over suppliers and consumers
        for plist, needs, mnp, mxp, mxq, price in (
            (
                suppliers,
                needed_supplies,
                mninp,
                mxinp,
                mxinq,
                self.buy_price,
            ),
            (
                consumers,
                needed_sales,
                mnoutp,
                mxoutp,
                mxoutq,
                self.sell_price,
            ),
        ):
            # if there are no good prices, just do nothing
            if mnp > mxp:
                continue
            # try to satisfy my future needs in order
            for t, q in needs.items():
                # if I have no partners, do nothing
                if not plist:
                    continue
                # distribute the needs over the partners
                d = distribute(
                    int(q),
                    len(plist),
                    mx=mxq,
                    concentrated=random.random() < self.future_concentration,
                    equal=random.random() > 0.5,
                    allow_zero=self.awi.allow_zero_quantity,
                )
                # find relative time to the end of simulation to estimate good prices
                # Notice that nearer times will entail higher concessions
                r = 1 - max(0, min(1, (t - c) / n))
                offers |= {
                    plist[i]: (q, t, price(r, mnp, mxp, today=t == c))
                    for i, q in enumerate(d)
                    if q > 0
                }
                plist = list(set(plist).difference(offers.keys()))
        return offers

    def is_supplier(self, negotiator_id):
        return negotiator_id in self.awi.my_suppliers

    def is_consumer(self, negotiator_id):
        return negotiator_id in self.awi.my_consumers

    def best_price(self, partner_id):
        """Best price for a negotiation today"""
        issue = self.get_nmi(partner_id).issues[UNIT_PRICE]
        return issue.max_value if self.is_consumer(partner_id) else issue.min_value

    def good_price(self, partner_id, today: bool):
        """A good price to use"""
        nmi = self.get_nmi(partner_id)
        mn = nmi.issues[UNIT_PRICE].min_value
        mx = nmi.issues[UNIT_PRICE].max_value
        if self.is_supplier(partner_id):
            return self.buy_price(nmi.state.relative_time, mn, mx, today=today)
        return self.sell_price(
            self.get_nmi(partner_id).state.relative_time, mn, mx, today=today
        )

    def buy_price(self, t: float, mn: float, mx: float, today: bool) -> float:
        """Return a good price to buy at"""
        e = self.today_exp if today else self.future_exp
        return max(mn, min(mx, int(mn + (mx - mn) * (t**e) + 0.5)))

    def sell_price(self, t: float, mn: float, mx: float, today: bool) -> float:
        """Return a good price to sell at"""
        e = self.today_exp if today else self.future_exp
        if not today:
            mn = mn + self.fmin * (mx - mn)
        return max(mn, min(mx, int(0.5 + mx - (mx - mn) * (t**e))))

    def good2buy(self, p: float, t: float, mn, mx, today: bool):
        """Is p a good price to buy at?"""
        if not today:
            mx = mx - self.fmin * (mx - mn)
        return p - 0.0001 <= self.buy_price(t, mn, mx, today)

    def good2sell(self, p: float, t: float, mn, mx, today: bool):
        """Is p a good price to sell at?"""
        return p + 0.0001 >= self.sell_price(t, mn, mx, today)
