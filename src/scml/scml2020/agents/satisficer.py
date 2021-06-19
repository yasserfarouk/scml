# required for typing
from collections import defaultdict
import random
import math
from typing import Any, Dict, List, Optional

import numpy as np
from negmas import (
    AgentMechanismInterface,
    Contract,
    Issue,
    Outcome,
    MechanismState,
    Negotiator,
    ResponseType,
)
from negmas.sao import SAONegotiator, SAOState, SAOAMI

from scml.scml2020 import SCML2020Agent, AWI
from scml.scml2020 import TIME, UNIT_PRICE, QUANTITY, NO_COMMAND

__all__ = ["SatisficerAgent"]


class ObedientNegotiator(SAONegotiator):
    """
    A negotiator that controls a single negotiation with a single partner.

    Args:

        selling: Whether this negotiator is engaged on selling or buying
        requested: Whether this negotiator is created to manage a negotiation
                   requested by its owner (as opposed to one that is created
                   to respond to one created by the partner).

    Remarks:

        - This negotiator does nothing. It just passes calls to its owner.

    """

    def __init__(self, *args, selling, requested, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_selling = selling
        self.is_requested = requested

    # =====================
    # Negotiation Callbacks
    # =====================

    def propose(self, state: MechanismState) -> Optional[Outcome]:
        """Simply calls the corresponding method on the owner"""
        return self.owner.propose(state, self.ami, self.is_selling, self.is_requested)

    def respond(self, state: MechanismState, offer: Outcome) -> ResponseType:
        """Simply calls the corresponding method on the owner"""
        return self.owner.respond(
            state, self.ami, offer, self.is_selling, self.is_requested
        )


class SatisficerAgent(SCML2020Agent, SAONegotiator):
    """
    A simple monolithic agent that tries to *carefully* make small profit
    every step.

    Args:

        target_productivity: The productivity level targeted by the agent defined
                             as the fraction of its lines to be active per step.
        target_profit: A profit amount considered enough. Once the agent achieves
                       this level of profit, it will just stop trading.
        satisfying_profit: A profit amount considered satisfactory. Used when
                           deciding negotiation agenda and signing to decide if
                           a price is a good price (see `_good_price()`). A
                           fraction of the trading price.
        acceptable_loss: A fraction of trading price that the seller/buyer is
                         willing to go under/over the current trading price during
                         negotiation.
        horizon: The number of time-steps in the future to consider in negotiations.
        price_range: The total range around the trading price for negotiation agendas.
        exponent: The exponent of the consession curve used during negotiation.
    """

    def __init__(
        self,
        *args,
        target_productivity=1.0,
        target_profit=0.2,
        satisfying_profit=0.15,
        acceptable_loss=0.05,
        horizon=5,
        price_range=0.4,
        exponent=1.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.target_profit = target_profit
        self.satisfying_profit = satisfying_profit
        self.target_productivity = target_productivity
        self.__end_all = False
        self.target_sales = self.target_supplies = 0
        self.max_sales, self.max_supplies = None, None
        self.min_sales, self.min_supplies = None, None
        self.tentative_sales = self.tentative_supplies = 0
        self.horizon = horizon
        self.price_range = price_range
        self.e = exponent
        self.acceptable_loss = acceptable_loss
        self.last_q = defaultdict(int)
        self.last_t = dict()

    # =====================
    # Negotiation functions
    # =====================

    def respond(
        self,
        state: SAOState,
        ami: SAOAMI,
        offer: Outcome,
        is_selling: bool,
        is_requested: bool,
    ):
        """
        Responds to an offer from one partner.

        Args:
            state: mechanism state including current round
            ami: Agent-mechanism-interface for accessing the negotiation mechanism
            offer: The offer proposed by the partner
            is_selling: Whether the agent is selling to this partner
            is_requested: Whether the agent requested this negotiation

        Remarks:

            - The main idea is to accept offers that are within the quantity limits
              for the delivery day if its price is good enough for the current stage
              of the negotiation.
            - During negotiation, the agent starts accepting highest/lowest prices
              for selling/buying and gradually conceeds to the minimally acceptable
              price (`good_price`) defined as being `acceptable_loss` above/below
              the trading price for buying/selling.

        """
        # Find the price range for this negotiation
        p0, p1 = ami.issues[UNIT_PRICE].min_value, ami.issues[UNIT_PRICE].max_value

        # Find current trading prices (catalog price if trading prices are not available)
        awi: AWI = self.awi
        s = awi.current_step
        prices = awi.trading_prices
        if prices is None:
            prices = awi.catalog_prices

        # Find the last offer we sent to this partner
        partner = [_ for _ in ami.agent_ids if _ != self.id][0]
        t = self.last_t.get(partner, None)

        # read limits of quantities and our tentative offers based on whether
        # we are selling/buygin
        if is_selling:
            tentative, mx, mn = (
                self.tentative_sales,
                self.max_sales,
                self.min_sales,
            )
            good_price = prices[awi.my_output_product] * (1 - self.acceptable_loss)
        else:
            tentative, mx, mn = (
                self.tentative_supplies,
                self.max_supplies,
                self.min_supplies,
            )
            good_price = prices[awi.my_input_product] * (1 + self.acceptable_loss)

        # remove the last offer we sent to the partner from the tentative list
        # because it is implicitly rejected by the partner.
        if t is not None:
            if is_selling:
                self.last_q[partner] = 0
                tentative[t:] -= self.last_q[partner]
            else:
                self.last_q[partner] = 0
                tentative[s:t] -= self.last_q[partner]

        # If the agent has already decided to stop trading, end the ngotiation.
        if self.__end_all:
            return ResponseType.END_NEGOTIATION

        # if it is not possible to get a good price in this negotiation, just end.
        if not is_requested and (
            (is_selling and p1 < good_price) or (not is_selling and p0 > good_price)
        ):
            return ResponseType.END_NEGOTIATION

        # parse the offer
        q, u, t = (
            offer[QUANTITY],
            offer[UNIT_PRICE],
            offer[TIME],
        )

        # r will go from one to zero over the negotiation time and controls our
        # concession
        r = math.pow(1 - state.step / ami.n_steps, self.e)

        if is_selling:
            # If selling we conceed down from the highest price
            p = (p1 - good_price) * r + good_price
        else:
            # If buing we conceed up from the lowest price
            p = (good_price - p0) * r + good_price

        # if the quantity offers is not within the range we want for this time-step
        # reject the offer
        if q > mx[t] or q < mn[t]:
            return ResponseType.REJECT_OFFER

        # if price is OK accept the offer
        if (is_selling and u >= p) or (not is_selling and u <= p):
            return ResponseType.ACCEPT_OFFER
        # otherwise, reject it
        return ResponseType.REJECT_OFFER

    def propose(
        self, state: SAOState, ami: SAOAMI, is_selling: bool, is_requested: bool
    ):
        """
        Used to propose to the opponent

        Args:
            state: mechanism state including current round
            ami: Agent-mechanism-interface for accessing the negotiation mechanism
            offer: The offer proposed by the partner
            is_selling: Whether the agent is selling to this partner
            is_requested: Whether the agent requested this negotiation
        """
        if self.__end_all:
            return None
        awi: AWI = self.awi
        s = awi.current_step
        prices = awi.trading_prices
        if prices is None:
            prices = awi.catalog_prices

        if is_selling:
            tentative, mx, mn = (
                self.tentative_sales,
                self.max_sales,
                self.min_sales,
            )
            good_price = prices[awi.my_output_product] * 0.9
        else:
            tentative, mx, mn = (
                self.tentative_supplies,
                self.max_supplies,
                self.min_supplies,
            )
            good_price = prices[awi.my_input_product] * 1.1

        t0, t1 = ami.issues[TIME].min_value, ami.issues[TIME].max_value
        q0, q1 = ami.issues[QUANTITY].min_value, ami.issues[QUANTITY].max_value
        p0, p1 = ami.issues[UNIT_PRICE].min_value, ami.issues[UNIT_PRICE].max_value

        r = math.pow(1 - state.relative_time, self.e)

        if is_selling:
            p = (p1 - good_price) * r + good_price
        else:
            p = (good_price - p0) * r + good_price

        mx, mn = mx[t0 : t1 + 1] - tentative[t0 : t1 + 1], mn[t0 : t1 + 1]
        mx[mx > q1] = q1
        mn[mn < q0] = q0
        options = []

        for t in range(t0, t1 + 1):
            if mx[t - t0] >= mn[t - t0]:
                for _ in range(mn[t - t0], mx[t - t0] + 1):
                    options.append((t, _))

        if len(options) < 1:
            return None

        t, q = random.choice(options)
        offer = [0, 0, 0]
        offer[TIME], offer[QUANTITY], offer[UNIT_PRICE] = t, int(q), int(p + 0.5)

        partner = [_ for _ in ami.agent_ids if _ != self.id][0]

        self.last_q[partner] = q
        self.last_t[partner] = t

        if is_selling:
            tentative[t:] += q
        else:
            tentative[s:t] += q

        return tuple(offer)

    # =====================
    # Time-Driven Callbacks
    # =====================
    def init(self):
        """Called once"""
        awi: AWI = self.awi
        self.initial_balance = awi.current_balance
        self.horizon = min(
            awi.settings.get("exogenous_horizon", self.horizon),
            min(awi.n_steps, self.horizon),
        )
        self.production_cost = float(awi.profile.costs[:, awi.my_input_product].max())
        self.secured_supplies = np.zeros(awi.n_steps, dtype=int)
        self.secured_sales = np.zeros(awi.n_steps, dtype=int)

    def before_step(self):
        """Called at at the BEGINNING of every production step (day)"""
        awi: AWI = self.awi
        s = awi.current_step
        steps, products, lines, level = (
            awi.n_steps,
            awi.n_products,
            awi.n_lines,
            awi.my_input_product,
        )

        available_input = awi.current_inventory[awi.my_input_product]
        available_output = awi.current_inventory[awi.my_output_product]

        # find the time of tirst and last allowed sale and supply
        first_sale, last_sale = min(s + 1, level), steps - 1
        first_supply, last_supply = max(s + 1, level), steps - level - 1

        sale_period = last_sale - first_sale + 1
        supply_period = last_supply - first_supply + 1

        # find total target sales and supplies to the end of the simulation.
        # I need to sell everything in my inventory but buy only what is not
        # already in it.

        self.target_sales = (
            int(lines * (sale_period) * self.target_productivity)
            + available_output
            + available_input
        )
        self.target_supplies = self.target_sales - available_input

        self.max_sales = np.zeros(steps, dtype=np.int64)
        self.max_supplies = np.zeros(steps, dtype=np.int64)

        if sale_period > 0:
            self.max_sales[first_sale : last_sale + 1] = (
                self.target_sales // sale_period
            )
            # max_sales[s] is the maximum allowed TO this step
            self.max_sales[first_sale : last_sale + 1] = (
                self.max_sales[first_sale : last_sale + 1].cumsum()
                - self.secured_sales[first_sale : last_sale + 1]
            )
            # I need to also sell all available items anytime
            self.max_sales[first_sale : last_sale + 1] += available_output
            # TODO check that I need one step to produce and then sell
            self.max_sales[first_sale + 1 : last_sale + 1] += available_input

        if supply_period > 0:
            self.max_supplies[first_supply : last_supply + 1] = (
                self.target_supplies // supply_period
            )
            # max_supplies[s] is the maximum allowed FROM this step
            self.max_supplies[first_supply : last_supply + 1] = (
                self.max_supplies[first_supply : last_supply + 1].cumsum()[::-1]
                - self.secured_supplies[first_supply : last_supply + 1]
            )

        # TODO use minimums to make sure that I sell everything at the end and I get all my needs for production
        # items here should be sold/bought at any price or at least with some margin of loss
        self.min_sales = np.zeros(steps, dtype=np.int64)
        self.min_supplies = np.zeros(steps, dtype=np.int64)

        self.tentative_sales = np.zeros(steps, dtype=np.int64)
        self.tentative_supplies = np.zeros(steps, dtype=np.int64)

    def do_production(self):
        commands = NO_COMMAND * np.ones(self.awi.n_lines, dtype=int)
        inputs = min(self.awi.state.inventory[self.awi.my_input_product], len(commands))
        commands[:inputs] = self.awi.my_input_product
        commands[inputs:] = NO_COMMAND
        self.awi.set_commands(commands)

    def step(self):
        """Called at the end of the day. Will request all negotiations"""
        awi: AWI = self.awi
        s = awi.current_step
        prices = awi.trading_prices
        if prices is None:
            prices = awi.catalog_prices

        profit = (
            awi.current_balance
            + awi.current_inventory[awi.my_input_product] * prices[awi.my_input_product]
            + awi.current_inventory[awi.my_output_product]
            * prices[awi.my_output_product]
            - self.initial_balance
        ) / self.initial_balance

        if profit > self.target_profit:
            self.__end_all = True
            return

        self.do_production()

        consumers = awi.my_consumers
        suppliers = awi.my_suppliers
        qrange = (1, int(awi.n_lines))
        trange = (s, min(s + self.horizon, awi.n_steps - 1))
        # # if output trading price is too low, just do not even try
        # if (
        #     prices[awi.my_output_product]
        #     < prices[awi.my_output_product] + self.production_cost
        # ):
        #     return
        prices[awi.my_output_product] = int(
            max(
                prices[awi.my_output_product],
                (1 + self.satisfying_profit)
                * (prices[awi.my_input_product] + self.production_cost),
            )
        )
        dp = prices[awi.my_input_product] * self.price_range / 2.0
        urange = (
            int(prices[awi.my_input_product] - dp),
            int(prices[awi.my_input_product] + dp + 0.5),
        )
        awi.request_negotiations(
            True,
            awi.my_input_product,
            qrange,
            urange,
            trange,
            None,
            [ObedientNegotiator(selling=False, requested=True) for _ in suppliers],
            suppliers,
        )
        urange = (
            int(prices[awi.my_output_product] - dp),
            int(prices[awi.my_output_product] + dp + 0.5),
        )
        awi.request_negotiations(
            False,
            awi.my_output_product,
            qrange,
            urange,
            trange,
            None,
            [ObedientNegotiator(selling=True, requested=True) for _ in consumers],
            consumers,
        )

        # remove all secured amounts for this time-step from expectation of the future
        self.secured_sales[s:] -= self.secured_sales[s]
        self.secured_supplies[s:] -= self.secured_supplies[s]

    # ================================
    # Negotiation Control and Feedback
    # ================================

    def _need_to_negotiate(self, is_selling: bool, issues: List[Issue]) -> bool:
        # TODO check limits for buying and selling
        return not self.__end_all

    def respond_to_negotiation_request(
        self,
        initiator: str,
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
    ) -> Optional[Negotiator]:
        """Called whenever an agent requests a negotiation with you.
        Return either a negotiator to accept or None (default) to reject it"""
        is_selling = annotation["seller"] == self.id
        if not self._need_to_negotiate(is_selling, issues):
            return None
        return ObedientNegotiator(selling=is_selling, requested=False)

    # =============================
    # Contract Control and Feedback
    # =============================

    def _is_good_price(self, is_selling: bool, u: float, slack: float = 0.0):
        awi: AWI = self.awi
        prices = awi.trading_prices
        if prices is None:
            prices = awi.catalog_prices
        if is_selling:
            return u > (1 + self.satisfying_profit - slack) * (
                prices[awi.my_input_product] + self.production_cost
            )
        return u < (1 - self.satisfying_profit + slack) * (
            prices[awi.my_output_product] - self.production_cost
        )

    def sign_all_contracts(self, contracts: List[Contract]) -> List[Optional[str]]:
        """Called to ask you to sign all contracts that were concluded in
        one step (day)"""
        signatures: List[Optional[str]] = [None] * len(contracts)
        awi: AWI = self.awi
        sell_contracts = sorted(
            [
                (i, _)
                for i, _ in enumerate(contracts)
                if _.annotation["seller"] == self.id
            ],
            key=lambda x: -x[1].agreement["unit_price"],
        )
        buy_contracts = sorted(
            [
                (i, _)
                for i, _ in enumerate(contracts)
                if _.annotation["seller"] != self.id
            ],
            key=lambda x: x[1].agreement["unit_price"],
        )
        bought, sold = np.zeros(awi.n_steps), np.zeros(awi.n_steps)
        total_bought = total_sold = 0
        for i, c in buy_contracts:
            # If I already signed above my total needs, do not sign any more.
            if total_bought >= self.target_supplies:
                break
            q, u, t = (
                c.agreement["quantity"],
                c.agreement["unit_price"],
                c.agreement["time"],
            )
            # If I already signed above my total needs FOR THE DAY, do not sign any more.
            if bought[t] >= self.max_supplies[t]:
                break
            # End if prices go too high
            if not self._is_good_price(False, u, 0.0):
                break
            signatures[i] = self.id
            bought[t] += q

        for i, c in sell_contracts:
            # If I already signed above my total needs, do not sign any more.
            if total_sold >= self.target_supplies:
                break
            q, u, t = (
                c.agreement["quantity"],
                c.agreement["unit_price"],
                c.agreement["time"],
            )
            # If I already signed above my total needs FOR THE DAY, do not sign any more.
            if sold[t] >= self.max_supplies[t]:
                break
            # End if prices go too high
            if not self._is_good_price(True, u, 0.0):
                break
            signatures[i] = self.id
            sold[t] += q

        return signatures

    def on_contracts_finalized(
        self,
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        """Called to inform you about the final status of all contracts in
        a step (day)"""
        awi: AWI = self.awi
        s = awi.current_step
        sell_contracts = [
            (i, _) for i, _ in enumerate(signed) if _.annotation["seller"] == self.id
        ]
        buy_contracts = [
            (i, _) for i, _ in enumerate(signed) if _.annotation["seller"] != self.id
        ]

        for i, c in buy_contracts:
            # If I already signed above my total needs, do not sign any more.
            q, u, t = (
                c.agreement["quantity"],
                c.agreement["unit_price"],
                c.agreement["time"],
            )
            self.secured_supplies[t] += q

        for i, c in sell_contracts:
            # If I already signed above my total needs, do not sign any more.
            q, u, t = (
                c.agreement["quantity"],
                c.agreement["unit_price"],
                c.agreement["time"],
            )
            self.secured_sales[t:] += q

    # ====================
    # Production Callbacks
    # ====================

    def confirm_production(
        self, commands: np.ndarray, balance: int, inventory: np.ndarray
    ) -> np.ndarray:
        """
        Called just before production starts at every step allowing the
        agent to change what is to be produced in its factory on that step.
        """
        return commands

    # ==========================
    # Callback about Bankruptcy
    # ==========================

    def on_agent_bankrupt(
        self,
        agent: str,
        contracts: List[Contract],
        quantities: int,
        compensation_money: int,
    ) -> None:
        """Called whenever any agent goes bankrupt. It informs you about changes
        in future contracts you have with you (if any)."""

    # ========================
    # Callbacks we do not need
    # ========================

    def on_negotiation_failure(
        self,
        partners: List[str],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
        state: MechanismState,
    ) -> None:
        """Called when a negotiation the agent is a party of ends without
        agreement"""

    def on_negotiation_success(
        self, contract: Contract, mechanism: AgentMechanismInterface
    ) -> None:
        """Called when a negotiation the agent is a party of ends with
        agreement"""
