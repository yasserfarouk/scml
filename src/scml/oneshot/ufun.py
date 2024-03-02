from __future__ import annotations

from collections import namedtuple
from functools import cache
from typing import Iterable, Literal, overload

from attr import define
from negmas import Contract
from negmas.outcomes import Issue, Outcome, OutcomeSpace, make_issue, make_os
from negmas.preferences import StationaryMixin, UtilityFunction

from .common import QUANTITY, TIME, UNIT_PRICE, is_system_agent

__all__ = ["OneShotUFun", "UFunLimit", "UtilityInfo"]

UFunLimit = namedtuple(
    "UFunLimit",
    [
        "utility",
        "input_quantity",
        "input_price",
        "output_quantity",
        "output_price",
        "exogenous_input_quantity",
        "exogenous_input_price",
        "exogenous_output_quantity",
        "exogenous_output_price",
        "inventory_input",
        "inventory_output",
        "producible",
    ],
)
"""Information about one utility limit (either highest or lowest). See `OnShotUFun.find_limit` for details."""


@define
class UtilityInfo:
    producible: int
    total_input: int
    total_output: int
    shortfall_quantity: int
    shortfall_penalty: float
    remaining_quantity: int
    disposal_cost: float
    storage_cost: float
    utility: float


class OneShotUFun(StationaryMixin, UtilityFunction):  # type: ignore
    """
    Calculates the utility function of a list of contracts or offers.

    Args:
        force_exogenous: Is the agent forced to accept exogenous contracts
                         given through `ex_*` arguments?
        ex_pin: total price of exogenous inputs for this agent
        ex_qin: total quantity of exogenous inputs for this agent
        ex_pout: total price of exogenous outputs for this agent
        ex_qout: total quantity of exogenous outputs for this agent.
        cost: production cost of the agent.
        disposal_cost: disposal cost per unit of input/output.
        shortfall_penalty: penalty for failure to deliver one unit of output.
        input_agent: Is the agent an input agent which means that its input
                     product is the raw material
        output_agent: Is the agent an output agent which means that its output
                      product is the final product
        n_lines: Number of production lines. If None, will be read through the AWI.
        input_product: Index of the input product. If None, will be read through
                       the AWI
        input_qrange: A 2-int tuple giving the range of input quantities negotiated.
                      If not given will be read through the AWI
        input_prange: A 2-int tuple giving the range of input unit prices negotiated.
                      If not given will be read through the AWI
        output_qrange: A 2-int tuple giving the range of output quantities negotiated.
                      If not given will be read through the AWI
        output_prange: A 2-int tuple giving the range of output unit prices negotiated.
                      If not given will be read through the AWI
        n_input_negs: How many input negotiations are allowed. If not given, it
                      will be the number of suppliers as given by the AWI
        n_output_negs: How many output negotiations are allowed. If not given, it
                      will be the number of consumers as given by the AWI
        current_step: Current simulation step. Needed only for `ufun_range`
                      when returning best outcomes
        normalized: If given the values returned by `from_*`, `utility_range`
                    and `__call__` will all be normalized between zero and one.

    Remarks:
        - The utility function assumes that the agent will have to pay for
          all its input products but will receive money only for the output
          products it could generate and sell.
        - The utility function respects production capacity (n. lines). The
          agent cannot produce more than the number of lines it has.
        - disposal cost is paid for items bought but not produced only. Items
          consumed in production (i.e. sold) are not counted.
    """

    def __init__(
        self,
        ex_pin: int,
        ex_qin: int,
        ex_pout: int,
        ex_qout: int,
        input_product: int,
        input_agent: bool,
        output_agent: bool,
        production_cost: float,
        disposal_cost: float,
        storage_cost: float,
        shortfall_penalty: float,
        input_penalty_scale: float | None,
        output_penalty_scale: float | None,
        storage_penalty_scale: float | None,
        n_input_negs: int,
        n_output_negs: int,
        current_step: int,
        agent_id: str | None,
        time_range: tuple[int, int],
        inventory_in: int = 0,
        inventory_out: int = 0,
        input_qrange: tuple[int, int] = (0, 0),
        input_prange: tuple[int, int] = (0, 0),
        output_qrange: tuple[int, int] = (0, 0),
        output_prange: tuple[int, int] = (0, 0),
        force_exogenous: bool = True,
        n_lines: int = 10,
        normalized: bool = False,
        current_balance: int | float = float("inf"),
        suppliers: set[str] = set(),
        consumers: set[str] = set(),
        perishable=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.agent_id = agent_id
        self.time_range = time_range
        self._best, self._worst = None, None
        self.suppliers = suppliers
        self.consumers = consumers
        self.current_balance = current_balance
        self.normalized = normalized
        self.input_penalty_scale = input_penalty_scale
        self.storage_penalty_scale = storage_penalty_scale
        self.output_penalty_scale = output_penalty_scale
        self.current_step = current_step
        self.ex_pin, self.ex_pout = ex_pin, ex_pout
        self.ex_qin, self.ex_qout = ex_qin, ex_qout
        self.inventory_in = inventory_in
        self.inventory_out = inventory_out
        self.n_input_negs = n_input_negs
        self.perishable = perishable
        self.n_output_negs = n_output_negs
        self.input_qrange, self.input_prange = input_qrange, input_prange
        self.output_qrange, self.output_prange = output_qrange, output_prange
        if production_cost is None:
            production_cost = 0.0
        if disposal_cost is None:
            disposal_cost = 0.0
        if storage_cost is None:
            storage_cost = 0.0
        if shortfall_penalty is None:
            shortfall_penalty = 0.0
        (
            self.production_cost,
            self.disposal_cost,
            self.storage_cost,
            self.shortfall_penalty,
        ) = (
            production_cost,
            disposal_cost,
            storage_cost,
            shortfall_penalty,
        )
        self.input_agent, self.output_agent = input_agent, output_agent
        self.force_exogenous = force_exogenous
        if not force_exogenous:
            self.ex_pin = self.ex_qin = self.ex_pout = self.ex_qout = 0
        self.n_lines = n_lines
        if input_product is None and input_agent:
            input_product = 0
        self.input_product = input_product
        if self.input_product is not None:
            self.output_product = self.input_product + 1
        else:
            self.output_product = None
        if self.normalized:
            self._best = self.find_limit(True, None, None)
            self._worst = self.find_limit(False, None, None)
        else:
            self._best = UFunLimit(*tuple([None] * 12))
            self._worst = UFunLimit(*tuple([None] * 12))
        if self.input_agent or self.output_agent:
            # if this is an edge agent, all negotiations will be on the same product so we can define its outcome-space
            qrange = self.input_qrange if self.output_agent else self.output_qrange
            prange = self.input_prange if self.output_agent else self.output_prange
            self.outcome_space = make_os(
                [
                    make_issue(qrange, name="quantity"),
                    make_issue(time_range, name="time"),
                    make_issue(prange, name="unit_price"),
                ]
            )
        else:
            # if this is not an edge agent, we have a different outcome space for each side
            self.outcome_spaces = [
                make_os(
                    [
                        make_issue(qrange, name="quantity"),
                        make_issue(time_range, name="time"),
                        make_issue(prange, name="unit_price"),
                    ]
                )
                for qrange, prange in (
                    (self.input_qrange, self.input_prange),
                    (self.output_qrange, self.output_qrange),
                )
            ]
        # slightly bias toward agreements
        self.reserved_value = self.from_contracts([], ignore_exogenous=False) - 1e-3
        self._signed_agreements: list[tuple[int, int, int]] = []
        self._signed_is_output: list[bool] = []
        self._registered_sale_failures: set[str] = set()
        self._registered_supply_failures: set[str] = set()
        if perishable:
            assert (
                self.storage_cost == 0
            ), f"Perishable ufun but {self.storage_cost=} ({self.disposal_cost})"
        else:
            assert (
                self.disposal_cost == 0
            ), f"Non-perishable ufun but {self.disposal_cost=} ({self.storage_cost})"

    @property
    def best_option(self) -> UFunLimit:
        """Best possible options"""
        if self._best is None:
            self._best = self.find_limit(True)
        return self._best

    @property
    def worst_option(self) -> UFunLimit:
        """Best possible options"""
        if self._worst is None:
            self._worst = self.find_limit(False)
        return self._worst

    def register_supply_failure(self, supplier_id: str):
        self.find_limit_brute_force.cache_clear()
        self._registered_supply_failures.add(supplier_id)

    def register_sale_failure(self, consumer_id: str):
        self.find_limit_brute_force.cache_clear()
        self._registered_sale_failures.add(consumer_id)

    def register_sale(self, q: int, p: int, t: int):
        """Registers a sale to be considered when calculating utilities"""
        if t != self.current_step:
            return
        self.find_limit_brute_force.cache_clear()
        self._signed_agreements.append((q, t if t >= 0 else self.current_step, p))
        self._signed_is_output.append(True)

    def register_supply(self, q: int, p: int, t: int):
        """Registers a supply to be considered when calculating utilities"""
        if t != self.current_step:
            return
        self.find_limit_brute_force.cache_clear()
        self._signed_agreements.append((q, t if t >= 0 else self.current_step, p))
        self._signed_is_output.append(False)

    def xml(self, issues) -> str:
        raise NotImplementedError(f"Cannot convert the ufun to xml: {issues}")

    def eval(self, offer: tuple[int, int, int] | None) -> float:  # type: ignore
        """
        Calculates the utility function given a single contract.

        Remarks:
            - This method calculates the utility value of a single offer assuming all other negotiations end in failure.
            - It can only be called for agents that exist in the first or last layer of the production graph.
        """
        if not self.input_agent and not self.output_agent:
            return float("-inf")
        if offer is not None:
            offer = tuple(offer)  # type: ignore
        return self.from_offers((offer if offer else None,), (self.input_agent,))

    @overload
    def from_contracts(
        self,
        contracts: Iterable[Contract],
        return_info: Literal[False] = False,
        ignore_exogenous=True,
    ) -> float:
        ...

    @overload
    def from_contracts(
        self,
        contracts: Iterable[Contract],
        return_info: Literal[True],
        ignore_exogenous=True,
    ) -> UtilityInfo:
        ...

    def from_contracts(
        self,
        contracts: Iterable[Contract],
        return_info: bool = False,
        ignore_exogenous=True,
    ) -> float | UtilityInfo:
        """
        Calculates the utility function given a list of contracts

        Args:
            contracts: A list/tuple of contracts
            ignore_exogenous: If given, any contracts with a system agent will
                              be ignored.

        Remarks:
            - This method ignores any unsigned contracts passed to it.
            - We do not consider time at all so it is implicitly assumed that
              all contracts have the same delivery time value.
            - The reason for having the `ignore_exogenous` parameter is to avoid
              double counting exogenous contracts if their information is passed
              during construction of the ufun and they also exist in the list of
              `contracts` passed here.
        """
        offers, outputs = [], []
        output_product = self.output_product
        for c in contracts:
            if c.signed_at < 0:
                continue
            if c.nullified_at >= 0:
                continue
            if ignore_exogenous and any(is_system_agent(_) for _ in c.partners):
                continue
            product = c.annotation["product"]
            is_output = product == output_product
            assert (
                c.annotation["buyer"] != c.annotation["seller"]
            ), f"{self.agent_id=}: Buyer == Seller == {c.annotation['buyer']}"
            assert (is_output and c.annotation["buyer"] != self.agent_id) or (
                not is_output and c.annotation["seller"] != self.agent_id
            ), (
                f"{self.agent_id=}: Got contract in which I am either buying "
                f"my output product or selling my input product: {self.input_product=}"
                f", {self.output_product=}\n{c}"
            )
            outputs.append(is_output)
            offers.append(self.outcome_as_tuple(c.agreement))
        return self.from_offers(  # type: ignore
            tuple(offers),
            tuple(outputs),
            return_info=return_info,  # type: ignore
        )

    @staticmethod
    def outcome_as_tuple(offer):
        if isinstance(offer, dict):
            outcome = [None] * 3
            outcome[QUANTITY] = offer["quantity"]
            outcome[TIME] = offer["time"]
            outcome[UNIT_PRICE] = offer["unit_price"]
            return tuple(outcome)
        return tuple(offer)

    @overload
    def from_offers(
        self,
        offers: tuple[tuple[int, int, int | float] | None, ...]
        | dict[str, tuple[int, int, int] | None],
        outputs: tuple[bool, ...] | None = None,
        return_info: Literal[False] = False,
        ignore_signed_contracts: bool = True,
    ) -> float:
        ...

    @overload
    def from_offers(
        self,
        offers: tuple[tuple[int, int, int | float] | None, ...]
        | dict[str, tuple[int, int, int] | None],
        outputs: tuple[bool, ...] | None,
        return_info: Literal[True],
        ignore_signed_contracts: bool = True,
    ) -> UtilityInfo:
        ...

    def from_offers(
        self,
        offers: tuple[tuple[int, int, int | float] | None, ...]
        | dict[str, tuple[int, int, int] | None],
        outputs: tuple[bool, ...] | None = None,
        return_info: bool = False,
        ignore_signed_contracts: bool = True,
    ) -> float | UtilityInfo:
        """
        Calculates the utility value given a list of offers and whether each
        offer is for output or not (= input).

        Args:
            offers: An iterable (e.g. list) of tuples each with three values:
                    (quantity, time, unit price) IN THAT ORDER. Time is ignored
                    and can be set to any value.
            outputs: An iterable of the same length as offers of booleans
                     specifying for each offer whether it is an offer for buying
                     the agent's output product.
            return_info: If true, detailed utility information is returned as Utility Info
            ignore_signed_contracts: If true, ignores the registered signed contracts.
                                     This means that only exogenous contracts and offers
                                     will be used in evaluating the utility.
        Remarks:
            - This method takes into account the exogenous contract information
              passed when constructing the ufun.
            - You can pass a dictionary mapping partner ID to an offer and the system
              will use the correct value for the corresponding outputs array.
        """
        if isinstance(offers, dict):
            partners: list[str] = list(offers.keys())
            offers = tuple(offers.values())
            outputs = tuple(p in self.consumers for p in partners)
            # assert all(
            #     (p in self.consumers and not p in self.suppliers)
            #     or (p in self.suppliers and not p in self.consumers)
            #     for p in partners
            # )
            return self.from_offers(
                offers,
                outputs,
                return_info=return_info,  # type: ignore
                ignore_signed_contracts=ignore_signed_contracts,
            )
        # copy inputs because we are going to modify them.
        offers = list(offers)  # type: ignore
        if outputs is None:
            if self.input_agent:
                outputs = tuple([True] * len(offers))
            elif self.output_agent:
                outputs = tuple([False] * len(offers))
            else:
                raise RuntimeError(
                    "You cannot pass outputs=None if the agent is neither a first or last level agent"
                )

        def order(x):
            """A helper function to order contracts in the following fashion:
            1. input contracts are ordered from cheapest to most expensive.
            2. output contracts are ordered from highest price to cheapest.
            3. The relative order of input and output contracts is indeterminate.
            """
            offer, is_output, _ = x
            # if is_exogenous and self.force_exogenous:
            #     return float("-inf")
            return -offer[UNIT_PRICE] if is_output else offer[UNIT_PRICE]

        # copy inputs because we are going to modify them.
        outputs = list(outputs)  # type: ignore
        # add registered sales and supplies if needed
        if not ignore_signed_contracts and self._signed_agreements:
            offers += self._signed_agreements  # type: ignore
            outputs += self._signed_is_output  # type: ignore
        # indicate that all inputs are not exogenous and that we are adding two
        # exogenous contracts after them.
        exogenous = [False] * len(offers) + [True, True]
        # add exogenous contracts as offers one for input and another for output
        invin = self.ex_qin + self.inventory_in
        invout = self.ex_qout + self.inventory_out
        offers += [  # type: ignore
            (
                invin,
                self.current_step,
                self.ex_pin / invin if invin else 0,
            ),
            (
                invout,
                self.current_step,
                self.ex_pout / invout if invout else 0,
            ),
        ]
        outputs += [False, True]  # type: ignore
        # initialize some variables
        qin, qout, pin, pout = 0, 0, 0, 0
        qin_bar, going_bankrupt = 0, self.current_balance < 0
        pout_bar = 0
        # we are going to collect output contracts in output_offers
        output_offers = []
        # sort contracts in the optimal order of execution: from cheapest when
        # buying and from the most expensive when selling. See `order` above.
        assert outputs is not None
        sorted_offers = sorted(
            list(zip(offers, outputs, exogenous, strict=True)),
            key=order,
        )

        # we calculate the total quantity we are are required to pay for `qin` and
        # the associated amount of money we are going to pay `pin`. Moreover,
        # we calculate the total quantity we can actually buy given our limited
        # money balance (`qin_bar`).
        for offer, is_output, is_exogenous in sorted_offers:  # type: ignore
            if not offer:
                continue
            # ignore any offers that are not about this time
            if offer[TIME] != self.current_step:
                continue
            offer: tuple[int, int, int]
            if is_output:
                output_offers.append((offer, is_exogenous))
                continue
            topay_this_time = offer[UNIT_PRICE] * offer[QUANTITY]
            if not going_bankrupt and (
                pin + topay_this_time + offer[QUANTITY] * self.production_cost
                > self.current_balance
            ):
                unit_total_cost = offer[UNIT_PRICE] + self.production_cost
                can_buy = int((self.current_balance - pin) // unit_total_cost)
                qin_bar = qin + can_buy
                going_bankrupt = True
            pin += topay_this_time
            qin += offer[QUANTITY]

        if not going_bankrupt:
            qin_bar = qin

        # calculate the maximum amount we can produce given our limited production
        # capacity and the input we CAN BUY
        n_lines = self.n_lines
        producible = min(qin_bar, n_lines)

        # No need to do this test now because we test for the ability to produce with
        # the ability to buy items. The factory buys cheaper items and produces them
        # before attempting more expensive ones. This may or may not be optimal but
        # who cares. It is consistent and that is all that matters.
        # # if we do not have enough money to pay for production in full, we limit
        # # the producible quantity to what we can actually produce
        # if (
        #     self.production_cost
        #     and producible * self.production_cost > self.current_balance
        # ):
        #     producible = int(self.current_balance // self.production_cost)

        # find the total sale quantity (qout) and money (pout). Moreover find
        # the actual amount of money we will receive
        done_selling = False
        for offer, is_exogenous in output_offers:
            if not done_selling:
                if qout + offer[QUANTITY] >= producible:
                    assert producible >= qout, f"{producible=}, {qout=}"
                    can_sell = producible - qout
                    done_selling = True
                else:
                    can_sell = offer[QUANTITY]
                pout_bar += can_sell * offer[UNIT_PRICE]
            pout += offer[UNIT_PRICE] * offer[QUANTITY]
            qout += offer[QUANTITY]

        # should never produce more than we signed to sell
        producible = min(producible, qout)

        # we cannot produce more than our capacity or inputs and we should not
        # produce more than our required outputs
        producible = min(qin, self.n_lines, producible)

        # the scale with which to multiply disposal_cost and shortfall_penalty
        # if no scale is given then the unit price will be used.
        output_penalty = self.output_penalty_scale
        if output_penalty is None:
            output_penalty = pout / qout if qout else 0
        shortfall = max(0, qout - producible)
        output_penalty *= self.shortfall_penalty * shortfall
        disposal_cost = self.input_penalty_scale
        if disposal_cost is None:
            disposal_cost = pin / qin if qin else 0
        disposal_cost *= self.disposal_cost * max(0, qin - producible)
        storage_cost = self.storage_penalty_scale
        if storage_cost is None:
            storage_cost = pin / qin if qin else 0
        remainingq = max(0, qin - producible)
        storage_cost *= self.storage_cost * remainingq

        # call a helper method giving it the total quantity and money in and out.
        u = self.from_aggregates(
            qin,
            qout,
            producible,
            pin,
            pout_bar,
            disposal_cost,
            output_penalty,
            storage_cost,
        )
        if return_info:
            # the real producible quantity is the minimum of what we can produce
            # given supplies and production capacity and what we can sell.
            return UtilityInfo(
                utility=u,
                remaining_quantity=remainingq,
                storage_cost=storage_cost,
                disposal_cost=disposal_cost,
                producible=producible,
                total_input=qin,
                total_output=qout,
                shortfall_penalty=output_penalty,
                shortfall_quantity=shortfall,
            )
        return u

    @cache
    def from_aggregates(
        self,
        qin: int,
        qout_signed: int,
        qout_sold: int,
        pin: int,
        pout: int,
        input_penalty: float,
        output_penalty: float,
        storage_penalty: float,
    ) -> float:
        """
        Calculates the utility from aggregates of input/output quantity/prices

        Args:
            qin: Input quantity (total including all exogenous contracts).
            qout_signed: Output quantity (total including all exogenous contracts)
                         that the agent agreed to sell.
            qout_sold: Output quantity (total including all exogenous contracts)
                       that the agent will actually sell.
            pin: Input total price (i.e. unit price * qin).
            pout: Output total price (i.e. unit price * qin).
            input_penalty: total disposal cost
            output_penalty: total shortfall penalty
            storage_penalty: total storage penalty

        Remarks:
            - Most likely, you do not need to directly call this method. Consider
              `from_offers` and `from_contracts` that take current balance and
              exogenous contract information (passed during ufun construction)
              into account.
            - The method respects production capacity (n. lines). The
              agent cannot produce more than the number of lines it has.
            - This method does not take exogenous contracts or current balance
              into account.
            - The method assumes that the agent CAN pay for all input
              and production.

        """
        assert qout_sold <= qout_signed, f"sold: {qout_sold}, signed: {qout_signed}"

        # production capacity
        lines = self.n_lines

        # we cannot produce more than our capacity or inputs and we should not
        # produce more than our required outputs
        produced = min(qin, lines, qout_sold)

        # self explanatory. right?  few notes:
        # 1. You pay disposal costs for anything that you buy and do not produce
        #    and sell. Because we know that you sell no more than what you produce
        #    we can multiply the disposal cost with the difference between input
        #    quantity and the amount produced
        # 2. You pay shortfall penalty for anything that you should have sold but
        #    did not. The only reason you cannot sell something is if you cannot
        #    produce it. That is why the shortfall penalty is multiplied by the
        #    difference between what you should have sold and the produced amount.
        # 3. You pay storage penalty for anything that remains in your storage at the end.
        # You can either have disposal penalty or storage penalty not both
        u = (
            pout
            - pin
            - self.production_cost * produced
            - input_penalty
            - storage_penalty
            - output_penalty
        )
        if not self.normalized:
            return u
        # normalize values between zero and one if needed.
        rng = self.max_utility - self.min_utility
        if rng < 1e-12:
            return 1.0
        return (u - self.min_utility) / rng

    def breach_level(self, qin: int = 0, qout: int = 0):
        """Calculates the breach level that would result from a given quantities"""
        qin += self.ex_qin + self.inventory_in
        qin = min(qin, self.n_lines)
        qout += self.ex_qout + self.inventory_out
        return 0 if qin >= qout else (qout - qin) / qout

    def is_breach(self, qin: int = 0, qout: int = 0):
        """Whether the given quantities would lead to a breach."""
        qin += self.ex_qin + self.inventory_in
        qout += self.ex_qout + self.inventory_out
        return qout > min(qin, self.n_lines)

    @property
    def max_utility(self):
        """The maximum possible utility value"""
        if self._best is None:
            self._best = self.find_limit(True)
        return self._best.utility

    @property
    def min_utility(self):
        """The minimum possible utility value"""
        if self._worst is None:
            self._worst = self.find_limit(False)
        return self._worst.utility

    def minmax(self, *args, **kwargs) -> tuple[float, float]:
        worst, best = self.extreme_outcomes(*args, **kwargs)
        return (self(worst), self(best))

    def extreme_outcomes(
        self,
        outcome_space: OutcomeSpace | None = None,
        issues: Iterable[Issue] | None = None,
        outcomes: Iterable[Outcome] | None = None,
        max_cardinality=1000,
    ) -> tuple[Outcome, Outcome]:
        assert outcome_space is None and issues is None and outcomes is None
        _ = max_cardinality
        product = (
            self.output_product
            if self.input_agent
            else self.input_product
            if self.output_agent
            else None
        )
        if product is None:
            raise ValueError(
                f"Cannot find the utility range of a midlevel agent: {self.id}\n{vars(self)}"
            )
        t = self.current_step
        is_input = int(product == self.input_product)
        best = self.find_limit(
            True,
            n_input_negs=is_input,
            n_output_negs=1 - is_input,
        )
        worst = self.find_limit(
            False,
            n_input_negs=is_input,
            n_output_negs=1 - is_input,
        )
        if self.input_agent:
            worst_outcome = (worst.output_quantity, t, worst.output_price)
            best_outcome = (best.output_quantity, t, best.output_price)
        else:
            worst_outcome = (worst.input_quantity, t, worst.input_price)
            best_outcome = (best.input_quantity, t, best.input_price)
        return worst_outcome, best_outcome

    def utility_range(
        self,
        outcome_space: OutcomeSpace | None = None,
        issues: list[Issue] | None = None,
        outcomes: list[Outcome] | None = None,
        return_outcomes=False,
        max_n_outcomes=1000,
    ) -> tuple[float, float] | tuple[float, float, Outcome, Outcome]:
        """
        Finds the utility range and optionally returns the corresponding outcomes
        from a given issue space or in a single negotiation.

        Args:
            issues: The set of issues of the negotiation. If not given it will
                    be read from the AWI. Note that you cannot specify these
                    issues except for agent in the first or last layer of the
                    production graph (because otherwise, the agent cannot know
                    whether this negotiation is for buying of selling).
            outcomes: A list of outcomes to consider. Using outcomes is much slower
                      than using issues and you should never pass both.
            infeasible_cutoff: A utility value under which we consider the outcome
                               infeasible.
            return_outcomes: If given the worst and best outcomes (in that order)
                             will be returned.
            max_n_outcomes: Maximum number of outcomes to try. Not used.

        Returns:
            A tuple of worst and best utility values if `return_outcomes` is `False`.
            otherwise, the worst and best outcomes are appended to the returned
            utilities leading to a 4-items tuple instead of two.

        Remarks:
            - You will get a warning if you use a list of outcomes here because
              it is too slow.
            - You should only pass `issues` if you know that the agent is either
              an input agent or an output agent. Agents in the middle of the
              production graph cannot know whether these issues are for buying
              of for selling. To find the utility range for these agents, you
              can use `worst` and `best` that allow specifying input and output
              issues separately.
            - It is always assumed that the range required is for a single
              negotiation not a set of negotiations and under the assumption that
              all other negotiations if any will end in failure
        """
        if not return_outcomes:
            return self.minmax(outcome_space, issues, list(outcomes), max_n_outcomes)  # type: ignore
        worst, best = self.extreme_outcomes(
            outcome_space, issues, outcomes, max_n_outcomes
        )
        return (self(worst), self(best), worst, best)

    def _is_midlevel(self):
        return not self.input_agent and not self.output_agent

    def find_limit(
        self,
        best: bool,
        n_input_negs=None,
        n_output_negs=None,
        secured_input_quantity=0,
        secured_input_unit_price=0.0,
        secured_output_quantity=0,
        secured_output_unit_price=0.0,
        ignore_signed_contracts: bool = True,
    ) -> UFunLimit:
        """
        Finds either the maximum or the minimum of the ufun.

        Args:
             best: Best(max) or worst (min) ufun value?
             n_input_negs: How many input negs are we to consider? None means all
             n_output_negs: How many output negs are we to consider? None means all
             secured_input_quantity: A quantity that MUST be bought
             secured_input_unit_price: The (average) unit price of the quantity
                                       that MUST be bought.
             secured_output_quantity: A quantity that MUST be sold.
             secured_output_unit_price: The (average) unit price of the quantity
                                        that MUST be sold.
             ignore_signed_contracts: If True all signed contracts will be ignored.
                                      Use secured_* to pass this information if you need
                                      to in this case.
        Remarks:
            - You can use the `secured_*` arguments and control over the number
              of negotiations to consider to find the utility limits **given**
              some already concluded and signed contracts
        """
        default_params = (
            n_input_negs is None
            and n_output_negs is None
            and secured_input_quantity == 0
            and secured_input_unit_price < 1e-5
            and secured_output_quantity == 0
            and secured_output_unit_price < 1e-5
        )
        set_best, set_worst = best and default_params, not best and default_params
        result = self.find_limit_brute_force(
            best,
            n_input_negs,
            n_output_negs,
            secured_input_quantity,
            secured_input_unit_price,
            secured_output_quantity,
            secured_output_unit_price,
            ignore_signed_contracts=ignore_signed_contracts,
        )
        actual_util = self.from_offers(
            (
                (result.output_quantity, self.current_step, result.output_price),
                (result.input_quantity, self.current_step, result.input_price),
            ),
            (True, False),
            ignore_signed_contracts=ignore_signed_contracts,
        )
        assert (
            abs(result.utility - actual_util) < 1e-2
        ), f"UFunLimit with utility {result.utility} != actual utility {actual_util} of the outcome in it!!\n{result}"
        if set_best:
            self._best = result
        elif set_worst:
            self._worst = result
        return result

    def best(self) -> Outcome:
        if self._best is None:
            self._best = self.find_limit(True)
        if not self._best.output_quantity:
            return (
                self._best.input_quantity,
                self.current_step,
                self._best.input_price,
            )
        if not self._best.input_quantity:
            return (
                self._best.output_quantity,
                self.current_step,
                self._best.output_price,
            )
        raise ValueError(
            f"We need to buy and sell which means there is single contract that makes sense {self._best=}."
        )

    def worst(self) -> Outcome:
        if self._worst is None:
            self._worst = self.find_limit(False)
        if not self._worst.output_quantity:
            return (
                self._worst.input_quantity,
                self.current_step,
                self._worst.input_price,
            )
        if not self._worst.input_quantity:
            return (
                self._worst.output_quantity,
                self.current_step,
                self._worst.output_price,
            )
        raise ValueError(
            f"We need to buy and sell which means there is single contract that makes sense {self._worst=}."
        )

    @cache
    def find_limit_brute_force(
        self,
        best,
        n_input_negs=None,
        n_output_negs=None,
        secured_input_quantity=0,
        secured_input_unit_price=0.0,
        secured_output_quantity=0,
        secured_output_unit_price=0.0,
        ignore_signed_contracts=True,
    ) -> UFunLimit:
        """
        Finds either the maximum and the minimum of the ufun.

        Args:
             best: Best(max) or worst (min) ufun value?
             n_input_negs: How many input negs are we to consider? None means all
             n_output_negs: How many output negs are we to consider? None means all
             secured_input_quantity: A quantity that MUST be bought
             secured_input_unit_price: The (average) unit price of the quantity
                                       that MUST be bought.
             secured_output_quantity: A quantity that MUST be sold.
             secured_output_unit_price: The (average) unit price of the quantity
                                        that MUST be sold.
        Remarks:
            - You can use the `secured_*` arguments and control over the number
              of negotiations to consider to find the utility limits **given**
              some already concluded and signed contracts
            - Note that this function CANNOT take into account the sales or supplies
              already signed (and registered via `register_sale` and/or `register_supply`).
              You MUST pass the quantities and prices for signed contracts through the secured_*
              parameters.

        Returns:
            worst and best outcome information in the form of `UFunLimit` tuple.

        """
        if n_input_negs is None:
            n_input_negs = self.n_input_negs
            if not ignore_signed_contracts:
                n_input_negs -= sum(int(_) for _ in self._signed_is_output if not _)
                n_input_negs -= len(self._registered_supply_failures)
                assert n_input_negs >= 0, f"{n_input_negs=} cannot be negative"

        if n_output_negs is None:
            n_output_negs = self.n_output_negs
            if not ignore_signed_contracts:
                n_output_negs -= sum(int(_) for _ in self._signed_is_output if _)
                n_output_negs -= len(self._registered_sale_failures)
                assert n_output_negs >= 0, f"{n_output_negs=} cannot be negative"

        if not ignore_signed_contracts:
            sales = [
                c for c, o in zip(self._signed_agreements, self._signed_is_output) if o
            ]
            supplies = [
                c
                for c, o in zip(self._signed_agreements, self._signed_is_output)
                if not o
            ]
            secured_input_quantity = sum(_[0] for _ in supplies)
            secured_input_unit_price = sum(_[-1] * _[0] for _ in supplies) / (
                secured_input_quantity if secured_input_quantity else 1
            )
            secured_output_quantity = sum(_[0] for _ in sales)
            secured_output_unit_price = sum(_[-1] * _[0] for _ in sales) / (
                secured_output_quantity if secured_output_quantity else 1
            )
        imax = n_input_negs * self.input_qrange[1] + 1
        omax = n_output_negs * self.output_qrange[1] + 1

        # we know that the prices of inputs for the best and worst solutions.
        ip = self.input_prange[0] if best else self.input_prange[1]
        op = self.output_prange[1] if best else self.output_prange[0]
        limit_io, limit_u = None, (float("-inf") if best else float("inf"))
        limit_p, limit_p = 0, 0
        for i in range(imax):
            for o in range(omax):
                info = self.from_offers(
                    (
                        (i, self.current_step, ip),
                        (o, self.current_step, op),
                        (
                            secured_input_quantity,
                            self.current_step,
                            secured_input_unit_price,
                        ),
                        (
                            secured_output_quantity,
                            self.current_step,
                            secured_output_unit_price,
                        ),
                    ),
                    (False, True, False, True),
                    return_info=True,
                    ignore_signed_contracts=True,
                )
                u, p = info.utility, info.producible
                if (best and u >= limit_u) or (not best and u <= limit_u):
                    limit_io, limit_u, limit_p = (
                        (i, ip, o, op),
                        u,
                        p,
                    )
        # this method cannot find the exogenous quantities at the limit found
        # if force_exogenous was false and will return None for them.
        assert limit_io is not None
        return UFunLimit(
            utility=limit_u,
            input_quantity=limit_io[0],
            input_price=limit_io[1],
            output_quantity=limit_io[2],
            output_price=limit_io[3],
            exogenous_input_price=self.ex_pin / self.ex_qin if self.ex_qin else 0,
            exogenous_output_price=self.ex_pout / self.ex_qout if self.ex_qout else 0,
            exogenous_input_quantity=self.ex_qin if self.force_exogenous else None,
            exogenous_output_quantity=self.ex_qout if self.force_exogenous else None,
            inventory_input=self.inventory_in,
            inventory_output=self.inventory_out,
            producible=limit_p,
        )

    def ok_to_buy_at(self, unit_price: float) -> bool:
        """
        Checks if the unit price can -- even in principle -- be acceptable for buying

        Remarks:
            - This method is **very** optimistic. If it returns `False`, an
              agent should **never** buy at this price. If it returns `True`, it
              may *still be a bad idea* to buy at this price.
            - If we **buy** at this price, the **best** case scenario is that we pay it
              and pay production cost then receive the unit price of one output.
            - If we do **not** buy at this price, the **worst** case scenario is that
              we will pay shortfall penalty for one item
            - We should **NOT** buy if the best case scenario when buying is worse than
              the worst case scenario when not buying.
            - If called for agents not at the end of the production chain, it will
              always return `True` because in these cases we do not know what the
              the unit price for the output so there is nothing to compare with.
        """
        if not self.perishable:
            return True
        assert self.inventory_in <= 0, f"A perishable case but {self.inventory_in=}"
        assert self.inventory_out <= 0, f"A perishable case but {self.inventory_out=}"
        # can reject a price only if we know the output unit price
        # (i.e. we have an output agent)
        if not self.output_agent:
            return True
        total_out = self.ex_qout
        # If we are not selling, we should not buy
        if total_out < 1:
            return False
        # do not buy at this price if it is **guaranteed** to lead to a loss
        return (
            unit_price + self.production_cost - self.ex_pout // total_out
        ) < self.shortfall_penalty

    def ok_to_sell_at(self, unit_price: float) -> bool:
        """
        Checks if the unit price can -- even in principle -- be acceptable for selling

        Remarks:
            - This method is **very** optimistic. If it returns `False`, an
              agent should **never** sell at this price. If it returns `True`, it
              may *still be a bad idea* to sell at this price.
            - Sales decisions does not affect in any way the amount we pay for
              input materials. It only affects the amount we produce, the amout we
              get paid in sales and the amount we pay as disposal cost and
              shortfall penalty.
            - If we agree to sell an item at this price, the best case scenario
              is that we can actually produce this item and sell it. We pay production
              cost and receive the given unit price.
            - If we do **not** sell at this price, the worst case scenario is that
              we really needed that sale. In this case, we will pay disposal cost
              for one item.
            - We should **NOT** sell if the best case scenario when selling is worse than
              the worst case scenario when not selling.
            - If called for agents not at the beginning of the production chain, it will
              always return `True` because in these cases we do not know what the
              the unit price for the input so there is nothing to compare with.
        """
        if not self.perishable:
            return True
        assert self.inventory_in <= 0, f"A perishable case but {self.inventory_in=}"
        assert self.inventory_out <= 0, f"A perishable case but {self.inventory_out=}"
        # can reject a price only if we know the input unit price
        # (i.e. we have an input agent)
        if not self.input_agent:
            return True
        # If we are not buying, we cannot sell
        if self.ex_qin < 1:
            return False
        # do not sell at this price if it is **guaranteed** to lead to a loss
        return (self.production_cost - unit_price) < self.disposal_cost
