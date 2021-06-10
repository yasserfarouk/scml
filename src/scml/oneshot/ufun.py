from collections import namedtuple
from copy import deepcopy
import warnings
from typing import Iterable, Tuple, Union, List, Collection, Optional

from negmas import Contract
from negmas.utilities import UtilityFunction, UtilityValue
from negmas.outcomes import Outcome, Issue

from scml.scml2020.common import is_system_agent

from .common import QUANTITY, UNIT_PRICE, TIME

__all__ = ["OneShotUFun", "UFunLimit"]

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
        "producible",
    ],
)
"""Information about one utility limit (either highest or lowest). See `OnShotUFun.find_limit` for details."""


class OneShotUFun(UtilityFunction):
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
        output_agent: Is the agent an input agent which means that its input
                      product is the raw material
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
        shortfall_penalty: float,
        input_penalty_scale: Optional[float],
        output_penalty_scale: Optional[float],
        n_input_negs: int,
        n_output_negs: int,
        current_step: int,
        input_qrange: Tuple[int, int] = (0, 0),
        input_prange: Tuple[int, int] = (0, 0),
        output_qrange: Tuple[int, int] = (0, 0),
        output_prange: Tuple[int, int] = (0, 0),
        force_exogenous: bool = True,
        n_lines: int = 10,
        normalized: bool = False,
        current_balance: Union[int, float] = float("inf"),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.current_balance = current_balance
        self.normalized = normalized
        self.input_penalty_scale = input_penalty_scale
        self.output_penalty_scale = output_penalty_scale
        self.current_step = current_step
        self.ex_pin, self.ex_pout = ex_pin, ex_pout
        self.ex_qin, self.ex_qout = ex_qin, ex_qout
        self.n_input_negs = n_input_negs
        self.n_output_negs = n_output_negs
        self.input_qrange, self.input_prange = input_qrange, input_prange
        self.output_qrange, self.output_prange = output_qrange, output_prange
        self.production_cost, self.disposal_cost, self.shortfall_penalty = (
            production_cost,
            disposal_cost,
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
        if self.normalized:
            self.best = self.find_limit(True, None, None)
            self.worst = self.find_limit(False, None, None)
        else:
            self.best = UFunLimit(*tuple([None] * 10))
            self.worst = UFunLimit(*tuple([None] * 10))

    def xml(self, issues) -> str:
        raise NotImplementedError("Cannot convert the ufun to xml")

    def __call__(self, offer) -> float:
        """
        Calculates the utility function given a single contract.

        Remarks:
            - This method calculates the utility value of a single offer assuming all other negotiations end in failure.
            - It can only be called for agents that exist in the first or last layer of the production graph.
        """
        if not self.input_agent and not self.output_agent:
            return float("-inf")
        return self.from_offers([offer], [self.input_agent])

    def from_contracts(
        self, contracts: Iterable[Contract], ignore_exogenous=True
    ) -> float:
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
            if ignore_exogenous and any(is_system_agent(_) for _ in c.partners):
                continue
            product = c.annotation["product"]
            is_output = product == output_product
            outputs.append(is_output)
            offers.append(self.outcome_as_tuple(c.agreement))
        return self.from_offers(offers, outputs)

    @staticmethod
    def outcome_as_tuple(offer):
        if isinstance(offer, dict):
            outcome = [None] * 3
            outcome[QUANTITY] = offer["quantity"]
            outcome[TIME] = offer["time"]
            outcome[UNIT_PRICE] = offer["unit_price"]
            return tuple(outcome)
        return tuple(offer)

    def from_offers(
        self, offers: Iterable[Tuple], outputs: Iterable[bool], return_producible=False
    ) -> Union[float, Tuple[float, int]]:
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
            return_producible: If true, the producible quantity will be returned
        Remarks:
            - This method takes into account the exogenous contract information
              passed when constructing the ufun.
        """

        def order(x):
            """A helper function to order contracts in the following fashion:
            1. input contracts are ordered from cheapest to most expensive.
            2. output contracts are ordered from highest price to cheapest.
            3. The relative order of input and output contracts is indeterminate.
            """
            offer, is_output, is_exogenous = x
            # if is_exogenous and self.force_exogenous:
            #     return float("-inf")
            return -offer[UNIT_PRICE] if is_output else offer[UNIT_PRICE]

        # copy inputs because we are going to modify them.
        offers, outputs = deepcopy(list(offers)), deepcopy(list(outputs))
        # indicate that all inputs are not exogenous and that we are adding two
        # exogenous contracts after them.
        exogenous = [False] * len(offers) + [True, True]
        # add exogenous contracts as offers one for input and another for output
        offers += [
            (self.ex_qin, 0, self.ex_pin / self.ex_qin if self.ex_qin else 0),
            (self.ex_qout, 0, self.ex_pout / self.ex_qout if self.ex_qout else 0),
        ]
        outputs += [False, True]
        # initialize some variables
        qin, qout, pin, pout = 0, 0, 0, 0
        qin_bar, going_bankrupt = 0, self.current_balance < 0
        pout_bar = 0
        # we are going to collect output contracts in output_offers
        output_offers = []
        # sort contracts in the optimal order of execution: from cheapest when
        # buying and from the most expensive when selling. See `order` above.
        sorted_offers = list(sorted(zip(offers, outputs, exogenous), key=order))

        # we calculate the total quantity we are are required to pay for `qin` and
        # the associated amount of money we are going to pay `pin`. Moreover,
        # we calculate the total quantity we can actually buy given our limited
        # money balance (`qin_bar`).
        for offer, is_output, is_exogenous in sorted_offers:
            offer = self.outcome_as_tuple(offer)
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

        # No need to this test now because we test for the ability to produce with
        # the ability to buy items. The factory buys cheaper items and produces them
        # before attempting more expensive ones. This may or may not be optimal but
        # who cars. It is consistent that it is all that matters.
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
                    assert producible >= qout, f"producible {producible}, qout {qout}"
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
        output_penalty *= self.shortfall_penalty * max(0, qout - producible)
        input_penalty = self.input_penalty_scale
        if input_penalty is None:
            input_penalty = pin / qin if qin else 0
        input_penalty *= self.disposal_cost * max(0, qin - producible)

        # call a helper method giving it the total quantity and money in and out.
        u = self.from_aggregates(
            qin, qout, producible, pin, pout_bar, input_penalty, output_penalty
        )
        if return_producible:
            # the real producible quantity is the minimum of what we can produce
            # given supplies and production capacity and what we can sell.
            return u, producible
        return u

    def from_aggregates(
        self,
        qin: int,
        qout_signed: int,
        qout_sold: int,
        pin: int,
        pout: int,
        input_penalty,
        output_penalty,
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
        u = (
            pout
            - pin
            - self.production_cost * produced
            - input_penalty
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
        qin += self.ex_qin
        qin = min(qin, self.n_lines)
        qout += self.ex_qout
        return 0 if qin >= qout else (qout - qin) / qout

    def is_breach(self, qin: int = 0, qout: int = 0):
        """Whether the given quantities would lead to a breach."""
        qin += self.ex_qin
        qout += self.ex_qout
        return qout > min(qin, self.n_lines)

    @property
    def max_utility(self):
        """The maximum possible utility value"""
        if self.best is None:
            self.best = self.find_limit(True)
        return self.best.utility

    @property
    def min_utility(self):
        """The minimum possible utility value"""
        if self.worst is None:
            self.worst = self.find_limit(False)
        return self.worst.utility

    def utility_range(
        self,
        issues: List[Issue] = None,
        outcomes: Collection[Outcome] = None,
        infeasible_cutoff: Optional[float] = None,
        return_outcomes=False,
        max_n_outcomes=1000,
        ami=None,
    ) -> Union[
        Tuple[UtilityValue, UtilityValue],
        Tuple[UtilityValue, UtilityValue, Outcome, Outcome],
    ]:
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
        if outcomes is not None:
            warnings.warn(
                "Using utility_range with outcomes instead of issues is "
                "extremely inefficient for OneShotUFun"
            )
            return super().utility_range(
                issues, outcomes, infeasible_cutoff, return_outcomes, max_n_outcomes
            )
        product = (
            self.output_product
            if self.input_agent
            else self.input_product
            if self.output_agent
            else None
        )
        if product is None and ami:
            product = ami.annotation["product"]
        if product is None and self.ami:
            product = self.ami.annotation.get("product", None)
        if product is None:
            raise ValueError("Cannot find the utility range of a midlevel agent")
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
        if not return_outcomes:
            return worst.utility, best.utility
        if self.input_agent:
            worst_outcome = (worst.output_quantity, t, worst.output_price)
            best_outcome = (best.output_quantity, t, best.output_price)
        else:
            worst_outcome = (worst.input_quantity, t, worst.input_price)
            best_outcome = (best.input_quantity, t, best.input_price)
        return (  # typing: ignore
            worst.utility,
            best.utility,
            worst_outcome,
            best_outcome,
        )  # typing: ignore

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
        )
        actual_util = self.from_offers(
            [
                (result.output_quantity, 0, result.output_price),
                (result.input_quantity, 0, result.input_price),
            ],
            [True, False],
        )
        assert (
            abs(result.utility - actual_util) < 1e-2
        ), f"UFunLimit with utility {result.utility} != actual utility {actual_util} of the outcome in it!!"
        if set_best:
            self.best = result
        elif set_worst:
            self.worst = result
        return result

    def register_agrerement(is_output: bool, quantity: int, unit_price: int) -> None:
        pass

    def find_limit_brute_force(
        self,
        best,
        n_input_negs=None,
        n_output_negs=None,
        secured_input_quantity=0,
        secured_input_unit_price=0.0,
        secured_output_quantity=0,
        secured_output_unit_price=0.0,
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

        Returns:
            worst and best outcome information in the form of `UFunLimit` tuple.

        """
        if n_input_negs is None:
            n_input_negs = self.n_input_negs
        if n_output_negs is None:
            n_output_negs = self.n_output_negs
        imax = n_input_negs * self.input_qrange[1] + 1
        omax = n_output_negs * self.output_qrange[1] + 1

        # we know that the prices of inputs for the best and worst solutions.
        ip = self.input_prange[0] if best else self.input_prange[1]
        op = self.output_prange[1] if best else self.output_prange[0]
        limit_io, limit_u = None, (float("-inf") if best else float("inf"))
        limit_p, limit_p = 0, 0
        for i in range(imax):
            for o in range(omax):
                u, p = self.from_offers(
                    [(i, 0, ip), (o, 0, op)],
                    [False, True],
                    return_producible=True,
                )
                if (best and u >= limit_u) or (not best and u <= limit_u):
                    limit_io, limit_u, limit_p = (
                        (i, ip, o, op),
                        u,
                        p,
                    )
        # this method cannot find the exogenous quantities at the limit found
        # if force_exogenous was false and will return None for them.
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
            producible=limit_p,
        )
