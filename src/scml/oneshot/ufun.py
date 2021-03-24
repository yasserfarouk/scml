from collections import namedtuple
from copy import deepcopy
import warnings
from typing import Iterable, Tuple, Union, List, Collection, Optional

from negmas import Contract
from negmas.utilities import UtilityFunction, UtilityValue
from negmas.outcomes import Outcome, Issue


# import quadprog
# import cvxpy as cp
import mip as mp

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
        storage_cost: storage cost per unit of input/output.
        delivery_penalty: penalty for failure to deliver one unit of output.
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
        - storage cost is paid for items bought but not produced only. Items
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
        storage_cost: float,
        delivery_penalty: float,
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
        self.production_cost, self.storage_cost, self.delivery_penalty = (
            production_cost,
            storage_cost,
            delivery_penalty,
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
            1. input contracts are ordered from cheapest to most expensive. The
               exception to this is that exogenous buy contracts will appear
               before anything else if force-exogenous is set.
            2. output contracts are ordered from highest price to cheapest. The
               exception to this is that exogenous sell contracts will appear
               before anything else if force-exogenous is set.
            3. The relative order of input and output contracts is indeterminate.
            """
            offer, is_output, is_exogenous = x
            if is_exogenous and self.force_exogenous:
                return float("-inf")
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
        qin, qout, pin = 0, 0, 0
        qin_bar, going_bankrupt = 0, False
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
            topay = offer[UNIT_PRICE] * offer[QUANTITY]
            if not going_bankrupt and (
                pin + topay + offer[QUANTITY] * self.production_cost
                > self.current_balance
            ):
                assert self.current_balance >= pin
                unit_total_cost = offer[UNIT_PRICE] + self.production_cost
                can_buy = (self.current_balance - pin) // unit_total_cost
                if can_buy >= offer[QUANTITY]:
                    assert (
                        can_buy < offer[QUANTITY]
                    ), f"can-buy {can_buy} of {offer[QUANTITY]} yet fails!!"
                qin_bar = qin + can_buy
                going_bankrupt = True
            pin += topay
            qin += offer[QUANTITY]
        if not going_bankrupt:
            qin_bar = qin
        # breakpoint()
        # calculate the maximum amount we can produce given our limited production
        # capacity and the input we CAN BUY
        n_lines = self.n_lines
        producible = min(qin_bar, n_lines)
        # if we do not have enough money to pay for production in full, we limit
        # the producible quantity to what we can actually produce
        if (
            self.production_cost
            and producible * self.production_cost > self.current_balance
        ):
            producible = int(self.current_balance // self.production_cost)

        # find the total sale quantity (qout) and money (pout). Moreover find
        # the actual amount of money we will receive
        for offer, is_exogenous in output_offers:
            qout += offer[QUANTITY]
            if qout >= producible:
                pout_bar += (producible - qout) * offer[UNIT_PRICE]
            else:
                pout_bar += offer[QUANTITY] * offer[UNIT_PRICE]

        # call a helper method giving it the total quantity and money in and out.
        u = self.from_aggregates(qin, qout, pin, pout_bar)
        if return_producible:
            # the real producible quantity is the minimum of what we can produce
            # given supplies and production capacity and what we can sell.
            producible = min(producible, qout)
            return u, producible
        return u

    def from_aggregates(
        self,
        qin: int = 0,
        qout: int = 0,
        pin: int = 0,
        pout: int = 0,
    ) -> float:
        """
        Calculates the utility from aggregates of input/output quantity/prices

        Args:
            qin: Input quantity (total including all exogenous contracts).
            qout: Output quantity (total including all exogenous contracts).
            pin: Input total price (i.e. unit price * qin).
            pout: Output total price (i.e. unit price * qin).

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
        # production capacity
        lines = self.n_lines

        # we cannot produce more than our capacity or inputs and we should not
        # produce more than our required outputs
        produced = min(qin, lines, qout)

        # self explanatory. right?  few notes:
        # 1. You pay storage costs for anything that you buy and do not produce
        #    and sell. Because we know that you sell no more than what you produce
        #    we can multiply the storage cost with the difference between input
        #    quantity and the amount produced
        # 2. You pay delivery penalty for anything that you should have sold but
        #    did not. The only reason you cannot sell something is if you cannot
        #    produce it. That is why the delivery penalty is multiplied by the
        #    difference between what you should have sold and the produced amount.

        u = (
            pout
            - pin
            - self.production_cost * produced
            - self.storage_cost * max(0, qin - produced)
            - self.delivery_penalty * max(0, qout - produced)
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
        return self.best.utility

    @property
    def min_utility(self):
        """The minimum possible utility value"""
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
        try:
            result = self.find_limit_optimal(
                best,
                n_input_negs,
                n_output_negs,
                secured_input_quantity,
                secured_input_unit_price,
                secured_output_quantity,
                secured_output_unit_price,
            )
        except Exception as e:
            warnings.warn(f"Optimal ufun limit calculation failed with exception {e}")
            result = None
        if result is None:
            result = self.find_limit_greedy(
                best,
                n_input_negs,
                n_output_negs,
                secured_input_quantity,
                secured_input_unit_price,
                secured_output_quantity,
                secured_output_unit_price,
            )
        if result is None:
            warnings.warn("Greedy ufun limit calculation failed")
        result: UFunLimit
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

    def find_limits_brute_force(
        self,
        n_input_negs=None,
        n_output_negs=None,
        secured_input_quantity=0,
        secured_input_unit_price=0.0,
        secured_output_quantity=0,
        secured_output_unit_price=0.0,
    ) -> Tuple[UFunLimit, UFunLimit]:
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
        best_ip = self.input_prange[0]
        best_op = self.output_prange[1]
        worst_ip = self.input_prange[1]
        worst_op = self.output_prange[0]
        best_limit_io, best_limit_u = None, float("-inf")
        worst_limit_io, worst_limit_u = None, float("inf")
        best_limit_p, worst_limit_p = 0, 0
        for i in range(imax):
            for o in range(omax):
                u, p = self.from_offers(
                    [(i, 0, best_ip), (o, 0, best_op)],
                    [False, True],
                    return_producible=True,
                )
                if u >= best_limit_u:
                    best_limit_io, best_limit_u, best_limit_p = (
                        (i, best_ip, o, best_op),
                        u,
                        p,
                    )
                u, p = self.from_offers(
                    [(i, 0, worst_ip), (o, 0, worst_op)],
                    [False, True],
                    return_producible=True,
                )
                if u <= worst_limit_u:
                    worst_limit_io, worst_limit_u, worst_limit_p = (
                        (i, worst_ip, o, worst_op),
                        u,
                        p,
                    )
        # this method cannot find the exogenous quantities at the limit found
        # if force_exogenous was false and will return None for them.
        return (
            UFunLimit(
                utility=worst_limit_u,
                input_quantity=worst_limit_io[0],
                input_price=worst_limit_io[1],
                output_quantity=worst_limit_io[2],
                output_price=worst_limit_io[3],
                exogenous_input_price=self.ex_pin / self.ex_qin if self.ex_qin else 0,
                exogenous_output_price=self.ex_pout / self.ex_qout
                if self.ex_qout
                else 0,
                exogenous_input_quantity=self.ex_qin if self.force_exogenous else None,
                exogenous_output_quantity=self.ex_qout
                if self.force_exogenous
                else None,
                producible=worst_limit_p,
            ),
            UFunLimit(
                utility=best_limit_u,
                input_quantity=best_limit_io[0],
                input_price=best_limit_io[1],
                output_quantity=best_limit_io[2],
                output_price=best_limit_io[3],
                exogenous_input_price=self.ex_pin / self.ex_qin if self.ex_qin else 0,
                exogenous_output_price=self.ex_pout / self.ex_qout
                if self.ex_qout
                else 0,
                exogenous_input_quantity=self.ex_qin if self.force_exogenous else None,
                exogenous_output_quantity=self.ex_qout
                if self.force_exogenous
                else None,
                producible=best_limit_p,
            ),
        )

    def find_limit_optimal(
        self,
        best: bool,
        n_input_negs=None,
        n_output_negs=None,
        secured_input_quantity=0,
        secured_input_unit_price=0.0,
        secured_output_quantity=0,
        secured_output_unit_price=0.0,
        check=False,
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
             check: If given more assertions will be used
        Remarks:
            - You can use the `secured_*` arguments and control over the number
              of negotiations to consider to find the utility limits **given**
              some already concluded and signed contracts
        """

        def make_program(
            best: bool, allow_oversales, n_input_negs=None, n_output_negs=None
        ):
            """Creates and solves a LIP that finds one limit of the ufun

            Args:
                best: best or worst?
                allow_oversales: If given, the output quantity will be allowed
                                 to be larger than the input quantity which means
                                 that the agent may pay delivery penalty but will
                                 never pay storage cost and vice versa.
                n_input_negs: Number of input negotiations
                n_output_negs: Number of output negotiations

            """
            # PREPARING HELPER VALUES
            # =======================
            # if n. negs is not given use the number passed to the constructor
            if n_input_negs is None:
                n_input_negs = self.n_input_negs
            if n_output_negs is None:
                n_output_negs = self.n_output_negs

            # The most appropriate unit price for the given limit is easy to
            # find. Buy cheap, sell expensive if best and vice versa. No need
            # to search over unit prices.
            uin = self.input_prange[1 - int(best)]
            uout = self.output_prange[int(best)]
            # for exogenous contracts, the unit price is already given
            ex_uin = self.ex_pin / self.ex_qin if self.ex_qin else 0
            ex_uout = self.ex_pout / self.ex_qout if self.ex_qout else 0

            # the scale with which to multiply storage_cost and delivery_penalty
            scale = (
                self.output_penalty_scale
                if allow_oversales
                else self.input_penalty_scale
            )
            # if no scale is given then the unit price will be used. Note that
            # we only pay delivery_penalty if we allow oversales and in this case
            # the scale should be the unit price of the output
            if scale is None:
                scale = uout if allow_oversales else uin

            # create the model
            m = mp.Model(sense=mp.MAXIMIZE if best else mp.MINIMIZE)
            m.verbose = False

            # OPTIMIZATION VARIABLES
            # ======================

            # Variable qin: The NEGOTIATED quantity of input
            qin = m.add_var(var_type=mp.INTEGER, name="qin")
            # Variable out: The NEGOTIATED quantity of output
            qout = m.add_var(var_type=mp.INTEGER, name="qout")
            # Variable: produced: the quantity that will be produced
            produced = m.add_var(var_type=mp.INTEGER, name="produced")
            # Variable: ex_q*_signed: A boolean indicating whether the exogenous
            # contract is to be signed in the limit to be found. Note that this
            # is set to 1 when exogenous contracts are forced.
            if self.force_exogenous:
                ex_qin_signed = 1
                ex_qout_signed = 1
            else:
                ex_qin_signed = m.add_var(var_type=mp.BINARY, name="ex_qin_signed")
                ex_qout_signed = m.add_var(var_type=mp.BINARY, name="ex_qout_signed")

            # HELPER VARIABLES
            # ================

            # [value] total cost of producing one unit.
            total_cost = uin + self.production_cost

            # [variable] the total quantity in all sales that the agent SHOULD produce
            # to avoid paying delivery penalty.
            qout_total = qout + ex_qout_signed * self.ex_qout

            # [variable] the total quantity in all supplies that the agent SHOULD pay for
            # to avoid paying storage_cost.
            qin_total = qin + ex_qin_signed * self.ex_qin

            # Variable: real_qout: The actual negotiated quantity sold
            # Variable: real_ex_qout: The actual exogenous quantity sold
            # we know that to get the worst possible result, we should not really
            # sell anything to make sure we get no money from our consumers while
            # paying as much as possible
            if best:
                real_qout = m.add_var(var_type=mp.INTEGER, name="real_qout")
                real_ex_qout = m.add_var(var_type=mp.INTEGER, name="real_ex_qout")
            else:
                real_qout = 0
                real_ex_qout = 0

            # Constraints
            # ===========

            # input negotiated quantity cannot negative or more than what can be negotiated
            m += qin >= 0  # typing: ignore
            m += qin <= self.input_qrange[1] * self.n_input_negs  # typing: ignore

            # output negotiated quantity cannot negative or more than what can be negotiated
            m += qout >= 0  # typing: ignore
            m += qout <= self.output_qrange[1] * self.n_output_negs  # typing: ignore

            # production must be positive and below capacity and inputs
            m += produced >= 0  # typing: ignore
            m += produced <= self.n_lines  # typing: ignore
            m += produced <= qin_total

            # we cannot sell what we cannot produce
            m += real_qout <= produced

            # if it costs anything to produce, then we cannot produce more than
            # what we can pay for. Note that total_cost here takes account of
            # both buying inputs and production
            if total_cost:
                m += produced <= self.current_balance // total_cost

            # we know that the actual negotiated-amount we will sell is nonnegative and
            # higher than the amount negotiated
            m += real_qout <= qout
            if not isinstance(real_qout, int):
                m += real_qout >= 0
            # the exogenous contracts we honor cannot exceed the total. We need
            # these constraints ONLY if we are optimizing over real_ex_qout
            if not isinstance(real_ex_qout, int) or not isinstance(ex_qout_signed, int):
                m += real_ex_qout <= ex_qout_signed * self.ex_qout
            if not isinstance(real_ex_qout, int):
                m += real_ex_qout >= 0

            # # if exogenous contracts are not force, we have a choice of signing
            # # them.
            # # TODO: we already define these variables as booleans
            # if not self.force_exogenous:
            #     m += ex_qin_signed >= 0  # typing: ignore
            #     m += ex_qin_signed <= 1  # typing: ignore
            #     m += ex_qout_signed >= 0  # typing: ignore
            #     m += ex_qout_signed <= 1  # typing: ignore

            # implement allow_oversales: When given total input cannot be greater
            # than total output and vice versa
            if allow_oversales:
                m += qout_total >= qin_total
            else:
                m += qout_total <= qin_total
                # if best and (not self.force_exogenous or (self.ex_qout < self.n_lines)):
                #     m += produced == qout_total

            # these constraints are unnecessary. They are here to speed up computation
            # when we know that we are finding the best possible utility value
            if best:
                # m += real_ex_qout <= produced
                if allow_oversales:
                    # Should produce everything I buy to avoid paying storage_cost
                    m += produced == qin_total
                else:
                    # Should produce everything I buy to avoid paying delivery_penalty
                    m += real_qout == qout_total
                # should never sign sell contracts that I cannot honor (if possible)
                # m += qout_total <= produced
                # should never commit to sales above my production capacity
                if not self.force_exogenous or (self.ex_qout < self.n_lines):
                    m += qout_total <= self.n_lines
                if not self.force_exogenous or (self.ex_qin < self.n_lines):
                    m += qin_total <= self.n_lines

            # RUN the optimization
            op = mp.maximize if best else mp.minimize

            exp = (
                # income from selling negotiated products
                real_qout * uout
                # income from selling exogenous contracts that can be honored
                + real_ex_qout * ex_uout
                # income from secured outputs. TODO remove this may be
                + secured_output_quantity * secured_output_unit_price
                # pay for all inputs I commit to through negotiation
                - qin * uin  # typing: ignore
                # pay for all exogenous contracts I sign
                - ex_qin_signed * self.ex_qin * ex_uin  # typing: ignore
                # pay for secured quantities. TODO remove this may be
                - secured_input_quantity * secured_input_unit_price
                # pay for production
                - self.production_cost * produced  # typing: ignore
            )
            if allow_oversales:
                # we may pay delivery penalty but never storage cost
                # pay for delivery penalty if I could not honor all sale contracts
                exp = (
                    exp - self.delivery_penalty * (qout_total - produced) * scale
                )  # typing: ignore
            else:
                # we may pay storage cost but never delivery penalty
                # pay for storage cost if I could not sell the produce of all inputs I received
                exp = (
                    exp
                    - self.storage_cost * (qin_total - real_qout - real_ex_qout) * scale
                )  # typing: ignore

            m.objective = op(exp)
            status = m.optimize()
            # breakpoint()
            if (
                status != mp.OptimizationStatus.OPTIMAL
                and status != mp.OptimizationStatus.FEASIBLE
            ):
                warnings.warn("Infeasible solution to ufun max/min")
                return None, [None] * 9
            qin, qout, produced = qin.x, qout.x, produced.x
            if not self.force_exogenous:
                ex_qin, ex_qout = (
                    ex_qin_signed.x * self.ex_qin,
                    self.ex_qout * ex_qout_signed.x,
                )
            else:
                ex_qin, ex_qout = self.ex_qin, self.ex_qout
            return m.objective_value, [
                int(qin),
                int(uin),
                int(qout),
                int(uout),
                int(ex_qin),
                ex_uin,
                int(ex_qout),
                ex_uout,
                int(produced),
            ]

        u1, vals1 = make_program(
            best, False, n_input_negs, n_output_negs
        )  # typing: ignore
        u2, vals2 = make_program(
            best, True, n_input_negs, n_output_negs
        )  # typing: ignore
        if u1 is None and u2 is None:
            raise ValueError("Cannot find the limit outcome")
        if u2 is None or (not best and u1 < u2) or (best and u1 > u2):
            utility, vals = u1, vals1
        else:
            utility, vals = u2, vals2
        if check:
            # breakpoint()
            (qin, uin, qout, uout, ex_qin, ex_uin, ex_qout, ex_uout, produced) = vals
            actual = self.from_offers(
                [
                    (qin, 0, uin),
                    (ex_qin, 0, ex_qin),
                    (secured_input_quantity, 0, secured_input_unit_price),
                    (qout, 0, uout),
                    (ex_qout, 0, ex_qout),
                    (secured_output_quantity, 0, secured_output_unit_price),
                ],
                [False] * 3 + [True] * 3,
            )
            assert (
                utility == actual
            ), f"Optimal value {utility} != {actual}\n{utility}-{vals}"
        return UFunLimit(*tuple([utility] + vals))

    def find_limit_greedy(
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
        if n_input_negs is None:
            n_input_negs = self.n_input_negs
        if n_output_negs is None:
            n_output_negs = self.n_output_negs
        if best:
            in_price, out_price = (
                self.input_prange[0],
                self.output_prange[1],
            )
            u, (
                in_quantity,
                out_quantity,
                producible,
                ex_in,
                ex_out,
            ) = self._best_greedy(
                n_input_negs,
                n_output_negs,
                secured_input_quantity,
                secured_input_unit_price,
                secured_output_quantity,
                secured_output_unit_price,
            )
        else:
            in_price, out_price = (
                self.input_prange[1],
                self.output_prange[0],
            )
            u, (
                in_quantity,
                out_quantity,
                producible,
                ex_in,
                ex_out,
            ) = self._worst_greedy(
                n_input_negs,
                n_output_negs,
                secured_input_quantity,
                secured_input_unit_price,
                secured_output_quantity,
                secured_output_unit_price,
            )
        ex_uin = self.ex_pin / self.ex_qin if self.ex_qin else 0
        ex_uout = self.ex_pout / self.ex_qout if self.ex_qout else 0
        if in_quantity == 0:
            in_price = self.input_prange[0] if best else self.input_prange[1]
        if out_quantity == 0:
            out_price = self.output_prange[1] if best else self.output_prange[0]
        return UFunLimit(
            utility=u,
            input_quantity=int(in_quantity),
            input_price=int(in_price),
            output_quantity=int(out_quantity),
            output_price=int(out_price),
            exogenous_input_quantity=int(ex_in),
            exogenous_output_quantity=int(ex_out),
            exogenous_input_price=ex_uin,
            exogenous_output_price=ex_uout,
            producible=int(producible),
        )

    def _best_greedy(
        self,
        n_input_negs,
        n_output_negs,
        secured_input_quantity,
        secured_input_unit_price,
        secured_output_quantity,
        secured_output_unit_price,
    ) -> Tuple[float, Tuple[int, int, int, int, int]]:
        """
        Returns the highest possible utility with the corresponding total
        input, output quantities

        Args:
            n_input_negs: Number of input negotiations. If not given it defaults
                          to the number of suppliers
            n_output_negs: Number of output negotiations. If not given it defaults
                          to the number of consumers

        Returns:
            A tuple with highest utility value and corresponding input/output
            quantities.

        Remarks:
            - Best input / output prices are always the minimum / maximum in
              the negotiation outcome space.
            - Most of the time you do not need to pass any arguments. Nevertheless,
              if you want the system to find the best utility and outcome for a
              single negotiation or a set of negotiations (i.e. some but not
              all currently running negotiations), you can pass them using the
              optional parameters.
        """

        nlines = self.n_lines
        (
            min_in,
            max_in,
            min_price_in,
            max_price_in,
            min_out,
            max_out,
            min_price_out,
            max_price_out,
        ) = self._ranges(
            self.input_qrange,
            self.input_prange,
            self.output_qrange,
            self.output_prange,
            n_input_negs,
            n_output_negs,
        )
        min_in += secured_input_quantity
        max_in += secured_input_quantity
        min_out += secured_output_quantity
        max_out += secured_output_quantity

        min_price_in += secured_input_quantity * secured_input_unit_price
        max_price_in += secured_input_quantity * secured_input_unit_price
        min_price_out += secured_output_quantity * secured_output_unit_price
        max_price_out += secured_output_quantity * secured_output_unit_price

        if nlines <= 0 or max_in <= 0 or max_out <= 0:
            return 0.0, (0, 0, 0, self.ex_qin, self.ex_qout)
        unit_price_in = max_price_in / max_in
        unit_price_out = max_price_out / max_out
        # we avoid bankruptcy by never producing more than what we can buy
        production_cost = min_price_in + self.production_cost
        can_be_bought = (
            self.current_balance // production_cost if production_cost else float("inf")
        )
        producible = min(
            max_out,
            nlines,
            max_in,
            can_be_bought,
        )
        max_price_out = max_price_out * producible // max_out
        max_out = producible
        best_margin = unit_price_out - unit_price_in - self.production_cost
        if best_margin > 0:
            qin, qout = max(producible, min_in), max(producible, min_out)
            return (
                self.from_aggregates(
                    qin=qin,
                    qout=qout,
                    pin=min_price_in * qin,
                    pout=max_price_out * producible,
                ),
                (qin, qout, min(qin, qout, self.n_lines), self.ex_qin, self.ex_qout),
            )
        producible = min(min_in, min_out, self.n_lines, can_be_bought)
        qin, qout = max(producible, min_in), max(producible, min_out)
        u1 = self.from_aggregates(
            qin=min_in,
            qout=min_out,
            pin=min_price_in * qin,
            pout=max_price_out * producible,
        )
        q = max(min_in, min_out)
        q = min(q, self.n_lines)
        u2 = self.from_aggregates(
            qin=q,
            qout=q,
            pin=min_price_in * qin,
            pout=max_price_out * producible,
        )
        if u1 > u2:
            return u1, (min_in, min_out, producible, self.ex_qin, self.ex_qout)
        return u2, (q, q, q, self.ex_qin, self.ex_qout)

    def _worst_greedy(
        self,
        n_input_negs,
        n_output_negs,
        secured_input_quantity,
        secured_input_unit_price,
        secured_output_quantity,
        secured_output_unit_price,
    ) -> Tuple[float, Tuple[int, int, int, int, int]]:
        """
        Returns the lowest possible utility with the corresponding total
        input, output quantities

        Args:
            n_input_negs: Number of input negotiations. If not given it defaults
                          to the number of suppliers
            n_output_negs: Number of output negotiations. If not given it defaults
                          to the number of consumers

        Returns:
            A tuple with highest utility value and corresponding input/output
            quantities.

        Remarks:
            - Worst input / output prices are always the maximum / minimum in
              the negotiation outcome space.
            - Most of the time you do not need to pass any arguments. Nevertheless,
              if you want the system to find the worst utility and outcome for a
              single negotiation or a set of negotiations (i.e. some but not
              all currently running negotiations), you can pass them using the
              optional parameters.
        """
        (_, max_in, _, max_price_in, min_out, _, min_price_out, _,) = self._ranges(
            self.input_qrange,
            self.input_prange,
            self.output_qrange,
            self.output_prange,
            n_input_negs,
            n_output_negs,
        )
        # min_in += secured_input_quantity
        max_in += secured_input_quantity
        min_out += secured_output_quantity
        # max_out += secured_output_quantity

        # min_price_in += secured_input_quantity * secured_input_unit_price
        max_price_in += secured_input_quantity * secured_input_unit_price
        min_price_out += secured_output_quantity * secured_output_unit_price
        # max_price_out += secured_output_quantity * secured_output_unit_price
        producible = min(max_in, min_out)
        return (
            self.from_aggregates(
                qin=max_in,
                qout=min_out,
                pin=max_price_in,
                pout=min_price_out,
            ),
            (max_in, min_out, producible, self.ex_qin, self.ex_qout),
        )

    def _ranges(
        self,
        input_qrange,
        input_prange,
        output_qrange,
        output_prange,
        n_input_negs=None,
        n_output_negs=None,
    ):
        # if I can sign exogenous contracts, then I can get no exogenous quantities
        # otherwise I must get at least the quantity in the exogenous contracts
        if self.force_exogenous:
            min_ex_in = self.ex_qin
            min_ex_out = self.ex_qout
            min_ex_price_in = self.ex_pin
            min_ex_price_out = self.ex_pout
        else:
            min_ex_in = min_ex_out = 0
            min_ex_price_in = min_ex_price_out = 0
        # maximum exogenous quantity is simply the exogenous quantity given
        max_ex_in = self.ex_qin
        # max_ex_out = self.ex_qout
        max_ex_price_in = self.ex_pin
        max_ex_price_out = self.ex_pout

        # if there are no negotiations for inputs/outputs, the corresponding
        # maximum quantity/price will be zero
        # max_neg_in = max_neg_out = 0
        # max_neg_price_in = max_neg_price_out = 0
        # I can always refuse to sign/accept everything which means the minimum
        # negotiated quantities and prices is always zero
        min_neg_in = min_neg_out = 0
        min_neg_price_in = min_neg_price_out = 0

        # If I can negotiate quantities, then the maximum possible value is the
        # maximum quantity per negotiation multiplied by the number of negotiations
        max_neg_in = input_qrange[1] * n_input_negs
        max_neg_price_in = input_prange[1] * n_input_negs
        max_neg_out = output_qrange[1] * n_output_negs
        max_neg_price_out = output_prange[1] * n_output_negs
        # totals are just exogenous + negotiated
        min_in = min_ex_in + min_neg_in
        max_in = max_ex_in + max_neg_in
        min_price_in = min_ex_price_in + min_neg_price_in
        min_price_out = min_ex_price_out + min_neg_price_out
        max_price_in = max_ex_price_in + max_neg_price_in
        max_price_out = max_ex_price_out + max_neg_price_out
        return (
            min_in,
            max_in,
            min_price_in,
            max_price_in,
            min_neg_out,
            max_neg_out,
            min_price_out,
            max_price_out,
        )
