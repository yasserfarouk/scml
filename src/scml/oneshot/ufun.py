import warnings
from typing import Iterable, Tuple, Union, List, Collection, Optional

from negmas import Contract
from negmas.utilities import UtilityFunction, UtilityValue
from negmas.outcomes import Outcome, Issue

from .common import QUANTITY, UNIT_PRICE, TIME

__all__ = ["OneShotUFun"]


class OneShotUFun(UtilityFunction):
    """
    Calculates the utility function of a list of contracts or offers.

    Args:
        owner: The `OneShotAgent` agent owning this utility function.
        pin: total price of exogenous inputs for this agent
        qin: total quantity of exogenous inputs for this agent
        pout: total price of exogenous outputs for this agent
        qout: total quantity of exogenous outputs for this agent.
        cost: production cost of the agent.
        storage_cost: storage cost per unit of input/output.
        delivery_penalty: penalty for failure to deliver one unit of output.
        input_agent: Is the agent an input agent which means that its input
                     product is the raw material
        output_agent: Is the agent an input agent which means that its input
                     product is the raw material
        normalize: If given the values returned will range between zero and one
                   Note that the minimum utility is not no-profit but maximum
                   loss

    Remarks:
        - The utility function assumes that the agent will have to pay for
          all its input products but will receive money only for the output
          products it could generate and sell.
        - The utility function respects production capacity (n. lines). The
          agent cannot produce more than the number of lines it has.
    """

    def __init__(
        self,
        owner: "OneShotAgent" = None,  # type: ignore
        awi: "OneShotAWI" = None,  # type: ignore
        pin: int = 0,
        qin: int = 0,
        pout: int = 0,
        qout: int = 0,
        production_cost: float = 0.0,
        storage_cost: float = 0.0,
        delivery_penalty: float = 0.0,
        input_agent: bool = False,
        output_agent: bool = False,
        normalize: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.__owner = owner
        self._awi = awi
        if not self._awi:
            self._awi = owner.awi
        self._pin, self._pout = pin, pout
        self._qin, self._qout = qin, qout
        self._production_cost, self._storage_cost, self._delivery_penalty = (
            production_cost,
            storage_cost,
            delivery_penalty,
        )
        self._input_agent, self._output_agent = input_agent, output_agent
        self._force_exogenous = self._awi.bb_read(
            "settings", "force_signing"
        ) or self._awi.bb_read("settings", "exogenous_force_max")
        self._public_trading_prices = self._awi.bb_read(
            "settings", "public_trading_prices"
        )

    def xml(self, issues) -> str:
        raise NotImplementedError("Cannot convert the ufun to xml")

    def __call__(self, offer) -> float:
        """
        Calculates the utility function given a single contract.

        Remarks:
            - This method calculates the utility value of a single offer assuming all other negotiations end in failure.
            - It can only be called for agents that exist in the first or last layer of the production graph.
        """
        if not self._input_agent and not self._output_agent:
            return float("-inf")
        return self.from_offers([offer], [self._input_agent])

    def from_contracts(self, contracts: Iterable[Contract]) -> float:
        """
        Calculates the utility function given a list of contracts
        """
        offers, outputs = [], []
        for c in contracts:
            if c.signed_at < 0:
                continue
            product = c.annotation["product"]
            is_output = product == self._awi.my_output_product
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

    def from_offers(self, offers: Iterable[Tuple], outputs: Iterable[bool]) -> float:
        """
        Calculates the utility value given a list of offers and whether each offer is for output or not.
        """
        qin, qout, pin, pout = 0, 0, 0, 0
        output_offers = []
        for offer, is_output in zip(offers, outputs):
            offer = self.outcome_as_tuple(offer)
            if is_output:
                output_offers.append(offer)
            else:
                qin += offer[QUANTITY]
                pin += offer[UNIT_PRICE] * offer[QUANTITY]
        output_offers = sorted(output_offers, key=lambda x: -x[UNIT_PRICE])
        producible = min(qin, self._awi.n_lines)
        for offer in output_offers:
            qout += offer[QUANTITY]
            if qout >= producible:
                qout = producible
                break
        return self.from_aggregates(qin, qout, pin, pout)

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
            qin: Input quantity.
            qout: Output quantity.
            pin: Input total price (i.e. unit price * qin).
            pout: Output total price (i.e. unit price * qin).
            production_cost: production cost per unit of output manufactured.
            storage_cost: storage cost per unit for inputs not used.
            delivery_penalty: deletion penalty per unit for outputs not delivered.

        Remarks:
            - The utility function assumes that the agent will have to pay for
              all its input products but will receive money only for the output
              products it could generate and sell.
            - The utility function respects production capacity (n. lines). The
              agent cannot produce more than the number of lines it has.

        """
        qin += self._qin
        qout += self._qout
        pin += self._pin
        pout += self._pout
        paid = pin
        lines = self._awi.n_lines
        produced = min(qin, lines, qout)
        received = pout * produced / qout if qout else 0
        if self._public_trading_prices:
            tpi = self._awi._world.trading_prices[self._awi.my_input_product]  # type: ignore
            tpo = self._awi._world.trading_prices[self._awi.my_output_product]  # type: ignore
        else:
            tpi = self._awi._world.catalog_prices[self._awi.my_input_product]  # type: ignore
            tpo = self._awi._world.catalog_prices[self._awi.my_output_product]  # type: ignore
        return (
            received
            - paid
            - self._production_cost * produced
            - self._storage_cost * tpi * max(0, qin - qout)
            - self._delivery_penalty * tpo * max(0, qout - qin)
        )

    def breach_level(self, qin: int = 0, qout: int = 0):
        """Calculates the breach level that would result from a given quantities"""
        qin += self._qin
        qout += self._qout
        if max(qin, qout) < 1:
            return 0
        return abs(qin - qout) / max(qin, qout)

    def is_breach(self, qin: int = 0, qout: int = 0):
        """Whether the given quantities would lead to a breach."""
        qin += self._qin
        qout += self._qout
        return qin != qout

    def best(
        self,
        input_issues=None,
        output_issues=None,
        n_input_negs=None,
        n_output_negs=None,
    ) -> Tuple[float, Tuple[int, int]]:
        """
        Returns the highest possible utility with the corresponding total
        input, output quantities

        Args:
            input_issues: The input issues in the same order defined in
                          `scml.scml2020.common` (quantity, time, unit price).
                          If None, then the `AWI` will be used to find them.
            output_issues: The output issues in the same order defined in
                          `scml.scml2020.common` (quantity, time, unit price).
                          If None, then the `AWI` will be used to find them.
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

        nlines = self._awi.n_lines
        (
            min_in,
            max_in,
            min_price_in,
            max_price_in,
            min_out,
            max_out,
            min_price_out,
            max_price_out,
        ) = self._ranges(input_issues, output_issues, n_input_negs, n_output_negs)
        if nlines <= 0 or max_in <= 0 or max_out <= 0:
            return 0.0, (0, 0)
        unit_price_in = max_price_in / max_in
        unit_price_out = max_price_out / max_out
        producible = min(max_out, nlines, max_in)
        max_price_out = max_price_out * producible // max_out
        max_out = producible
        best_margin = unit_price_out - unit_price_in - self._production_cost
        if best_margin > 0:
            qin = qout = producible
            qin, qout = max(qin, min_in), max(qout, min_out)
            return (
                self.from_aggregates(
                    qin=qin - self._qin,
                    qout=qout - self._qout,
                    pin=min_price_in - self._pin,
                    pout=max_price_out - self._pout,
                ),
                (qin, qout),
            )
        u1 = self.from_aggregates(
            qin=min_in - self._qin,
            qout=min_out - self._qout,
            pin=min_price_in - self._pin,
            pout=max_price_out - self._pout,
        )
        q = max(min_in, min_out)
        u2 = self.from_aggregates(
            qin=q - self._qin,
            qout=q - self._qout,
            pin=min_price_in - self._pin,
            pout=max_price_out - self._pout,
        )
        if u1 > u2:
            return u1, (min_in, min_out)
        return u2, (q, q)

    @property
    def max_utility(self):
        """The maximum possible utility value"""
        return self.best()[0]

    def worst(
        self,
        input_issues=None,
        output_issues=None,
        n_input_negs=None,
        n_output_negs=None,
    ) -> Tuple[float, Tuple[int, int]]:
        """
        Returns the lowest possible utility with the corresponding total
        input, output quantities

        Args:
            input_issues: The input issues in the same order defined in
                          `scml.scml2020.common` (quantity, time, unit price).
                          If None, then the `AWI` will be used to find them.
            output_issues: The output issues in the same order defined in
                          `scml.scml2020.common` (quantity, time, unit price).
                          If None, then the `AWI` will be used to find them.
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
        (
            _,
            max_in,
            _,
            max_price_in,
            min_out,
            _,
            min_price_out,
            _,
        ) = self._ranges(input_issues, output_issues, n_input_negs, n_output_negs)

        return (
            self.from_aggregates(
                qin=max_in - self._qin,
                qout=min_out - self._qout,
                pin=max_price_in - self._pin,
                pout=min_price_out - self._pout,
            ),
            (max_in, min_out),
        )

    @property
    def min_utility(self):
        """The minimum possible utility value"""
        return self.worst()[0]

    def _ranges(
        self,
        input_issues=None,
        output_issues=None,
        n_input_negs=None,
        n_output_negs=None,
    ):
        if input_issues is None:
            input_issues = self._awi.current_input_issues
        if output_issues is None:
            output_issues = self._awi.current_output_issues
        if n_input_negs is None:
            n_input_negs = len(self._awi.my_suppliers)
        if n_output_negs is None:
            n_output_negs = len(self._awi.my_consumers)
        # if I can sign exogenous contracts, then I can get no exogenous quantities
        # otherwise I must get at least the quantity in the exogenous contracts
        if self._force_exogenous:
            min_ex_in = self._qin
            min_ex_out = self._qout
            min_ex_price_in = self._pin
            min_ex_price_out = self._pout
        else:
            min_ex_in = min_ex_out = 0
            min_ex_price_in = min_ex_price_out = 0
        # maximum exogenous quantity is simply the exogenous quantity given
        max_ex_in = self._qin
        self._qout
        max_ex_price_in = self._pin
        max_ex_price_out = self._pout

        # if there are no negotiations for inputs/outputs, the corresponding
        # maximum quantity/price will be zero
        max_neg_in = max_neg_out = 0
        max_neg_price_in = max_neg_price_out = 0
        # I can always refuse to sign/accept everything which means the minimum
        # negotiated quantities and prices is always zero
        min_neg_in = min_neg_out = 0
        min_neg_price_in = min_neg_price_out = 0

        # If I can negotiate quantities, then the maximum possible value is the
        # maximum quantity per negotiation multiplied by the number of negotiations
        if input_issues:
            max_neg_in = int(input_issues[QUANTITY].max_value) * n_input_negs
            max_neg_price_in = input_issues[UNIT_PRICE].max_value * max_neg_in
        if output_issues:
            max_neg_out += int(output_issues[QUANTITY].max_value) * n_output_negs
            max_neg_price_out = output_issues[UNIT_PRICE].max_value * max_neg_out
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

    def utility_range(
        self,
        issues: List[Issue] = None,
        outcomes: Collection[Outcome] = None,
        infeasible_cutoff: Optional[float] = None,
        return_outcomes=False,
        max_n_outcomes=1000,
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
        if not return_outcomes:
            return self.min_utility, self.max_utility
        if self._input_agent and self._output_agent:
            raise ValueError(
                "Cannot find utility_range for middle agents "
                "because we do not whether or not this is a "
                "selling or buying negotiation"
            )
        if issues is not None:
            if self._input_agent:
                input_issues, output_issues = [], self._awi.current_output_issues
            else:
                output_issues, input_issues = [], self._awi.current_input_issues
        else:
            input_issues = self._awi.current_input_issues
            output_issues = self._awi.current_output_issues
        t = self._awi.current_step
        worst_in_price, best_in_price = (
            (
                input_issues[UNIT_PRICE].max_value,
                input_issues[UNIT_PRICE].min_value,
            )
            if input_issues
            else (0, 0)
        )
        worst_out_price, best_out_price = (
            (
                output_issues[UNIT_PRICE].min_value,
                output_issues[UNIT_PRICE].max_value,
            )
            if output_issues
            else (0, 0)
        )
        worst_u, (worst_in_quantity, worst_out_quantity) = self.worst(
            input_issues, output_issues, 1, 1
        )
        best_u, (best_in_quantity, best_out_quantity) = self.worst(
            input_issues, output_issues, 1, 1
        )
        return (
            worst_u,
            best_u,
            (worst_out_quantity, t, worst_out_price)
            if self._input_agent
            else (worst_in_quantity, t, worst_in_price),
            (best_out_quantity, t, best_out_price)
            if self._input_agent
            else (best_in_quantity, t, best_in_price),
        )
