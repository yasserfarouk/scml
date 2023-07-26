"""
Defines ways to encode and decode actions.
"""
from __future__ import annotations

import random
import warnings
from abc import ABC, abstractmethod

import numpy as np
from attr import define, field
from gymnasium import Space, spaces
from negmas.gb.common import ResponseType
from negmas.sao.common import SAOResponse

from scml.common import integer_cut
from scml.oneshot.awi import OneShotAWI
from scml.oneshot.rl.common import isin
from scml.oneshot.rl.factory import (
    FixedPartnerNumbersOneShotFactory,
    LimitedPartnerNumbersOneShotFactory,
    OneShotWorldFactory,
)
from scml.scml2019.common import QUANTITY, UNIT_PRICE

__all__ = [
    "ActionManager",
    "FixedPartnerNumbersActionManager",
    "LimitedPartnerNumbersActionManager",
    "DefaultActionManager",
]


@define(frozen=True)
class ActionManager(ABC):
    """
    Manges actions of an agent in an RL environment.
    """

    factory: OneShotWorldFactory

    @abstractmethod
    def make_space(self) -> Space:
        """Creates the action space"""
        ...

    @abstractmethod
    def decode(self, awi: OneShotAWI, action: np.ndarray) -> dict[str, SAOResponse]:
        """Decodes an action from an array to a `PurchaseOrder` and a `CounterMessage`."""
        ...

    def encode(self, awi: OneShotAWI, responses: dict[str, SAOResponse]) -> np.ndarray:
        """Encodes an action as an array. This is only used for testing so it is optional"""
        ...


@define(frozen=True)
class FixedPartnerNumbersActionManager(ActionManager):
    n_prices: int = 2
    n_partners: int = field(init=False)
    n_suppliers: int = field(init=False)
    n_consumers: int = field(init=False)
    max_quantity: int = field(init=False)

    def __attrs_post_init__(self):
        assert isinstance(self.factory, FixedPartnerNumbersOneShotFactory)
        object.__setattr__(self, "n_suppliers", self.factory.n_suppliers)
        object.__setattr__(self, "n_consumers", self.factory.n_consumers)
        object.__setattr__(self, "max_quantity", self.factory.n_lines)
        object.__setattr__(self, "n_partners", self.n_suppliers + self.n_consumers)

    def make_space(self) -> Space:
        """Creates the action space"""
        return spaces.MultiDiscrete(
            np.asarray(
                [self.max_quantity + 1, self.n_prices] * self.n_partners
            ).flatten()
        )

    def decode(self, awi: OneShotAWI, action: np.ndarray) -> dict[str, SAOResponse]:
        """
        Generates offers to all partners from an encoded action. Default is to return the action as it is assuming it is a `dict[str, SAOResponse]`
        """
        action = action.reshape((self.n_partners, 2))
        assert (
            QUANTITY == 0 and UNIT_PRICE == 2
        ), f"We assume that quantity and price has indices 0, 2. If not, you need to modify the tuples below to put them in the correct index"
        responses = dict()
        partners = awi.my_partners
        assert (
            len(partners) == self.n_partners
        ), f"{len(partners)=} while {self.n_partners=}:\n{partners=}"
        nmis = awi.current_nmis
        for partner, response in zip(partners, action, strict=True):
            nmi = nmis.get(partner, None)
            if not nmi:
                continue
            partner_offer = nmi.state.current_offer  # type: ignore
            minprice = nmi.issues[UNIT_PRICE].min_value
            if partner_offer is None:
                rtype = ResponseType.REJECT_OFFER
                outcome = (
                    response[0],
                    awi.current_step,
                    response[1] + minprice,
                )
            elif response[0] <= 0:
                rtype = ResponseType.END_NEGOTIATION
                outcome = None
            elif (
                response[0] == partner_offer[QUANTITY]
                and response[1] + minprice == partner_offer[UNIT_PRICE]
            ):
                rtype = ResponseType.ACCEPT_OFFER
                outcome = partner_offer
            else:
                rtype = ResponseType.REJECT_OFFER
                outcome = (
                    response[0],
                    awi.current_step,
                    response[1] + minprice,
                )

            responses[partner] = SAOResponse(rtype, outcome)
        return responses

    def encode(self, awi: OneShotAWI, responses: dict[str, SAOResponse]) -> np.ndarray:
        """
        Receives offers for all partners and generates the corresponding action. Used mostly for debugging and testing.
        """
        action = []
        partners = awi.my_partners
        assert len(partners) == len(responses)
        nmis = awi.current_nmis
        for partner in partners:
            response = responses[partner]
            nmi = nmis.get(partner, None)
            if not nmi:
                warnings.warn(
                    f"Cannot encode an action with a response for {partner} because no such partner currently exist. Will ignore it."
                )
                action.append([0, 0])
                continue
            current_offer = nmi.state.current_offer  # type: ignore
            if response.response == ResponseType.END_NEGOTIATION:
                action.append([0, 0])
            elif response.response == ResponseType.ACCEPT_OFFER:
                assert (
                    current_offer == response.outcome
                ), f"Accepting an outcome different from the current offer!! {current_offer=}, {response.outcome=}"
                action.append([current_offer[QUANTITY], current_offer[UNIT_PRICE]])
            elif response.response == ResponseType.REJECT_OFFER:
                if response.outcome is None:
                    action.append([0, 0])
                else:
                    action.append(
                        [
                            response.outcome[QUANTITY],
                            response.outcome[UNIT_PRICE],
                        ]
                    )
            else:
                raise ValueError(f"Unacceptable response type {response}")
        return np.asarray(action).flatten()


@define(frozen=True)
class LimitedPartnerNumbersActionManager(ActionManager):
    n_prices: int = 2
    n_partners: tuple[int, int] = field(init=False)
    n_suppliers: tuple[int, int] = field(init=False)
    n_consumers: tuple[int, int] = field(init=False)
    max_quantity: tuple[int, int] = field(init=False)

    def __attrs_post_init__(self):
        if isinstance(self.factory, LimitedPartnerNumbersOneShotFactory):
            object.__setattr__(self, "n_suppliers", self.factory.n_suppliers)
            object.__setattr__(self, "n_consumers", self.factory.n_consumers)
            object.__setattr__(self, "max_quantity", self.factory.n_lines)
        elif isinstance(self.factory, FixedPartnerNumbersOneShotFactory):
            object.__setattr__(
                self,
                "n_suppliers",
                (self.factory.n_suppliers, self.factory.n_suppliers),
            )
            object.__setattr__(
                self,
                "n_consumers",
                (self.factory.n_consumers, self.factory.n_consumers),
            )
            object.__setattr__(
                self, "max_quantity", (self.factory.n_lines, self.factory.n_lines)
            )
        else:
            raise AssertionError(
                f"{self.__class__} does not support factories of type {self.factory.__class__}"
            )
        object.__setattr__(
            self,
            "n_partners",
            (
                self.n_suppliers[0] + self.n_consumers[0],
                self.n_suppliers[1] + self.n_consumers[1],
            ),
        )

    def make_space(self) -> Space:
        """Creates the action space"""
        return spaces.MultiDiscrete(
            np.asarray(
                [self.max_quantity[-1] + 1, self.n_prices] * self.n_partners[0]
            ).flatten()
        )

    def decode(self, awi: OneShotAWI, action: np.ndarray) -> dict[str, SAOResponse]:
        """
        Generates offers to all partners from an encoded action. Default is to return the action as it is assuming it is a `dict[str, SAOResponse]`
        """
        action = action.reshape((self.n_partners[0], 2))
        assert (
            QUANTITY == 0 and UNIT_PRICE == 2
        ), f"We assume that quantity and price has indices 0, 2. If not, you need to modify the tuples below to put them in the correct index"
        responses = dict()
        partners = awi.my_partners
        n_partners, extra = len(partners), []
        if n_partners > len(action):
            extra = partners[len(action) :]
            partners = partners[: len(action)]
        elif n_partners < len(action):
            qtotal = min(
                sum(_[0] for _ in action[n_partners:]),
                awi.quantity_range * n_partners,
            )
            action = action[:n_partners]
            random.seed(0)
            qs = integer_cut(qtotal, len(action), 0, awi.quantity_range)
            action = np.asarray(
                [(_[0] + q, _[1]) for _, q in zip(action, qs)], dtype=int
            )
        assert isin(
            len(partners), self.n_partners
        ), f"{len(partners)=} while {self.n_partners=}:\n{partners=}"
        for partner, response in zip(partners, action, strict=True):
            neg = awi.current_negotiation_details["buy"].get(partner, None)
            if not neg:
                neg = awi.current_negotiation_details["sell"].get(partner, None)
            if not neg:
                continue
            partner_offer = neg.nmi.state.current_offer  # type: ignore
            minprice = neg.nmi.issues[UNIT_PRICE].min_value
            if partner_offer is None:
                rtype = ResponseType.REJECT_OFFER
                outcome = (
                    response[0],
                    awi.current_step,
                    response[1] + minprice,
                )
            elif response[0] <= 0:
                rtype = ResponseType.END_NEGOTIATION
                outcome = None
            elif (
                response[0] == partner_offer[QUANTITY]
                and response[1] + minprice == partner_offer[UNIT_PRICE]
            ):
                rtype = ResponseType.ACCEPT_OFFER
                outcome = partner_offer
            else:
                rtype = ResponseType.REJECT_OFFER
                outcome = (
                    response[0],
                    awi.current_step,
                    response[1] + minprice,
                )

            responses[partner] = SAOResponse(rtype, outcome)
        for p in extra:
            responses[p] = SAOResponse(ResponseType.END_NEGOTIATION, None)
        return responses

    def encode(self, awi: OneShotAWI, responses: dict[str, SAOResponse]) -> np.ndarray:
        """
        Receives offers for all partners and generates the corresponding action. Used mostly for debugging and testing.
        """
        action = []
        partners = awi.my_partners

        for partner in partners:
            response = responses.get(partner, None)
            if not response:
                action.append([0, 0])
                continue
            neg = awi.current_negotiation_details["buy"].get(partner, None)
            if not neg:
                neg = awi.current_negotiation_details["sell"].get(partner, None)
            if not neg:
                warnings.warn(
                    f"Cannot encode an action with a response for {partner} because no such partner currently exist. Will ignore it."
                )
                action.append([0, 0])
                continue
            current_offer = neg.nmi.state.current_offer  # type: ignore
            if response.response == ResponseType.END_NEGOTIATION:
                action.append([0, 0])
            elif response.response == ResponseType.ACCEPT_OFFER:
                assert (
                    current_offer == response.outcome
                ), f"Accepting an outcome different from the current offer!! {current_offer=}, {response.outcome=}"
                action.append([current_offer[QUANTITY], current_offer[UNIT_PRICE]])
            elif response.response == ResponseType.REJECT_OFFER:
                if response.outcome is None:
                    action.append([0, 0])
                else:
                    action.append(
                        [
                            response.outcome[QUANTITY],
                            response.outcome[UNIT_PRICE],
                        ]
                    )
            else:
                raise ValueError(f"Unacceptable response type {response}")
        return np.asarray(action).flatten()


@define(frozen=True)
class UnconstrainedActionManager(ActionManager):
    """
    An action manager that matches any factory.

    Args:
        n_prices: Number of distinct prices allowed in the action.
        max_quantity: Maximum allowed quantity to offer in any negotiation. The number of quantities is one plus that because zero is allowed to model ending negotiation.
        n_partners: Maximum of partners allowed in the action.

    Remarks:
        - This action manager will always generate offers that are within the
          price and quantity limits given in its parameters. Wen decoding them,
          it will scale them up so that the maximum corresponds to the actual value
          in the world it finds itself. For example, if `n_prices` is 10 and the world
          has only two prices currently in the price issue, it will use any value less than
          5 as the minimum price and any value above 5 as the maximum price. If on the other
          hand the current price issue has 20 values, then it will scale by multiplying the
          number given in the encoded action (ranging from 0 to 9) by 19/9 which makes it range
          from 0 to 19 which is what is expected by the world.
        - This action manager will adjust offers for different number of partners as follows:
          - If the true number of partners is larger than `n_partners` used by this action manager,
            it will simply use `n_partners` of them and always end negotiations with the rest of them.
          - If the true number of partners is smaller than `n_partners`, it will use the first `n_partners`
            values in the encoded action and increase the quantities of any counter offers (i.e. ones in
            which the response is REJECT_OFFER) by the amount missing from the ignored partners in the encoded
            action up to the maximum quantities allowed by the current negotiation context. For example, if
            `n_partneers` is 4 and we have only 2 partners in reality, and the received quantities from
            partners were [4, 3] while the maximum quantity allowed is 10 and the encoded action was
            [2, *, 3, *, 2, *, 1, *] (where we ignored prices), then the encoded action will be converted to
            [(Reject, 5, *), (Accept, 3, *)] where the 3 extra units that were supposed to be offered to the
            last two partners are moved to the first partner. If the maximum quantity allowed was 4 in that
            example, the result will be [(Reject, 4, *), (Accept, 3, *)].

    """

    n_prices = 10
    max_quantity = 10
    n_partners = 8

    def make_space(self) -> Space:
        """Creates the action space"""
        return spaces.MultiDiscrete(
            np.asarray(
                [self.max_quantity + 1, self.n_prices] * self.n_partners
            ).flatten()
        )

    def adjust_reponses_to_partners(
        self, awi: OneShotAWI, action: np.ndarray, partners: list[str], randomize=False
    ) -> tuple[np.ndarray, list[str]]:
        """Called to adjust responses to partners.

        Returns:
            A tuple of action and selected partners.

        Remarks:
            - The input is of shape (N, 2) where N can be any nonzero positive integer
            - The first output must have the shape (n_partners,  2) and all values in the first column must be less than n_quantities while all
              values in the second column must be greater than n_prices.
            - The second output must lave length n_partners and specifies the partners with which not to end the negotiation necessarily.
        """
        partner_offers = awi.current_offers
        n_partners, n_action = len(partners), len(action)
        if n_partners == n_action:
            return action, partners
        if n_partners > n_action:
            return action, partners[:n_action]
        # find the total quantity to be distributed.
        qtotal = sum(q for q, _ in action[n_partners:])
        if qtotal <= 0:
            return action[:n_partners], partners
        # We need to distribute some quantities from the part of the encoded action we ignore if possible
        # find all partners we can distribute to and their corresponding availability to receive extra quantity
        # If the action for a given partner specifies acceptance (same as the offer from that partner), we cannot
        # distribute to it
        action = action[:n_partners]
        available_partners, available_quantities = [], []
        received_offers = []
        for _ in range(n_partners):
            o = partner_offers.get(partners[_], (float("inf"), 0, float("inf")))
            if o is None:
                received_offers.append((float("inf"), float("inf")))
                continue
            received_offers.append((o[QUANTITY], o[UNIT_PRICE]))
        for i, (my_offer, partner_offer) in enumerate(zip(action, received_offers)):
            if my_offer[0] == 0 or (
                my_offer[0] == partner_offer[0] and my_offer[1] == partner_offer[1]
            ):
                continue
            # quantities cannot exceed the maximum quantity allowed
            q = self.max_quantity - my_offer[0]
            if q <= 0:
                continue
            available_partners.append(i)
            available_quantities.append(q)
        if not available_partners:
            # if we cannot distribute anything, just return the first n_partners offers as the new action
            return action, partners
        qtotal = min(qtotal, sum(available_quantities))
        assert qtotal > 0
        # we can distribute something here
        qs = integer_cut(
            qtotal,
            len(available_quantities),
            0,
            available_quantities,
            randomize=randomize,
        )
        added = np.zeros(n_partners, dtype=int)
        for i, q in zip(available_partners, qs, strict=True):
            added[i] = q
        action = np.asarray(
            [(_[0] + q, _[1]) for _, q in zip(action, added)], dtype=int
        )
        return action, partners

    def decode(self, awi: OneShotAWI, action: np.ndarray) -> dict[str, SAOResponse]:
        """
        Generates offers to all partners from an encoded action. Default is to return the action as it is assuming it is a `dict[str, SAOResponse]`
        """
        action = action.reshape((action.size // 2, 2))
        assert (
            len(action) == self.n_partners
        ), f"{len(action)=} while {self.n_partners=}"
        assert (
            QUANTITY == 0 and UNIT_PRICE == 2
        ), f"We assume that quantity and price has indices 0, 2. If not, you need to modify the tuples below to put them in the correct index"
        nmis = awi.current_nmis
        partners = awi.my_partners
        action, partners = self.adjust_reponses_to_partners(awi, action, partners)
        n_partners = len(partners)
        assert (
            len(action) == n_partners
        ), f"{len(action)=} but {len(partners)=} even after adjustment"
        # scale quantities and prices in action to match the current issues
        scaled = []
        for partner, (q, p) in zip(partners, action, strict=True):
            nmi = nmis.get(partner, None)
            if not nmi:
                warnings.warn(
                    f"Did not find {partner} in the list of partners"
                    f"\n{partners=}\n{awi.my_partners=}\n{action=}"
                )
                scaled.append((0, 0))
                continue
            qscale = nmi.issues[QUANTITY].max_value / (self.max_quantity - 1)
            pscale = (nmi.issues[UNIT_PRICE].max_value + 1) / self.n_prices
            scaled.append(
                (
                    min(self.max_quantity, int(q * qscale + 0.5)),
                    min(self.n_prices - 1, int(p * pscale + 0.5)),
                )
            )
        action = np.asarray(scaled, dtype=int)
        responses = dict()
        # assert (
        #     n_partners == self.n_partners
        # ), f"{len(awi.my_partners)=} while {self.n_partners=}:\n{awi.my_partners=}"
        for partner, response in zip(partners, action, strict=True):
            nmi = nmis.get(partner, None)
            if not nmi:
                continue
            minprice = nmi.issues[UNIT_PRICE].min_value
            partner_offer = nmi.state.current_offer  # type: ignore
            if partner_offer is None and not (response[0] == 0 and response[1] == 0):
                rtype = ResponseType.REJECT_OFFER
                outcome = (
                    response[0],
                    awi.current_step,
                    response[1] + minprice,
                )
            elif partner_offer is None and (response[0] == 0 and response[1] == 0):
                rtype = ResponseType.END_NEGOTIATION
                outcome = None
            elif response[0] <= 0 and response[1] <= 0:
                rtype = ResponseType.END_NEGOTIATION
                outcome = None
            elif (
                response[0] == partner_offer[QUANTITY]
                and response[1] + minprice == partner_offer[UNIT_PRICE]
            ) or (response[0] <= 0 and response[1] > 0):
                # acceptance is encoded as either returning same offer as the partner's or 0 quantity and nonzero price
                rtype = ResponseType.ACCEPT_OFFER
                outcome = partner_offer
            else:
                rtype = ResponseType.REJECT_OFFER
                outcome = (
                    response[0],
                    awi.current_step,
                    response[1] + minprice,
                )

            responses[partner] = SAOResponse(rtype, outcome)
        # end negotiation with anyone ignored
        ignored_partners = {_ for _ in awi.my_partners if _ not in set(partners)}
        for p in ignored_partners:
            responses[p] = SAOResponse(ResponseType.END_NEGOTIATION, None)
        return responses

    def encode(self, awi: OneShotAWI, responses: dict[str, SAOResponse]) -> np.ndarray:
        """
        Receives offers for all partners and generates the corresponding action. Used mostly for debugging and testing.
        """
        action = []
        partners = awi.my_partners
        nmis = awi.current_nmis

        for partner in partners:
            response = responses.get(partner, None)
            if not response:
                action.append([0, 0])
                continue
            nmi = nmis.get(partner, None)
            if not nmi:
                warnings.warn(
                    f"Cannot encode an action with a response for {partner} because no such partner currently exist. Will ignore it."
                )
                action.append([0, 0])
                continue
            current_offer = nmi.state.current_offer  # type: ignore
            if response.response == ResponseType.END_NEGOTIATION:
                action.append([0, 0])
            elif response.response == ResponseType.ACCEPT_OFFER:
                assert (
                    current_offer == response.outcome
                ), f"Accepting an outcome different from the current offer!! {current_offer=}, {response.outcome=}"
                action.append([current_offer[QUANTITY], current_offer[UNIT_PRICE]])
            elif response.response == ResponseType.REJECT_OFFER:
                if response.outcome is None:
                    action.append([0, 0])
                else:
                    action.append(
                        [
                            response.outcome[QUANTITY],
                            response.outcome[UNIT_PRICE],
                        ]
                    )
            else:
                raise ValueError(f"Unacceptable response type {response}")
        return np.asarray(action).flatten()


DefaultActionManager = UnconstrainedActionManager
"""The default action manager"""
