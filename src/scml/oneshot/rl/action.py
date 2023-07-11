"""
Defines ways to encode and decode actions.
"""
from __future__ import annotations

import random
import warnings
from abc import ABC, abstractmethod
from typing import Protocol

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
        """Encodes an action as an array"""
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
        action = action.reshape((self.n_partners, self.n_prices))
        assert (
            QUANTITY == 0 and UNIT_PRICE == 2
        ), f"We assume that quantity and price has indices 0, 2. If not, you need to modify the tuples below to put them in the correct index"
        responses = dict()
        partners = [_ for _ in awi.my_suppliers if not awi.is_system(_)]
        partners += [_ for _ in awi.my_consumers if not awi.is_system(_)]
        assert (
            len(partners) == self.n_partners
        ), f"{len(partners)=} while {self.n_partners=}:\n{partners=}"
        for partner, response in zip(partners, action, strict=True):
            neg = awi.current_negotiation_details["buy"].get(partner, None)
            if not neg:
                neg = awi.current_negotiation_details["sell"].get(partner, None)
            if not neg:
                continue
            partner_offer = neg.nmi.state.current_offer  # type: ignore
            if partner_offer is None:
                rtype = ResponseType.REJECT_OFFER
                outcome = (
                    response[0],
                    awi.current_step,
                    response[1],
                )
            elif response[0] <= 0:
                rtype = ResponseType.END_NEGOTIATION
                outcome = None
            elif (
                response[0] == partner_offer[QUANTITY]
                and response[1] == partner_offer[UNIT_PRICE]
            ):
                rtype = ResponseType.ACCEPT_OFFER
                outcome = partner_offer
            else:
                rtype = ResponseType.REJECT_OFFER
                outcome = (
                    response[0],
                    awi.current_step,
                    response[0],
                )

            responses[partner] = SAOResponse(rtype, outcome)
        return responses

    def encode(self, awi: OneShotAWI, responses: dict[str, SAOResponse]) -> np.ndarray:
        """
        Receives offers for all partners and generates the corresponding action. Used mostly for debugging and testing.
        """
        action = []
        partners = [_ for _ in awi.my_suppliers if not awi.is_system(_)]
        partners += [_ for _ in awi.my_consumers if not awi.is_system(_)]
        assert len(partners) == len(responses)
        for partner in partners:
            response = responses[partner]
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
        action = action.reshape((self.n_partners[0], self.n_prices))
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
            if partner_offer is None:
                rtype = ResponseType.REJECT_OFFER
                outcome = (
                    response[0],
                    awi.current_step,
                    response[1],
                )
            elif response[0] <= 0:
                rtype = ResponseType.END_NEGOTIATION
                outcome = None
            elif (
                response[0] == partner_offer[QUANTITY]
                and response[1] == partner_offer[UNIT_PRICE]
            ):
                rtype = ResponseType.ACCEPT_OFFER
                outcome = partner_offer
            else:
                rtype = ResponseType.REJECT_OFFER
                outcome = (
                    response[0],
                    awi.current_step,
                    response[0],
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


DefaultActionManager = FixedPartnerNumbersActionManager
"""The default action manager"""
