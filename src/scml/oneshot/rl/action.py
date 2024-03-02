"""
Defines ways to encode and decode actions.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
from attr import define
from gymnasium import Space, spaces
from negmas.gb.common import ResponseType, field
from negmas.helpers import distribute_integer_randomly
from negmas.outcomes.issue_ops import itertools
from negmas.sao.common import SAOResponse

from scml.oneshot.awi import OneShotAWI
from scml.oneshot.context import BaseContext
from scml.scml2019.common import QUANTITY
from .helpers import recover_offers, encode_given_offers

__all__ = [
    "ActionManager",
    "FlexibleActionManager",
    "DefaultActionManager",
]


@define(frozen=True)
class ActionManager(ABC):
    """
    Manges actions of an agent in an RL environment.
    """

    context: BaseContext
    continuous: bool = False
    n_suppliers: int = field(init=False, default=8)
    n_consumers: int = field(init=False, default=8)
    n_partners: int = field(init=False, default=16)

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
        _ = awi, responses
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement `encode`."
        )


def safemin(x: Iterable | int | float | str):
    if isinstance(x, Iterable):
        return min(x)
    return x


@define(frozen=True)
class FlexibleActionManager(ActionManager):
    """
    An action manager that matches any context.

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

    capacity_multiplier: int = 1
    n_prices: int = 2
    max_group_size: int = 2
    reduce_space_size: bool = True
    extra_checks: bool = False
    max_quantity: int = field(init=False, default=10)

    def __attrs_post_init__(self):
        p = self.context.extract_context_params(self.reduce_space_size)
        if p.nlines:
            object.__setattr__(
                self, "max_quantity", self.capacity_multiplier * p.nlines
            )
            object.__setattr__(self, "n_consumers", p.nconsumers)
            object.__setattr__(self, "n_suppliers", p.nsuppliers)
        object.__setattr__(self, "n_partners", self.n_suppliers + self.n_consumers)

    def make_space(self) -> spaces.MultiDiscrete | spaces.Box:
        """Creates the action space"""
        return (
            spaces.MultiDiscrete(
                np.asarray(
                    [self.max_quantity + 1, self.n_prices] * self.n_partners
                ).flatten()
            )
            if not self.continuous
            else spaces.Box(0.0, 1.0, shape=(self.n_partners * 2,))
        )

    def decode(self, awi: OneShotAWI, action: np.ndarray) -> dict[str, SAOResponse]:
        """
        Generates offers to all partners from an encoded action. Default is to return the action as it is assuming it is a `dict[str, SAOResponse]`
        """
        action = action.reshape((action.size // 2, 2))
        if not (len(action) == self.n_partners):
            raise AssertionError(
                f"{len(action)=}, {self.n_partners=} ({self.n_suppliers=}, {self.n_consumers=})"
            )
        offers = recover_offers(
            action,
            awi,
            self.n_suppliers,
            self.n_consumers,
            self.max_group_size,
            self.continuous,
            self.n_prices,
        )
        separated_offers, responses = dict(), dict()
        nmis = awi.current_nmis
        for k, v in offers.items():
            if "+" not in k:
                separated_offers[k] = tuple(int(_) for _ in v) if v else v
                continue
            partners = k.split("+")
            if v is None:
                separated_offers |= dict(zip(partners, itertools.repeat(None)))
                continue
            q = v[QUANTITY]
            dist = distribute_integer_randomly(q, len(partners))
            separated_offers |= dict(zip(partners, ((_, v[1], v[-1]) for _ in dist)))

        for k, v in separated_offers.items():
            nmi = nmis.get(k, None)
            if nmi is None:
                continue
            if v is None:
                responses[k] = SAOResponse(ResponseType.END_NEGOTIATION, None)
                continue
            partner_offer = nmi.state.current_offer  # type: ignore
            if v == partner_offer:
                responses[k] = SAOResponse(ResponseType.ACCEPT_OFFER, partner_offer)
                continue
            responses[k] = SAOResponse(ResponseType.REJECT_OFFER, v)

        return responses

    def encode(self, awi: OneShotAWI, responses: dict[str, SAOResponse]) -> np.ndarray:
        """
        Receives offers for all partners and generates the corresponding action. Used mostly for debugging and testing.
        """
        offers = dict()
        for k, v in responses.items():
            if v.response == ResponseType.END_NEGOTIATION:
                offers[k] = None
                continue
            offers[k] = v.outcome
        encoded = encode_given_offers(
            offers,
            awi,
            self.n_suppliers,
            self.n_consumers,
            self.max_group_size,
            self.continuous,
        )

        return np.asarray(encoded, dtype=np.float32 if self.continuous else np.int32)


DefaultActionManager = FlexibleActionManager
"""The default action manager"""
