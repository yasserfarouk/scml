"""
Defines ways to encode and decode actions.
"""
from __future__ import annotations

import warnings
from typing import Protocol

import numpy as np
from attr import define, field
from gymnasium import Space, spaces
from negmas.gb.common import ResponseType
from negmas.sao.common import SAOResponse

from scml.oneshot.awi import OneShotAWI
from scml.scml2019.common import QUANTITY, UNIT_PRICE

__all__ = [
    "ActionManager",
]


class ActionManager(Protocol):
    """
    Manges actions of an agent in an RL environment.
    """

    def make_space(self) -> Space:
        """Creates the action space"""
        ...

    def encode(self, awi: OneShotAWI, responses: dict[str, SAOResponse]) -> np.ndarray:
        """Encodes an action as an array"""
        ...

    def decode(self, awi: OneShotAWI, action: np.ndarray) -> dict[str, SAOResponse]:
        """Decodes an action from an array to a `PurchaseOrder` and a `CounterMessage`."""
        ...

    def is_valid(self, env) -> bool:
        """Checks that it is OK to use this observation manager with a given `OneShotEnv`"""
        ...


@define(frozen=True)
class DefaultActionManager:
    n_suppliers: int
    n_consumers: int
    n_quantities: int
    n_prices: int = 2
    n_partners: int = field(init=False)

    def __attrs_post_init__(self):
        object.__setattr__(self, "n_partners", self.n_suppliers + self.n_consumers)

    def make_space(self) -> Space:
        """Creates the action space"""
        print(self)
        return spaces.MultiDiscrete(
            np.asarray(
                [self.n_quantities + 1, self.n_prices] * self.n_partners
            ).flatten()
        )

    def decode(self, awi: OneShotAWI, action: np.ndarray) -> dict[str, SAOResponse]:
        """
        Generates offers to all partners from an encoded action. Default is to return the action as it is assuming it is a `dict[str, SAOResponse]`
        """
        action = action.reshape((self.n_partners, self.n_prices))
        # assert (
        #     action.shape[0] == self.n_partners
        # ), f"Action shape should be {(self.n_partners, self.n_prices)} but found {action.shape=}"
        # assert (
        #     action.shape[1] == self.n_prices
        # ), f"Action shape should be {(self.n_partners, self.n_prices)} but found {action.shape=}"
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

    def is_valid(self, env) -> bool:
        """Checks that it is OK to use this observation manager with a given `OneShotEnv`"""
        if env._n_suppliers != self.n_suppliers:
            return False
        if env._n_consumers != self.n_consumers:
            return False
        return True
