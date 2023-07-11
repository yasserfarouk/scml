import warnings
from typing import Any

from negmas.gb.common import ResponseType
from negmas.sao.common import SAOResponse

from scml.scml2019.common import QUANTITY, UNIT_PRICE

from ..policy import OneShotPolicy

MyState = Any

__all__ = ["OneShotDummyAgent"]


class OneShotDummyAgent(OneShotPolicy):
    def act(self, state: Any) -> Any:
        raise RuntimeError(f"This agent is not supposed to ever be called")

    # def decode_action(self, action: list[tuple[int, int]]) -> dict[str, SAOResponse]:
    #     """
    #     Generates offers to all partners from an encoded action. Default is to return the action as it is assuming it is a `dict[str, SAOResponse]`
    #     """
    #     responses = dict()
    #     partners = [_ for _ in self.awi.my_suppliers if not self.awi.is_system(_)]
    #     partners += [_ for _ in self.awi.my_consumers if not self.awi.is_system(_)]
    #     for partner, response in zip(partners, action, strict=True):
    #         neg = self.awi.current_negotiation_details["buy"].get(partner, None)
    #         if not neg:
    #             neg = self.awi.current_negotiation_details["sell"].get(partner, None)
    #         if not neg:
    #             continue
    #         current_offer = neg.nmi.state.current_offer  # type: ignore
    #         if response[0] <= 0:
    #             rtype = ResponseType.END_NEGOTIATION
    #             outcome = None
    #         elif response == (current_offer[QUANTITY], current_offer[UNIT_PRICE]):
    #             rtype = ResponseType.ACCEPT_OFFER
    #             outcome = current_offer
    #         else:
    #             rtype = ResponseType.REJECT_OFFER
    #             assert QUANTITY == 0 and UNIT_PRICE == 2
    #             outcome = (
    #                 current_offer[QUANTITY],
    #                 self.awi.current_step,
    #                 current_offer[UNIT_PRICE],
    #             )
    #
    #         responses[partner] = SAOResponse(rtype, outcome)
    #     return responses
    #
    # def encode_action(self, responses: dict[str, SAOResponse]) -> list[tuple[int, int]]:
    #     """
    #     Receives offers for all partners and generates the corresponding action. Used mostly for debugging and testing.
    #     """
    #     action = []
    #     partners = [_ for _ in self.awi.my_suppliers if not self.awi.is_system(_)]
    #     partners += [_ for _ in self.awi.my_consumers if not self.awi.is_system(_)]
    #     assert len(partners) == len(responses)
    #     for partner in partners:
    #         response = responses[partner]
    #         neg = self.awi.current_negotiation_details["buy"].get(partner, None)
    #         if not neg:
    #             neg = self.awi.current_negotiation_details["sell"].get(partner, None)
    #         if not neg:
    #             warnings.warn(
    #                 f"Cannot encode an action with a response for {partner} because no such partner currently exist. Will ignore it."
    #             )
    #             action.append((0, 0))
    #             continue
    #         current_offer = neg.nmi.state.current_offer  # type: ignore
    #         if response.response == ResponseType.END_NEGOTIATION:
    #             action.append((0, 0))
    #         elif response.response == ResponseType.ACCEPT_OFFER:
    #             assert (
    #                 current_offer == response.outcome
    #             ), f"Accepting an outcome different from the current offer!! {current_offer=}, {response.outcome=}"
    #             action.append((current_offer[QUANTITY], current_offer[UNIT_PRICE]))
    #         else:
    #             if response.outcome is None:
    #                 action.append((0, 0))
    #             else:
    #                 action.append(
    #                     (
    #                         response.outcome[QUANTITY],
    #                         response.outcome[UNIT_PRICE],
    #                     )
    #                 )
    #     return action
