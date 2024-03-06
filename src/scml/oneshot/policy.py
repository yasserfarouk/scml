import random
from abc import ABC, abstractmethod
from typing import Any

from negmas.gb.common import ResponseType
from negmas.helpers.strings import itertools
from negmas.outcomes import Outcome
from negmas.sao.common import SAOResponse, SAOState

from .agent import OneShotSyncAgent
from .common import QUANTITY, UNIT_PRICE

__all__ = ["OneShotPolicy"]


class OneShotPolicy(OneShotSyncAgent, ABC):
    """
    A oneshot agent structured in three components, state encoder, policy (action) and action decoder.

    The agent is divided into three components:

    1. State encoder (encode_state()) which takes the current state of all
       negotiation mechanisms, access the awi as needed, and generates a
       **state** which can be of any type to be passed to the next component.
    2. Policy (act()) which takes the state generated from the state encoder
       and returns an action which may be encoded as any type to be passed to
       the next component. *The policy (i.e. `act` () method) is not supposed
       to access the AWI or any other members of the class. It is preferred to
       be a pure function*. This makes it easy to test the policy at predefined
       conditions (i.e. states) without having to construct a simulation.
    3. Action decoder (decode_action()) which takes the action generated from
       the policy and generates the appropriate set of responses to all partners.

    Remarks:
        - The simplest form of state encoder which is implemented by default is
          to return the `state` member of the AWI.
        - The simplest form of action encoding is to simply return the responses as
          a `dict[str, SAOResponse]` from `act` which is then passed as it is by
          `decode_action` . This is the default implementation of `decode_action`
    """

    def encode_state(self, mechanism_states: dict[str, SAOState]) -> Any:
        """
        Called to generate a state to be passed to the act() method. The default is all of `awi` of type `OneShotState`
        """
        return self.awi

    @abstractmethod
    def act(self, state: Any) -> Any:
        """
        The main policy. Generates an action given a state
        """
        offers = []
        for partner in itertools.chain(state.my_suppliers, state.my_consumers):
            # End the negotiation if it is already ended or randomly with some small probability
            if (
                random.random() < 0.025
                or partner not in state.mechanism_states.keys()
                or state.mechanism_states[partner].ended
            ):
                offers[partner] = (0, 0)
                continue
            outcome = self.awi.current_input_outcome_space.random_outcome()
            offers.append((outcome[QUANTITY], outcome[UNIT_PRICE]))
        return offers

    def decode_action(self, action: Any) -> dict[str, SAOResponse]:
        """
        Generates offers to all partners from an encoded action. Default is to return the action as it is assuming it is a `dict[str, SAOResponse]`
        """
        return action

    def encode_action(
        self, responses: dict[str, SAOResponse]
    ) -> dict[str, SAOResponse]:
        """
        Receives offers for all partners and generates the corresponding action. Used mostly for debugging and testing.
        """
        return responses

    def __call__(self, state):
        """A policy is a callable that receives a state and generates an action"""
        return self.act(state)

    def counter_all(
        self, offers: dict[str, Outcome | None], states: dict[str, SAOState]
    ) -> dict[str, SAOResponse]:
        """Calculate a response to all offers from all negotiators
        (negotiator ID is the key).

        Args:
            offers: Maps negotiator IDs to offers
            states: Maps negotiator IDs to offers AT the time the offers were made.

        Returns:
            A dictionary mapping negotiator ID to an `SAOResponse`. The response
            per agent consist of a tuple. In case of acceptance or ending the
            negotiation the second item of the tuple should be None. In case of
            rejection, the second item should be the counter offer.


        Remarks:
            - The response type CANNOT be WAIT.
            - If the system determines that a loop is formed, the agent may
            receive this call for a subset of negotiations not all of them.

        """
        return self.decode_action(self.act(self.encode_state(states)))

    def first_proposals(self) -> dict[str, Outcome | None]:
        """
        Gets a set of proposals to use for initializing the negotiation.

        Returns:
            A dictionary mapping each negotiator (in self.negotiators dict) to
            an outcome to be used as the first proposal if the agent is to start
            a negotiation.

        """
        partners = self.awi.my_partners

        def _state() -> SAOState:
            return SAOState(started=True, n_negotiators=2)

        responses = self.counter_all(
            offers=dict(zip(partners, itertools.repeat(None))),
            states=dict(zip(partners, itertools.repeat(_state()))),
        )
        return dict(
            zip(
                responses.keys(),
                [
                    None if k == ResponseType.END_NEGOTIATION else v.outcome
                    for k, v in responses.items()
                ],
            )
        )
