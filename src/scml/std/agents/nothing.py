from negmas import ResponseType

from ..agent import StdAgent
from ..policy import StdPolicy

__all__ = ["StdDoNothingAgent", "StdDummyAgent"]


class StdDoNothingAgent(StdAgent):
    """An agent that does nothing.

    Remarks:

        Note that this agent will lose money whenever it is at the edges (i.e.
        it is an input or an output agent trading in raw material or final
        product).
    """

    def propose(self, negotiator_id, state):
        return None

    def respond(self, negotiator_id, state):
        return ResponseType.END_NEGOTIATION


class StdDummyAgent(StdPolicy):
    """An agent that always raises an exception if called to negotiate. It is useful as a placeholder (for example for RL and MARL exposition)"""

    def act(self, state):
        raise RuntimeError(f"This agent is not supposed to ever be called")
