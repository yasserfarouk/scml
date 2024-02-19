from negmas import ResponseType

from ..agent import OneShotAgent
from ..policy import OneShotPolicy

__all__ = ["OneshotDoNothingAgent", "Placeholder"]


class OneshotDoNothingAgent(OneShotAgent):
    """An agent that does nothing.

    Remarks:

        Note that this agent will lose money whenever it is at the edges (i.e.
        it is an input or an output agent trading in raw material or final
        product).
    """

    def propose(self, negotiator_id, state):
        return None

    def respond(self, negotiator_id, state, source=None):
        return ResponseType.END_NEGOTIATION


class Placeholder(OneShotPolicy):
    """An agent that always raises an exception if called to negotiate. It is useful as a placeholder (for example for RL and MARL exposition)"""

    def act(self, state):
        raise RuntimeError("This agent is not supposed to ever be called")
