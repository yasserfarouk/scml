from scml.oneshot.agents.nothing import OneshotDoNothingAgent
from scml.std.policy import StdPolicy

__all__ = ["StdDoNothingAgent", "StdPlaceholder"]
StdDoNothingAgent = OneshotDoNothingAgent


class StdPlaceholder(StdPolicy):
    """An agent that always raises an exception if called to negotiate. It is useful as a placeholder (for example for RL and MARL exposition)"""

    def act(self, state):
        raise RuntimeError("This agent is not supposed to ever be called")
