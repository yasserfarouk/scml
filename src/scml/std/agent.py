from scml.oneshot.agent import EndingNegotiator, OneShotAgent, OneShotSyncAgent

__all__ = [
    "StdAgent",
    "StdSyncAgent",
    "EndingNegotiator",
]


class StdAgent(OneShotAgent):
    """
    Base class for all agents in the standard game.

    Remarks:
        - You can access all of the negotiators associated with the agent using
          `self.negotiators` which is a dictionary mapping the `negotiator_id` to
          a tuple of two values: The `SAONegotiator` object and a key-value context
          dictionary.
        - The `negotiator_id` associated with a negotiation with some partner will
          be the same as the agent ID of that partner. This means that all negotiators
          engaged with some partner over all simulation steps will have the same ID
          which is useful if you are keeping information about past negotiations and
          partner behavior.
    """


class StdSyncAgent(OneShotSyncAgent, StdAgent):
    """
    Base class for agents that negotiate synchronously by receiving all offers at once then responding to all of them at once.
    """
