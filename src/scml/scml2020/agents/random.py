"""Implements a randomly behaving agent"""

from negmas import Issue
from negmas import RandomUtilityFunction

from .indneg import IndependentNegotiationsAgent

__all__ = ["RandomAgent"]


class RandomAgent(IndependentNegotiationsAgent):
    """An agent that negotiates randomly."""

    def create_ufun(self, is_seller: bool, issues=None, outcomes=None):
        return RandomUtilityFunction(
            outcomes if outcomes is not None else Issue.enumerate(issues, astype=tuple),
            reserved_value=0.0,
        )
