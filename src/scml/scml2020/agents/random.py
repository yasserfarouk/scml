"""Implements a randomly behaving agent"""

from negmas import RandomUtilityFunction

from .indneg import IndependentNegotiationsAgent

__all__ = ["RandomAgent"]


class RandomAgent(IndependentNegotiationsAgent):
    """An agent that negotiates randomly."""

    def create_ufun(self, is_seller: bool, issues=None, outcomes=None):
        if issues:
            outcomes = None
        return RandomUtilityFunction(
            issues=issues,
            outcomes=outcomes,
            reserved_value=0.0,
        )
