"""Implements a randomly behaving agent"""
from typing import List, Optional, Dict, Any

import numpy as np
from negmas import Contract, Breach, AgentMechanismInterface, MechanismState, Issue, Negotiator, RandomUtilityFunction
from negmas import AspirationNegotiator

from .indneg import IndependentNegotiationsAgent

__all__ = ["RandomAgent"]


class RandomAgent(IndependentNegotiationsAgent):
    """An agent that negotiates randomly."""

    def create_ufun(self, is_seller: bool, issues=None, outcomes=None):
        return RandomUtilityFunction(outcomes if outcomes is not None else Issue.enumerate(issues, astype=tuple)
                                     , reserved_value=0.0)
