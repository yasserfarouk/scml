from abc import ABC, abstractmethod
from typing import Any, Callable, Union

import numpy as np
from negmas import Agent, AgentWorldInterface, World

__all__ = ["isin", "Context", "RLState", "RLAction", "RLModel", "model_wrapper"]


def isin(x: int | tuple[int, int], y: tuple[int, int] | int):
    """Checks that x is within the range specified by y. Ugly but works"""
    if isinstance(x, tuple):
        if isinstance(y, tuple):
            return y[0] <= x[0] <= y[-1]
        return x[0] == y == x[-1]
    if isinstance(y, tuple):
        return y[0] <= x <= y[-1]
    return x == y


class Context(ABC):
    """A context used for generating worlds satisfying predefined conditions and testing for them"""

    @abstractmethod
    def generate(
        self,
        types: tuple[type[Agent], ...] = tuple(),
        params: tuple[dict[str, Any], ...] | None = None,
    ) -> tuple[World, tuple[Agent]]:
        """
        Generates a world with one or more agents to be controlled externally and returns both

        Args:
            agent_types: The types of a list of agents to be guaranteed to exist in the world
            agent_params: The parameters to pass to the constructors of these agents. None means no parameters for any agents

        Returns:
            The constructed world and a tuple of the agents created corresponding (in order) to the given agent types/params
        """
        ...

    @abstractmethod
    def is_valid_world(
        self,
        world: World,
        types: tuple[type[Agent], ...] = tuple(),
    ) -> bool:
        """Checks that the given world could have been generated from this context"""
        ...

    @abstractmethod
    def is_valid_awi(self, awi: AgentWorldInterface) -> bool:
        """Checks that the given AWI is connected to a world that could have been generated from this context"""
        ...

    @abstractmethod
    def contains_context(self, context: "Context") -> bool:
        """Checks that the any world generated from the given `context` could have been generated from this context"""
        ...

    def __contains__(self, other: "Union[World, AgentWorldInterface, Context]") -> bool:
        if isinstance(other, Context):
            return self.contains_context(other)
        if isinstance(other, AgentWorldInterface):
            return self.is_valid_awi(other)
        return self.is_valid_world(other)


RLState = np.ndarray
"""We assume that RL states are numpy arrays"""
RLAction = np.ndarray
"""We assume that RL actions are numpy arrays"""
RLModel = Callable[[RLState], RLAction]
"""A policy is a callable that receives a state and returns an action"""


def model_wrapper(model, deterministic: bool = False) -> RLModel:
    """Wraps a stable_baselines3 model as an RL model"""

    return lambda obs: model.predict(obs, deterministic=deterministic)[0]
