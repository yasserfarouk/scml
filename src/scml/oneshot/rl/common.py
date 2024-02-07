from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Union

import numpy as np
from negmas.helpers import get_class

from scml.oneshot.agent import OneShotAgent
from scml.oneshot.awi import OneShotAWI
from scml.oneshot.world import SCMLBaseWorld

__all__ = [
    "isin",
    "isinfloat",
    "isinclass",
    "isinobject",
    "Context",
    "RLState",
    "RLAction",
    "RLModel",
    "model_wrapper",
]

IterableOrValue = tuple[int, int] | set[int] | list[int] | int | np.ndarray
IterableOrValueFloat = (
    tuple[float, float] | set[float] | list[float] | float | np.ndarray
)
IterableOrValueClass = Iterable[str | type] | type | str
IterableOrValueObject = Iterable[str | Any] | Any
EPSILON = 1e-5


def isinobject(x: IterableOrValueClass, y: IterableOrValueClass):
    return isinclass(
        type(x) if not isinstance(x, Iterable) else [type(_) for _ in x], y
    )


def isinclass(x: IterableOrValueClass, y: IterableOrValueClass):
    """Checks that x is within the range specified by y. Ugly but works"""
    if not isinstance(x, Iterable) and not isinstance(y, Iterable):
        return issubclass(get_class(x), get_class(y))
    if not isinstance(x, Iterable):
        x = [x]
    if not isinstance(y, Iterable):
        y = [y]
    x = [get_class(_) for _ in x]
    y = [get_class(_) for _ in y]
    for a in x:
        for b in y:
            if issubclass(a, b):  # type: ignore
                break
        else:
            return False
    return True


def isin(x: IterableOrValue, y: IterableOrValue):
    """Checks that x is within the range specified by y. Ugly but works"""
    if not isinstance(x, Iterable) and not isinstance(y, Iterable):
        return x == y
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(y, np.ndarray):
        y = y.tolist()
    if isinstance(x, list):
        x = {_ for _ in x}
    if isinstance(y, list):
        y = {_ for _ in y}
    if isinstance(x, tuple):
        if isinstance(y, tuple):
            return y[0] <= x[0] <= y[-1]
        if not isinstance(y, Iterable):
            return x[0] == y == x[-1]
        x = set(list(range(x[0], x[-1])))
    if isinstance(y, tuple):
        if not isinstance(x, Iterable):
            return y[0] <= x <= y[-1]
        y = set(list(range(y[0], y[-1])))
    if not isinstance(x, Iterable):
        x = {x}
    if not isinstance(y, Iterable):
        y = {y}
    assert isinstance(x, set) and isinstance(
        y, set
    ), f"{x=} ({type(x)=}), {y=} ({type(y)})"
    return not x.difference(y)


def isinfloat(x: IterableOrValueFloat, y: IterableOrValueFloat):
    """Checks that x is within the range specified by y. Ugly but works"""
    if not isinstance(x, Iterable) and not isinstance(y, Iterable):
        return abs(x - y) < EPSILON
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(y, np.ndarray):
        y = y.tolist()
    if isinstance(x, tuple):
        if isinstance(y, tuple):
            return y[0] - EPSILON <= x[0] <= y[-1] + EPSILON
        if not isinstance(y, Iterable):
            return abs(x[0] - y) < EPSILON and abs(y - x[-1]) < EPSILON
    if isinstance(y, tuple):
        if not isinstance(x, Iterable):
            return y[0] - EPSILON <= x <= y[-1] + EPSILON
    if not isinstance(x, Iterable):
        x = [x]
    if not isinstance(y, Iterable):
        y = [y]
    for a in x:
        for b in y:
            if abs(a - b) < EPSILON:
                break
        else:
            return False
    return True


class Context(ABC):
    """A context used for generating worlds satisfying predefined conditions and testing for them"""

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    @abstractmethod
    def generate(
        self,
        types: tuple[type[OneShotAgent], ...] = tuple(),
        params: tuple[dict[str, Any], ...] | None = None,
    ) -> tuple[SCMLBaseWorld, tuple[OneShotAgent]]:
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
        world: SCMLBaseWorld,
        types: tuple[type[OneShotAgent], ...] = tuple(),
    ) -> bool:
        """Checks that the given world could have been generated from this context"""
        ...

    @abstractmethod
    def is_valid_awi(self, awi: OneShotAWI) -> bool:
        """Checks that the given AWI is connected to a world that could have been generated from this context"""
        ...

    @abstractmethod
    def contains_context(self, context: "Context") -> bool:
        """Checks that the any world generated from the given `context` could have been generated from this context"""
        ...

    def __contains__(self, other: "Union[SCMLBaseWorld, OneShotAWI, Context]") -> bool:
        if isinstance(other, Context):
            return self.contains_context(other)
        if isinstance(other, OneShotAWI):
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
