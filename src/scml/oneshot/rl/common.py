from abc import ABC, abstractmethod
from typing import Union

from negmas import Agent, World


def isin(x: int | tuple[int, int], y: tuple[int, int] | int):
    """Checks that x is within the range specified by y. Ugly but works"""
    if isinstance(x, tuple):
        if isinstance(y, tuple):
            return y[0] <= x[0] <= y[-1]
        return x[0] == y == x[-1]
    if isinstance(y, tuple):
        return y[0] <= x <= y[-1]
    return x == y


class WorldFactory(ABC):
    """Generates worlds satisfying predefined conditions and tests for them"""

    def __call__(self) -> tuple[World, Agent]:
        """Generates a world with one agent to be controlled externally and returns both"""
        ...

    @abstractmethod
    def is_valid_world(self, world: World) -> bool:
        """Checks that the given world could have been generated from this generator"""
        ...

    @abstractmethod
    def contains_factory(self, generator: "WorldFactory") -> bool:
        """Checks that the any world generated from the given `generator` could have been generated from this generator"""
        ...

    def __contains__(self, other: "Union[World, WorldFactory]") -> bool:
        if isinstance(other, WorldFactory):
            return self.contains_factory(other)
        return self.is_valid_world(other)
