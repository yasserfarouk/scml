from typing import Any, Protocol

from scml.std.awi import StdAWI
from scml.std.common import StdState


class RewardFunction(Protocol):
    def before_action(self, awi: StdAWI) -> Any:
        ...

    def __call__(
        self, awi: StdAWI, action: dict[str, tuple[int, int, int]], info: Any
    ) -> float:
        ...


class DefaultRewardFunction(RewardFunction):
    def before_action(self, awi: StdAWI) -> float:
        return awi.current_balance

    def __call__(
        self, awi: StdAWI, action: dict[str, tuple[int, int, int]], info: float
    ):
        _ = action
        return awi.current_balance - info
