from typing import Any, Protocol

from scml.oneshot.awi import OneShotAWI
from scml.oneshot.common import OneShotState


class RewardFunction(Protocol):
    def before_action(self, awi: OneShotAWI) -> Any:
        ...

    def __call__(
        self, awi: OneShotAWI, action: dict[str, tuple[int, int, int]], info: Any
    ) -> float:
        ...


class DefaultRewardFunction(RewardFunction):
    def before_action(self, awi: OneShotAWI) -> float:
        return awi.current_balance

    def __call__(
        self, awi: OneShotAWI, action: dict[str, tuple[int, int, int]], info: float
    ):
        _ = action
        return awi.current_balance - info
