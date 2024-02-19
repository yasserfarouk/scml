from typing import Any, Protocol, runtime_checkable

from negmas import SAOResponse

from scml.oneshot.awi import OneShotAWI

__all__ = ["RewardFunction", "DefaultRewardFunction"]


@runtime_checkable
class RewardFunction(Protocol):
    """
    Represents a reward function.

    Remarks:
        - `before_action` is called before the action is executed for initialization and should return info to be passed to the call
        - `__call__` is called with the awi (to get the state), action and info and should return the reward

    """

    def before_action(self, awi: OneShotAWI) -> Any:
        """
        Called before executing the action from the RL agent to save any required information for
        calculating the reward in its return

        Remarks:
            The returned value will be passed as `info` to `__call__()` when it is time to calculate
            the reward.
        """
        ...

    def __call__(
        self, awi: OneShotAWI, action: dict[str, SAOResponse], info: Any
    ) -> float:
        """
        Called to calculate the reward to be given to the agent at the end of a step.

        Args:
            awi: `OneShotAWI` to access the agent's state
            action: The action (decoded) as a mapping from partner ID to responses to their last offer.
            info: Information generated from `before_action()`. You an use this to store baselines for calculating the reward

        Returns:
            The reward (a number) to be given to the agent at the end of the step.
        """
        ...


class DefaultRewardFunction(RewardFunction):
    """
    The default reward function of SCML

    Remarks:
        - The reward is the difference between the balance before the action and after it.

    """

    def before_action(self, awi: OneShotAWI) -> float:
        return awi.current_score

    def __call__(self, awi: OneShotAWI, action: dict[str, SAOResponse], info: float):
        _ = action
        return awi.current_score - info
