from typing import Any

from scml.oneshot.rl.action import ActionManager
from scml.oneshot.context import GeneralContext
from scml.oneshot.rl.env import OneShotEnv
from scml.oneshot.rl.observation import ObservationManager
from scml.oneshot.rl.reward import DefaultRewardFunction, RewardFunction
from scml.std.agent import StdAgent
from scml.std.agents.nothing import StdPlaceholder
from scml.std.context import FixedPartnerNumbersStdContext

__all__ = ["StdEnv"]


class StdEnv(OneShotEnv):
    def __init__(
        self,
        action_manager: ActionManager,
        observation_manager: ObservationManager,
        reward_function: RewardFunction = DefaultRewardFunction(),
        render_mode=None,
        context: GeneralContext = FixedPartnerNumbersStdContext(),
        agent_type: type[StdAgent] = StdPlaceholder,
        agent_params: dict[str, Any] | None = None,
        extra_checks: bool = True,
        skip_after_negotiations: bool = True,
    ):
        super().__init__(
            action_manager,
            observation_manager,
            reward_function,
            render_mode,
            context,
            agent_type,
            agent_params,
            extra_checks,
            skip_after_negotiations,
        )
