from random import shuffle
from typing import Any

from negmas.gb.common import ResponseType
from negmas.helpers import instantiate
from negmas.outcomes import Outcome
from negmas.sao.common import SAOResponse, SAOState

from scml.oneshot.agent import OneShotAgent
from scml.oneshot.agents import GreedyOneShotAgent
from scml.oneshot.context import ANACOneShotContext, Context

from ..policy import OneShotPolicy
from .action import ActionManager, UnconstrainedActionManager
from .common import RLAction, RLModel, RLState
from .observation import ObservationManager

__all__ = ["OneShotRLAgent"]


class OneShotRLAgent(OneShotPolicy):
    """A oneshot agent that can execute  trained RL models in appropriate worlds. It falls back to the given agent type otherwise

    Args:
        models: List of models to choose from.
        observation_managers: List of observation managers. Must be the same length as `models`
        action_managers: List of action managers of the same length as `models` or `None` to use the default action manager.
        fallback_type: A `OneShotAgent` type to use as a fall-back if the current world is not compatible with any observation/action managers
        fallback_params: Parameters of the `fallback_type`
        dynamic_context_switching: If `True`, the world is tested each step (instead of only at init) to find the appropriate model
        randomize_test_order: If `True`, the order at which the observation/action managers are checked for compatibility with the current world
                              is randomized.
        **kwargs: Any other OneShotPolicy parameters
    """

    def __init__(
        self,
        *args,
        models: list[RLModel] | tuple[RLModel, ...] = tuple(),
        observation_managers: list[ObservationManager]
        | tuple[ObservationManager, ...] = tuple(),
        action_managers: list[ActionManager] | tuple[ActionManager, ...] | None = None,
        fallback_type: type[OneShotAgent] = GreedyOneShotAgent,
        fallback_params: dict[str, Any] | None = None,
        dynamic_context_switching: bool = False,
        randomize_test_order: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._models = models
        if action_managers is None:
            action_managers = [
                UnconstrainedActionManager(ANACOneShotContext())
                for _ in observation_managers
            ]
        self._action_managers = action_managers
        self._obs_managers = observation_managers
        self._fallback_type = fallback_type
        self._dynamic_context_switching = dynamic_context_switching
        self._randomize_test_order = randomize_test_order
        self._fallback_params = (
            fallback_params if fallback_params is not None else dict()
        )
        self._valid_context: Context = None  # type: ignore
        self._valid_action_manager: ActionManager = None  # type: ignore
        self._valid_obs_manager: ObservationManager = None  # type: ignore
        self._valid_index: int = -1
        self._fallback_agent: OneShotAget = None  # type: ignore

    def context_switch(self):
        aolist = zip(
            self._action_managers, self._obs_managers, range(len(self._obs_managers))
        )
        if self._randomize_test_order:
            aolist = list(aolist)
            shuffle(aolist)
        for a, o, i in aolist:
            if self.awi in a.context and self.awi in o.context:
                self._valid_index = i
                break
        if self._valid_index < 0 and self._fallback_agent is None:
            self._fallback_agent = instantiate(
                self._fallback_type, **self._fallback_params
            )
            # replace me with the newly created agent
            self._fallback_agent.id = self.id
            self._fallback_agent.name = self.name
            self._fallback_agent._awi = self._awi
            self._fallback_agent._owner = self._owner
            self._owner._obj = self._fallback_agent  # type: ignore
            self._fallback_agent.init()

    def init(self):
        super().init()
        self.context_switch()

    def encode_state(self, mechanism_states: dict[str, SAOState]) -> RLState:
        _ = mechanism_states
        if self._valid_index >= 0:
            return self._obs_managers[self._valid_index].encode(self.awi.state)
        raise RuntimeError(
            f"This is an RL agent running in fallback mode and its encode_state should never be called"
        )

    def decode_action(self, action: RLAction) -> dict[str, SAOResponse]:
        if self._valid_index >= 0:
            return self._action_managers[self._valid_index].decode(self.awi, action)
        raise RuntimeError(
            f"This is an RL agent running in fallback mode and its decode_action should never be called"
        )

    def act(self, state: RLState) -> RLAction:
        if self._valid_index >= 0:
            return self._models[self._valid_index](state)
        raise RuntimeError(
            f"This is an RL agent running in fallback mode and its act() method should never be called"
        )

    # =====================
    # Negotiation Callbacks
    # =====================

    def propose(self, *args, **kwargs) -> Outcome | None:
        """Called when the agent is asking to propose in one negotiation"""
        if self._valid_index >= 0:
            return super().propose(*args, **kwargs)
        return self._fallback_agent.propose(*args, **kwargs)

    def respond(self, *args, **kwargs) -> ResponseType:
        """Called when the agent is asked to respond to an offer"""
        if self._valid_index >= 0:
            return super().respond(*args, **kwargs)
        return self._fallback_agent.respond(*args, **kwargs)

    # =====================
    # Time-Driven Callbacks
    # =====================

    def before_step(self):
        """Called at at the BEGINNING of every production step (day)"""
        if self._valid_index >= 0:
            return super().before_step()
        return self._fallback_agent.before_step()

    def step(self):
        """Called at at the END of every production step (day)"""
        if self._dynamic_context_switching:
            self.context_switch()
        if self._valid_index >= 0:
            return super().step()
        return self._fallback_agent.step()

    # ================================
    # Negotiation Control and Feedback
    # ================================

    def on_negotiation_failure(self, *args, **kwargs) -> None:
        """Called when a negotiation the agent is a party of ends without agreement"""
        if self._valid_index >= 0:
            return super().on_negotiation_failure(*args, **kwargs)
        return self._fallback_agent.on_negotiation_failure(*args, **kwargs)

    def on_negotiation_success(self, *args, **kwargs) -> None:
        """Called when a negotiation the agent is a party of ends with agreement"""
        if self._valid_index >= 0:
            return super().on_negotiation_success(*args, **kwargs)
        return self._fallback_agent.on_negotiation_success(*args, **kwargs)
