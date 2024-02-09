from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Union

import numpy as np
from negmas.helpers import get_class

from scml.oneshot.agent import OneShotAgent
from scml.oneshot.awi import OneShotAWI
from scml.oneshot.world import SCMLBaseWorld

__all__ = [
    "RLState",
    "RLAction",
    "RLModel",
    "model_wrapper",
]


RLState = np.ndarray
"""We assume that RL states are numpy arrays"""
RLAction = np.ndarray
"""We assume that RL actions are numpy arrays"""
RLModel = Callable[[RLState], RLAction]
"""A policy is a callable that receives a state and returns an action"""


def model_wrapper(model, deterministic: bool = False) -> RLModel:
    """Wraps a stable_baselines3 model as an RL model"""

    return lambda obs: model.predict(obs, deterministic=deterministic)[0]
