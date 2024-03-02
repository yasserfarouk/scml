from typing import Callable

import numpy as np

from scml.oneshot.common import is_system_agent

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


def group_partners(
    my_partners: list[str], n_partners: int, max_group_size: int, extend: bool = True
) -> list[list[str]]:
    """Combines a list of partners/consumers into the given number of groups"""
    if n_partners == 0:
        return []
    partners = [_ for _ in my_partners if not is_system_agent(_)]
    partner_sets = [[] for _ in range(n_partners)]
    for i, partner in enumerate(partners):
        partner_sets[i % n_partners].append(partner)
    n = len(partners)
    if extend and n:
        for i in range(n, n_partners):
            partner_sets[i].append(partners[(i - n) % n])

    assert not partner_sets or max(len(_) for _ in partner_sets) <= max_group_size, (
        f"Too many partners {len(partners)} needing to combine more "
        f"than {max_group_size} which is not supported by "
        f"the observation space:\n{partner_sets=}"
    )

    return partner_sets
