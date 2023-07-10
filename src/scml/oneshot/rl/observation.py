"""
Defines ways to encode and decode observations.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
from attr import define, field
from gymnasium import spaces
from negmas.helpers.strings import itertools

from scml.oneshot.awi import OneShotAWI
from scml.oneshot.common import NegotiationDetails, OneShotState
from scml.scml2019.common import QUANTITY, UNIT_PRICE

__all__ = [
    "ObservationManager",
]


class ObservationManager(Protocol):
    """Manages the observations of an agent in an RL environment"""

    def make_space(self) -> spaces.Space:
        """Creates the observation space"""
        ...

    def make_first_observation(self, awi: OneShotAWI) -> np.ndarray:
        """Creates the initial observation (returned from gym's reset())"""
        ...

    def encode(self, state: OneShotState) -> np.ndarray:
        """Encodes an observation from the agent's state"""
        ...

    def is_valid(self, env) -> bool:
        """Checks that it is OK to use this observation manager with a given `OneShotEnv`"""
        ...

    # def decode(self, obs: Any) -> AgentFullState:
    #     """Decodes an observation to agent's state"""
    #     ...


@define(frozen=True)
class DefaultObservationManager:
    n_suppliers: int
    n_consumers: int
    n_quantities: int
    n_lines: int
    n_prices: int = 2
    n_bins: int = 10
    n_sigmas: int = 2
    n_partners: int = field(init=False)

    def __attrs_post_init__(self):
        object.__setattr__(self, "n_partners", self.n_suppliers + self.n_consumers)

    def make_space(self) -> spaces.Space:
        """Creates the action space"""
        return spaces.MultiDiscrete(
            np.asarray(
                list(
                    itertools.chain(
                        [self.n_quantities + 1, self.n_prices] * self.n_partners
                    )
                )
                + [self.n_quantities + 1] * 2
                + [max(self.n_suppliers, self.n_consumers) + 1] * 2
                + [self.n_lines]
                + [self.n_bins + 1] * 6
            ).flatten()
        )

    def make_first_observation(self, awi: OneShotAWI) -> np.ndarray:
        """Creates the initial observation (returned from gym's reset())"""
        mechanism_states = dict()
        for partner, info in itertools.chain(
            awi.current_negotiation_details["buy"].items(),
            awi.current_negotiation_details["sell"].items(),
        ):
            mechanism_states[partner] = info.nmi.state
        return self.encode(awi.state)

    def encode(self, state: OneShotState) -> np.ndarray:
        """Encodes the state as an array"""
        partners = state.my_suppliers + state.my_consumers
        partner_index = dict(zip(partners, range(len(partners))))
        infos: list[NegotiationDetails] = [None] * len(partners)  # type: ignore
        neg_relative_time = 0.0
        for partner, info in itertools.chain(
            state.current_negotiation_details["buy"].items(),
            state.current_negotiation_details["sell"].items(),
        ):
            infos[partner_index[partner]] = info
            neg_relative_time = max(neg_relative_time, info.nmi.state.relative_time)
        offers = [
            (0, 0)
            if info is None or info.nmi.state.current_offer is None  # type: ignore
            else (
                int(info.nmi.state.current_offer[QUANTITY]),  # type: ignore
                int(info.nmi.state.current_offer[UNIT_PRICE] - info.nmi.outcome_space.issues[UNIT_PRICE].min_value),  # type: ignore
            )
            for info in infos
        ]

        # TODO add more state values here and remember to add corresponding limits in the make_space function
        def _normalize(x, mu, sigma, n_sigmas=self.n_sigmas):
            """
            Normalizes x between 0 and 1 given that it is sampled from a normal (mu, sigma).
            This is actually a very stupid way to do it.
            """
            mn = mu - n_sigmas * sigma
            mx = mu + n_sigmas * sigma
            if abs(mn - mx) < 1e-6:
                return 1
            return max(0, min(1, (x - mn) / (mx - mn)))

        extra = [
            max(0, state.needed_sales),
            max(0, state.needed_supplies),
            state.n_input_negotiations,
            state.n_output_negotiations,
            state.n_lines - 1,
            int(self.n_bins * (state.level / state.n_processes) + 0.5),
            int(neg_relative_time * self.n_bins + 0.5),
            int(state.relative_simulation_time * self.n_bins + 0.5),
            int(
                _normalize(
                    state.disposal_cost,
                    state.profile.disposal_cost_mean,
                    state.profile.disposal_cost_dev,
                )
                * self.n_bins
                + 0.5
            ),
            int(
                _normalize(
                    state.shortfall_penalty,
                    state.profile.shortfall_penalty_mean,
                    state.profile.shortfall_penalty_dev,
                )
                * self.n_bins
                + 0.5
            ),
            int(
                self.n_bins
                * (
                    state.trading_prices[state.my_output_product]
                    - state.trading_prices[state.my_input_product]
                )
                + 0.5
            ),
        ]

        return np.asarray(np.asarray(offers).flatten().tolist() + extra)

    def is_valid(self, env) -> bool:
        """Checks that it is OK to use this observation manager with a given `OneShotEnv`"""
        if env._n_lines != self.n_lines:
            return False
        if env._n_suppliers != self.n_suppliers:
            return False
        if env._n_consumers != self.n_consumers:
            return False
        return True
