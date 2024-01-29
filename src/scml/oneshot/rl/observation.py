"""
Defines ways to encode and decode observations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np
from attr import define, field
from gymnasium import spaces
from negmas import ResponseType, SAOResponse
from negmas.helpers.strings import itertools

from scml.oneshot.awi import OneShotAWI
from scml.oneshot.common import NegotiationDetails, OneShotState
from scml.oneshot.rl.common import isin
from scml.oneshot.rl.context import (
    Context,
    FixedPartnerNumbersContext,
    FixedPartnerNumbersOneShotContext,
    LimitedPartnerNumbersContext,
)
from scml.scml2019.common import QUANTITY, UNIT_PRICE

__all__ = [
    "ObservationManager",
    "FixedPartnerNumbersObservationManager",
    "LimitedPartnerNumbersObservationManager",
    "DefaultObservationManager",
]


@define(frozen=True)
class ObservationManager(Protocol):
    """Manages the observations of an agent in an RL environment"""

    context: Context

    def make_space(self) -> spaces.Space:
        """Creates the observation space"""
        ...

    def encode(self, state: OneShotState) -> np.ndarray:
        """Encodes an observation from the agent's state"""
        ...

    def make_first_observation(self, awi: OneShotAWI) -> np.ndarray:
        """Creates the initial observation (returned from gym's reset())"""
        ...

    def get_offers(
        self, awi: OneShotAWI, encoded: np.ndarray
    ) -> dict[str, SAOResponse]:
        """Gets the offers from an encoded state"""
        ...


@define(frozen=True)
class BaseObservationManager(ABC):
    """Base class for all observation managers that use a context"""

    context: Context

    @abstractmethod
    def make_space(self) -> spaces.Space:
        """Creates the observation space"""
        ...

    @abstractmethod
    def encode(self, state: OneShotState) -> np.ndarray:
        """Encodes an observation from the agent's state"""
        ...

    def get_offers(
        self, awi: OneShotAWI, encoded: np.ndarray
    ) -> dict[str, SAOResponse]:
        """Gets the offers from an encoded state"""
        ...

    def make_first_observation(self, awi: OneShotAWI) -> np.ndarray:
        """Creates the initial observation (returned from gym's reset())"""
        return self.encode(awi.state)


@define(frozen=True)
class FixedPartnerNumbersObservationManager(BaseObservationManager):
    n_bins: int = 40
    n_sigmas: int = 2
    extra_checks: bool = False
    n_prices: int = 2
    n_partners: int = field(init=False)
    n_suppliers: int = field(init=False)
    n_consumers: int = field(init=False)
    max_quantity: int = field(init=False)
    n_lines: int = field(init=False)

    def __attrs_post_init__(self):
        assert isinstance(self.context, FixedPartnerNumbersContext)
        object.__setattr__(self, "n_suppliers", self.context.n_suppliers)
        object.__setattr__(self, "n_consumers", self.context.n_consumers)
        object.__setattr__(self, "max_quantity", self.context.n_lines)
        object.__setattr__(self, "n_lines", self.context.n_lines)
        object.__setattr__(self, "n_partners", self.n_suppliers + self.n_consumers)

    def make_space(self) -> spaces.Space:
        """Creates the action space"""
        return spaces.MultiDiscrete(
            np.asarray(
                list(
                    itertools.chain(
                        [self.max_quantity + 1, self.n_prices] * self.n_partners
                    )
                )
                + [self.max_quantity + 1] * 2
                + [self.n_lines]
                + [self.n_bins + 1] * 2
                + [self.n_bins * 10 + 1]
                + [self.n_bins + 1] * 3
            ).flatten()
        )

    def make_first_observation(self, awi: OneShotAWI) -> np.ndarray:
        """Creates the initial observation (returned from gym's reset())"""
        return self.encode(awi.state)

    def encode(self, state: OneShotState) -> np.ndarray:
        """Encodes the state as an array"""
        partners = state.my_partners
        assert (
            len(partners) >= self.n_partners
        ), f"{len(partners)=} but I can only support {self.n_partners=}"
        if len(partners) > self.n_partners:
            partners = partners[: self.n_partners]
        partner_index = dict(zip(partners, range(self.n_partners)))
        partnerset = set(partners)
        infos: list[NegotiationDetails | None] = [None] * self.n_partners  # type: ignore
        neg_relative_time = 0.0
        actual_partners = list(
            itertools.chain(
                state.current_negotiation_details["buy"].items(),
                state.current_negotiation_details["sell"].items(),
            )
        )
        for partner, info in actual_partners:
            if partner not in partnerset:
                continue
            infos[partner_index[partner]] = info
            if not info.nmi.state.ended:
                neg_relative_time = max(neg_relative_time, info.nmi.state.relative_time)
        offers = [
            (0, 0)
            if info is None or info.nmi.state.current_offer is None  # type: ignore
            else (
                int(info.nmi.state.current_offer[QUANTITY]),  # type: ignore
                int(
                    info.nmi.state.current_offer[UNIT_PRICE]  # type: ignore
                    - info.nmi.outcome_space.issues[UNIT_PRICE].min_value  # type: ignore
                ),  # type: ignore
            )
            for info in infos
        ]
        if self.extra_checks:
            assert (
                len(partners) == self.n_partners
            ), f"{len(partners)=} while {self.n_partners=}: {partners=}"
            assert (
                len(infos) == self.n_partners
            ), f"{len(infos)=} while {self.n_partners=}: {infos=}"
            assert (
                len(offers) == self.n_partners
            ), f"{len(infos)=} while {self.n_partners=}: {offers=}"
            assert (
                state.total_sales == 0 or state.total_supplies == 0
            ), f"{state.total_sales=}, {state.total_supplies=}, {state.exogenous_input_quantity=}, {state.exogenous_output_quantity=}"

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
            state.needed_sales,
            state.needed_supplies,
            # state.n_input_negotiations,
            # state.n_output_negotiations,
            state.n_lines - 1,
            int(self.n_bins * (state.level / state.n_processes) + 0.5),
            int(neg_relative_time * self.n_bins + 0.5),
            int(state.relative_simulation_time * self.n_bins * 10 + 0.5),
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
                * min(
                    1,
                    (
                        state.trading_prices[state.my_output_product]
                        - state.trading_prices[state.my_input_product]
                    )
                    / state.trading_prices[state.my_output_product],
                )
                + 0.5
            ),
        ]

        v = np.asarray(np.asarray(offers).flatten().tolist() + extra)
        if self.extra_checks:
            space = self.make_space()
            assert space is not None and space.shape is not None
            exp = space.shape[0]
            assert (
                len(v) == exp
            ), f"{len(v)=}, {len(extra)=}, {len(offers)=}, {exp=}, {self.n_partners=}\n{state.current_negotiation_details=}"
            assert all(
                -1 < a < b for a, b in zip(v, space.nvec)  # type: ignore
            ), f"{v=}\n{space.nvec=}\n{space.nvec - v =}\n{ (state.exogenous_input_quantity , state.total_supplies , state.total_sales , state.exogenous_output_quantity) }"  # type: ignore

        return v

    def is_valid(self, env) -> bool:
        """Checks that it is OK to use this observation manager with a given `OneShotEnv`"""
        if env._n_lines != self.n_lines:
            return False
        if env._n_suppliers != self.n_suppliers:
            return False
        if env._n_consumers != self.n_consumers:
            return False
        return True

    def get_offers(
        self, awi: OneShotAWI, encoded: np.ndarray
    ) -> dict[str, SAOResponse]:
        """Gets the offers from an encoded state"""
        offers = encoded[: 2 * self.n_partners].reshape((self.n_partners, 2))
        partners = awi.my_partners[: self.n_partners]
        responses = dict()
        minprice = (
            awi.current_output_issues
            if awi.is_first_level
            else awi.current_input_issues
        )[UNIT_PRICE].min_value
        for p, w in zip(partners, offers):
            if w[0] == w[1] == 0:
                responses[p] = SAOResponse(ResponseType.END_NEGOTIATION, None)
                continue

            outcome = (w[0], awi.current_step, w[1] + minprice)
            if self.extra_checks:
                os = (
                    awi.current_input_outcome_space
                    if awi.is_first_level
                    else awi.current_output_outcome_space
                )
                assert (
                    outcome in os
                ), f"received {outcome} from {p} ({w}) in step {awi.current_step} for OS {os}\n{encoded=}"
            responses[p] = SAOResponse(ResponseType.REJECT_OFFER, outcome)
        return responses


@define(frozen=True)
class LimitedPartnerNumbersObservationManager(BaseObservationManager):
    n_bins: int = 40
    n_sigmas: int = 2
    extra_checks: bool = True
    n_prices: int = 2
    n_partners: tuple[int, int] = field(init=False)
    n_suppliers: tuple[int, int] = field(init=False)
    n_consumers: tuple[int, int] = field(init=False)
    max_quantity: tuple[int, int] = field(init=False)
    n_lines: tuple[int, int] = field(init=False)
    sub_manager: FixedPartnerNumbersObservationManager = field(init=False)
    submanager_context: type[FixedPartnerNumbersContext] = field(
        init=False, default=FixedPartnerNumbersOneShotContext
    )

    def __attrs_post_init__(self):
        if isinstance(self.context, LimitedPartnerNumbersContext):
            object.__setattr__(self, "n_suppliers", self.context.n_suppliers)
            object.__setattr__(self, "n_consumers", self.context.n_consumers)
            object.__setattr__(self, "max_quantity", self.context.n_lines)
            object.__setattr__(self, "n_lines", self.context.n_lines)
        elif isinstance(self.context, FixedPartnerNumbersContext):
            object.__setattr__(
                self,
                "n_suppliers",
                (self.context.n_suppliers, self.context.n_suppliers),
            )
            object.__setattr__(
                self,
                "n_consumers",
                (self.context.n_consumers, self.context.n_consumers),
            )
            object.__setattr__(
                self, "max_quantity", (self.context.n_lines, self.context.n_lines)
            )
            object.__setattr__(
                self, "n_lines", (self.context.n_lines, self.context.n_lines)
            )
        else:
            raise AssertionError(
                f"{self.__class__} does not support factories of type {self.context.__class__}"
            )
        object.__setattr__(
            self,
            "n_partners",
            (
                self.n_suppliers[0] + self.n_consumers[0],
                self.n_suppliers[1] + self.n_consumers[1],
            ),
        )
        object.__setattr__(
            self,
            "sub_manager",
            FixedPartnerNumbersObservationManager(
                context=self.submanager_context(
                    year=self.context.year,
                    level=self.context.level,
                    n_processes=self.context.n_processes[-1]
                    if isinstance(self.context.n_processes, tuple)
                    else self.context.n_processes,
                    n_consumers=self.context.n_consumers[0]
                    if isinstance(self.context.n_consumers, tuple)
                    else self.context.n_consumers,
                    n_suppliers=self.context.n_suppliers[0]
                    if isinstance(self.context.n_suppliers, tuple)
                    else self.context.n_suppliers,
                    n_competitors=self.context.n_competitors[0]
                    if isinstance(self.context.n_competitors, tuple)
                    else self.context.n_competitors,
                    n_agents_per_level=self.context.n_agents_per_level,
                    n_lines=self.context.n_lines[0]
                    if isinstance(self.context.n_lines, tuple)
                    else self.context.n_lines,
                    non_competitors=self.context.non_competitors,
                ),
                n_bins=self.n_bins,
                n_sigmas=self.n_sigmas,
                extra_checks=self.extra_checks,
                n_prices=self.n_prices,
            ),
        )

    def get_offers(
        self, awi: OneShotAWI, encoded: np.ndarray
    ) -> dict[str, SAOResponse]:
        """Gets the offers from an encoded state"""
        return self.sub_manager.get_offers(awi, encoded)

    def make_space(self) -> spaces.Space:
        """Creates the action space"""
        return self.sub_manager.make_space()

    def make_first_observation(self, awi: OneShotAWI) -> np.ndarray:
        """Creates the initial observation (returned from gym's reset())"""
        return self.encode(awi.state)

    def encode(self, state: OneShotState) -> np.ndarray:
        """Encodes the state as an array"""
        return self.sub_manager.encode(state)

    def is_valid(self, env) -> bool:
        """Checks that it is OK to use this observation manager with a given `OneShotEnv`"""
        if isin(env._n_lines, self.n_lines):
            return False
        if isin(env._n_suppliers, self.n_suppliers):
            return False
        if isin(env._n_consumers, self.n_consumers):
            return False
        return True


DefaultObservationManager = FixedPartnerNumbersObservationManager
"""The default observation manager"""
