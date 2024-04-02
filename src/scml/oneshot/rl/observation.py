"""
Defines ways to encode and decode observations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Iterable, Protocol, runtime_checkable

import numpy as np
from attr import define, field
from gymnasium import spaces
from negmas.helpers.strings import itertools
from negmas.outcomes import Outcome
from scml.oneshot.context import BaseContext

from scml.oneshot.awi import OneShotAWI
from scml.oneshot.rl.helpers import (
    discretize_and_clip,
    read_offers,
    clip,
    recover_offers,
)

__all__ = [
    "ObservationManager",
    "FlexibleObservationManager",
    "DefaultObservationManager",
]


def astuple(x: Iterable | int | float | str) -> tuple:
    if not isinstance(x, Iterable):
        return (x, x)
    return tuple(x)


@runtime_checkable
class ObservationManager(Protocol):
    """Manages the observations of an agent in an RL environment"""

    @property
    def context(self) -> BaseContext:
        ...

    def make_space(self) -> spaces.Space:
        """Creates the observation space"""
        ...

    def encode(self, awi: OneShotAWI) -> np.ndarray:
        """Encodes an observation from the agent's awi"""
        ...

    def make_first_observation(self, awi: OneShotAWI) -> np.ndarray:
        """Creates the initial observation (returned from gym's reset())"""
        ...

    def get_offers(
        self, awi: OneShotAWI, encoded: np.ndarray
    ) -> dict[str, Outcome | None]:
        """Gets the offers from an encoded awi"""
        ...


@define
class BaseObservationManager(ABC):
    """Base class for all observation managers that use a context"""

    context: BaseContext
    continuous: bool = False
    n_partners: int = field(init=False, default=16)
    n_suppliers: int = field(init=False, default=8)
    n_consumers: int = field(init=False, default=8)

    @abstractmethod
    def make_space(self) -> spaces.Space:
        """Creates the observation space"""
        ...

    @abstractmethod
    def encode(self, awi: OneShotAWI) -> np.ndarray:
        """Encodes an observation from the agent's awi"""
        ...

    @abstractmethod
    def get_offers(
        self, awi: OneShotAWI, encoded: np.ndarray
    ) -> dict[str, Outcome | None]:
        """Gets the offers from an encoded awi"""
        ...

    def make_first_observation(self, awi: OneShotAWI) -> np.ndarray:
        """Creates the initial observation (returned from gym's reset())"""
        return self.encode(awi)


@define
class FlexibleObservationManager(BaseObservationManager):
    """
    An observation manager that can be used with any SCML world.

    Args:
        capacity_multiplier: A factor to multiply by the number of lines to give the maximum quantity allowed in offers
        exogenous_multiplier: A factor to multiply maximum production capacity with when encoding exogenous quantities
        continuous: If given the observation space will be a Box otherwise it will be a MultiDiscrete
        n_prices: The number of prices to use for encoding the unit price (if not `continuous`)
        max_production_cost: The limit for production cost. Anything above that will be mapped to this max
        max_group_size: Maximum size used for grouping observations from multiple partners. This will be
                        used in the number of partners in the simulation is larger than the number used
                        for training.
        n_past_received_offers: Number of past received offers to add to the observation.
        n_bins: N. bins to use for discretization (if not `continuous`)
        n_sigmas: The number of sigmas used for limiting the range of randomly distributed variables
        extra_checks: If given, extra checks are applied to make sure encoding and decoding make sense
    Remarks:
        ...
    """

    capacity_multiplier: int = 1
    n_prices: int = 2
    max_group_size: int = 2
    reduce_space_size: bool = True
    n_past_received_offers: int = 1
    extra_checks: bool = False
    n_bins: int = 40
    n_sigmas: int = 2
    max_production_cost: int = 10
    exogenous_multiplier: int = 1
    max_quantity: int = field(init=False, default=10)
    _chosen_partner_indices: list[int] | None = field(init=False, default=None)
    _previous_offers: deque = field(init=False)
    _dims: list[int] | None = field(init=False, default=None)

    def __attrs_post_init__(self):
        p = self.context.extract_context_params(self.reduce_space_size)
        if p.nlines:
            object.__setattr__(self, "n_suppliers", p.nsuppliers)
            object.__setattr__(self, "n_consumers", p.nconsumers)
            object.__setattr__(
                self, "max_quantity", p.nlines * self.capacity_multiplier
            )
            if not self.exogenous_multiplier:
                object.__setattr__(self, "exogenous_multiplier", p.nlines)
            object.__setattr__(self, "n_partners", p.nsuppliers + p.nconsumers)
        n = (2 * self.n_partners) * self.n_past_received_offers
        self._previous_offers = deque([0] * n, maxlen=n) if n else deque()

    def get_dims(self) -> list[int]:
        """Get the sizes of all dimensions in the observation space. Used if not continuous."""
        return (
            list(
                itertools.chain(
                    [self.max_group_size * self.max_quantity + 1, self.n_prices]
                    * self.n_partners
                )
            )
            + list(
                itertools.chain(
                    [self.max_group_size * self.max_quantity + 1, self.n_prices]
                    * self.n_partners
                    * self.n_past_received_offers
                )
            )
            + [self.max_quantity + 1] * 2  # needed sales and supplies
            + [self.n_bins] * 2  # level, relative_simulation
            + [self.n_bins * 2]  # neg_relative
            + [self.n_bins] * 3  # production cost, penalties and other costs
            + [self.n_bins] * 2  # exogenous_contract quantity summary
        )

    def make_space(self) -> spaces.MultiDiscrete | spaces.Box:
        """Creates the action space"""
        dims = self.get_dims()
        if self._dims is None:
            self._dims = dims
        elif self.extra_checks:
            assert all(
                a == b for a, b in zip(dims, self._dims, strict=True)
            ), f"Surprising dims while making space\n{self._dims=}\n{dims=}"
        if self.continuous:
            return spaces.Box(0.0, 1.0, shape=(len(dims),))
        return spaces.MultiDiscrete(np.asarray(dims))

    def make_first_observation(self, awi: OneShotAWI) -> np.ndarray:
        """Creates the initial observation (returned from gym's reset())"""
        return self.encode(awi)

    def encode(self, awi: OneShotAWI) -> np.ndarray:
        """Encodes the awi as an array"""

        offers = read_offers(
            awi,
            self.n_suppliers,
            self.n_consumers,
            self.max_group_size,
            self.continuous,
        )

        current_offers = np.asarray(offers).flatten().tolist()

        if self.extra_checks:
            assert (
                len(current_offers) == self.n_partners * 2
            ), f"{len(current_offers)=} but {self.n_partners=}"
            assert (
                len(self._previous_offers)
                == self.n_past_received_offers * self.n_partners * 2
            ), f"{self._previous_offers=} but {self.n_partners=}"

        extra = self.extra_obs(awi)
        v = np.asarray(
            current_offers
            + list(self._previous_offers)
            + (
                [min(1, max(0, v[0] if isinstance(v, Iterable) else v)) for v in extra]
                if self.continuous
                else [
                    discretize_and_clip(
                        clip(v[0]) if isinstance(v, Iterable) else clip(v),
                        clip(v[1]) if isinstance(v, Iterable) else self.n_bins,
                    )
                    for v in extra
                ]
            ),
            dtype=np.float32 if self.continuous else np.int32,
        )
        if self.continuous:
            v = np.minimum(np.maximum(v, 0.0), 1.0)

        if self._previous_offers:
            for _ in current_offers:
                self._previous_offers.append(_)
        if self.extra_checks:
            space = self.make_space()
            assert self.continuous or isinstance(space, spaces.MultiDiscrete)
            assert not self.continuous or isinstance(space, spaces.Box)
            assert space is not None and space.shape is not None
            exp = space.shape[0]
            assert (
                len(v) == exp
            ), f"{len(v)=}, {len(extra)=}, {len(offers)=}, {exp=}, {self.n_partners=}\n{awi.current_negotiation_details=}"
            if self._dims is None:
                self._dims = self.get_dims()
            assert self.continuous or all(
                a <= b for a, b in zip(v, self._dims, strict=True)
            ), f"Surprising dims\n{v=}\n{self._dims=}"
            assert not self.continuous or all(
                [0 <= x <= 1 for x in v]
            ), f"Surprising dims (continuous)\n{v=}"
            if isinstance(space, spaces.MultiDiscrete):
                if not all(0 <= a < b for a, b in zip(v, space.nvec)):
                    print(
                        f"{v=}\n{space.nvec=}\n{space.nvec - v =}\n{ (awi.current_exogenous_input_quantity , awi.total_supplies , awi.total_sales , awi.current_exogenous_output_quantity) }"
                    )
                assert all(
                    0 <= a < b for a, b in zip(v, space.nvec)
                ), f"{offers=}\n{extra=}\n{v=}\n{space.nvec=}\n{space.nvec - v =}\n{ (awi.current_exogenous_input_quantity , awi.total_supplies , awi.total_sales , awi.current_exogenous_output_quantity) }"  # type: ignore

        return v

    def extra_obs(self, awi: OneShotAWI) -> list[tuple[float, int] | float]:
        """
        The observation values other than offers and previous offers.

        Returns:
            A list of tuples. Each is some observation variable as a
            real number between zero and one and a number of bins to
            use for discrediting this variable. If a single value, the
            number of bins will be self.n_bin

        """
        # adding extra components to the observation
        neg_relative_time = min(
            awi.current_states.values(), key=lambda x: x.relative_time
        ).relative_time
        exogenous = awi.exogenous_contract_summary
        incost = (
            awi.current_disposal_cost if awi.is_perishable else awi.current_storage_cost
        )

        return [
            (awi.needed_sales / self.max_quantity, self.max_quantity + 1),
            (awi.needed_supplies / self.max_quantity, self.max_quantity + 1),
            awi.level / (awi.n_processes - 1),
            awi.relative_time,
            (neg_relative_time, 2 * self.n_bins),
            awi.profile.cost / self.max_production_cost,
            incost / (incost + awi.current_shortfall_penalty),
            (
                awi.trading_prices[awi.my_output_product]
                - awi.trading_prices[awi.my_input_product]
            )
            / awi.trading_prices[awi.my_output_product],
            exogenous[0][0]
            / (self.exogenous_multiplier * awi.production_capacities[0]),
            exogenous[-1][0]
            / (self.exogenous_multiplier * awi.production_capacities[-1]),
        ]

    def get_offers(
        self, awi: OneShotAWI, encoded: np.ndarray
    ) -> dict[str, Outcome | None]:
        """
        Gets offers from an encoded awi.
        """
        return recover_offers(
            encoded,
            awi,
            self.n_suppliers,
            self.n_consumers,
            self.max_group_size,
            self.continuous,
            n_prices=self.n_prices,
        )


DefaultObservationManager = FlexibleObservationManager
"""The default observation manager"""
