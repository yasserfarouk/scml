"""
Defines ways to encode and decode observations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Protocol, runtime_checkable

import numpy as np
from attr import define, field
from gymnasium import spaces
from negmas import DiscreteCartesianOutcomeSpace
from negmas.helpers.strings import itertools
from negmas.outcomes import Outcome

from scml.oneshot.awi import OneShotAWI
from scml.oneshot.common import OneShotState
from scml.oneshot.context import BaseContext, extract_context_params
from scml.oneshot.rl.common import group_partners
from scml.scml2019.common import QUANTITY, UNIT_PRICE

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

    def encode(self, state: OneShotState) -> np.ndarray:
        """Encodes an observation from the agent's state"""
        ...

    def make_first_observation(self, awi: OneShotAWI) -> np.ndarray:
        """Creates the initial observation (returned from gym's reset())"""
        ...

    def get_offers(
        self, awi: OneShotAWI, encoded: np.ndarray
    ) -> dict[str, Outcome | None]:
        """Gets the offers from an encoded state"""
        ...


@define
class BaseObservationManager(ABC):
    """Base class for all observation managers that use a context"""

    context: BaseContext

    @abstractmethod
    def make_space(self) -> spaces.Space:
        """Creates the observation space"""
        ...

    @abstractmethod
    def encode(self, state: OneShotState) -> np.ndarray:
        """Encodes an observation from the agent's state"""
        ...

    @abstractmethod
    def get_offers(
        self, awi: OneShotAWI, encoded: np.ndarray
    ) -> dict[str, Outcome | None]:
        """Gets the offers from an encoded state"""
        ...

    def make_first_observation(self, awi: OneShotAWI) -> np.ndarray:
        """Creates the initial observation (returned from gym's reset())"""
        return self.encode(awi.state)


@define
class FlexibleObservationManager(BaseObservationManager):
    n_bins: int = 40
    n_sigmas: int = 2
    n_prices: int = 2
    max_production_cost: int = 10
    max_group_size: int = 2
    reduce_space_size: bool = True
    n_partners: int = field(init=False, default=16)
    n_suppliers: int = field(init=False, default=8)
    n_consumers: int = field(init=False, default=8)
    max_quantity: int = field(init=False, default=10)
    n_lines: int = field(init=False, default=10)
    extra_checks: bool = False
    _chosen_partner_indices: list[int] | None = field(init=False, default=None)
    _previous_offers: list[int] | None = field(init=False, default=None)
    _dims: list[int] | None = field(init=False, default=None)

    def __attrs_post_init__(self):
        p = extract_context_params(
            self.context, self.reduce_space_size, raise_on_failure=False
        )
        if p.nlines:
            object.__setattr__(self, "n_suppliers", p.nsuppliers)
            object.__setattr__(self, "n_consumers", p.nconsumers)
            object.__setattr__(self, "max_quantity", p.nlines)
            object.__setattr__(self, "n_lines", p.nlines)
            object.__setattr__(self, "n_partners", p.nsuppliers + p.nconsumers)

    def get_dims(self) -> list[int]:
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
                )
            )
            + [self.max_quantity + 1] * 2  # needed sales and supplies
            # + [self.n_lines]
            + [self.n_bins + 1] * 2  # level, relative_simulation
            + [self.n_bins * 2 + 1]  # neg_relative
            + [self.n_bins + 1] * 4  # production cost, penalties and other costs
            + [self.n_bins + 1] * 2  # exogenous_contract quantity summary
        )

    def make_space(self) -> spaces.MultiDiscrete:
        """Creates the action space"""
        dims = self.get_dims()
        if self._dims is None:
            self._dims = dims
        elif self.extra_checks:
            assert all(
                a == b for a, b in zip(dims, self._dims, strict=True)
            ), f"Surprising dims\n{self._dims=}\n{dims=}"
        space = spaces.MultiDiscrete(np.asarray(dims))
        return space

    def make_first_observation(self, awi: OneShotAWI) -> np.ndarray:
        """Creates the initial observation (returned from gym's reset())"""
        return self.encode(awi.state)

    def encode(self, state: OneShotState) -> np.ndarray:
        """Encodes the state as an array"""
        suppliers = group_partners(
            state.my_suppliers, self.n_suppliers, self.max_group_size
        )
        consumers = group_partners(
            state.my_consumers, self.n_consumers, self.max_group_size
        )

        if self.extra_checks:
            partners = suppliers + consumers
            assert len(partners) == self.n_partners, (
                f"{len(partners)=} but I can only support {self.n_partners=}"
                f"\n{suppliers=}\n{consumers=}\n{type(self.context)=}\n{self.context=}\n{self=}"
            )
        neg_relative_time = 1.0
        for s in state.current_states.values():
            neg_relative_time = min(neg_relative_time, s.relative_time)
        offer_map = state.current_offers

        def read_offers(N, lstoflst, min_price, offer_map=offer_map):
            offers = [[0, 0] for _ in range(N)]
            for i, lst in enumerate(lstoflst):
                n_read = 0
                for partner in lst:
                    outcome = offer_map.get(partner, (0, 0, 0))
                    if outcome is None:
                        continue
                    offers[i][0] += outcome[QUANTITY]
                    offers[i][1] += outcome[UNIT_PRICE] * outcome[QUANTITY]
                    n_read += 1
                if n_read and offers[i][0]:
                    offers[i][1] = offers[i][1] / offers[i][0] - min_price
            return offers

        min_price = state.current_input_outcome_space.issues[UNIT_PRICE].min_value
        offers = read_offers(self.n_suppliers, suppliers, min_price)

        min_price = state.current_output_outcome_space.issues[UNIT_PRICE].min_value
        offers += read_offers(self.n_consumers, consumers, min_price)

        if self.extra_checks:
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

        offerslist = np.asarray(offers).flatten().tolist()
        if self._previous_offers is None:
            previous_offers = [0] * len(offerslist)
        else:
            previous_offers = self._previous_offers
        assert (
            len(previous_offers) == self.n_partners * 2
        ), f"{len(previous_offers)=} but {self.n_partners=}"
        assert (
            len(offerslist) == self.n_partners * 2
        ), f"{len(offerslist)=} but {self.n_partners=}"
        exogenous = state.exogenous_contract_summary
        exogenous = [
            min(
                self.n_bins,
                max(
                    0,
                    int(
                        self.n_bins
                        * (
                            exogenous[0][0]
                            / (self.max_quantity * len(state.all_consumers[0]))
                        )
                    ),
                ),
            ),
            min(
                self.n_bins,
                max(
                    0,
                    int(
                        self.n_bins
                        * (
                            exogenous[-1][0]
                            / (self.max_quantity * len(state.all_suppliers[-1]))
                        )
                    ),
                ),
            ),
        ]
        extra = [
            max(0, state.needed_sales),
            max(0, state.needed_supplies),
            # state.n_lines - 1,
            max(
                0,
                min(
                    self.n_bins,
                    int(self.n_bins * (state.level / (state.n_processes - 1))),
                ),
            ),
            max(0, min(self.n_bins, int(state.relative_simulation_time * self.n_bins))),
            max(0, min(self.n_bins * 2, int(neg_relative_time * self.n_bins * 2))),
            max(
                0,
                min(
                    self.n_bins,
                    int(self.n_bins * state.profile.cost / self.max_production_cost),
                ),
            ),
            max(
                0,
                min(
                    self.n_bins,
                    int(
                        _normalize(
                            state.disposal_cost,
                            state.profile.disposal_cost_mean,
                            state.profile.disposal_cost_dev,
                        )
                        * self.n_bins
                    ),
                ),
            ),
            max(
                0,
                min(
                    self.n_bins,
                    int(
                        _normalize(
                            state.shortfall_penalty,
                            state.profile.shortfall_penalty_mean,
                            state.profile.shortfall_penalty_dev,
                        )
                        * self.n_bins
                    ),
                ),
            ),
            max(
                0,
                min(
                    self.n_bins,
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
                    ),
                ),
            ),
        ] + exogenous
        v = np.asarray(offerslist + previous_offers + extra)

        if self.extra_checks:
            if self._dims is None:
                self._dims = self.get_dims()
            assert all(
                a <= b for a, b in zip(v, self._dims, strict=True)
            ), f"Surprising dims\n{v=}\n{self._dims=}"
            space = self.make_space()
            assert isinstance(space, spaces.MultiDiscrete)
            assert space is not None and space.shape is not None
            exp = space.shape[0]
            assert (
                len(v) == exp
            ), f"{len(v)=}, {len(extra)=}, {len(offers)=}, {exp=}, {self.n_partners=}\n{state.current_negotiation_details=}"
            if not all(0 <= a < b for a, b in zip(v, space.nvec)):  # type: ignore
                print(
                    f"{v=}\n{space.nvec=}\n{space.nvec - v =}\n{ (state.exogenous_input_quantity , state.total_supplies , state.total_sales , state.exogenous_output_quantity) }"
                )
                # breakpoint()
            assert all(
                0 <= a < b
                for a, b in zip(v, space.nvec)  # type: ignore
            ), f"{offers=}\n{extra=}\n{v=}\n{space.nvec=}\n{space.nvec - v =}\n{ (state.exogenous_input_quantity , state.total_supplies , state.total_sales , state.exogenous_output_quantity) }"  # type: ignore

        self._previous_offers = offerslist
        return v

    def get_offers(
        self, awi: OneShotAWI, encoded: np.ndarray, previous=False
    ) -> dict[str, Outcome | None]:
        """
        Gets offers from an encoded state.
        """
        suppliers = group_partners(
            awi.my_suppliers, self.n_suppliers, self.max_group_size
        )
        consumers = group_partners(
            awi.my_consumers, self.n_consumers, self.max_group_size
        )
        buyos = awi.current_input_outcome_space
        sellos = awi.current_output_outcome_space
        min_buy_price = buyos.issues[UNIT_PRICE].min_value
        min_sell_price = sellos.issues[UNIT_PRICE].min_value
        return self.decode_offers(
            encoded,
            suppliers,
            consumers,
            previous,
            min_buy_price,
            min_sell_price,
            awi.current_step,
            buyos,
            sellos,
        )

    def decode_offers(
        self,
        encoded: np.ndarray,
        suppliers: list[list[str]],
        consumers: list[list[str]],
        mine: bool,
        min_buy_price: float,
        min_sell_price: float,
        current_step: int,
        buyos: DiscreteCartesianOutcomeSpace | None = None,
        sellos: DiscreteCartesianOutcomeSpace | None = None,
    ) -> dict[str, Outcome | None]:
        strt = 2 * self.n_partners if mine else 0
        offers = encoded[strt : strt + 2 * self.n_partners].reshape(
            (self.n_partners, 2)
        )
        partners = suppliers + consumers
        assert len(offers) == len(partners), f"{len(offers)=}, {len(partners)=}"
        responses = dict()
        for plst, w, is_supplier in zip(
            partners,
            offers,
            [True] * len(suppliers) + [False] * len(consumers),
            strict=True,
        ):
            p = "+".join(plst)
            minprice = min_buy_price if is_supplier else min_sell_price
            if w[0] == w[1] == 0:
                responses[p] = None
                continue

            outcome = (w[0], current_step, w[1] + minprice)
            if self.extra_checks:
                os = buyos if is_supplier else sellos
                assert (
                    os is None or outcome in os
                ), f"received {outcome} from {p} ({w}) in step {current_step} for OS {os}\n{encoded=}"
            responses[p] = outcome
        return responses


DefaultObservationManager = FlexibleObservationManager
"""The default observation manager"""
