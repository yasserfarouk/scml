from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from negmas.sao import SAOController, SAONegotiator

from ..scml2020.common import (
    ANY_LINE,
    ANY_STEP,
    INFINITE_COST,
    NO_COMMAND,
    FactoryProfile,
    FactoryState,
    is_system_agent,
)

if TYPE_CHECKING:
    from .adapter import OneShotSCML2020Adapter


class AWIHelper:
    """The Agent SCML2020World Interface for SCML2020 world allowing a single process per agent"""

    def __init__(self, owner: OneShotSCML2020Adapter):
        self._owner = owner
        self._world = owner.awi._world

    # --------
    # Actions
    # --------

    def request_negotiations(
        self,
        is_buy: bool,
        product: int,
        quantity: int | tuple[int, int],
        unit_price: int | tuple[int, int],
        time: int | tuple[int, int],
        controller: SAOController | None = None,
        negotiators: list[SAONegotiator] | None = None,
        partners: list[str] | None = None,
        extra: dict[str, Any] | None = None,
        copy_partner_id=True,
    ) -> bool:
        if not is_buy:
            self._world.logwarning(
                f"{self.agent.name} requested selling on {product}. This is not allowed in oneshot"
            )
            return False
        buyable, sellable = self.my_input_products, self.my_output_products
        if (product not in buyable and is_buy) or (
            product not in sellable and not is_buy
        ):
            self._world.logwarning(
                f"{self.agent.name} requested ({'buying' if is_buy else 'selling'}) on {product}. This is not allowed"
            )
            return False
        unit_price, time, quantity = self._world._make_issues(product)
        return self._world._request_negotiations(
            self._owner.id,
            # product,
            # quantity,
            # unit_price,
            # time,
            controller,
            negotiators,
            extra,
        )

    def request_negotiation(
        self,
        is_buy: bool,
        product: int,
        quantity: int | tuple[int, int],
        unit_price: int | tuple[int, int],
        time: int | tuple[int, int],
        partner: str,
        negotiator: SAONegotiator,
        extra: dict[str, Any] | None = None,
    ) -> bool:
        if not is_buy:
            return False

        if extra is None:
            extra = dict()
        buyable, sellable = self.my_input_products, self.my_output_products
        if (product not in buyable and is_buy) or (
            product not in sellable and not is_buy
        ):
            self._world.logwarning(
                f"{self.agent.name} requested ({'buying' if is_buy else 'selling'}) on {product}. This is not allowed"
            )
            return False
        unit_price, time, quantity = self._world._make_issues(product)
        return (
            self._world._request_negotiation(
                self._owner.id,
                product,
                # quantity,
                # unit_price,
                # time,
                partner,
                negotiator,
                extra,
                is_buy,
            )
            is not None
        )

    def schedule_production(
        self,
        process: int,
        repeats: int,
        step: int | tuple[int, int] = ANY_STEP,
        line: int = ANY_LINE,
        override: bool = True,
        method: str = "latest",
        partial_ok: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        return (np.asarray([]), np.asarray([]))

    def order_production(
        self, process: int, steps: np.ndarray, lines: np.ndarray
    ) -> None:
        return None

    def available_for_production(
        self,
        repeats: int,
        step: int | tuple[int, int] = ANY_STEP,
        line: int = ANY_LINE,
        override: bool = True,
        method: str = "latest",
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray([self._world.current_step] * self._owner.awi.n_lines),
            np.arange(self._owner.awi.n_lines),
        )

    def set_commands(self, commands: np.ndarray, step: int = -1) -> None:
        return None

    def cancel_production(self, step: int, line: int) -> bool:
        return False

    # ---------------------
    # Information Gathering
    # ---------------------

    @property
    def trading_prices(self) -> np.ndarray | None:
        """Returns the current trading prices of all products"""
        return (
            self._world.trading_prices if self._world.publish_trading_prices else None
        )

    @property
    def current_balance(self) -> float:
        return self._world.current_balance(self.agent.id)

    @property
    def exogenous_contract_summary(self):
        if not self._world.publish_exogenous_summary:
            return None
        summary = self._world.exogenous_contracts_summary
        if summary is None:
            return None
        exogenous = np.zeros((self.n_products, self.n_steps, 2))
        step = self._world.current_step
        for product in range(exogenous.shape[0]):
            exogenous[product, step, 0] = summary[product][0]
            exogenous[product, step, 1] = summary[product][1]
        return exogenous

    def __getattr__(self, attr):
        return getattr(self._owner.awi, attr)

    @property
    def inputs(self) -> np.ndarray:
        """Returns the number of inputs to every production process"""
        return np.ones(self.n_products - 1)

    @property
    def outputs(self) -> np.ndarray:
        """Returns the number of outputs to every production process"""
        return np.ones(self.n_products - 1)

    @property
    def my_input_products(self) -> np.ndarray:
        """Returns a list of products that are inputs to at least one process the agent can run"""
        return np.asarray([self._owner.awi.my_input_product])

    @property
    def my_output_products(self) -> np.ndarray:
        """Returns a list of products that are outputs to at least one process the agent can run"""
        return np.asarray([self._owner.awi.my_output_product])

    def is_system(self, aid: str) -> bool:
        """
        Checks whether an agent is a system agent or not

        Args:
            aid: Agent ID
        """
        return is_system_agent(aid)

    @property
    def state(self) -> FactoryState:
        """Receives the factory state"""
        bchanges = np.zeros(self.n_steps, dtype=float)
        if self._world.current_step:
            profits = self._world._profits[self._owner.id]
            bchanges[: len(profits)] = profits
        return FactoryState(
            inventory=np.zeros(self.n_products, dtype=int),
            commands=NO_COMMAND * np.ones((self.n_steps, self.n_lines), dtype=int),
            inventory_changes=np.zeros(self.n_products, dtype=int),
            balance_change=bchanges[-1],
            balance=int(self._world.scores()[self._owner.id]),
            contracts=[],
        )

    @property
    def profile(self) -> FactoryProfile:
        """Gets the profile (static private information) associated with the agent"""
        costs = INFINITE_COST * np.ones((self.n_lines, self.n_products), dtype=int)
        costs[:, self.my_input_product] = self._owner.awi.profile.cost
        return FactoryProfile(costs)

    @property
    def allow_zero_quantity(self) -> bool:
        """
        Does negotiations allow zero quantity?
        """
        return False
