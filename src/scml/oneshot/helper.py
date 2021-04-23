from typing import Union, Optional, Tuple, List, Any, Dict
import numpy as np
from negmas import Negotiator
from negmas.sao import SAOController, SAONegotiator
from ..scml2020.common import (
    FactoryState,
    FactoryProfile,
    ANY_LINE,
    ANY_STEP,
    INFINITE_COST,
    NO_COMMAND,
    is_system_agent,
)


class AWIHelper:
    """The Agent SCML2020World Interface for SCML2020 world allowing a single process per agent"""

    def __init__(self, owner: "OneShotSCML2020Adapter"):
        self._owner = owner
        self._world = owner.awi._world

    # --------
    # Actions
    # --------

    def request_negotiations(
        self,
        is_buy: bool,
        product: int,
        quantity: Union[int, Tuple[int, int]],
        unit_price: Union[int, Tuple[int, int]],
        time: Union[int, Tuple[int, int]],
        controller: Optional[SAOController] = None,
        negotiators: List[Negotiator] = None,
        partners: List[str] = None,
        extra: Dict[str, Any] = None,
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
            product,
            quantity,
            unit_price,
            time,
            controller,
            negotiators,
            extra,
        )

    def request_negotiation(
        self,
        is_buy: bool,
        product: int,
        quantity: Union[int, Tuple[int, int]],
        unit_price: Union[int, Tuple[int, int]],
        time: Union[int, Tuple[int, int]],
        partner: str,
        negotiator: SAONegotiator,
        extra: Dict[str, Any] = None,
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
        return self._world._request_negotiation(
            self._owner.id,
            product,
            quantity,
            unit_price,
            time,
            partner,
            negotiator,
            extra,
        )

    def schedule_production(
        self,
        process: int,
        repeats: int,
        step: Union[int, Tuple[int, int]] = ANY_STEP,
        line: int = ANY_LINE,
        override: bool = True,
        method: str = "latest",
        partial_ok: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return [[], []]

    def order_production(
        self, process: int, steps: np.ndarray, lines: np.ndarray
    ) -> None:
        return None

    def available_for_production(
        self,
        repeats: int,
        step: Union[int, Tuple[int, int]] = ANY_STEP,
        line: int = ANY_LINE,
        override: bool = True,
        method: str = "latest",
    ) -> Tuple[np.ndarray, np.ndarray]:
        return [
            np.asarray([self._world.current_step] * self._owner.awi.n_lines),
            np.arange(self._owner.awi.n_lines),
        ]

    def set_commands(self, commands: np.ndarray, step: int = -1) -> None:
        return None

    def cancel_production(self, step: int, line: int) -> bool:
        return None

    # ---------------------
    # Information Gathering
    # ---------------------

    @property
    def trading_prices(self) -> np.ndarray:
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
        return np.ones((self.n_products - 1))

    @property
    def outputs(self) -> np.ndarray:
        """Returns the number of outputs to every production process"""
        return np.ones((self.n_products - 1))

    @property
    def my_input_products(self) -> np.ndarray:
        """Returns a list of products that are inputs to at least one process the agent can run"""
        return [self._owner.awi.my_input_product]

    @property
    def my_output_products(self) -> np.ndarray:
        """Returns a list of products that are outputs to at least one process the agent can run"""
        return [self._owner.awi.my_output_product]

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
            balance_change=bchanges,
            balance=self._world.scores()[self._owner.id],
            contracts=[],
        )

    @property
    def profile(self) -> FactoryProfile:
        """Gets the profile (static private information) associated with the agent"""
        costs = INFINITE_COST * np.ones((self.n_lines, self.n_products), dtype=int)
        costs[:, self.my_input_product] = self._owner.awi.profile.cost
        return FactoryProfile(costs)
