"""Simulators module implementing factory simulation"""

import math
import sys
from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

from scml.scml2020 import ANY_LINE
from scml.scml2020 import ANY_STEP
from scml.scml2020 import NO_COMMAND
from scml.scml2020 import FactoryProfile

__all__ = ["FactorySimulator", "transaction", "temporary_transaction"]


@dataclass
class _Bookmark:
    id: int
    jobs: Dict[int, List[int]] = field(
        default_factory=lambda: defaultdict(list), init=False
    )
    buy_contracts: Dict[int, List[int]] = field(
        default_factory=lambda: defaultdict(list), init=False
    )
    sell_contracts: Dict[int, List[int]] = field(
        default_factory=lambda: defaultdict(list), init=False
    )
    payment_updates: Dict[int, int] = field(
        default_factory=lambda: defaultdict(int), init=False
    )
    inventory_updates: Dict[int, Dict[int, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int)), init=False
    )


NEVER = sys.maxsize  # indicates infinite future time


@dataclass
class _State:
    t: int
    inventory: np.array
    balance: int
    commands: np.array
    bankrupt_at: Optional[int]


@dataclass
class _FullBookmark:
    id: int
    balance: np.array
    inventory: np.array
    commands: np.array
    bankrupt_at: Optional[int]


class FactorySimulator:
    """
    A simulator that can be used to predict future state of a factory given some combination of operations (sell, buy,
    schedule).

    """

    def __init__(
        self,
        profile: FactoryProfile,
        initial_balance: int,
        bankruptcy_limit: int,
        spot_market_global_loss: float,
        catalog_prices: np.ndarray,
        n_steps: int,
        initial_inventory: np.ndarray = None,
    ):
        self._n_steps = n_steps
        n_products = profile.n_products
        self._catalog_prices = catalog_prices
        self._initial_balance = initial_balance
        self._initial_inventory = np.zeros(n_products)
        self._profile = profile
        self._n_products = n_products
        self._reserved_inventory = np.zeros(shape=(n_products, n_steps))
        self._bankrupt_at = NEVER
        self.bankruptcy_limit = bankruptcy_limit
        self.spot_market_global_loss = spot_market_global_loss
        n_steps, n_products, n_processes = (
            self._n_steps,
            self._n_products,
            self._n_products - 1,
        )
        self._n_lines = profile.n_lines
        self._balance = np.ones(n_steps) * initial_balance
        if initial_inventory is None:
            initial_inventory = np.zeros(n_products, dtype=int)

        self._inventory = initial_inventory
        self._inventory = np.repeat(
            initial_inventory.reshape((n_products, 1)), n_steps, axis=1
        )
        self._profile = profile
        self.commands = (
            np.ones(shape=(self._n_lines, self._n_steps), dtype=int) * NO_COMMAND
        )
        self.commands = np.zeros(shape=(self._n_lines, self._n_steps), dtype=int)
        self._fixed_before = 0
        self._bookmarks: List[_FullBookmark] = []
        self._active_bookmark: Optional[_FullBookmark] = None

    # -----------------
    # FIXED PROPERTIES
    # -----------------

    @property
    def n_steps(self) -> int:
        """Number of steps to predict ahead."""
        return self._n_steps

    @property
    def initial_balance(self) -> int:
        """Initial cash in balance"""
        return self._initial_balance

    @property
    def initial_inventory(self) -> np.array:
        """Initial inventory"""
        return self._initial_inventory

    @property
    def n_lines(self):
        """Number of lines"""
        return self._n_lines

    @property
    def final_balance(self) -> int:
        """Returns the final balance of the agent at the end of the simulation"""
        return self._balance[-1]

    def final_score(self, prices: Optional[np.ndarray]) -> int:
        """Returns the final balance of the agent at the end of the simulation"""
        return self._balance[-1]

    # -------------------------------
    # DYNAMIC PROPERTIES (READ STATE)
    # -------------------------------

    def inventory_at(self, t: int) -> np.array:
        """
        Returns the inventory of all products *at* time t

        Args:

            t: Time

        Returns:

            An array of size `n_products` giving the quantity of each product in inventory at time-step `t`.

        See Also:

            `inventory_to` `balance_at`

        """
        return self.inventory_to(t)[:, -1]

    def line_schedules_at(self, t: int) -> np.array:
        """
        Returns the schedule of each line at a given timestep

        Args:

            t: time

        Returns:

            An array of `n_lines` values giving the schedule up at `t`.

        Remarks:

            - A `NO_COMMAND` value means no production, otherwise the index of the process being run
        """
        return self.commands[:, :t]

    def reserved_inventory_to(self, t: int) -> np.array:
        """
        Returns the *reserved* inventory of all products *up to* time t

        Args:

            t: Time

        Returns:

            An array of size `n_products` * `t` giving the quantity of each product reserved at every step up to `t`.

        Remarks:

            - Reserved inventory *is counted* in calls to `inventory_at` , `total_inventory_at` , `inventory_to`
              , `total_inventory_to`
            - Reserving quantities of products is a tool that can be used to avoid double counting availability of given
              products in the inventory for multiple contracts.

        See Also:

            `total_inventory_at` `inventory_at` `reserved_inventory_at`

        """
        return self._reserved_inventory[:, : t + 1]

    def reserved_inventory_at(self, t: int) -> np.array:
        """
        Returns the *reserved* inventory of all products *at* time t

        Args:

            t: Time

        Returns:

            An array of size `n_products` giving the quantity of each product reserved at time-step `t`.

        Remarks:

            - Reserved inventory *is counted* in calls to `inventory_at` , `total_inventory_at` , `inventory_to`
              , `total_inventory_to`
            - Reserving quantities of products is a tool that can be used to avoid double counting availability of given
              products in the inventory for multiple contracts.

        See Also:

            `total_inventory_to` `inventory_to` `reserved_inventory_at`

        """
        return self._reserved_inventory[:, t]

    def available_inventory_to(self, t: int) -> np.array:
        """
        Returns the *available* inventory of all products *up to* time t.

        Args:

            t: Time

        Returns:

            An array of size `n_products` * `t` giving the quantity of each product available at every step up to `t`.

        Remarks:

            - Available inventory is defined as the difference between inventory and reserved inventory.
            - Reserved inventory *is counted* in calls to `inventory_at` , `total_inventory_at` , `inventory_to`
              , `total_inventory_to`
            - Reserving quantities of products is a tool that can be used to avoid double counting availability of given
              products in the inventory for multiple contracts.

        See Also:

            `total_inventory_to` `inventory_to` `reserved_inventory_to`

        """
        return self.inventory_to(t) - self.reserved_inventory_to(t)

    def available_inventory_at(self, t: int) -> np.array:
        """
        Returns the *available* inventory of all products *at* time t

        Args:

            t: Time

        Returns:

            An array of size `n_products` giving the quantity of each product available at time-step `t`.

        Remarks:

            - Available inventory is defined as the difference between inventory and reserved inventory.
            - Reserved inventory *is counted* in calls to `inventory_at` , `total_inventory_at` , `inventory_to`
              , `total_inventory_to`
            - Reserving quantities of products is a tool that can be used to avoid double counting availability of given
              products in the inventory for multiple contracts.

        See Also:

            `total_inventory_to` `inventory_to` `reserved_inventory_at`

        """
        return self.inventory_at(t) - self.reserved_inventory_at(t)

    def is_bankrupt(self) -> bool:
        """Checks if the agent will go bankrupt given all the info so far"""
        return np.min(self._balance) < self.bankruptcy_limit

    def balance_to(self, t: int) -> np.array:
        """
        Returns the balance fo the factory until and including time t.

        Args:
            t: time

        Remarks:

            - The balance is defined as the cash in balance

        """

        return self._balance[: t + 1]

    def score(self, inventory_weight=0.5) -> float:
        """
        Estimates the final score of the agent

        Args:
            inventory_weight: The weight of the inventory that remains at
                              the end of the simulation

        Remarks:
            - It uses the catalog prices for price estimation. This
              may be inaccurate. There is no way to know the actual
              trading prices of the market that are used to calculate
              the real score
        """
        return self.balance_at(
            self.n_steps - 1
        ) + inventory_weight * self._catalog_prices * self.inventory_at(
            self.n_steps - 1
        )

    def balance_at(self, t: int) -> np.array:
        """
        Returns the balance of the factory at time t.

        Args:
            t: time

        Remarks:

            - The balance is defined as the cash in balance

        """

        return self._balance[t]

    def inventory_to(self, t: int) -> np.array:
        """
        Returns the balance fo the factory *up to* time t.

        Args:

            t: time

        Remarks:

            - The balance is defined as the cash in balance

        """

        return self._inventory[:, : t + 1]

    def line_schedules_to(self, t: int) -> np.array:
        return self.commands[:, : t + 1]

    # -------------------------
    # OPERATIONS (UPDATE STATE)
    # -------------------------

    def receive(self, payment: int, t: int) -> bool:
        """
        Simulates receiving payment at time t

        Args:

            payment: Amount received
            t: time

        Returns:

            Success or failure

        """
        return self.pay(-payment, t)

    def reserve(self, product: int, quantity: int, t: int) -> bool:
        """
        Simulates reserving the given quantity of the given product at times >= t.

        Args:

            product: Index/ID of the product being reserved
            quantity: quantity being reserved
            t: time

        Returns:

            Success/failure

        Remarks:

            - Reserved products do not show in calls to  `inventory_at` , `inventory_to` etc.
            - Reserving a product does nothing more than mark some quantity as reserved for calls to
              `reserved_inventory_at` and `available_inventory_at`.
            - This feature can be used to simulate inventory hiding commands in the real factory and to avoid
              double counting of inventory when calculating needs for future contracts.

        """
        self._reserved_inventory[product, t] += quantity
        return True

    def pay(self, payment: int, t: int, ignore_money_shortage: bool = True) -> bool:
        """
        Simulate payment at time t

        Args:

            payment: Amount payed
            t: time
            ignore_money_shortage: If True, shortage in money will be ignored and the balance can go negative

        Returns:
            Success or failure
        """
        if t < self._fixed_before:
            raise ValueError(
                f"Cannot run operations in the past (t={t}, fixed before {self._fixed_before})"
            )
        b = self._balance[t:]
        if b.size < 1:
            return False
        if ignore_money_shortage:
            b -= payment
            return True
        b -= payment
        if b.min() < self.bankruptcy_limit:
            b += payment
            return False
        return True
        # interest rate computation. Ignored for now
        # backup = b.copy()
        # for i in range(len(b)):
        #     b[i] -= payment
        #     if b[i] < self.bankruptcy_limit:
        #         self._balance[t:] = backup
        #         return False
        #     if b[i] < 0 <= b[i] + payment:
        #         payment -= int(math.ceil(self.interest_rate * b[i]))
        # return True

    def transport_to(
        self,
        product: int,
        quantity: int,
        t: int,
        ignore_inventory_shortage: bool = True,
    ) -> bool:
        """
        Simulates transporting products to/from inventory at time t

        Args:

            product: product ID (index)
            quantity: quantity to transport
            t: time
            ignore_inventory_shortage: Ignore shortage in the `product` which may lead to negative inventory[product]

        Returns:

            Success or failure

        """
        if t < self._fixed_before:
            raise ValueError(
                f"Cannot run operations in the past (t={t}, fixed before {self._fixed_before})"
            )
        s = self._inventory[product, t:].view()
        if s.size < 1:
            return False
        s += quantity
        if ignore_inventory_shortage:
            return True
        if s.min() < 0:
            s -= quantity
            return False
        return True

    def buy(
        self,
        product: int,
        quantity: int,
        price: int,
        t: int,
        ignore_money_shortage: bool = True,
    ) -> bool:
        """
        Buy a given quantity of a product for a given price at some time t

        Args:

            product: Product to buy (ID/index)
            quantity: quantity to buy
            price: unit price
            t: time
            ignore_money_shortage: If True, shortage in money will be ignored and the balance can go negative

        Returns:

            Success or failure

        Remarks:

            - buy cannot ever have inventory shortage

        See Also:

            `sell`

        """
        if t < self._fixed_before:
            raise ValueError(
                f"Cannot run operations in the past (t={t}, fixed before {self._fixed_before})"
            )
        balance = self._balance.copy()
        if not self.pay(price * quantity, t, ignore_money_shortage):
            self._balance = balance
            return False
        return self.transport_to(product, quantity, t, True)

    def sell(
        self,
        product: int,
        quantity: int,
        price: int,
        t: int,
        ignore_inventory_shortage: bool = True,
    ) -> bool:
        """
        sell a given quantity of a product for a given price at some time t

        Args:

            product: Index/ID of the product to be sold
            quantity: quantity to be sold
            price: unit price
            t: time
            ignore_inventory_shortage: If True, shortage in inventory will be ignored and the inventory can go negative

        Returns:

            Success or failure


        Remarks:

            - sell cannot ever have space shortage

        See Also:

            `buy`

        """
        if t < self._fixed_before:
            raise ValueError(
                f"Cannot run operations in the past (t={t}, fixed before {self._fixed_before})"
            )
        inventory = self._inventory.copy()
        if not self.transport_to(product, -quantity, t, ignore_inventory_shortage):
            self._inventory = inventory
            return False
        return self.pay(-price * quantity, t, True)

    def available_for_production(
        self,
        repeats: int,
        step: Union[int, Tuple[int, int]] = ANY_STEP,
        line: int = ANY_LINE,
        override: bool = True,
        method: str = "latest",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Finds available times and lines for scheduling production.

        Args:

            repeats: How many times to repeat the process
            step: The simulation step or a range of steps. The special value ANY_STEP gives the factory the freedom to
                  schedule production at any step in the present or future.
            line: The production line. The special value ANY_LINE gives the factory the freedom to use any line
            override: Whether to override any existing commands at that line at that time.
            method: When to schedule the command if step was set to a range. Options are latest, earliest, all

        Returns:

            Tuple[np.ndarray, np.ndarray] The steps and lines at which production is scheduled.

        Remarks:

            - You cannot order production in the past or in the current step
            - Ordering production, will automatically update inventory and balance for all simulation steps assuming
              that this production will be carried out. At the indicated `step` if production was not possible (due
              to insufficient funds or insufficient inventory of the input product), the predictions for the future
              will be corrected.

        """
        current_step = self._fixed_before + 1
        if not isinstance(step, tuple):
            if step < 0:
                step = (current_step, self._n_steps)
            else:
                step = (step, step + 1)
        else:
            step = (step[0], step[1] + 1)
        step = (max(current_step, step[0]), step[1])
        step = (min(step[0], self._fixed_before), min(step[1], self._fixed_before))
        if step[1] <= step[0]:
            return np.empty(shape=0, dtype=int), np.empty(shape=0, dtype=int)
        if override:
            if line < 0:
                steps, lines = np.nonzero(
                    self.commands[step[0] : step[1], :] >= NO_COMMAND
                )
            else:
                steps = np.nonzero(
                    self.commands[step[0] : step[1], line] >= NO_COMMAND
                )[0]
                lines = [line]
        else:
            if line < 0:
                steps, lines = np.nonzero(
                    self.commands[step[0] : step[1], :] == NO_COMMAND
                )
            else:
                steps = np.nonzero(
                    self.commands[step[0] : step[1], line] == NO_COMMAND
                )[0]
                lines = [line]
        steps += step[0]
        possible = min(repeats, len(steps))
        if possible < repeats:
            return np.empty(shape=0, dtype=int), np.empty(shape=0, dtype=int)
        if method.startswith("l"):
            steps, lines = steps[-possible + 1 :], lines[-possible + 1 :]
        elif method == "all":
            pass
        else:
            steps, lines = steps[:possible], lines[:possible]

        return steps, lines

    def order_production(
        self, process: int, steps: np.ndarray, lines: np.ndarray
    ) -> None:
        """
        Orders production of the given process

        Args:
            process: The process to run
            steps: The time steps to run the process at as an np.ndarray
            lines: The corresponding lines to run the process at

        Remarks:

            - len(steps) must equal len(lines)
            - No checks are done in this function. It is expected to be used after calling `available_for_production`
        """
        if len(steps) == 0:
            return
        if np.min(steps) < self._fixed_before:
            raise ValueError(
                f"Cannot run operations in the past (t={np.min(steps)}, fixed before {self._fixed_before})"
            )
        self.commands[steps, lines] = process

    def schedule(
        self,
        process: int,
        quantity: int,
        t: Union[int, Tuple[int, int]] = ANY_STEP,
        line: int = ANY_LINE,
        override=True,
        method: str = "latest",
        ignore_inventory_shortage=True,
        ignore_money_shortage=True,
    ) -> bool:
        """
        Simulates scheduling the given job at its `time` and `line` optionally overriding whatever was already scheduled

        Args:

            process: The process to run
            quantity: The quantity to be produced
            t: The time-step step
            line: The line
            ignore_inventory_shortage: If true shortages in inputs will be ignored
            ignore_money_shortage: If true, shortage in money will be ignored
            override: Whether the job should override any already registered job at its time-step
            method: The method employed for scheduling. Supported methods are latest, earliest

        Returns:

            Success/failure
        """
        steps, lines = self.available_for_production(quantity, t, line, override, "all")
        if len(steps) < quantity:
            return False
        cost = self._profile.costs[process]
        # confirm that there is enough money to start production
        if (not ignore_money_shortage) and np.any(self._balance[t:] < cost):
            return False
        # bookmark to be able to rollback at any error
        with transaction(self) as bookmark:
            if not self.pay(cost, t):
                self.rollback(bookmark)
                return False
            scheduled = 0
            for s, l in zip(steps, lines):
                if not (
                    (ignore_inventory_shortage or self._inventory[process, s] >= 1)
                    and (ignore_money_shortage or (self._balance[s] >= cost))
                ):
                    continue
                scheduled += 1
                self.commands[s, l] = process
                self._inventory[process, s] -= 1
                self._inventory[process + 1, s] += 1
                self._balance[s] -= cost
            if scheduled < quantity:
                self.rollback(bookmark)
                return False
        return True

    # ------------------
    # HISTORY MANAGEMENT
    # ------------------

    def fix_before(self, t: int) -> bool:
        """
        Fix the history before this point

        Args:

            t: time

        Returns:

            Success/failure

        Remarks:

            - After this function is called at any time-step `t`, there is no way to change any component of the factory
              state at any timestep before `t`.
            - This function is useful for *fixing* any difference between the simulator and the real state (in
              conjunction with `set_state`).

        See Also:

            `set_state` `fixed_before`

        """
        self._fixed_before = t
        return True

    @property
    def fixed_before(self):
        """
        Gives the time before which the schedule is fixed.

        See Also:
            `fix_before`

        """
        return self._fixed_before

    def delete_bookmark(self, bookmark_id: int) -> bool:
        """
        Commits everything since the bookmark so it cannot be rolled back

        Args:

            bookmark_id The bookmark ID returned from bookmark

        Returns:

            Success/failure

        Remarks:

            - You can delete bookmarks in the reverse order of their creation only. If the bookmark ID given here is
              not the one at the top of the bookmarks stack, the deletion will fail (return False).

        See Also:

            `delete_bookmark` `rollback` `transaction` `temporary_transaction`
        """
        if self._active_bookmark is None or self._active_bookmark.id != bookmark_id:
            raise ValueError(f"there is no active bookmark to delete")
        self._bookmarks = self._bookmarks[:-1]
        self._active_bookmark = (
            self._bookmarks[-1] if len(self._bookmarks) > 0 else None
        )
        return True

    def bookmark(self) -> int:
        """Sets a bookmark to the current location

        Returns:

            bookmark ID

        Remarks:

            - Bookmarks can be used to implement transactions.


        See Also:

            `delete_bookmark` `rollback` `transaction` `temporary_transaction`
        """
        bookmark = _FullBookmark(
            id=len(self._bookmarks),
            balance=self._balance.copy(),
            inventory=self._inventory.copy(),
            commands=self.commands.copy(),
            bankrupt_at=self._bankrupt_at,
        )
        self._bookmarks.append(bookmark)
        self._active_bookmark = bookmark
        return bookmark.id

    def rollback(self, bookmark_id: int) -> bool:
        """Rolls back to the given bookmark ID

        Args:
            bookmark_id The bookmark ID returned from bookmark

        Remarks:

            - You can only rollback in the reverse order of bookmarks. If the bookmark ID given here is not the one
              at the top of the bookmarks stack, the rollback will fail (return False)

        See Also:

            `delete_bookmark` `rollback` `transaction` `temporary_transaction`
        """
        if self._active_bookmark is None or self._active_bookmark.id != bookmark_id:
            raise ValueError(f"there is no active bookmark to rollback")
        b = self._active_bookmark
        self._balance, self._inventory = b.balance, b.inventory
        self.commands = b.commands
        return True

    def set_state(
        self, t: int, inventory: np.array, balance: int, commands: np.array
    ) -> None:
        """
        Sets the current state at the given time-step. It implicitly causes a fix_before(t + 1)

        Args:

            t: Time step to set the state at
            inventory: quantity of every product (array of integers of size `n_products`)
            balance: Cash in balance
            commands: Line schedules (array of process numbers/NO_PRODUCTION of size `n_lines`)

        """
        self._inventory[:, t:] += inventory.reshape(
            self._n_products, 1
        ) - self._inventory[:, t].reshape(self._n_products, 1)
        self._balance[t:] += balance - self._balance[t]
        self.commands[:, t] = commands
        self.fix_before(t)


@contextmanager
def transaction(simulator):
    """Runs the simulated actions then confirms them if they are not rolled back"""
    _bookmark = simulator.bookmark()
    yield _bookmark
    simulator.delete_bookmark(_bookmark)


@contextmanager
def temporary_transaction(simulator):
    """Runs the simulated actions then rolls them back"""
    _bookmark = simulator.bookmark()
    yield _bookmark
    simulator.rollback(_bookmark)
    simulator.delete_bookmark(_bookmark)
