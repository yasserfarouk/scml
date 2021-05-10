"""Implements the world class for the SCML2020 world """
import copy
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

from .common import ANY_LINE
from .common import ANY_STEP
from .common import INFINITE_COST
from .common import NO_COMMAND
from .common import is_system_agent, FactoryProfile, ContractInfo, FactoryState, Failure

__all__ = [
    "Factory",
]


class Factory:
    """A simulated factory"""

    def __init__(
        self,
        profile: FactoryProfile,
        initial_balance: int,
        inputs: np.ndarray,
        outputs: np.ndarray,
        catalog_prices: np.ndarray,
        world: "SCML2020World",
        compensate_before_past_debt: bool,
        buy_missing_products: bool,
        production_buy_missing: bool,
        production_penalty: float,
        production_no_bankruptcy: bool,
        production_no_borrow: bool,
        agent_id: str,
        agent_name: Optional[str] = None,
        confirm_production: bool = True,
        initial_inventory: Optional[np.ndarray] = None,
        disallow_concurrent_negs_with_same_partners=False,
    ):
        self.confirm_production = confirm_production
        self.production_buy_missing = production_buy_missing
        self.compensate_before_past_debt = compensate_before_past_debt
        self.buy_missing_products = buy_missing_products
        self.production_penalty = production_penalty
        self.production_no_bankruptcy = production_no_bankruptcy
        self.production_no_borrow = production_no_borrow
        self.catalog_prices = catalog_prices
        self.initial_balance = initial_balance
        self.__profile = profile
        self.world = world
        self.profile = copy.deepcopy(profile)
        self._disallow_concurrent_negs_with_same_partners = (
            disallow_concurrent_negs_with_same_partners
        )
        """The readonly factory profile (See `FactoryProfile` )"""
        self.commands = NO_COMMAND * np.ones(
            (world.n_steps, profile.n_lines), dtype=int
        )
        """An n_steps * n_lines array giving the process scheduled for each line at every step. -1 indicates an empty
        line. """
        self._balance = initial_balance
        """Current balance"""
        self._inventory = (
            np.zeros(profile.n_products, dtype=int)
            if initial_inventory is None
            else initial_inventory
        )
        """Current inventory"""
        self.agent_id = agent_id
        """A unique ID for the agent owning the factory"""
        self.inputs = inputs
        """An n_process array giving the number of inputs needed for each process
        (of the product with the same index)"""
        self.outputs = outputs
        """An n_process array giving the number of outputs produced by each process
        (of the product with the next index)"""
        self.inventory_changes = np.zeros(len(inputs) + 1, dtype=int)
        """Changes in the inventory in the last step"""
        self.balance_change = 0
        """Change in the balance in the last step"""
        self.min_balance = self.world.bankruptcy_limit
        """The minimum balance possible"""
        self.is_bankrupt = False
        """Will be true when the factory is bankrupt"""
        self.agent_name = (
            self.world.agents[agent_id].name
            if agent_name is None and world
            else agent_name
        )
        """SCML2020Agent names used for logging purposes"""
        self.contracts: List[List[ContractInfo]] = [[] for _ in range(world.n_steps)]
        """A list of lists of contracts per time-step (len == n_steps)"""

    @property
    def state(self) -> FactoryState:
        return FactoryState(
            self._inventory.copy(),
            self._balance,
            self.commands,
            self.inventory_changes,
            self.balance_change,
            [copy.copy(_.contract) for times in self.contracts for _ in times],
        )

    @property
    def current_inventory(self) -> np.ndarray:
        """Current inventory contents"""
        return self._inventory

    @property
    def current_balance(self) -> int:
        """Current wallet balance"""
        return self._balance

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
        """
        Orders production of the given process on the given step and line.

        Args:

            process: The process index
            repeats: How many times to repeat the process
            step: The simulation step or a range of steps. The special value ANY_STEP gives the factory the freedom to
                  schedule production at any step in the present or future.
            line: The production line. The special value ANY_LINE gives the factory the freedom to use any line
            override: Whether to override any existing commands at that line at that time.
            method: When to schedule the command if step was set to a range. Options are latest, earliest, all
            partial_ok: If true, it is OK to produce only a subset of repeats

        Returns:

            Tuple[np.ndarray, np.ndarray] The steps and lines at which production is scheduled.

        Remarks:

            - You cannot order production in the past or in the current step
            - Ordering production, will automatically update inventory and balance for all simulation steps assuming
              that this production will be carried out. At the indicated `step` if production was not possible (due
              to insufficient funds or insufficient inventory of the input product), the predictions for the future
              will be corrected.

        """
        if self.is_bankrupt:
            return np.empty(0, dtype=int), np.empty(0, dtype=int)
        steps, lines = self.available_for_production(
            repeats, step, line, override, method
        )
        if len(steps) < 1:
            return np.empty(0, dtype=int), np.empty(0, dtype=int)
        if len(steps) < repeats:
            if not partial_ok:
                return np.empty(0, dtype=int), np.empty(0, dtype=int)
            repeats = len(steps)
        self.order_production(process, steps[:repeats], lines[:repeats])
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
        if self.is_bankrupt:
            return
        if len(steps) > 0:
            self.commands[steps, lines] = process

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
        if self.is_bankrupt:
            return np.empty(shape=0, dtype=int), np.empty(shape=0, dtype=int)
        current_step = self.world.current_step
        if not isinstance(step, tuple):
            if step < 0:
                step = (current_step, self.world.n_steps)
            else:
                step = (step, step + 1)
        else:
            step = (step[0], step[1] + 1)
        step = (max(current_step, step[0]), step[1])
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
        try:
            if method.startswith("l"):
                steps, lines = steps[-possible + 1 :], lines[-possible + 1 :]
            elif method == "all":
                pass
            else:
                steps, lines = steps[:possible], lines[:possible]
        except:
            return np.empty(shape=0, dtype=int), np.empty(shape=0, dtype=int)
        return steps, lines

    def cancel_production(self, step: int, line: int) -> bool:
        """
        Cancels pre-ordered production given that it did not start yet.

        Args:

            step: Step to cancel at
            line: Line to cancel at

        Returns:

            True if step >= self.current_step

        Remarks:

            - Cannot cancel a process in the past or present.
        """
        if self.is_bankrupt:
            return False
        if step < self.world.current_step or line < 0:
            return False
        self.commands[step, line] = NO_COMMAND
        return True

    def step(self) -> List[Failure]:
        """
        Override this method to modify stepping logic.
        """
        if self.is_bankrupt:
            return []
        step = self.world.current_step
        profile = self.__profile
        failures = []
        initial_balance = self._balance
        initial_inventory = self._inventory.copy()

        if self.confirm_production:
            self.commands[step, :] = self.world.call(
                self.world.agents[self.agent_id],
                self.world.agents[self.agent_id].confirm_production,
                self.commands[step, :],
                self.current_balance,
                self.current_inventory.copy(),
            )

        # do production
        for line in np.nonzero(self.commands[step, :] != NO_COMMAND)[0]:
            p = self.commands[step, line]
            cost = profile.costs[line, p]
            ins, outs = self.inputs[p], self.outputs[p]

            # if execution will lead to bankruptcy or the cost is infinite, ignore this command
            if self._balance - cost < self.min_balance or cost == INFINITE_COST:
                failures.append(
                    Failure(is_inventory=False, line=line, step=step, process=p)
                )
                # self._register_failure(step, p, cost, ins, outs)
                continue
            inp, outp = p, p + 1

            # if we do not have enough inputs, ignore this command
            if self._inventory[inp] < ins:
                failures.append(
                    Failure(is_inventory=True, line=line, step=step, process=p)
                )
                continue

            # execute the command
            self._balance -= cost
            self.store(
                inp,
                -ins,
                self.production_buy_missing,
                self.production_penalty,
                self.production_no_bankruptcy,
                self.production_no_borrow,
            )
            self.store(
                outp,
                outs,
                self.production_buy_missing,
                self.production_penalty,
                self.production_no_bankruptcy,
                self.production_no_borrow,
            )

        assert self._balance >= self.min_balance
        assert np.min(self._inventory) >= 0
        self.inventory_changes = self._inventory - initial_inventory
        self.balance_change = self._balance - initial_balance
        return failures

    def spot_price(self, product: int, spot_loss: float) -> int:
        """
        Get the current spot price for buying the given product on the spot market

        Args:
            product: Product
            spot_loss: Spot loss specific to that agent

        Returns:
            The unit price
        """
        return int(np.ceil(self.world.trading_prices[product] * (1 + spot_loss)))

    def store(
        self,
        product: int,
        quantity: int,
        buy_missing: bool,
        spot_price: float,
        no_bankruptcy: bool = False,
        no_borrowing: bool = False,
    ) -> int:
        """
        Stores the given amount of product (signed) to the factory.

        Args:
            product: Product
            quantity: quantity to store/take out (-ve means take out)
            buy_missing: If the quantity is negative and not enough product exists in the market, it buys the product
                         from the spot-market at an increased price of penalty
            spot_price: The fraction of unit_price added because we are buying from the spot market. Only effective if
                     quantity is negative and not enough of the product exists in the inventory
            no_bankruptcy: Never bankrupt the agent on this transaction
            no_borrowing: Never borrow for this transaction

        Returns:
            The quantity actually stored or taken out (always positive)
        """
        if self.is_bankrupt:
            self.world.logwarning(
                f"{self.agent_name} received a transaction "
                f"(product: {product}, q: {quantity}) after being bankrupt"
            )
            return 0
        available = self._inventory[product]
        if available + quantity >= 0:
            self._inventory[product] += quantity
            self.inventory_changes[product] += quantity
            return int(quantity if quantity > 0 else -quantity)
        # we have an inventory breach here. We know that quantity < 0
        assert quantity < 0
        quantity = -quantity
        if not buy_missing:
            # if we are not buying from the spot market, pay the penalty for missing products and transfer all available
            to_pay = int(
                np.ceil(spot_price * (quantity - available) / quantity)
                * self.world.trading_prices[product]
            )
            self.pay(to_pay, no_bankruptcy, no_borrowing)
            self._inventory[product] = 0
            self.inventory_changes[product] -= available
            return int(available)
        # we have an inventory breach and should try to buy missing quantity from the spot market
        effective_unit = self.spot_price(product, spot_price)
        effective_total = (quantity - available) * effective_unit
        paid = self.pay(
            effective_total, no_bankruptcy, no_borrowing, unit=effective_unit
        )
        paid_for = int(paid // effective_unit)
        assert self._inventory[product] + paid_for >= 0, (
            f"{self.agent_name} had {self._inventory[product]} and paid for {paid_for} ("
            f"original quantity {quantity})"
        )
        self._inventory[product] += paid_for
        self.inventory_changes[product] += paid_for
        self.store(product, -quantity, False, 0.0, True, True)
        return int(paid_for)

    def buy(
        self,
        product: int,
        quantity: int,
        unit_price: int,
        buy_missing: bool,
        penalty: float,
        no_bankruptcy: bool = False,
        no_borrowing: bool = False,
    ) -> Tuple[int, int]:
        """
        Executes a transaction to buy/sell involving adding quantity and paying price (both are signed)

        Args:
            product: The product transacted on
            quantity: The quantity (added)
            unit_price: The unit price (paid)
            buy_missing: If true, attempt buying missing products from the spot market
            penalty: The penalty as a fraction to be paid for breaches
            no_bankruptcy: If true, this transaction can never lead to bankruptcy
            no_borrowing: If true, this transaction can never lead to borrowing

        Returns:
            Tuple[int, int] The actual quantities bought and the total cost
        """
        if self.is_bankrupt:
            self.world.logwarning(
                f"{self.agent_name} received a transaction "
                f"(product: {product}, q: {quantity}, u:{unit_price}) after being bankrupt"
            )
            return 0, 0
        if quantity < 0:
            # that is a sell contract
            taken = self.store(
                product, quantity, buy_missing, penalty, no_bankruptcy, no_borrowing
            )
            paid = self.pay(-taken * unit_price, no_bankruptcy, no_borrowing)
            return taken, paid
        # that is a buy contract
        paid = self.pay(quantity * unit_price, no_bankruptcy, no_borrowing)
        stored = self.store(
            product,
            paid // unit_price,
            buy_missing,
            penalty,
            no_bankruptcy,
            no_borrowing,
        )
        return stored, paid

    def pay(
        self,
        money: int,
        no_bankruptcy: bool = False,
        no_borrowing: bool = False,
        unit: int = 0,
    ) -> int:
        """
        Pays money

        Args:
            money: amount to pay
            no_bankruptcy: If true, this transaction can never lead to bankruptcy
            no_borrowing: If true, this transaction can never lead to borrowing
            unit: If nonzero then an integer multiple of unit will be paid

        Returns:
            The amount actually paid

        """
        if self.is_bankrupt:
            self.world.logwarning(
                f"{self.agent_name} was asked to pay {money} after being bankrupt"
            )
            return 0
        new_balance = self._balance - money
        if new_balance < self.min_balance:
            if no_bankruptcy:
                money = self._balance - self.min_balance
            else:
                money = self.bankrupt(money)
        elif no_borrowing and new_balance < 0:
            money = self._balance
        if unit > 0:
            money = (money // unit) * unit
        self._balance -= money
        self.balance_change -= money
        return money

    def bankrupt(self, required: int) -> int:
        """
        Bankruptcy processing for the given agent

        Args:
            required: The money required after the bankruptcy is processed

        Returns:
            The amount of money to pay back to the entity that should have been paid `money`

        """
        self.world.logdebug(
            f"bankrupting {self.agent_name} (has: {self._balance}, needs {required})"
        )

        # sell everything on the agent's inventory
        spot_loss = np.array(
            self.world._agent_spot_loss[
                self.world.a2i[self.agent_id], self.world.current_step
            ]
        )
        prices = self.world.trading_prices / (
            (1 + spot_loss) * (1 + self.world.spot_market_global_loss)
        )

        total = np.sum(self._inventory * self.world.liquidation_rate * prices)
        pay_back = min(required, total)
        available = total - required

        # If past debt is paid before compensation pay it
        original_balance = self._balance
        if not self.compensate_before_past_debt:
            available += original_balance

        compensations = self.world.compensate(available, self)
        self.is_bankrupt = True
        for agent in self.world.agents.values():
            if is_system_agent(agent.id) or agent.id == self.agent_id:
                continue
            if agent.id in compensations.keys():
                info = compensations[agent.id]
                agent.on_agent_bankrupt(
                    self.agent_id,
                    [_[0] for _ in info],
                    [_[1] for _ in info],
                    [_[2] for _ in info],
                )
            else:
                agent.on_agent_bankrupt(self.agent_id, [], [], 0)
        return pay_back
