"""Implements the world class for the SCML2020 world """
import copy
import functools
import itertools
import logging
import math
import numbers
import random
from abc import abstractmethod
from collections import defaultdict, namedtuple
import sys
from dataclasses import dataclass, field
from typing import (
    Optional,
    Dict,
    Any,
    Union,
    Tuple,
    Callable,
    List,
    Set,
    Collection,
    Type,
    Iterable,
)
import numpy as np

from negmas import (
    Contract,
    Action,
    Breach,
    AgentWorldInterface,
    Agent,
    RenegotiationRequest,
    Negotiator,
    AgentMechanismInterface,
    MechanismState,
    Issue,
    Entity,
    SAONegotiator,
    SAOController,
    PassThroughSAONegotiator,
)
from negmas.helpers import instantiate, unique_name
from negmas.situated import World, TimeInAgreementMixin, BreachProcessing

__all__ = [
    "FactoryState",
    "SCML2020Agent",
    "AWI",
    "World",
    "FinancialReport",
    "FactoryProfile",
    "INFINITE_COST",
    "ANY_LINE",
    "ANY_STEP",
    "NO_COMMAND",
    "Factory",
]


ANY_STEP = -1
"""Used to indicate any time-step"""


ANY_LINE = -1
"""Used to indicate any line"""


NO_COMMAND = -1
"""A constant indicating no command is scheduled on a factory line"""


INFINITE_COST = sys.maxsize // 2
"""A constant indicating an invalid cost for lines incapable of running some process"""


ContractInfo = namedtuple(
    "ContractInfo", ["q", "u", "product", "is_seller", "partner", "contract"]
)
"""Information about a contract including a pointer to it"""

CompensationRecord = namedtuple(
    "CompensationRecord", ["product", "quantity", "money", "seller_bankrupt", "factory"]
)
"""A record of delayed compensation used when a factory goes bankrupt to keep honoring its future contracts to the 
limit possible"""


@dataclass
class FinancialReport:
    """A report published periodically by the system showing the financial standing of an agent"""

    __slots__ = [
        "agent_id",
        "step",
        "cash",
        "assets",
        "breach_prob",
        "breach_level",
        "is_bankrupt",
        "agent_name",
    ]
    agent_id: str
    """Agent ID"""
    step: int
    """Simulation step at the beginning of which the report was published."""
    cash: int
    """Cash in the agent's wallet. Negative numbers indicate liabilities."""
    assets: int
    """Value of the products in the agent's inventory @ catalog prices. """
    breach_prob: float
    """Number of times the agent breached a contract over the total number of contracts it signed."""
    breach_level: float
    """Sum of the agent's breach levels so far divided by the number of contracts it signed."""
    is_bankrupt: bool
    """Whether the agent is already bankrupt (i.e. incapable of doing any more transactions)."""
    agent_name: str
    """Agent name for printing purposes"""

    def __str__(self):
        bankrupt = "BANKRUPT" if self.is_bankrupt else ""
        return (
            f"{self.agent_name} @ {self.step} {bankrupt}: Cash: {self.cash}, Assets: {self.assets}, "
            f"breach_prob: {self.breach_prob}, breach_level: {self.breach_level}"
        )


@dataclass
class FactoryProfile:
    """Defines all private information of a factory"""

    __slots__ = [
        "costs",
        "external_sales",
        "external_supplies",
        "external_sale_prices",
        "external_supply_prices",
    ]
    costs: np.ndarray
    """An n_lines * n_processes array giving the cost of executing any process (INVALID_COST indicates infinity)"""
    external_sales: np.ndarray
    """A n_steps * n_products array giving guaranteed sales of different products for the whole simulation time"""
    external_supplies: np.ndarray
    """A n_steps * n_products array giving guaranteed sales of different products for the whole simulation time"""
    external_sale_prices: np.ndarray
    """A n_steps * n_products array giving guaranteed unit prices for the `external_quantities` . It will be zero
    for times and products for which there are no guaranteed quantities (i.e. (external_quantities[...] == 0) =>
     (external_prices[...] == 0) )"""
    external_supply_prices: np.ndarray
    """A n_steps * n_products array giving guaranteed unit prices for the `external_quantities` . It will be zero
    for times and products for which there are no guaranteed quantities (i.e. (external_quantities[...] == 0) =>
     (external_prices[...] == 0) )"""

    @property
    def n_lines(self):
        return self.costs.shape[0]

    @property
    def n_products(self):
        return self.external_sales.shape[1]

    @property
    def n_steps(self):
        return self.external_sales.shape[0]

    @property
    def n_processes(self):
        return self.costs.shape[1]


@dataclass
class Failure:
    """A production failure"""

    __slots__ = ["is_inventory", "line", "step", "process"]
    is_inventory: bool
    """True if the cause of failure was insufficient inventory. If False, the cause was insufficient funds. Note that
    if both conditions were true, only insufficient funds (is_inventory=False) will be reported."""
    line: int
    """The line at which the failure happened"""
    step: int
    """The step at which the failure happened"""
    process: int
    """The process that failed to execute (if `external_contract_failure` and `is_inventory` , then this will be the 
    process that would have generated the needed product. and if `external_contract_failure` and not `is_inventory` 
    , then it is not valid)"""


@dataclass
class FactoryState:
    inventory: np.ndarray
    """An n_products vector giving current quantity of every product in storage"""
    balance: int
    """Current balance in the wallet"""
    commands: np.ndarray
    """n_steps * n_lines array giving the process scheduled on each line at every step for the 
    whole simulation"""
    inventory_changes: np.ndarray
    """Changes in the inventory in the last step"""
    balance_change: int
    """Change in the balance in the last step"""
    contracts: List[List[ContractInfo]]
    """The An n_steps list of lists containing the contracts of this agent by time-step"""

    @property
    def n_lines(self) -> int:
        return self.commands.shape[1]

    @property
    def n_steps(self) -> int:
        return self.commands.shape[0]

    @property
    def n_products(self) -> int:
        return len(self.inventory)

    @property
    def n_processes(self) -> int:
        return len(self.inventory) - 1


class Factory:
    """A simulated factory"""

    def __init__(
        self,
        profile: FactoryProfile,
        initial_balance: int,
        inputs: np.ndarray,
        outputs: np.ndarray,
        catalog_prices: np.ndarray,
        world: "World",
        compensate_before_past_debt: bool,
        buy_missing_products: bool,
        external_penalty: float,
        external_no_bankruptcy: bool,
        external_no_borrow: bool,
        production_penalty: float,
        production_no_bankruptcy: bool,
        production_no_borrow: bool,
        agent_id: str,
        agent_name: Optional[str] = None,
    ):
        self.compensate_before_past_debt = compensate_before_past_debt
        self.buy_missing_products = buy_missing_products
        self.external_penalty = external_penalty
        self.external_no_bankruptcy = external_no_bankruptcy
        self.external_no_borrow = external_no_borrow
        self.production_penalty = production_penalty
        self.production_no_bankruptcy = production_no_bankruptcy
        self.production_no_borrow = production_no_borrow
        self.catalog_prices = catalog_prices
        self.__profile = profile
        self.world = world
        self.profile = copy.deepcopy(profile)
        """The readonly factory profile (See `FactoryProfile` )"""
        self.commands = NO_COMMAND * np.ones(
            (profile.n_steps, profile.n_lines), dtype=int
        )
        """An n_steps * n_lines array giving the process scheduled for each line at every step. -1 indicates an empty
        line. """
        # self.predicted_inventory = profile.external_quantities.copy()
        """An n_steps * n_products array giving the inventory content at different steps. For steps in the past and 
        present, this is the *actual* value of the inventory at that time. For steps in the future, this is a 
        *prediction* of the inventory at that step."""
        self._balance = initial_balance
        """Current balance"""
        self._inventory = np.zeros(profile.n_products, dtype=int)
        """Current inventory"""
        # self.predicted_balance = initial_balance - np.sum(
        #     profile.external_quantities * profile.external_prices, axis=-1
        # )
        """An n_steps vector giving the wallet balance at different steps. For steps in the past and 
        present, this is the *actual* value of the balance at that time. For steps in the future, this is a 
        *prediction* of the balance at that step."""
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

        Returns:

            Tuple[np.ndarray, np.ndarray] The steps and lines at which production is scheduled.

        Remarks:

            - You cannot order production in the past or in the current step
            - Ordering production, will automatically update inventory and balance for all simulation steps assuming
              that this production will be carried out. At the indicated `step` if production was not possible (due
              to insufficient funds or insufficient inventory of the input product), the predictions for the future
              will be corrected.

        """
        steps, lines = self.available_for_production(
            repeats, step, line, override, method
        )
        self.order_production(process, steps, lines)
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
                step = (current_step, self.profile.n_steps)
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
        if possible < 1:
            return np.empty(shape=0, dtype=int), np.empty(shape=0, dtype=int)
        if method.startswith("l"):
            steps, lines = steps[-possible + 1 :], lines[-possible + 1 :]
        elif method == "all":
            pass
        else:
            steps, lines = steps[:possible], lines[:possible]

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

    def step(
        self, accepted_sales: np.ndarray, accepted_supplies: np.ndarray
    ) -> List[Failure]:
        """
        Override this method to modify stepping logic.

        Args:
            accepted_sales: Sales per product accepted by the factory manager
            accepted_supplies: Supplies per product accepted by the factory manager

        Returns:

        """
        if self.is_bankrupt:
            return []
        step = self.world.current_step
        profile = self.__profile
        failures = []
        initial_balance = self._balance
        initial_inventory = self._inventory.copy()

        # buy guaranteed supplies as much as possible
        # if it is possible to pay for all the supplies, do that directly without checks
        #    otherwise do normal transactions to check for bankruptcy (either way we do not report breaches)
        if np.max(accepted_supplies) > 0:
            prices = profile.external_supply_prices[step, :] * accepted_supplies
            supply_money = np.sum(prices)
            if self._balance - supply_money >= self.min_balance:
                self._balance -= supply_money
                self._inventory += accepted_supplies
            else:
                for p, (q, u) in enumerate(
                    zip(
                        accepted_supplies.tolist(),
                        profile.external_supply_prices[step, :].tolist(),
                    )
                ):
                    self.buy(
                        p,
                        q,
                        u,
                        self.buy_missing_products,
                        self.external_penalty,
                        self.external_no_bankruptcy,
                        self.external_no_borrow,
                    )
                    if self.is_bankrupt:
                        break

            if self.is_bankrupt:
                return []

        # Sell guaranteed sales as much as possible
        if np.max(accepted_sales) > 0:
            in_inventory = (self._inventory - accepted_sales) >= 0

            if np.all(in_inventory):
                self._balance += np.sum(
                    accepted_sales * profile.external_sale_prices[step, :]
                )
                self._inventory -= accepted_sales
            else:
                for p, (q, u) in enumerate(
                    zip(
                        accepted_sales.tolist(),
                        profile.external_sale_prices[step, :].tolist(),
                    )
                ):
                    if q < 1:
                        continue
                    self.buy(
                        p,
                        -q,
                        u,
                        self.buy_missing_products,
                        self.external_penalty,
                        self.external_no_bankruptcy,
                        self.external_no_borrow,
                    )
                    if self.is_bankrupt:
                        break

            if self.is_bankrupt:
                return []

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
            self.store(
                inp,
                -ins,
                0,
                self.buy_missing_products,
                self.production_penalty,
                self.production_no_bankruptcy,
                self.production_no_borrow,
            )
            self.store(
                outp,
                outs,
                0,
                self.buy_missing_products,
                self.production_penalty,
                self.production_no_bankruptcy,
                self.production_no_borrow,
            )

        assert self._balance >= self.min_balance
        assert np.min(self._inventory) >= 0
        self.inventory_changes = self._inventory - initial_inventory
        self.balance_change = self._balance - initial_balance
        return failures

    def store(
        self,
        product: int,
        quantity: int,
        unit_price: int,
        buy_missing: bool,
        penalty: float,
        no_bankruptcy: bool = False,
        no_borrowing: bool = False,
    ) -> int:
        """
        Stores the given amount of product (signed) to the factory.

        Args:
            product: Product
            quantity: quantity to store/take out (-ve means take out)
            unit_price: Unit price
            buy_missing: If the quantity is negative and not enough product exists in the market, it buys the product
                         from the spot-market at an increased price of penalty
            penalty: The fraction of unit_price added because we are buying from the spot market. Only effectivec if
                     quantity is negative and not enough of the product exists in the inventory
            no_bankruptcy: Never bankrupt the agent on this transaction
            no_borrowing: Never borrow for this transaction

        Returns:
            The quantity actually stored or taken out (always positive)
        """
        if self.is_bankrupt:
            self.world.logwarning(
                f"{self.agent_name} received a transaction "
                f"(product: {product}, q: {quantity}, u:{unit_price}) after being bankrupt"
            )
            return 0
        available = self._inventory[product]
        if available + quantity >= 0:
            self._inventory[product] += quantity
            self.inventory_changes[product] += quantity
            return quantity if quantity > 0 else -quantity
        # we have an inventory breach here. We know that quantity < 0
        assert quantity < 0
        quantity = -quantity
        if not buy_missing:
            # if we are not buying from the spot market, pay the penalty for missing products and transfer all available
            to_pay = int(
                np.ceil(penalty * (quantity - available) / quantity)
                * max(self.catalog_prices[product], unit_price)
            )
            self.pay(to_pay, no_bankruptcy, no_borrowing)
            self._inventory[product] = 0
            self.inventory_changes[product] -= available
            return available
        # we have an inventory breach and should try to buy missing quantity from the spot market
        real_price = (quantity - available) * max(
            self.catalog_prices[product], unit_price
        )
        to_pay = int(np.ceil(real_price * (1 + penalty)))
        paid = int(self.pay(to_pay, no_bankruptcy, no_borrowing) / (1 + penalty))
        paid_for = paid // unit_price
        assert self._inventory[product] + paid_for >= 0, (
            f"Had {self._inventory[product]} and paid for {paid_for} ("
            f"original quantity {quantity})"
        )
        self._inventory[product] += paid_for
        self.inventory_changes[product] += paid_for
        return self.store(
            product, -quantity, unit_price, False, penalty, no_bankruptcy, no_borrowing
        )

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
                product,
                quantity,
                unit_price,
                buy_missing,
                penalty,
                no_bankruptcy,
                no_borrowing,
            )
            paid = self.pay(-taken * unit_price, no_bankruptcy, no_borrowing)
            return taken, paid
        # that is a buy contract
        paid = self.pay(quantity * unit_price, no_bankruptcy, no_borrowing)
        stored = self.store(
            product,
            quantity * paid // unit_price,
            unit_price,
            buy_missing,
            penalty,
            no_bankruptcy,
            no_borrowing,
        )
        return stored, paid

    def pay(
        self, money: int, no_bankruptcy: bool = False, no_borrowing: bool = False
    ) -> int:
        """
        Pays money

        Args:
            money: amount to pay
            no_bankruptcy: If true, this transaction can never lead to bankruptcy
            no_borrowing: If true, this transaction can never lead to borrowing

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
        total = int(np.sum(self._inventory * self.catalog_prices))
        pay_back = min(required, total)
        available = total - required

        # If past debt is paid before compensation pay it
        original_balance = self._balance
        if not self.compensate_before_past_debt:
            available += original_balance

        self.world.compensate(available, self)
        self.is_bankrupt = True
        return pay_back


class AWI(AgentWorldInterface):
    """The Agent World Interface for SCML2020 world allowing a single process per agent"""

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
        controller: SAOController,
        partners: List[str] = None,
        extra: Dict[str, Any] = None,
    ) -> bool:
        """
        Requests a negotiation

        Args:

            is_buy: If True the negotiation is about buying otherwise selling.
            product: The product to negotiate about
            quantity: The minimum and maximum quantities. Passing a single value q is equivalent to passing (q,q)
            unit_price: The minimum and maximum unit prices. Passing a single value u is equivalent to passing (u,u)
            time: The minimum and maximum delivery step. Passing a single value t is equivalent to passing (t,t)
            controller: The controller to manage the complete set of negotiations
            partners: ID of all the partners to negotiate with.
            extra: Extra information accessible through the negotiation annotation to the caller

        Returns:

            `True` if the partner accepted and the negotiation is ready to start

        Remarks:

            - All negotiations will use the following issues **in order**: quantity, time, unit_price
            - Negotiations with bankrupt agents or on invalid products (see next point) will be automatically rejected
            - Valid products for a factory are the following (any other products are not valid):
                1. Buying an input product (i.e. product $\in$ `my_input_products`
                1. Seeling an output product (i.e. product $\in$ `my_output_products`


        """
        if extra is None:
            extra = dict()
        if (product not in self.my_input_products and is_buy) or (
            product not in self.my_output_products and not is_buy
        ):
            self._world.logwarning(
                f"{self.agent.name} requested negotiation on {product} "
                f"({'buying' if is_buy else 'selling'}) but this is not in "
                f"its ({'inputs' if is_buy else 'outputs'})"
            )
            return False
        if partners is None:
            partners = (
                self.all_suppliers[product] if is_buy else self.all_consumers[product]
            )
        negotiators = [
            controller.create_negotiator(PassThroughSAONegotiator) for _ in partners
        ]
        results = [
            self.request_negotiation(
                is_buy, product, quantity, unit_price, time, partner, negotiator, extra
            )
            if not self._world.a2f[partner].is_bankrupt
            else False
            for partner, negotiator in zip(partners, negotiators)
        ]
        return any(results)

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
        """
        Requests a negotiation

        Args:

            is_buy: If True the negotiation is about buying otherwise selling.
            product: The product to negotiate about
            quantity: The minimum and maximum quantities. Passing a single value q is equivalent to passing (q,q)
            unit_price: The minimum and maximum unit prices. Passing a single value u is equivalent to passing (u,u)
            time: The minimum and maximum delivery step. Passing a single value t is equivalent to passing (t,t)
            partner: ID of the partner to negotiate with.
            negotiator: The negotiator to use for this negotiation (if the partner accepted to negotiate)
            extra: Extra information accessible through the negotiation annotation to the caller

        Returns:

            `True` if the partner accepted and the negotiation is ready to start

        Remarks:

            - All negotiations will use the following issues **in order**: quantity, time, unit_price
            - Negotiations with bankrupt agents or on invalid products (see next point) will be automatically rejected
            - Valid products for a factory are the following (any other products are not valid):
                1. Buying an input product (i.e. product $\in$ `my_input_products`
                1. Seeling an output product (i.e. product $\in$ `my_output_products`


        """
        if extra is None:
            extra = dict()
        if (product not in self.my_input_products and is_buy) or (
            product not in self.my_output_products and not is_buy
        ):
            self._world.logwarning(
                f"{self.agent.name} requested negotiation on {product} "
                f"({'buying' if is_buy else 'selleing'}) but this is not in "
                f"its ({'inputs' if is_buy else 'outputs'})"
            )
            return False
        if self._world.a2f[partner].is_bankrupt:
            return False

        def values(x: Union[int, Tuple[int, int]]):
            if not isinstance(x, Iterable):
                return int(x), int(x)
            return int(x[0]), int(x[1])

        self._world.logdebug(
            f"{self.agent.name} requested to {'buy' if is_buy else 'sell'} {product} to {partner}"
            f" q: {quantity}, u: {unit_price}, t: {time}"
        )

        annotation = {
            "product": product,
            "is_buy": is_buy,
            "buyer": self.agent.id if is_buy else partner,
            "seller": partner if is_buy else self.agent.id,
            "caller": self.agent.id,
        }
        issues = [
            Issue(values(quantity), name="quantity", value_type=int),
            Issue(values(time), name="time", value_type=int),
            Issue(values(unit_price), name="unit_price", value_type=int),
        ]
        partners = [self.agent.id, partner]
        extra["negotiator_id"] = negotiator.id
        req_id = self.agent.create_negotiation_request(
            issues=issues,
            partners=partners,
            negotiator=negotiator,
            annotation=annotation,
            extra=dict(**extra),
        )
        return self.request_negotiation_about(
            issues=issues, partners=partners, req_id=req_id, annotation=annotation
        )

    def schedule_production(
        self,
        process: int,
        repeats: int,
        step: Union[int, Tuple[int, int]] = ANY_STEP,
        line: int = ANY_LINE,
        override: bool = True,
        method: str = "latest",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Orders the factory to run the given process at the given line at the given step

        Args:

            process: The process to run
            repeats: How many times to repeat the process
            step: The simulation step or a range of steps. The special value ANY_STEP gives the factory the freedom to
                  schedule production at any step in the present or future.
            line: The production line. The special value ANY_LINE gives the factory the freedom to use any line
            override: Whether to override existing production commands or not
            method: When to schedule the command if step was set to a range. Options are latest, earliest

        Returns:
            Tuple[int, int] giving the steps and lines at which production is scheduled.

        Remarks:

            - The step cannot be in the past. Production can only be ordered for current and future steps
            - ordering production of process -1 is equivalent of `cancel_production` only if both step and line are
              given
        """
        return self._world.a2f[self.agent.id].schedule_production(
            process, repeats, step, line, override, method
        )

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
        return self._world.a2f[self.agent.id].order_production(process, steps, lines)

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
        return self._world.a2f[self.agent.id].available_for_production(
            repeats, step, line, override, method
        )

    def cancel_production(self, step: int, line: int) -> bool:
        """
        Cancels any production commands on that line at this step

        Args:
            step: The step to cancel production at (must be in the future).
            line: The production line

        Returns:

            success/failure

        Remarks:

            - The step cannot be in the past or the current step. Cancellation can only be ordered for future steps
        """
        return self._world.a2f[self.agent.id].cancel_production(step, line)

    # ---------------------
    # Information Gathering
    # ---------------------

    @property
    def state(self) -> FactoryState:
        """Receives the factory state"""
        return self._world.a2f[self.agent.id].state

    @property
    def profile(self) -> FactoryProfile:
        """Gets the profile (static private information) associated with the agent"""
        return self._world.a2f[self.agent.id].profile

    @property
    def all_suppliers(self) -> List[List[str]]:
        """Returns a list of agent IDs for all suppliers for every product"""
        return self._world.suppliers

    @property
    def all_consumers(self) -> List[List[str]]:
        """Returns a list of agent IDs for all consumers for every product"""
        return self._world.consumers

    @property
    def catalog_prices(self) -> np.ndarray:
        """Returns the catalog prices of all products"""
        return self._world.catalog_prices

    @property
    def inputs(self) -> np.ndarray:
        """Returns the number of inputs to every production process"""
        return self._world.process_inputs

    @property
    def outputs(self) -> np.ndarray:
        """Returns the number of outputs to every production process"""
        return self._world.process_outputs

    @property
    def n_products(self) -> int:
        """Returns the number of products in the system"""
        return len(self._world.catalog_prices)

    @property
    def n_processes(self) -> int:
        """Returns the number of processes in the system"""
        return self.n_products - 1

    @property
    def my_input_product(self) -> int:
        """Returns a list of products that are inputs to at least one process the agent can run"""
        return self._world.agent_inputs[self.agent.id][0]

    @property
    def my_output_product(self) -> int:
        """Returns a list of products that are outputs to at least one process the agent can run"""
        return self._world.agent_outputs[self.agent.id][0]

    @property
    def my_input_products(self) -> np.ndarray:
        """Returns a list of products that are inputs to at least one process the agent can run"""
        return self._world.agent_inputs[self.agent.id]

    @property
    def my_output_products(self) -> np.ndarray:
        """Returns a list of products that are outputs to at least one process the agent can run"""
        return self._world.agent_outputs[self.agent.id]

    @property
    def my_suppliers(self) -> List[str]:
        """Returns a list of IDs for all of the agent's suppliers (agents that can supply at least one product it may
        need).

        Remarks:

            - If the agent have multiple input products, suppliers of a specific product $p$ can be found using:
              **self.all_suppliers[p]**.
        """
        return self._world.agent_suppliers[self.agent.id]

    @property
    def my_consumers(self) -> List[str]:
        """Returns a list of IDs for all the agent's consumers (agents that can consume at least one product it may
        produce).

        Remarks:

            - If the agent have multiple output products, consumers of a specific product $p$ can be found using:
              **self.all_consumers[p]**.
        """
        return self._world.agent_consumers[self.agent.id]

    @property
    def n_lines(self) -> int:
        """The number of lines in the corresponding factory. You can read `state` to get this among other information"""
        return self.state.n_lines

    @property
    def n_products(self) -> int:
        """Number of products in the world"""
        return self.state.n_products

    @property
    def n_processes(self) -> int:
        """Number of processes in the world"""
        return self.state.n_processes


class SCML2020Agent(Agent):
    """Base class for all SCML2020 agents (factory managers)"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _respond_to_negotiation_request(
        self,
        initiator: str,
        partners: List[str],
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
        role: Optional[str],
        req_id: Optional[str],
    ) -> Optional[Negotiator]:
        return self.respond_to_negotiation_request(
            initiator, issues, annotation, mechanism
        )

    def set_renegotiation_agenda(
        self, contract: Contract, breaches: List[Breach]
    ) -> Optional[RenegotiationRequest]:
        return None

    def respond_to_renegotiation_request(
        self, contract: Contract, breaches: List[Breach], agenda: RenegotiationRequest
    ) -> Optional[Negotiator]:
        return None

    def on_neg_request_rejected(self, req_id: str, by: Optional[List[str]]):
        pass

    def on_neg_request_accepted(self, req_id: str, mechanism: AgentMechanismInterface):
        pass

    @abstractmethod
    def on_contract_nullified(
        self, contract: Contract, compensation_money: int, compensation_fraction: float
    ) -> None:
        """
        Called whenever a contract is nullified (because the partner is bankrupt)

        Args:

            contract: The contract being nullified
            compensation_money: The compensation money that is already added to the agent's wallet
            compensation_fraction: The fraction of the contract's total to be compensated. The rest is lost.

        """

    @abstractmethod
    def on_failures(self, failures: List[Failure]) -> None:
        """
        Called whenever there are failures either in production or in execution of guaranteed transactions

        Args:

            failures: A list of `Failure` s.
        """

    @abstractmethod
    def confirm_external_sales(
        self, quantities: np.ndarray, unit_prices: np.ndarray
    ) -> np.ndarray:
        """
        Called to confirm the amount of guaranteed sales the agent is willing to accept

        Args:

            quantities: An n_products vector giving the maximum quantity that can sold (without negotiation)
            unit_prices: An n_products vector giving the guaranteed unit prices

        Returns:

            An n_products vector specifying the quantities to be sold (up to the given `quantities` limit).
        """

    @abstractmethod
    def confirm_external_supplies(
        self, quantities: np.ndarray, unit_prices: np.ndarray
    ) -> np.ndarray:
        """
        Called to confirm the amount of guaranteed supplies the agent is willing to accept

        Args:

            quantities: An n_products vector giving the maximum quantity that can bought (without negotiation)
            unit_prices: An n_products vector giving the guaranteed unit prices

        Returns:

            An n_products vector specifying the quantities to be bought (up to the given `quantities` limit).
        """

    @abstractmethod
    def respond_to_negotiation_request(
        self,
        initiator: str,
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
    ) -> Optional[Negotiator]:
        """
        Called whenever another agent requests a negotiation with this agent.

        Args:
            initiator: The ID of the agent that requested this negotiation
            issues: Negotiation issues
            annotation: Annotation attached with this negotiation
            mechanism: The `AgentMechanismInterface` interface to the mechanism to be used for this negotiation.

        Returns:
            None to reject the negotiation, otherwise a negotiator
        """


class World(TimeInAgreementMixin, World):
    """A Supply Chain World Simulation as described for the SCML league of ANAC @ IJCAI 2020.

        Args:

            process_inputs: An n_processes vector specifying the number of inputs from each product needed to execute
                            each process.
            process_outputs: An n_processes vector specifying the number of inputs from each product generated by
                            executing each process.
            catalog_prices: An n_products vector (i.e. n_processes+1 vector) giving the catalog price of all products
            profiles: An n_agents list of `FactoryProfile` objects specifying the private profile of the factory
                      associated with each agent.
            agent_types: An n_agents list of strings/ `SCML2020Agent` classes specifying the type of each agent
            agent_params: An n_agents dictionaries giving the parameters of each agent
            initial_balance: The initial balance in each agent's wallet. All agents will start with this same value.
            breach_penalty: The total penalty paid upon a breach will be calculated as (breach_level * breach_penalty *
                            contract_quantity * contract_unit_price).
            external_supply_limit: An n_steps * n_products array giving the total supply available of each product over time.
                          Only affects guaranteed supply.
            external_sales_limit: An n_steps * n_products array giving the total sales to happen for each product over time.
                         Only affects guaranteed sales.
            financial_report_period: The number of steps between financial reports. If < 1, it is a fraction of n_steps
            borrow_on_breach: If true, agents will be forced to borrow money on breach as much as possible to honor the
                              contract
            interest_rate: The interest at which loans grow over time (it only affect a factory when its balance is
                           negative)
            bankruptcy_limit: The maximum amount that be be borrowed (including interest). The balance of any factory cannot
                          go lower than - borrow_limit or the agent will go bankrupt immediately
            compensation_fraction: Fraction of a contract to be compensated (at most) if a partner goes bankrupt. Notice
                                   that this fraction is not guaranteed because the bankrupt agent may not have enough
                                   assets to pay all of its standing contracts to this level of compensation. In such
                                   cases, a smaller fraction will be used.
            compensate_immediately: If true, compensation will happen immediately when an agent goes bankrupt and in
                                    in money. This means that agents with contracts involving the bankrupt agent will
                                    just have these contracts be nullified and receive monetary compensation immediately
                                    . If false, compensation will not happen immediately but at the contract execution
                                    time. In this case, agents with contracts involving the bankrupt agent will be
                                    informed of the compensation fraction (instead of the compensation money) at the
                                    time of bankruptcy and will receive the compensation in kind (money if they are
                                    sellers and products if they are buyers) at the normal execution time of the
                                    contract. In the special case of no-compensation (i.e. `compensation_fraction` is
                                    zero or the bankrupt agent has no assets), the two options will behave similarity.
            compensate_before_past_debt: If true, then compensations will be paid before past debt is considered,
                                         otherwise, the money from liquidating bankrupt agents will first be used to
                                         pay past debt then whatever remains will be used for compensation. Notice that
                                         in all cases, the trigger of bankruptcy will be paid before compensation and
                                         past debts.
            external_force_max: If true, agents are not asked to confirm guaranteed transactions and they
                                are carried out up to bankruptcy
            external_no_borrow: If true, agents will not borrow if they fail to satisfy an external transaction. The
                                transaction will just fail silently
            external_no_bankruptcy: If true, agents will not go bankrupt because of an external transaction. The
                                    transaction will just fail silently
            external_penalty: The penalty paid for failure to honor external contracts
            production_no_borrow: If true, agents will not borrow if they fail to satisfy its production need to execute
                                  a scheduled production command
            production_no_bankruptcy: If true, agents will not go bankrupt because of an production related transaction.
            production_penalty: The penalty paid when buying from spotmarket to satisfy production needs
            compact: If True, no logs will be kept and the whole simulation will use a smaller memory footprint
            n_steps: Number of simulation steps (can be considered as days).
            time_limit: Total time allowed for the complete simulation in seconds.
            neg_n_steps: Number of negotiation steps allowed for all negotiations.
            neg_time_limit: Total time allowed for a complete negotiation in seconds.
            neg_step_time_limit: Total time allowed for a single step of a negotiation. in seconds.
            negotiation_speed: The number of negotiation steps that pass in every simulation step. If 0, negotiations
                               will be guaranteed to finish within a single simulation step
            signing_delay: The number of simulation steps to pass between a contract is concluded and signed
            name: The name of the simulations
            **kwargs: Other parameters that are passed directly to `World` constructor.

    """

    def __init__(
        self,
        # SCML2020 specific parameters
        process_inputs: np.ndarray,
        process_outputs: np.ndarray,
        catalog_prices: np.ndarray,
        profiles: List[FactoryProfile],
        agent_types: List[Type[SCML2020Agent]],
        agent_params: List[Dict[str, Any]] = None,
        initial_balance: int = 1000,
        # breach processing parameters
        buy_missing_products=True,
        borrow_on_breach=True,
        bankruptcy_limit=0.2,
        breach_penalty=0.15,
        financial_report_period=5,
        interest_rate=0.05,
        # compensation parameters (for victims of bankrupt agents)
        compensation_fraction=1.0,
        compensate_immediately=False,
        compensate_before_past_debt=False,
        # external contracts parameters
        external_force_max=True,
        external_no_borrow=False,
        external_no_bankruptcy=False,
        external_penalty=0.15,
        external_supply_limit: np.ndarray = None,
        external_sales_limit: np.ndarray = None,
        # production failure parameters
        production_no_borrow=False,
        production_no_bankruptcy=False,
        production_penalty=0.15,
        # General World Parameters
        compact=False,
        no_logs=False,
        n_steps=1000,
        time_limit=60 * 90,
        # mechanism params
        neg_n_steps=20,
        neg_time_limit=2 * 60,
        neg_step_time_limit=60,
        negotiation_speed=21,
        # simulation parameters
        signing_delay=1,
        name: str = None,
        **kwargs,
    ):
        self.buy_missing_products = buy_missing_products
        kwargs["log_to_file"] = not no_logs
        if compact:
            kwargs["log_screen_level"] = logging.CRITICAL
            kwargs["log_file_level"] = logging.ERROR
            kwargs["log_negotiations"] = False
            kwargs["log_ufuns"] = False
            # kwargs["save_mechanism_state_in_contract"] = False
            kwargs["save_cancelled_contracts"] = False
            kwargs["save_resolved_breaches"] = False
            kwargs["save_negotiations"] = False
        self.compact = compact
        if negotiation_speed == 0:
            negotiation_speed = neg_n_steps + 1
        super().__init__(
            bulletin_board=None,
            breach_processing=BreachProcessing.NONE,
            awi_type="scml.scml2020.AWI",
            start_negotiations_immediately=False,
            mechanisms={"negmas.sao.SAOMechanism": {}},
            default_signing_delay=signing_delay,
            n_steps=n_steps,
            time_limit=time_limit,
            negotiation_speed=negotiation_speed,
            neg_n_steps=neg_n_steps,
            neg_time_limit=neg_time_limit,
            neg_step_time_limit=neg_step_time_limit,
            name=name,
            **kwargs,
        )
        TimeInAgreementMixin.init(self, time_field="time")
        self.breach_penalty = breach_penalty
        self.bulletin_board.add_section("reports_time")
        self.bulletin_board.add_section("reports_agent")
        self.external_no_borrow = external_no_borrow
        self.external_no_bankruptcy = external_no_bankruptcy
        self.external_penalty = external_penalty
        self.production_no_borrow = production_no_borrow
        self.production_no_bankruptcy = production_no_bankruptcy
        self.production_penalty = production_penalty
        self.compensation_fraction = compensation_fraction
        if not isinstance(agent_types, Iterable):
            agent_types = [agent_types] * len(profiles)

        assert len(profiles) == len(agent_types)
        self.profiles = profiles
        self.catalog_prices = catalog_prices
        self.process_inputs = process_inputs
        self.process_outputs = process_outputs
        self.n_products = len(catalog_prices)
        self.n_processes = len(process_inputs)
        self.borrow_on_breach = borrow_on_breach
        self.interest_rate = interest_rate
        self.external_force_max = external_force_max
        self.compensate_before_past_debt = compensate_before_past_debt
        self.financial_reports_period = (
            financial_report_period
            if financial_report_period >= 1
            else int(0.5 + financial_report_period * n_steps)
        )
        self.compensation_fraction = compensation_fraction
        self.compensate_immediately = compensate_immediately
        self.bankruptcy_limit = (
            -bankruptcy_limit
            if isinstance(bankruptcy_limit, int)
            else -int(0.5 + bankruptcy_limit * initial_balance)
        )
        assert self.n_products == self.n_processes + 1

        if external_supply_limit is None:
            self.supply_limit = sys.maxsize * np.ones(
                (n_steps, self.n_products), dtype=int
            )
        else:
            self.supply_limit = external_supply_limit
        if external_sales_limit is None:
            self.sales_limit = sys.maxsize * np.ones(
                (n_steps, self.n_products), dtype=int
            )
        else:
            self.sales_limit = external_sales_limit

        if agent_params is None:
            agent_params = [
                dict(name=f"{_.__name__[:3]}{i:03}") for i, _ in enumerate(agent_types)
            ]
        agents = []
        for i, (atype, aparams) in enumerate(zip(agent_types, agent_params)):
            a = instantiate(atype, **aparams)
            self.join(a, i)
            agents.append(a)
        n_agents = len(agents)

        self.factories = [
            Factory(
                world=self,
                profile=profile,
                initial_balance=initial_balance,
                inputs=process_inputs,
                outputs=process_outputs,
                agent_id=agents[i].id,
                catalog_prices=catalog_prices,
                compensate_before_past_debt=self.compensate_before_past_debt,
                buy_missing_products=self.buy_missing_products,
                external_penalty=self.external_penalty,
                external_no_bankruptcy=self.external_no_bankruptcy,
                external_no_borrow=self.external_no_borrow,
                production_penalty=self.production_penalty,
                production_no_borrow=self.production_no_borrow,
                production_no_bankruptcy=self.production_no_bankruptcy,
            )
            for i, profile in enumerate(profiles)
        ]
        self.a2f = dict(zip((_.id for _ in agents), self.factories))
        self.f2a = dict(zip((_.agent_id for _ in self.factories), agents))
        self.afp = list(zip(agents, self.factories, profiles))
        self.a2i = dict(zip((_.id for _ in agents), range(n_agents)))
        self.i2a = agents
        self.f2i = dict(zip((_.agent_id for _ in self.factories), range(n_agents)))
        self.i2f = self.factories

        self.breach_prob = dict(zip((_.id for _ in agents), itertools.repeat(0.0)))
        self.breach_level = dict(zip((_.id for _ in agents), itertools.repeat(0.0)))
        self.agent_n_contracts = dict(zip((_.id for _ in agents), itertools.repeat(0)))

        n_processes = len(process_inputs)
        n_products = n_processes + 1

        self.suppliers: List[List[str]] = [[] for _ in range(n_products)]
        self.consumers: List[List[str]] = [[] for _ in range(n_products)]
        self.agent_processes: Dict[str, List[int]] = defaultdict(list)
        self.agent_inputs: Dict[str, List[int]] = defaultdict(list)
        self.agent_outputs: Dict[str, List[int]] = defaultdict(list)
        self.agent_consumers: Dict[str, List[str]] = defaultdict(list)
        self.agent_suppliers: Dict[str, List[str]] = defaultdict(list)

        for p in range(n_processes):
            for agent_id, profile in zip(self.agents.keys(), profiles):
                if np.all(profile.costs[:, p] == INFINITE_COST):
                    continue
                self.suppliers[p + 1].append(agent_id)
                self.consumers[p].append(agent_id)
                self.agent_processes[agent_id].append(p)
                self.agent_inputs[agent_id].append(p)
                self.agent_outputs[agent_id].append(p + 1)

        for p in range(n_products):
            for a in self.suppliers[p]:
                self.agent_consumers[a] = self.consumers[p]
            for a in self.consumers[p]:
                self.agent_suppliers[a] = self.suppliers[p]

        self.agent_processes = {k: np.array(v) for k, v in self.agent_processes.items()}
        self.agent_inputs = {k: np.array(v) for k, v in self.agent_inputs.items()}
        self.agent_outputs = {k: np.array(v) for k, v in self.agent_outputs.items()}

        self._n_production_failures = 0
        self.__n_nullified = 0
        self.__n_bankrupt = 0
        self.penalties = 0
        # self.is_bankrupt: Dict[str, bool] = dict(
        #     zip(self.agents.keys(), itertools.repeat(False))
        # )
        self.compensation_balance = 0
        self.compensation_records: Dict[str, List[CompensationRecord]] = defaultdict(
            list
        )
        self.compensation_factory = Factory(
            FactoryProfile(
                np.zeros((n_steps, n_processes), dtype=int),
                np.zeros((n_steps, n_products), dtype=int),
                np.zeros((n_steps, n_products), dtype=int),
                np.zeros((n_steps, n_products), dtype=int),
                np.zeros((n_steps, n_products), dtype=int),
            ),
            initial_balance=0,
            inputs=self.process_inputs,
            outputs=self.process_outputs,
            world=self,
            agent_id="SYSTEM",
            agent_name="SYSTEM",
            catalog_prices=catalog_prices,
            compensate_before_past_debt=self.compensate_before_past_debt,
            buy_missing_products=self.buy_missing_products,
            external_penalty=self.external_penalty,
            external_no_bankruptcy=self.external_no_bankruptcy,
            external_no_borrow=self.external_no_borrow,
            production_penalty=self.production_penalty,
            production_no_borrow=self.production_no_borrow,
            production_no_bankruptcy=self.production_no_bankruptcy,
        )

    @classmethod
    def generate(
        cls,
        agent_types: List[Type[SCML2020Agent]],
        agent_params: List[Dict[str, Any]] = None,
        depth=4,
        process_inputs: Union[np.ndarray, Tuple[int, int]] = (1, 4),
        process_outputs: Union[np.ndarray, Tuple[int, int]] = (1, 1),
        profit_mean=0.15,
        profit_stddev=0.05,
        initial_balance: int = 1000,
        breach_penalty=0.15,
        n_steps: int = 100,
        n_lines: int = 10,
        n_agents_per_process: int = 1,
        agent_name_reveals_location=False,
        signing_delay: int = 1,
    ):
        n_products = depth + 1
        n_processes = depth
        if not isinstance(agent_types, Iterable):
            agent_types = [agent_types] * (n_agents_per_process * n_processes)
        n_agents = len(agent_types)
        assert n_agents >= n_processes
        assert n_agents_per_process <= n_agents
        if isinstance(process_inputs, np.ndarray):
            process_inputs = process_inputs.flatten()
            assert (
                len(process_inputs) == depth
            ), f"depth is {depth} but got {len(process_inputs)} inputs"
            inputs = process_inputs
        else:
            inputs = np.random.randint(*process_inputs, size=depth)

        if isinstance(process_outputs, np.ndarray):
            process_outputs = process_outputs.flatten()
            assert (
                len(process_outputs) == depth
            ), f"depth is {depth} but got {len(process_outputs)} outputs"
            outputs = process_outputs
        else:
            outputs = np.random.randint(*process_outputs, size=depth)

        min_cost, max_cost = 1, 10
        min_unit, max_unit = 1, 10
        if n_agents_per_process < 1:
            costs = np.random.randint(
                min_cost, max_cost, (n_agents, n_lines, n_processes), dtype=float
            )
            n_agents_per_process = n_agents
        elif n_agents >= n_agents_per_process * n_processes:
            costs = float("inf") * np.ones(
                (n_agents, n_lines, n_processes), dtype=float
            )
            for p in range(n_processes):
                costs[
                    p * n_agents_per_process : (p + 1) * n_agents_per_process, :, p
                ] = np.random.randint(min_cost, max_cost, size=n_agents_per_process)
            if not agent_name_reveals_location:
                costs = np.random.permutation(costs)
        else:
            costs = INFINITE_COST * np.ones((n_agents, n_lines, n_processes), dtype=int)
            for p in range(n_processes):
                # guarantee that at least one agent is running each process (we know that n_agents>=n_processes)
                costs[p, :, p] = random.randrange(min_cost, max_cost)
                # generate random new agents for the process
                agents = np.random.randint(
                    0, n_agents - 2, size=n_agents_per_process - 1
                )
                # make sure that the agent with the same index as the process is not chosen twice
                agents[agents >= p] += 1
                # assign agents to processes
                costs[agents, :, p] = np.random.randint(
                    min_cost, max_cost, size=(n_agents_per_process, 1)
                )
            # permute costs so that agent i is not really guaranteed to run process i
            if not agent_name_reveals_location:
                costs = np.random.permutation(costs)

        # generate guaranteed contracts
        # -----------------------------
        # select the amount of random raw material available to every agent that can consume it
        quantity = np.zeros((n_products, n_agents, n_steps), dtype=int)
        a0 = np.nonzero(~np.isinf(costs[:, 0, 0].flatten()))[0].flatten()
        quantity[0, a0, :-1] = np.random.randint(
            0, process_inputs[0] * n_lines, size=(1, len(a0), n_steps - 1)
        )
        # initialize unit prices randomly within the allowed negotiation range
        price = np.random.randint(min_unit, max_unit, size=quantity.shape)
        for p in range(1, n_processes):
            # find the agents / lines that can run this process
            ins = quantity[p - 1, :, :-1]

    def generate_initial_contracts(
        self
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pass

    def get_private_state(self, agent: "SCML2020Agent") -> dict:
        return vars(self.a2f[agent.id].state)

    def add_financial_report(
        self, agent: SCML2020Agent, factory: Factory, reports_agent, reports_time
    ) -> None:
        """
        Records a financial report for the given agent in the agent indexed reports and time indexed reports

        Args:
            agent: The agent
            factory: Its factory
            reports_agent: A dictionary of financial reports indexed by agent id
            reports_time: A dictionary of financial reports indexed by time

        Returns:

        """
        bankrupt = factory.is_bankrupt
        inventory = (
            int(np.sum(self.catalog_prices * factory.current_inventory)) if not bankrupt else 0
        )
        report = FinancialReport(
            agent_id=agent.id,
            step=self.current_step,
            cash=factory.current_balance,
            assets=inventory,
            breach_prob=self.breach_prob[agent.id],
            breach_level=self.breach_level[agent.id],
            is_bankrupt=bankrupt,
            agent_name=agent.name,
        )
        repstr = str(report).replace("\n", " ")
        self.logdebug(f"{agent.name}: {repstr}")
        if reports_agent.get(agent.id, None) is None:
            reports_agent[agent.id] = {}
        reports_agent[agent.id][self.current_step] = report
        if reports_time.get(self.current_step, None) is None:
            reports_time[self.current_step] = {}
        reports_time[self.current_step][agent.id] = report

    def simulation_step(self):
        s = self.current_step

        # pay interests for negative balances
        # -----------------------------------
        if self.interest_rate > 0.0:
            for agent, factory, _ in self.afp:
                if factory.current_balance < 0 and not factory.is_bankrupt:
                    to_pay = -int(math.ceil(self.interest_rate * factory.current_balance))
                    factory.pay(to_pay)

        # publish financial reports
        # -------------------------
        if self.current_step % self.financial_reports_period == 0:
            reports_agent = self.bulletin_board.data["reports_agent"]
            reports_time = self.bulletin_board.data["reports_time"]
            for agent, factory, _ in self.afp:
                self.add_financial_report(agent, factory, reports_agent, reports_time)

        # do external transactions and step factories
        # -------------------------------------------
        if self.external_force_max:
            for a, f, p in self.afp:
                if f.is_bankrupt:
                    continue
                f.step(p.external_sales[s, :], p.external_supplies[s, :])
        else:
            afp_randomized = [
                self.afp[_] for _ in np.random.permutation(np.arange(len(self.afp)))
            ]
            for a, f, p in afp_randomized:
                if f.is_bankrupt:
                    continue
                supply = a.confirm_external_supplies(
                    p.external_supplies[s].copy(), p.external_supply_prices[s].copy()
                )
                sales = a.confirm_external_sales(
                    p.external_sales[s].copy(), p.external_sale_prices[s].copy()
                )
                f.step(sales, supply)

    def contract_size(self, contract: Contract) -> float:
        return contract.agreement["quantity"] * contract.agreement["unit_price"]

    def contract_record(self, contract: Contract) -> Dict[str, Any]:
        c = {
            "id": contract.id,
            "seller_name": self.agents[contract.annotation["seller"]].name,
            "buyer_name": self.agents[contract.annotation["buyer"]].name,
            "seller_type": self.agents[
                contract.annotation["seller"]
            ].__class__.__name__,
            "buyer_type": self.agents[contract.annotation["buyer"]].__class__.__name__,
            "delivery_time": contract.agreement["time"],
            "quantity": contract.agreement["quantity"],
            "unit_price": contract.agreement["unit_price"],
            "signed_at": contract.signed_at if contract.signed_at is not None else -1,
            "nullified_at": contract.nullified_at
            if contract.nullified_at is not None
            else -1,
            "concluded_at": contract.concluded_at,
            "signatures": "|".join(str(_) for _ in contract.signatures),
            "issues": contract.issues if not self.compact else None,
            "seller": contract.annotation["seller"],
            "buyer": contract.annotation["buyer"],
        }
        if not self.compact:
            c.update(contract.annotation)
        c["n_neg_steps"] = contract.mechanism_state.step
        return c

    def breach_record(self, breach: Breach) -> Dict[str, Any]:
        return {
            "perpetrator": breach.perpetrator,
            "perpetrator_name": breach.perpetrator,
            "level": breach.level,
            "type": breach.type,
            "time": breach.step,
        }

    def execute_action(
        self, action: Action, agent: "SCML2020Agent", callback: Callable = None
    ) -> bool:
        if action.type == "schedule":
            s, _ = self.a2f[agent.id].schedule_production(
                process=action.params["process"],
                step=action.params.get("step", (self.current_step, self.n_steps - 1)),
                line=action.params.get("line", -1),
                override=action.params.get("override", True),
                method=action.params.get("method", "latest"),
            )
            return s >= 0
        elif action.type == "cancel":
            return self.a2f[agent.id].cancel_production(
                step=action.params.get("step", -1), line=action.params.get("line", -1)
            )

    def post_step_stats(self):
        self._stats["n_contracts_nullified"].append(self.__n_nullified)
        self._stats["n_bankrupt"].append(self.__n_bankrupt)
        market_size = 0
        self._stats[f"_balance_society"].append(self.penalties)
        internal_market_size = self.penalties
        prod = []
        for a, f, _ in self.afp:
            self._stats[f"balance_{a.name}"].append(f.current_balance)
            for p in a.awi.my_input_products:
                self._stats[f"inventory_{a.name}_input_{p}"].append(
                    f.current_inventory[p]
                )
            for p in a.awi.my_output_products:
                self._stats[f"inventory_{a.name}_output_{p}"].append(
                    f.current_inventory[p]
                )
            prod.append(np.mean(f.commands[self.current_step, :] != NO_COMMAND))
            self._stats[f"productivity_{a.name}"].append(prod[-1])
            self._stats[f"assets_{a.name}"].append(
                np.sum(f.current_inventory * self.catalog_prices)
            )
            self._stats[f"bankrupt_{a.name}"].append(f.is_bankrupt)
            if not f.is_bankrupt:
                market_size += f.current_balance
        self._stats["productivity"].append(float(np.mean(prod)))
        self._stats["market_size"].append(market_size)
        self._stats["production_failures"].append(
            self._n_production_failures / len(self.factories)
            if len(self.factories) > 0
            else np.nan
        )
        self._stats["_market_size_total"].append(market_size + internal_market_size)

    def pre_step_stats(self):
        self._n_production_failures = 0
        self.__n_nullified = 0
        self.__n_bankrupt = 0

    @property
    def business_size(self) -> float:
        """The total business size defined as the total money transferred within the system"""
        return sum(self.stats["activity_level"])

    @property
    def agreement_rate(self) -> float:
        """Fraction of negotiations ending in agreement and leading to signed contracts"""
        n_negs = sum(self.stats["n_negotiations"])
        n_contracts = len(self._saved_contracts)
        return n_contracts / n_negs if n_negs != 0 else np.nan

    @property
    def cancellation_rate(self) -> float:
        """Fraction of negotiations ending in agreement and leading to signed contracts"""
        n_negs = sum(self.stats["n_negotiations"])
        n_contracts = len(self._saved_contracts)
        n_signed_contracts = len(
            [_ for _ in self._saved_contracts.values() if _["signed"]]
        )
        return (1.0 - n_signed_contracts / n_contracts) if n_contracts != 0 else np.nan

    @property
    def n_negotiation_rounds_successful(self) -> float:
        """Average number of rounds in a successful negotiation"""
        n_negs = sum(self.stats["n_contracts_concluded"])
        if n_negs == 0:
            return np.nan
        return sum(self.stats["n_negotiation_rounds_successful"]) / n_negs

    @property
    def n_negotiation_rounds_failed(self) -> float:
        """Average number of rounds in a successful negotiation"""
        n_negs = sum(self.stats["n_negotiations"]) - sum(
            self.stats["n_contracts_concluded"]
        )
        if n_negs == 0:
            return np.nan
        return sum(self.stats["n_negotiation_rounds_failed"]) / n_negs

    @property
    def contract_execution_fraction(self) -> float:
        """Fraction of signed contracts successfully executed"""
        n_executed = sum(self.stats["n_contracts_executed"])
        n_signed_contracts = len(
            [_ for _ in self._saved_contracts.values() if _["signed"]]
        )
        return n_executed / n_signed_contracts if n_signed_contracts > 0 else np.nan

    @property
    def breach_rate(self) -> float:
        """Fraction of signed contracts that led to breaches"""
        n_breaches = sum(self.stats["n_breaches"])
        n_signed_contracts = len(
            [_ for _ in self._saved_contracts.values() if _["signed"]]
        )
        if n_signed_contracts != 0:
            return n_breaches / n_signed_contracts
        return np.nan

    def order_contracts_for_execution(
        self, contracts: Collection[Contract]
    ) -> Collection[Contract]:
        return sorted(contracts, key=lambda x: x.annotation["product"])

    def _execute(
        self,
        product: int,
        q: int,
        p: int,
        u: int,
        buyer_factory: Factory,
        seller_factory: Factory,
        has_breaches: bool,
    ):
        """Executes the contract"""
        self.logdebug(
            f"Transferring {q} of {product} at price {u} ({'breached' if has_breaches else ''})"
        )
        if q == 0 or u == 0:
            self.logwarning(f"{buyer_factory.agent_name} bought {q} from {seller_factory.agent_name} at {u} dollars"
                            f" ({'with breaches' if has_breaches else 'no breaches'})!! Zero quantity or unit price")
        if has_breaches:
            money = (
                p
                if buyer_factory.current_balance - p > self.bankruptcy_limit
                else max(0, buyer_factory.current_balance - self.bankruptcy_limit)
            )
            quantity = min(seller_factory.current_inventory[product], q)
            if quantity == 0 or money == 0:
                return
            u, q = min(money // quantity, u), min(quantity, money // u)
        assert q >= 0, f"executing with quantity {q}"
        if q != 0:
            buyer_factory.buy(product, q, u, False, 0.0)
            seller_factory.buy(product, -q, u, False, 0.0)

    def __register_contract(self, agent_id: str, level: float) -> None:
        """Registers execution of the contract in the agent's stats"""
        n_contracts = self.agent_n_contracts[agent_id] - 1
        self.breach_prob[agent_id] = (
            self.breach_prob[agent_id] * n_contracts + (level > 0)
        ) / (n_contracts + 1)
        self.breach_level[agent_id] = (
            self.breach_prob[agent_id] * n_contracts + level
        ) / (n_contracts + 1)

    def record_bankrupt(self, factory: Factory) -> None:
        """Records agent bankruptcy"""

        agent_id = factory.agent_id

        # announce bankruptcy
        reports_agent = self.bulletin_board.data["reports_agent"]
        reports_time = self.bulletin_board.data["reports_time"]
        self.add_financial_report(
            self.agents[agent_id], factory, reports_agent, reports_time
        )
        self.__n_bankrupt += 1

    def on_contract_concluded(self, contract: Contract, to_be_signed_at: int) -> None:
        if (
            any(self.a2f[_].is_bankrupt for _ in contract.partners)
            or contract.agreement["time"] >= self.n_steps
        ):
            return
        super().on_contract_concluded(contract, to_be_signed_at)

    def on_contract_signed(self, contract: Contract):
        if (
            any(self.a2f[_].is_bankrupt for _ in contract.partners)
            or contract.agreement["time"] >= self.n_steps
        ):
            return
        super().on_contract_signed(contract)
        self.logdebug(f"SIGNED {str(contract)}")
        t = contract.agreement["time"]
        u, q = contract.agreement["unit_price"], contract.agreement["quantity"]
        product = contract.annotation["product"]
        agent, partner = contract.partners
        is_seller = agent == contract.annotation["seller"]
        self.a2f[agent].contracts[t].append(
            ContractInfo(q, u, product, is_seller, partner, contract)
        )
        self.a2f[partner].contracts[t].append(
            ContractInfo(q, u, product, not is_seller, agent, contract)
        )

    def nullify_contract(self, contract: Contract):
        self.__n_nullified += 1
        contract.nullified_at = self.current_step

    def __register_breach(
        self, agent_id: str, level: float, contract_total: float, factory: Factory
    ) -> int:
        """
        Registers a breach of the given level on the given agent. Assume that the contract is already added
        to the agent_contracts

        Args:
            agent_id: The perpetrator of the breach
            level: The breach level
            contract_total: The total of the contract breached (quantity * unit_price)
            factory: The factory corresponding to the perpetrator

        Returns:
            If nonzero, the agent should go bankrupt and this amount taken from them
        """
        self.logdebug(
            f"{self.agents[agent_id].name} breached {level} of {contract_total}"
        )
        if factory.is_bankrupt:
            return 0
        if level <= 0:
            return 0
        penalty = int(math.ceil(level * contract_total))
        if factory.current_balance - penalty < self.bankruptcy_limit:
            return penalty
        if penalty > 0:
            factory.pay(penalty)
            self.penalties += penalty
        return 0

    def start_contract_execution(self, contract: Contract) -> Set[Breach]:
        self.logdebug(f"Executing {str(contract)}")
        # get contract info
        breaches = set()
        if self.compensate_immediately and any(self.a2f[a].is_bankrupt for a in contract.partners):
            return breaches
        product = contract.annotation["product"]
        buyer_id, seller_id = (
            contract.annotation["buyer"],
            contract.annotation["seller"],
        )
        buyer, buyer_factory = self.agents[buyer_id], self.a2f[buyer_id]
        seller, seller_factory = self.agents[seller_id], self.a2f[seller_id]
        q, u, t = (
            contract.agreement["quantity"],
            contract.agreement["unit_price"],
            contract.agreement["time"],
        )
        if q <= 0 or u <= 0:
            self.logwarning(
                f"Contract {str(contract)} has zero quantity of unit price!!! will be ignored"
            )
            return breaches
        p = q * u
        assert t == self.current_step
        self.agent_n_contracts[buyer_id] += 1
        self.agent_n_contracts[seller_id] += 1
        missing_product = q - seller_factory.current_inventory[product]
        missing_money = p - buyer_factory.current_balance

        # if the contract is already nullified, take care of it
        if contract.nullified_at is not None:
            self.compensation_factory._inventory[product] = 0
            self.compensation_factory._balance = 0
            for c in self.compensation_records.get(contract.id, []):
                if c.product >= 0 and c.quantity > 0:
                    assert c.product == product
                    self.compensation_factory._inventory[product] += c.quantity
                self.compensation_factory._balance += c.money
                if c.seller_bankrupt:
                    seller_factory = self.compensation_factory
                else:
                    buyer_factory = self.compensation_factory
        elif seller_factory == buyer_factory:
            self.logwarning(
                f"Seller factory {seller_factory.agent_id} and Buyer factory {buyer_factory.agent_id} are the same."
                f" This is most likely happening because you have two compensation records for the "
                f"same contract!!"
            )
            return breaches
        # if there are no breaches, just execute the contract
        if missing_money <= 0 and missing_product <= 0:
            self._execute(
                product, q, p, u, buyer_factory, seller_factory, has_breaches=False
            )
            self.__register_contract(seller_id, 0)
            self.__register_contract(buyer_id, 0)
            return breaches

        # if there is a product breach (the seller does not have enough products), register it
        if missing_product <= 0:
            self.__register_contract(seller_id, 0)
        else:
            product_breach_level = missing_product / q
            breaches.add(
                Breach(
                    contract=contract,
                    perpetrator=seller_id,
                    victims=buyer_id,
                    level=product_breach_level,
                    type="product",
                )
            )
            self.__register_contract(seller_id, product_breach_level)
            self.__register_breach(seller_id, product_breach_level, p, seller_factory)
            if self.borrow_on_breach:
                paid_for = seller_factory.store(
                    product,
                    -missing_product,
                    u,
                    self.buy_missing_products,
                    self.breach_penalty,
                )
                missing_product -= paid_for

        # if there is a money breach (the buyer does not have enough money), register it
        if missing_money < 0:
            self.__register_contract(buyer_id, 0)
        else:
            money_breach_level = missing_money / p
            breaches.add(
                Breach(
                    contract=contract,
                    perpetrator=buyer_id,
                    victims=seller_id,
                    level=money_breach_level,
                    type="money",
                )
            )
            self.__register_contract(buyer_id, money_breach_level)
            self.__register_breach(buyer_id, money_breach_level, p, buyer_factory)
            if self.borrow_on_breach:
                # find out the amount to be paid to borrow the needed money
                to_pay = math.ceil(missing_money * self.breach_penalty)
                paid = buyer_factory.pay(to_pay)
                missing_money -= (missing_money * paid) // to_pay

        # execute the contract to the limit possible
        self._execute(
            product,
            q,
            p,
            u,
            buyer_factory,
            seller_factory,
            has_breaches=missing_product > 0 or missing_money > 0,
        )

        # return the list of breaches
        return breaches

    def complete_contract_execution(
        self, contract: Contract, breaches: List[Breach], resolution: Contract
    ) -> None:
        pass

    def compensate(self, available: int, factory: Factory) -> None:
        """
        Called by a factory when it is going bankrupt after liquidation

        Args:
            available: The amount available from liquidation
            factory: The factory being bankrupted

        Returns:

        """
        agent_id = factory.agent_id
        # get all future contracts of the bankrupt agent that are not executed
        contracts = list(
            itertools.chain(
                *(factory.contracts[s] for s in range(self.current_step, self.n_steps))
            )
        )

        owed = 0
        total_owed = 0
        nulled_contracts = []
        for contract in contracts:
            total_owed += contract.q * contract.u
            if (
                self.a2f[contract.partner].is_bankrupt
                or contract.contract.nullified_at is not None
            ):
                continue
            nulled_contracts.append(contract)
            owed += contract.q * contract.u

        if available <= 0 or owed <= 0:
            self.record_bankrupt(factory)
            return

        # calculate compensation fraction
        if available >= owed:
            fraction = self.compensation_fraction
        else:
            fraction = self.compensation_fraction * available / owed

        # calculate compensation and pay it as needed
        for contract in nulled_contracts:
            victim = self.agents[contract.partner]
            victim_factory = self.a2f.get(victim.id, None)
            # calculate compensation (as money)
            compensation = min(available, fraction * contract.q * contract.u)
            if compensation < 0:
                self.nullify_contract(contract)
                continue
            if self.compensate_immediately:
                # pay immediate compensation if indicated
                victim_factory.pay(-compensation)
                available -= compensation
            else:
                # add the required products/money to the internal compensation inventory/funds to be paid at the
                # contract execution time.
                if contract.is_seller:
                    self.compensation_records[contract.contract.id].append(
                        CompensationRecord(
                            contract.product,
                            int((compensation // contract.u) * contract.u),
                            0,
                            True,
                            victim_factory,
                        )
                    )
                else:
                    self.compensation_records[contract.contract.id].append(
                        CompensationRecord(-1, 0, compensation, False, victim_factory)
                    )
            victim.on_contract_nullified(
                contract.contract,
                compensation if self.compensate_immediately else 0,
                fraction,
            )
            self.nullify_contract(contract.contract)
            self.record_bankrupt(factory)
