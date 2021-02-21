from collections import namedtuple
from typing import List
import sys
from dataclasses import dataclass
import numpy as np

__all__ = [
    "SYSTEM_BUYER_ID",
    "SYSTEM_SELLER_ID",
    "COMPENSATION_ID",
    "ANY_STEP",
    "NO_COMMAND",
    "ANY_LINE",
    "INFINITE_COST",
    "QUANTITY",
    "TIME",
    "UNIT_PRICE",
    "is_system_agent",
    "FactoryState",
    "FinancialReport",
    "FactoryProfile",
    "Failure",
]

SYSTEM_SELLER_ID = "SELLER"
"""ID of the system seller agent"""

SYSTEM_BUYER_ID = "BUYER"
"""ID of the system buyer agent"""

COMPENSATION_ID = "COMPENSATOR"
"""ID of the takeover agent"""


ANY_STEP = -1
"""Used to indicate any time-step"""


ANY_LINE = -1
"""Used to indicate any line"""


NO_COMMAND = -1
"""A constant indicating no command is scheduled on a factory line"""


INFINITE_COST = sys.maxsize // 2
"""A constant indicating an invalid cost for lines incapable of running some process"""


QUANTITY = 0
"""Index of quantity in negotiation issues"""


TIME = 1
"""Index of time in negotiation issues"""


UNIT_PRICE = 2
"""Index of unit price in negotiation issues"""


def is_system_agent(aid: str) -> bool:
    """
    Checks whether an agent is a system agent or not

    Args:

        aid: Agent ID

    Returns:

        True if the ID is for a system agent.
    """
    return (
        aid.startswith(SYSTEM_SELLER_ID)
        or aid.startswith(SYSTEM_BUYER_ID)
        or aid.startswith(COMPENSATION_ID)
    )


ContractInfo = namedtuple(
    "ContractInfo", ["q", "u", "product", "is_seller", "partner", "contract"]
)
"""Information about a contract including a pointer to it"""


@dataclass
class ExogenousContract:
    """Represents a contract to be revealed at revelation_time to buyer and seller between them that is not agreed upon
    through negotiation but is endogenously given"""

    product: int
    """Product"""
    quantity: int
    """Quantity"""
    unit_price: int
    """Unit price"""
    time: int
    """Delivery time"""
    revelation_time: int
    """Time at which to reveal the contract to both buyer and seller"""
    seller: int = -1
    """Seller index in the agents array (-1 means "system")"""
    buyer: int = -1
    """Buyer index in the agents array (-1 means "system")"""


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
            f"breach_prob: {self.breach_prob}, breach_level: {self.breach_level} "
            f"{'(BANKRUPT)' if self.is_bankrupt else ''}"
        )


@dataclass
class FactoryProfile:
    """Defines all private information of a factory"""

    __slots__ = ["costs"]
    costs: np.ndarray
    """An n_lines * n_processes array giving the cost of executing any process (INVALID_COST indicates infinity)"""

    @property
    def n_lines(self):
        return self.costs.shape[0]

    @property
    def n_products(self):
        return self.costs.shape[1] + 1

    @property
    def n_processes(self):
        return self.costs.shape[1]

    @property
    def processes(self) -> np.ndarray:
        """The processes that have valid costs"""
        return np.nonzero(np.any(self.costs != INFINITE_COST, axis=0))[0]

    @property
    def input_products(self) -> np.ndarray:
        """The input products to all processes runnable (See `processes` )"""
        return np.nonzero(np.any(self.costs != INFINITE_COST, axis=0))[0]

    @property
    def output_products(self) -> np.ndarray:
        """The output products to all processes runnable (See `processes` )"""
        return np.nonzero(np.any(self.costs != INFINITE_COST, axis=0))[0] + 1


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
    """The process that failed to execute"""


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
