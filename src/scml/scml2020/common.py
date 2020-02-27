import sys

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
