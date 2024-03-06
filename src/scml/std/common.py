from scml.oneshot.common import (
    INFINITE_COST,
    QUANTITY,
    SYSTEM_BUYER_ID,
    SYSTEM_SELLER_ID,
    TIME,
    UNIT_PRICE,
    FinancialReport,
    OneShotExogenousContract,
    OneShotProfile,
    OneShotState,
    is_system_agent,
)

__all__ = [
    "QUANTITY",
    "UNIT_PRICE",
    "TIME",
    "StdState",
    "StdExogenousContract",
    "StdProfile",
    "FinancialReport",
    "is_system_agent",
    "INFINITE_COST",
    "SYSTEM_BUYER_ID",
    "SYSTEM_SELLER_ID",
    "is_system_agent",
]


class StdState(OneShotState):
    """State of a one-shot agent"""


class StdExogenousContract(OneShotExogenousContract):
    """Exogenous contract information"""


class StdProfile(OneShotProfile):
    """Defines all private information of a factory"""
