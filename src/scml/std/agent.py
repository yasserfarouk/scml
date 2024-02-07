from scml.oneshot.agent import (
    EndingNegotiator,
    OneShotAgent,
    OneShotIndNegotiatorsAgent,
    OneShotSingleAgreementAgent,
    OneShotSyncAgent,
)

__all__ = [
    "StdAgent",
    "StdSyncAgent",
    "EndingNegotiator",
]

StdAgent = OneShotAgent
StdSyncAgent = OneShotSyncAgent
