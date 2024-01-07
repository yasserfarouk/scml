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
    "StdSingleAgreementAgent",
    "StdIndNegotiatorsAgent",
    "EndingNegotiator",
]

StdAgent = OneShotAgent
StdIndNegotiatorsAgent = OneShotIndNegotiatorsAgent
StdSingleAgreementAgent = OneShotSingleAgreementAgent
StdSyncAgent = OneShotSyncAgent
