"""
Implements the one shot version of SCML
"""
from typing import Any, Optional

from negmas import (
    Contract,
    Issue,
    MechanismState,
    Negotiator,
    NegotiatorMechanismInterface,
)

from scml.oneshot.sysagents import DefaultOneShotAdapter


class DefaultStdAdapter(DefaultOneShotAdapter):
    """
    The base class of all agents running in Std based on StdAgent.

    Remarks:

        - It inherits from `Adapter` allowing it to just pass any calls not
          defined explicitly in it to the internal `_obj` object representing
          the SCML2024StdAgent.
    """


class _StdSystemAgent(DefaultStdAdapter):
    """Implements an agent for handling system operations"""

    def __init__(self, *args, role, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = role
        self.name = role
        self.profile = None

    @property
    def type_name(self):
        return "System"

    @property
    def short_type_name(self):
        return "System"

    def respond_to_negotiation_request(
        self,
        initiator: str,
        issues: list[Issue],
        annotation: dict[str, Any],
        mechanism: NegotiatorMechanismInterface,
    ) -> Optional[Negotiator]:
        pass

    def before_step(self):
        pass

    def step(self):
        pass

    def init(self):
        pass

    def on_negotiation_failure(
        self,
        partners: list[str],
        annotation: dict[str, Any],
        mechanism: NegotiatorMechanismInterface,
        state: MechanismState,
    ) -> None:
        pass

    def on_negotiation_success(
        self, contract: Contract, mechanism: NegotiatorMechanismInterface
    ) -> None:
        pass

    def sign_all_contracts(self, contracts: list[Contract]) -> list[Optional[str]]:
        """Signs all contracts"""
        return [self.id] * len(contracts)
