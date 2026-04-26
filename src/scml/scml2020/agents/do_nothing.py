"""Implements an agent that does nothing"""

from typing import Any

from negmas import (
    Breach,
    Contract,
    Issue,
    MechanismState,
    Negotiator,
    NegotiatorMechanismInterface,
)

from scml.scml2020.agent import SCML2020Agent
from scml.scml2020.common import Failure

__all__ = ["DoNothingAgent"]


class DoNothingAgent(SCML2020Agent):
    """An agent that does nothing for the whole length of the simulation"""

    def respond_to_negotiation_request(
        self,
        initiator: str,
        issues: list[Issue],
        annotation: dict[str, Any],
        mechanism: NegotiatorMechanismInterface,
    ) -> Negotiator | None:
        return None

    def sign_all_contracts(self, contracts: list[Contract]) -> list[str | None]:
        return [None] * len(contracts)

    def on_contracts_finalized(
        self,
        signed: list[Contract],
        cancelled: list[Contract],
        rejectors: list[list[str]],
    ) -> None:
        pass

    def step(self):
        pass

    def init(self):
        pass

    def on_agent_bankrupt(
        self,
        agent: str,
        contracts: list[Contract],
        quantities: list[int],
        compensation_money: int,
    ) -> None:
        pass

    def on_failures(self, failures: list[Failure]) -> None:
        pass

    def on_negotiation_failure(
        self,
        partners: list[str],
        annotation: dict[str, Any],
        mechanism: NegotiatorMechanismInterface,
        state: MechanismState,
    ) -> None:
        pass

    def on_negotiation_success(self, contract: Contract, mechanism: NegotiatorMechanismInterface) -> None:
        pass

    def on_contract_cancelled(self, contract: Contract, rejectors: list[str]) -> None:
        pass

    def on_contract_executed(self, contract: Contract) -> None:
        pass

    def on_contract_breached(self, contract: Contract, breaches: list[Breach], resolution: Contract | None) -> None:
        pass
