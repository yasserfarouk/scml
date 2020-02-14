"""Implements an agent that does nothing"""
from typing import List, Optional, Dict, Any

from negmas import (
    Contract,
    Breach,
    AgentMechanismInterface,
    MechanismState,
    Issue,
    Negotiator,
)
from scml.scml2020.world import SCML2020Agent, Failure

__all__ = ["DoNothingAgent"]


class DoNothingAgent(SCML2020Agent):
    """An agent that does nothing for the whole length of the simulation"""

    def respond_to_negotiation_request(
        self,
        initiator: str,
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
    ) -> Optional[Negotiator]:
        return None

    def sign_all_contracts(self, contracts: List[Contract]) -> List[Optional[str]]:
        return [None] * len(contracts)

    def on_contracts_finalized(
        self,
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        pass

    def step(self):
        pass

    def init(self):
        pass

    def on_agent_bankrupt(
        self,
        agent: str,
        contracts: List[Contract],
        quantities: List[int],
        compensation_money: int,
    ) -> None:
        pass

    def on_failures(self, failures: List[Failure]) -> None:
        pass

    def on_negotiation_failure(
        self,
        partners: List[str],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
        state: MechanismState,
    ) -> None:
        pass

    def on_negotiation_success(
        self, contract: Contract, mechanism: AgentMechanismInterface
    ) -> None:
        pass

    def on_contract_cancelled(self, contract: Contract, rejectors: List[str]) -> None:
        pass

    def on_contract_executed(self, contract: Contract) -> None:
        pass

    def on_contract_breached(
        self, contract: Contract, breaches: List[Breach], resolution: Optional[Contract]
    ) -> None:
        pass
