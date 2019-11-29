"""Implements an agent that does nothing"""
from typing import List, Optional, Dict, Any

import numpy as np
from negmas import Contract, Breach, AgentMechanismInterface, MechanismState, Issue, Negotiator, RandomUtilityFunction
from negmas import AspirationNegotiator

from scml.scml2020.world import SCML2020Agent, Failure

__all__ = ["DoNothingAgent"]


class DoNothingAgent(SCML2020Agent):
    """An agent that does nothing for the whole length of the simulation"""

    def confirm_external_sales(self, quantities: np.ndarray, unit_prices: np.ndarray) -> np.ndarray:
        return np.zeros_like(quantities)

    def confirm_external_supplies(self, quantities: np.ndarray, unit_prices: np.ndarray) -> np.ndarray:
        return np.zeros_like(quantities)

    def respond_to_negotiation_request(self, initiator: str, issues: List[Issue], annotation: Dict[str, Any],
                                       mechanism: AgentMechanismInterface) -> Optional[Negotiator]:
        return None

    def sign_contract(self, contract: Contract) -> Optional[str]:
        return self.id

    def on_contract_signed(self, contract: Contract) -> None:
        pass

    def step(self):
        pass

    def init(self):
        pass

    def on_contract_nullified(
        self, contract: Contract, compensation_money: int, compensation_fraction: float
    ) -> None:
        pass

    def on_failures(self, failures: List[Failure]) -> None:
        pass

    def on_negotiation_failure(self, partners: List[str], annotation: Dict[str, Any],
                               mechanism: AgentMechanismInterface, state: MechanismState) -> None:
        pass

    def on_negotiation_success(self, contract: Contract, mechanism: AgentMechanismInterface) -> None:
        pass

    def on_contract_cancelled(self, contract: Contract, rejectors: List[str]) -> None:
        pass

    def on_contract_executed(self, contract: Contract) -> None:
        pass

    def on_contract_breached(self, contract: Contract, breaches: List[Breach], resolution: Optional[Contract]) -> None:
        pass
