from negmas import (
    Negotiator,
    Contract,
    Breach,
    RenegotiationRequest,
    AgentMechanismInterface,
    MechanismState,
)
from typing import Optional, List, Dict, Any

from scml.scml2019 import SCML2019Agent, FinancialReport, Loan
from .ConsumerUtilityFunction import ConsumerUtilityFunction
from .MyNegotiator2 import MyNegotiator2


class MyConsumer(SCML2019Agent):
    def __init__(self, agent, name):
        self.alpha = 1
        self.beta = 0
        self.agent = agent
        self.name = name
        self.negotiator_id = 0
        self.negotiators = {}
        self.NEGOTIATOR_ID_FIXED_PART = "NEGOTIATOR_ID_BUYER"
        self._reserved_val = 1
        super(MyConsumer, self).__init__(name=name)

    def step(self):
        pass

    def init(self):
        pass

    def on_neg_request_rejected(self, req_id: str, by: Optional[List[str]]):
        pass

    def on_neg_request_accepted(self, req_id: str, mechanism: AgentMechanismInterface):
        pass

    def on_negotiation_failure(
        self,
        partners: List[str],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
        state: MechanismState,
    ) -> None:
        if self.agent.get_amount_of_raw_materials() > 200:
            self._reserved_val = max(0, self._reserved_val - 0.001)
        pass

    def on_negotiation_success(
        self, contract: Contract, mechanism: AgentMechanismInterface
    ) -> None:
        self._reserved_val = min(1, self._reserved_val + 0.01)
        pass

    def on_contract_signed(self, contract: Contract) -> None:
        pass

    def on_contract_cancelled(self, contract: Contract, rejectors: List[str]) -> None:
        pass

    def set_renegotiation_agenda(
        self, contract: Contract, breaches: List[Breach]
    ) -> Optional[RenegotiationRequest]:
        pass

    def respond_to_renegotiation_request(
        self, contract: Contract, breaches: List[Breach], agenda: RenegotiationRequest
    ) -> Optional[Negotiator]:
        pass

    def sign_contract(self, contract: Contract) -> Optional[str]:
        pass

    def on_contract_executed(self, contract: Contract) -> None:
        pass

    def on_contract_breached(
        self, contract: Contract, breaches: List[Breach], resolution: Optional[Contract]
    ) -> None:
        pass

    def confirm_loan(self, loan: Loan, bankrupt_if_rejected: bool) -> bool:
        pass

    def on_contract_nullified(
        self, contract: Contract, bankrupt_partner: str, compensation: float
    ) -> None:
        pass

    def on_agent_bankrupt(self, agent_id: str) -> None:
        pass

    def confirm_partial_execution(
        self, contract: Contract, breaches: List[Breach]
    ) -> bool:
        pass

    def confirm_contract_execution(self, contract: Contract) -> bool:
        pass

    def on_new_cfp(self, cfp: "CFP"):
        pass

    def on_remove_cfp(self, cfp: "CFP"):
        pass

    def on_new_report(self, report: FinancialReport):
        pass

    def on_inventory_change(self, product: int, quantity: int, cause: str) -> None:
        pass

    def on_cash_transfer(self, amount: float, cause: str) -> None:
        pass

    def respond_to_negotiation_request(
        self, cfp: "CFP", partner: str
    ) -> Optional[Negotiator]:
        if self.awi.is_bankrupt(partner):
            return None
        ufun = ConsumerUtilityFunction(target_price=self.get_target_price())
        negotiator_id = self.NEGOTIATOR_ID_FIXED_PART + " : " + str(self.negotiator_id)
        negotiator = MyNegotiator2(
            name=negotiator_id,
            ufun=ufun,
            strategy=MyNegotiator2.STRATEGY_TIME_BASED_CONCESSION,
            reserved_value=self._reserved_val,
        )
        self.negotiators[negotiator_id] = negotiator
        self.negotiator_id += 1
        return negotiator

    def get_target_price(self):
        return self.agent.get_target_price()

    def get_negotiator(self, negotiator_id):
        return self.negotiators.get(negotiator_id)

    def get_negotiator_id_fixed_part(self):
        return self.NEGOTIATOR_ID_FIXED_PART
