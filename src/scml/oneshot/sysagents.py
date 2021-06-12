"""
Implements the one shot version of SCML
"""
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from negmas import Adapter
from negmas import AgentMechanismInterface
from negmas import Breach
from negmas import Contract
from negmas import Issue
from negmas import MechanismState
from negmas import Negotiator
from negmas.sao import PassThroughSAONegotiator
from negmas import RenegotiationRequest

from .common import QUANTITY, UNIT_PRICE
from .ufun import OneShotUFun
from .agent import OneShotAgent
from .helper import AWIHelper
from .mixins import OneShotUFunCreatorMixin

__all__ = ["DefaultOneShotAdapter", "_SystemAgent"]


class DefaultOneShotAdapter(Adapter, OneShotUFunCreatorMixin):
    """
    The base class of all agents running in OneShot based on OneShotAgent.

    Remarks:

        - It inherits from `Adapter` allowing it to just pass any calls not
          defined explicity in it to the internal `_obj` object representing
          the SCML2020OneShotAgent.
    """

    def make_ufun(self, add_exogenous: bool):
        return super().make_ufun(add_exogenous, in_adapter=False)

    def on_negotiation_failure(self, partners, annotation, mechanism, state):
        return self._obj.on_negotiation_failure(partners, annotation, mechanism, state)

    def on_negotiation_success(self, contract, mechanism):
        return self._obj.on_negotiation_success(contract, mechanism)

    def on_contract_executed(self, contract: Contract) -> None:
        pass

    def on_contract_breached(
        self, contract: Contract, breaches: List[Breach], resolution: Optional[Contract]
    ) -> None:
        pass

    def init_(self):
        if isinstance(self._obj, OneShotAgent):
            if not self.ufun:
                self.make_ufun(add_exogenous=True)
        super().init_()

    def init(self):
        if isinstance(self._obj, OneShotAgent):
            self._obj.connect_to_oneshot_adapter(self)
        else:
            self._obj._awi = AWIHelper(self)
        super().init()

    def before_step(self):
        if hasattr(self._obj, "before_step"):
            self._obj.before_step()

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type_name,
            "level": self.awi.my_input_product if self.awi else None,
            "levels": [self.awi.my_input_product] if self.awi else None,
        }

    def _respond_to_negotiation_request(
        self,
        initiator: str,
        partners: List[str],
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
        role: Optional[str],
        req_id: Optional[str],
    ) -> Optional[Negotiator]:
        partner = [_ for _ in partners if _ != self.id][0]
        if not self._obj:
            return None
        neg = self._obj.create_negotiator(PassThroughSAONegotiator, name=partner, id=partner)
        return neg

    def set_renegotiation_agenda(
        self, contract: Contract, breaches: List[Breach]
    ) -> Optional[RenegotiationRequest]:
        return None

    def respond_to_renegotiation_request(
        self, contract: Contract, breaches: List[Breach], agenda: RenegotiationRequest
    ) -> Optional[Negotiator]:
        return None

    def on_neg_request_rejected(self, req_id: str, by: Optional[List[str]]):
        pass

    def on_neg_request_accepted(self, req_id: str, mechanism: AgentMechanismInterface):
        pass


class _SystemAgent(DefaultOneShotAdapter):
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
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
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

    def sign_all_contracts(self, contracts: List[Contract]) -> List[Optional[str]]:
        """Signs all contracts"""
        return [self.id] * len(contracts)
