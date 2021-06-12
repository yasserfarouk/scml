from typing import Union, Optional, Tuple, List, Any, Dict
import numpy as np
from negmas import Negotiator, Adapter, Contract, Breach
from negmas.sao import SAOController, SAONegotiator

from ..scml2020.common import (
    FactoryState,
    FactoryProfile,
    ANY_LINE,
    ANY_STEP,
    is_system_agent,
)
from .sysagents import DefaultOneShotAdapter
from .ufun import OneShotUFun
from .helper import AWIHelper
from .mixins import OneShotUFunCreatorMixin


class OneShotSCML2020Adapter(DefaultOneShotAdapter, Adapter):
    """Base class for adapters allowing SCML std agents to run as Oneshot agents in SCML2020Oneshot worlds"""

    def on_negotiation_failure(self, partners, annotation, mechanism, state):
        return self._obj.on_negotiation_failure(partners, annotation, mechanism, state)

    def on_negotiation_success(self, contract, mechanism):
        return self._obj.on_negotiation_success(contract, mechanism)

    def on_contract_executed(self, contract: Contract) -> None:
        return self._obj.on_contract_executed(contract)

    def on_contract_breached(
        self, contract: Contract, breaches: List[Breach], resolution: Optional[Contract]
    ) -> None:
        return self._obj.on_contract_breached(contract, breaches, resolution)

    def init(self):
        self._obj._awi = AWIHelper(owner=self)
        super().init()

    def before_step(self):
        if hasattr(self._obj, "before_step"):
            self._obj.before_step()

    def to_dict(self):
        return self._obj.to_dict()

    def _respond_to_negotiation_request(
        self,
        initiator,
        partners,
        issues,
        annotation,
        mechanism,
        role,
        req_id,
    ):
        return self._obj._respond_to_negotiation_request(
            initiator,
            partners,
            issues,
            annotation,
            mechanism,
            role,
            req_id,
        )

    def set_renegotiation_agenda(self, contract, breaches):
        return None

    def respond_to_renegotiation_request(self, contract, breaches, agenda):
        return None

    def on_neg_request_rejected(self, req_id, by):
        return self._obj.on_neg_request_rejected(req_id, by)

    def on_neg_request_accepted(self, req_id, mechanism):
        return self._obj.on_neg_request_rejected(req_id, mechanism)
