"""
Implements the one shot version of SCML
"""
import warnings
from typing import Any, Optional

from negmas import (
    Adapter,
    Breach,
    Contract,
    Issue,
    MechanismState,
    Negotiator,
    NegotiatorMechanismInterface,
    RenegotiationRequest,
)
from negmas.sao import ControlledSAONegotiator


from .agent import OneShotAgent
from .awi import OneShotAWI
from .helper import AWIHelper
from .mixins import OneShotUFunCreatorMixin

__all__ = ["DefaultOneShotAdapter", "_StdSystemAgent"]


class DefaultOneShotAdapter(Adapter, OneShotUFunCreatorMixin):
    """
    The base class of all agents running in OneShot based on OneShotAgent.

    Remarks:

        - It inherits from `Adapter` allowing it to just pass any calls not
          defined explicity in it to the internal `_obj` object representing
          the OneShotAgent.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._obj: OneShotAgent

    def make_ufun(self, add_exogenous: bool, in_adapter=False):
        return super().make_ufun(add_exogenous, in_adapter=False)

    def on_negotiation_failure(self, partners, annotation, mechanism, state):
        if self.awi._world._debug:
            if tuple(sorted(partners)) in self._negs_done.keys():
                warnings.warn(
                    f"{partners=} found in completed negs for {self.id} at step: "
                    f"{self.awi.current_step} with info {self._negs_done[tuple(sorted(partners))]}"
                    f" on mechanism {mechanism.id}\n{self._negs_done}\n{mechanism.annotation=}"
                )
            self._negs_done[tuple(sorted(partners))] = (
                "failed",
                self.awi.current_step,
                mechanism.id,
                mechanism.annotation,
            )
        if annotation["buyer"] == self.id:
            if self.ufun is not None:
                self.ufun.register_supply_failure(annotation["seller"])
            try:
                self.awi.current_negotiation_details["buy"].pop(annotation["seller"])
            except Exception as e:
                if self.awi._world._debug:
                    raise AssertionError(
                        f'Partners: {list(self.awi.current_negotiation_details["buy"].keys())} {annotation["seller"]} not found\n{self.awi.my_suppliers=}\n{self.awi.my_consumers=}\n{e}'
                    )
                # else:
                # warnings.warn(
                #     f'Partners: {list(self.awi.current_negotiation_details["buy"].keys())} {annotation["seller"]} not found\n{self.awi.my_suppliers=}\n{self.awi.my_consumers=}\n{e}'
                # )

        elif annotation["seller"] == self.id:
            if self.ufun is not None:
                self.ufun.register_sale_failure(annotation["buyer"])
            if (
                self.awi.current_negotiation_details["sell"].get(
                    annotation["buyer"], None
                )
                is None
            ):
                pass
            try:
                self.awi.current_negotiation_details["sell"].pop(annotation["buyer"])
            except Exception as e:
                if self.awi._world._debug:
                    raise AssertionError(
                        f'Partners: {list(self.awi.current_negotiation_details["sell"].keys())} {annotation["buyer"]} not found\n{self.awi.my_suppliers=}\n{self.awi.my_consumers=}\n{e}'
                    )
                # else:
                #     warnings.warn(
                #         f'Partners: {list(self.awi.current_negotiation_details["sell"].keys())} {annotation["buyer"]} not found\n{self.awi.my_suppliers=}\n{self.awi.my_consumers=}\n{e}\n{contract}'
                #     )
        else:
            raise ValueError(
                f"{self.id} received a  negotiation failure for which it is not a buyer nor a seller"
            )
        result = self._obj.on_negotiation_failure(
            partners,
            annotation,
            mechanism,
            state,  # type: ignore
        )
        # for k in ("sell", "buy"):
        #     self.awi._world._agent_negotiations[self._obj.id][k].pop(mechanism.id, None)
        return result

    def on_negotiation_success(self, contract: Contract, mechanism):
        from scml.oneshot.ufun import OneShotUFun

        self.ufun: OneShotUFun  # type: ignore
        if self.awi._world._debug:
            partners = contract.partners
            if tuple(sorted(partners)) in self._negs_done.keys():
                warnings.warn(
                    f"{partners=} found in completed negs for {self.id} "
                    f"at step: {self.awi.current_step} with info "
                    f"{self._negs_done[tuple(sorted(partners))]} on "
                    f"mechanism {mechanism.id}\n{self._negs_done}\n{contract.annotation=}"
                )
            self._negs_done[tuple(sorted(partners))] = (
                "failed",
                self.awi.current_step,
                mechanism.id,
                mechanism.annotation,
            )
        annotation, agreement = contract.annotation, contract.agreement
        if annotation["buyer"] == self.id:
            self.awi._register_supply(
                annotation["seller"],
                agreement["quantity"],
                agreement["unit_price"],
                agreement["time"],
            )
            if self.ufun is not None:
                self.ufun.register_supply(
                    agreement["quantity"],
                    agreement["unit_price"],
                    agreement["time"],
                )
            try:
                self.awi.current_negotiation_details["buy"].pop(annotation["seller"])
            except Exception as e:
                if self.awi._world._debug:
                    raise AssertionError(
                        f'Partners: {list(self.awi.current_negotiation_details["buy"].keys())} {annotation["seller"]} not found\n{self.awi.my_suppliers=}\n{self.awi.my_consumers=}\n{e}'
                    )
                # else:
                #     warnings.warn(
                #         f'Partners: {list(self.awi.current_negotiation_details["buy"].keys())} {annotation["seller"]} not found\n{self.awi.my_suppliers=}\n{self.awi.my_consumers=}\n{e}\n{contract}'
                #     )
        elif annotation["seller"] == self.id:
            self.awi._register_sale(
                annotation["buyer"],
                agreement["quantity"],
                agreement["unit_price"],
                agreement["time"],
            )
            if self.ufun is not None:
                self.ufun.register_sale(
                    agreement["quantity"],
                    agreement["unit_price"],
                    agreement["time"],
                )
            try:
                self.awi.current_negotiation_details["sell"].pop(annotation["buyer"])
            except Exception as e:
                if self.awi._world._debug:
                    raise AssertionError(
                        f'Partners: {list(self.awi.current_negotiation_details["sell"].keys())} {annotation["buyer"]} not found\n{self.awi.my_suppliers=}\n{self.awi.my_consumers=}\n{e}'
                    )
                # else:
                #     warnings.warn(
                #         f'Partners: {list(self.awi.current_negotiation_details["sell"].keys())} {annotation["buyer"]} not found\n{self.awi.my_suppliers=}\n{self.awi.my_consumers=}\n{e}\n{contract}'
                #     )
        else:
            raise ValueError(
                f"{self.id} received a  contract for which it is not a buyer nor a seller: {contract=}"
            )
        result = self._obj.on_negotiation_success(contract, mechanism)
        # for k in ("sell", "buy"):
        #     self.awi._world._agent_negotiations[self._obj.id][k].pop(mechanism.id, None)
        return result

    def on_contract_executed(self, contract: Contract) -> None:
        pass

    def on_contract_breached(
        self, contract: Contract, breaches: list[Breach], resolution: Optional[Contract]
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
            self._obj._awi = AWIHelper(self)  # type: ignore
        super().init()

    def reset(self):
        if hasattr(self._obj, "reset"):
            self._obj.reset()

    def before_step(self):
        if self.awi._world._debug:
            self._negs_done = dict()
        self.awi._reset_sales_and_supplies()
        if hasattr(self._obj, "before_step"):
            self._obj.before_step()

    def step(self):
        if self.awi._world._debug:
            self._negs_done = dict()
        self._obj.step()

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
        partners: list[str],
        issues: list[Issue],
        annotation: dict[str, Any],
        mechanism: NegotiatorMechanismInterface,
        role: Optional[str],
        req_id: Optional[str],
    ) -> Optional[Negotiator]:
        partner = [_ for _ in partners if _ != self.id][0]
        if not self._obj:
            return None
        neg = self._obj.create_negotiator(
            ControlledSAONegotiator,  # type: ignore
            name=partner,
            id=partner,
        )
        return neg

    def set_renegotiation_agenda(
        self, contract: Contract, breaches: list[Breach]
    ) -> Optional[RenegotiationRequest]:
        return None

    def respond_to_renegotiation_request(
        self, contract: Contract, breaches: list[Breach], agenda: RenegotiationRequest
    ) -> Optional[Negotiator]:
        return None

    def on_neg_request_rejected(self, req_id: str, by: Optional[list[str]]):
        pass

    def on_neg_request_accepted(
        self, req_id: str, mechanism: NegotiatorMechanismInterface
    ):
        pass

    @property
    def awi(self) -> OneShotAWI:
        return self._awi  # type: ignore

    @awi.setter
    def awi(self, awi: OneShotAWI):
        """Sets the Agent-world interface. Should only be called by the world."""
        self._awi = awi

    @property
    def short_type_name(self):
        name = self.type_name.split(":")[-1].split(".")[-1]
        if name:
            return name
        return super().short_type_name


class _StdSystemAgent(DefaultOneShotAdapter):
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
