from typing import List
from negmas import (
    Agent,
    AgentWorldInterface,
    Issue,
    AgentMechanismInterface,
    RenegotiationRequest,
    Negotiator,
    ResponseType,
    Contract,
    MechanismState,
    Breach,
    PassThroughSAONegotiator,
    SAOController,
    SAOSyncController,
    MappingUtilityFunction,
)
from .common import QUANTITY, UNIT_PRICE

__all__ = ["OneShotUFun"]


class OneShotUFun:
    def __init__(
        self,
        owner: "OneShotAgent",
        p_in: int = 0,
        q_in: int = 0,
        p_out: int = 0,
        q_out: int = 0,
        cost: float = 0.0,
        storage_cost: float = 0.0,
        delivery_penalty: float = 0.0,
    ):
        self.owner = owner
        self.p_in, self.p_out = p_in, p_out
        self.q_in, self.q_out = q_in, q_out
        self.cost, self.storage_cost, self.delivery_penalty = (
            cost,
            storage_cost,
            delivery_penalty,
        )

    def __call__(self, contracts: List[Contract]) -> float:
        qin, qout, pin, pout = 0, 0, 0, 0
        for c in contracts:
            if c.signed_at < 0:
                continue
            product = c.annotation["product"]
            is_seller = product == self.owner.awi.my_output_product
            if is_seller:
                qout += c.agreement[QUANTITY]
                pout += c.agreement[UNIT_PRICE] * c.agreement[QUANTITY]
            else:
                qin += c.agreement[QUANTITY]
                pin += c.agreement[UNIT_PRICE] * c.agreement[QUANTITY]
        return self.eval(
            self.q_in + qin,
            self.q_out + qout,
            self.p_in + pin,
            self.p_out + pout,
            self.cost,
            self.storage_cost,
            self.delivery_penalty,
        )

    @classmethod
    def eval(
        cls,
        qin: int = 0,
        qout: int = 0,
        pin: int = 0,
        pout: int = 0,
        production_cost: float = 0.0,
        storage_cost: float = 0.0,
        delivery_penalty: float = 0,
    ) -> float:
        return (
            pin
            - pout
            - production_cost * min(qin, qout)
            - storage_cost * max(0, qin - qout)
            - delivery_penalty * max(0, qout - qin)
        )

    @classmethod
    def breach_level(cls, qin: int = 0, qout: int = 0):
        if max(qin, qout) < 1:
            return 0
        return abs(qin - qout) / max(qin, qout)

    @classmethod
    def is_breach(cls, qin: int = 0, qout: int = 0):
        return qin != qout
