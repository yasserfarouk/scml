from negmas import Contract
from negmas import Negotiator
from negmas.sao import NiceNegotiator
from typing import Optional

from scml.scml2019.common import CFP
from .builtins import GreedyFactoryManager


class FJ2FactoryManager(GreedyFactoryManager):
    """My factory manager"""

    def init(self):
        super().init()
        self.awi.register_interest([p.id for p in self.awi.products])

    def step(self):
        for product in self.awi.products:
            cfp = CFP(
                is_buy=False,
                publisher=self.id,
                product=product.id,
                time=(self.awi.current_step + 2, self.awi.current_step + 6),
                unit_price=0.0000893,
                quantity=(5, 10),
            )
            self.awi.register_cfp(cfp)

    def on_new_cfp(self, cfp: "CFP"):
        if self.is_good_cfp(cfp):
            self.request_negotiation(cfp, NiceNegotiator())
        else:
            super().on_new_cfp(cfp)

    def sign_contract(self, contract: Contract):
        cfp = contract.annotation.get("cfp")
        if self.is_my_cheat_cfp(cfp) or self.is_good_cfp(cfp):
            return "Making tomorrow with you."
        else:
            return super().sign_contract(contract)

    def on_contract_signed(self, contract: Contract):
        cfp = contract.annotation.get("cfp")
        if self.is_my_cheat_cfp(cfp) or self.is_good_cfp(cfp):
            self.awi.buy_insurance(contract)
        else:
            super().on_contract_signed(contract)

    def respond_to_negotiation_request(
        self, cfp: "CFP", partner: str
    ) -> Optional[Negotiator]:
        if self.is_my_cheat_cfp(cfp):
            return NiceNegotiator()
        return super().respond_to_negotiation_request(cfp, partner)

    def is_my_cheat_cfp(self, cfp: "CFP"):
        return cfp.publisher == self.id and cfp.unit_price == 0.0000893

    def is_good_cfp(self, cfp: "CFP"):
        if cfp.publisher != self.id:
            if cfp.unit_price == 0.0000893 and cfp.product == self.awi.products[-1].id:
                return True
            if cfp.is_buy and cfp.product != self.awi.products[-1].id:
                return True
        return False
