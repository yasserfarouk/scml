from negmas import UtilityValue
from typing import Optional

from scml.scml2019 import SCMLAgreement
from scml.scml2019.factory_managers.builtins import NegotiatorUtility


class ConsumerUtilityFunction(NegotiatorUtility):
    def __init__(self, target_price):
        super().__init__(agent=None, annotation=None)
        self.target_price = target_price
        self.reduced_cost = 0.01

    def call(self, agreement: SCMLAgreement) -> Optional[UtilityValue]:
        unit_price = agreement.get("unit_price")
        quantity = agreement.get("quantity")
        if self.target_price - unit_price == 0:
            return self.reduced_cost * quantity
        return (self.target_price - unit_price) * quantity
