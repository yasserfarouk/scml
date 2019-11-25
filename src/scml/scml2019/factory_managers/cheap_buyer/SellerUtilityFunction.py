from typing import Optional

from negmas import UtilityValue
from scml.scml2019.common import SCMLAgreement
from scml.scml2019.factory_managers.builtins import NegotiatorUtility


class SellerUtilityFunction(NegotiatorUtility):
    def __init__(self, unit_cost):
        self.unit_cost = unit_cost

    def call(self, agreement: SCMLAgreement) -> Optional[UtilityValue]:
        quantity = agreement.get("quantity")
        unit_price = agreement.get("unit_price")
        unit_cost = self.unit_cost
        return max((unit_price - unit_cost) * quantity, 0)
