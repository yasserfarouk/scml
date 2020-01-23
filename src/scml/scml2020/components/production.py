

__all__ = [
    "SupplyDrivenProductionStrategy",
    "DemandDrivenProductionStrategy",
]

import numpy as np
from negmas import Contract
from typing import List, Dict

from scml.scml2020.common import NO_COMMAND


class SupplyDrivenProductionStrategy:
    """A production strategy that converts all inputs to outputs"""

    def confirm_production(
        self: "SCML2020Agent", commands: np.ndarray, balance: int, inventory: np.ndarray
    ) -> np.ndarray:
        inputs = min(self.awi.state.inventory[self.input_product], len(commands))
        commands[:inputs] = self.input_product
        commands[inputs:] = NO_COMMAND
        return commands


class DemandDrivenProductionStrategy:
    """A production strategy that produces ONLY when a contract is secured"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.earliest_schedule: Dict[str, int] = dict()
        """Stores the first time step at which the production for each sell contract is scheduled"""

    def on_contracts_finalized(
        self,
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        for contract in signed:
            is_seller = contract.annotation["seller"] == self.id
            if not is_seller:
                continue
            step = contract.agreement["time"]
            # find the earliest time I can do anything about this contract
            earliest_production = self.awi.current_step
            if step > self.awi.n_steps - 1 or step < earliest_production:
                continue
            # if I am a seller, I will schedule production
            output_product = contract.annotation["product"]
            input_product = output_product - 1
            if input_product >= 0:
                steps, _ = self.awi.schedule_production(
                    process=input_product,
                    repeats=contract.agreement["quantity"],
                    step=(earliest_production, step - 1),
                    line=-1,
                )
                self.earliest_schedule[contract.id] = min(steps) if len(steps) > 0 else -1

    def can_be_produced(self, contract_id: str):
        return self.earliest_schedule.get(contract_id, -1) >= 0
