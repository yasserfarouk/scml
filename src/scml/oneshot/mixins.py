from __future__ import annotations

from typing import TYPE_CHECKING

from .common import QUANTITY, TIME, UNIT_PRICE
from .ufun import OneShotUFun

if TYPE_CHECKING:
    from .agent import OneShotAgent
    from .sysagents import DefaultOneShotAdapter

__all__ = ["OneShotUFunCreatorMixin"]


class OneShotUFunCreatorMixin:
    def make_ufun(
        self: OneShotAgent | DefaultOneShotAdapter,  # type: ignore
        add_exogenous,
        in_adapter,
    ):  # type: ignore
        awi = self._obj.awi if in_adapter else self.awi  # type: ignore
        it = awi.current_input_issues[TIME] if awi.current_input_issues else None

        iq = awi.current_input_issues[QUANTITY] if awi.current_input_issues else None
        ip = awi.current_input_issues[UNIT_PRICE] if awi.current_input_issues else None
        oq = awi.current_output_issues[QUANTITY] if awi.current_output_issues else None
        op = (
            awi.current_output_issues[UNIT_PRICE] if awi.current_output_issues else None
        )
        self.ufun = OneShotUFun(
            agent_id=awi.agent.id,
            perishable=awi.is_perishable,
            ex_qin=awi.current_exogenous_input_quantity if add_exogenous else 0,
            ex_pin=awi.current_exogenous_input_price if add_exogenous else 0,
            ex_qout=awi.current_exogenous_output_quantity if add_exogenous else 0,
            ex_pout=awi.current_exogenous_output_price if add_exogenous else 0,
            production_cost=awi.profile.cost,
            disposal_cost=awi.current_disposal_cost,
            storage_cost=awi.current_storage_cost,
            shortfall_penalty=awi.current_shortfall_penalty,
            input_penalty_scale=awi.penalty_multiplier(True, None),
            storage_penalty_scale=awi.penalty_multiplier(True, None),
            output_penalty_scale=awi.penalty_multiplier(False, None),
            input_agent=awi.my_input_product == 0,
            output_agent=awi.my_output_product == awi.n_products - 1,
            input_product=awi.my_input_product,
            n_input_negs=awi.n_input_negotiations,
            n_output_negs=awi.n_output_negotiations,
            current_step=awi.current_step,
            time_range=(it.min_value, it.max_value)
            if it
            else (awi.current_step, awi.current_step),
            input_qrange=(iq.min_value, iq.max_value) if iq else (0, 0),
            input_prange=(ip.min_value, ip.max_value) if ip else (0, 0),
            output_qrange=(oq.min_value, oq.max_value) if oq else (0, 0),
            output_prange=(op.min_value, op.max_value) if op else (0, 0),
            force_exogenous=awi.is_exogenous_forced,
            n_lines=awi.n_lines,
            current_balance=awi.current_balance,
            suppliers=set(awi.my_suppliers),
            consumers=set(awi.my_consumers),
            inventory_in=awi.current_inventory_input,
            inventory_out=awi.current_inventory_output,
        )
        if hasattr(self, "_obj") and not in_adapter:
            self._obj.ufun = self.ufun  # type: ignore
        return self.ufun
