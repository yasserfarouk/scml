import functools
import itertools
import math
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from random import randint, random
from typing import TYPE_CHECKING

from negmas import MechanismState, NegotiatorMechanismInterface
from negmas.events import Notification
from negmas.helpers import get_class
from negmas.negotiators import Negotiator
from negmas.preferences import MappingUtilityFunction, WeightedUtilityFunction
from negmas.situated import Breach, Contract, RenegotiationRequest
from numpy.random import dirichlet

from .agent import SCML2019Agent
from .common import CFP, DEFAULT_NEGOTIATOR, UNIT_PRICE, FinancialReport
from .helpers import pos_gauss

if True:  #
    from typing import Any, Dict, List, Optional, Tuple, Union

    from .common import Loan

if TYPE_CHECKING:
    from .awi import SCMLAWI

__all__ = ["Consumer", "ConsumptionProfile", "JustInTimeConsumer"]


@dataclass
class ConsumptionProfile:
    schedule: Union[int, List[int]] = 0
    underconsumption: float = 0.1
    overconsumption: float = 0.01
    dynamicity: float = 0.0
    cv: float = 0.1

    alpha_q: float = 0.5
    alpha_u: float = 1.0

    beta_q: float = 10.0
    beta_u: float = 10.0

    tau_q: float = 2
    tau_u: float = 0.25

    @classmethod
    def random(cls):
        return ConsumptionProfile(
            schedule=randint(0, 5),
            overconsumption=2 * random(),
            underconsumption=2 * random(),
            cv=random(),
            beta_q=99 * random() + 1,
            beta_u=99 * random() + 1,
            tau_q=3 * random(),
            tau_u=random(),
        )

    def schedule_at(self, time: int) -> int:
        if isinstance(self.schedule, int):
            return self.schedule
        else:
            return self.schedule[time % len(self.schedule)]

    def schedule_within(self, time: Union[int, List[int], Tuple[int, int]]) -> int:
        if isinstance(time, int):
            return self.schedule_at(time)
        if isinstance(time, tuple):
            times = list(range(time[0], time[1] + 1))
        else:
            times = time
        if isinstance(self.schedule, int):
            return self.schedule * len(times)
        return sum(self.schedule_at(t) for t in times)

    def set_schedule_at(self, time: int, value: int, n_steps: int) -> None:
        if isinstance(self.schedule, int):
            self.schedule = [self.schedule] * n_steps
        elif len(self.schedule) < n_steps:
            self.schedule = list(
                itertools.chain(
                    *([self.schedule] * int(math.ceil(n_steps / len(self.schedule))))
                )
            )
        self.schedule[time % len(self.schedule)] = value


class Consumer(SCML2019Agent, ABC):
    """Base class of all consumer classes"""

    pass


class JustInTimeConsumer(Consumer):
    """Consumer class"""

    def on_contract_executed(self, contract: Contract) -> None:
        pass

    def on_contract_breached(
        self, contract: Contract, breaches: List[Breach], resolution: Optional[Contract]
    ) -> None:
        pass

    def on_inventory_change(self, product: int, quantity: int, cause: str) -> None:
        pass

    def on_cash_transfer(self, amount: float, cause: str) -> None:
        pass

    def on_new_report(self, report: FinancialReport):
        pass

    def on_neg_request_rejected(self, req_id: str, by: Optional[List[str]]):
        pass

    def on_neg_request_accepted(
        self, req_id: str, mechanism: NegotiatorMechanismInterface
    ):
        pass

    def on_negotiation_failure(
        self,
        partners: List[str],
        annotation: Dict[str, Any],
        mechanism: NegotiatorMechanismInterface,
        state: MechanismState,
    ) -> None:
        pass

    def on_negotiation_success(
        self, contract: Contract, mechanism: NegotiatorMechanismInterface
    ) -> None:
        pass

    def on_contract_cancelled(self, contract: Contract, rejectors: List[str]) -> None:
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
        return True

    def on_remove_cfp(self, cfp: "CFP"):
        pass

    MAX_UNIT_PRICE = 100.0
    RELATIVE_MAX_PRICE = 1.5

    def __init__(
        self,
        profiles: Dict[int, ConsumptionProfile] = None,
        negotiator_type=DEFAULT_NEGOTIATOR,
        consumption_horizon: Optional[int] = 20,
        immediate_cfp_update: bool = True,
        name=None,
    ):
        super().__init__(name=name)
        self.negotiator_type = get_class(negotiator_type, scope=globals())
        self.profiles: Dict[int, ConsumptionProfile] = defaultdict(ConsumptionProfile)
        self.secured_quantities: Dict[int, int] = defaultdict(int)
        if profiles is not None:
            self.set_profiles(profiles=profiles)
        self.consumption_horizon = consumption_horizon
        self.immediate_cfp_update = immediate_cfp_update

    def on_new_cfp(self, cfp: "CFP") -> None:
        pass  # consumers never respond to CFPs

    def init(self):
        if self.consumption_horizon is None:
            self.consumption_horizon = self.awi.n_steps
        self.awi.register_interest(list(self.profiles.keys()))

    def set_profiles(self, profiles: Dict[int, ConsumptionProfile]):
        self.profiles = defaultdict(ConsumptionProfile)
        if profiles is not None:
            for k, v in profiles.items():
                self.profiles[k] = v
        self.secured_quantities = defaultdict(int)
        if profiles is not None:
            for k, v in profiles.items():
                self.secured_quantities[k] = 0

    def register_product_cfps(self, p: int, t: int, profile: ConsumptionProfile):
        current_schedule = profile.schedule_at(t)
        product = self.products[p]
        awi: SCMLAWI = self.awi
        if current_schedule <= 0:
            awi.bb_remove(
                section="cfps",
                query={"publisher": self.id, "time": t, "product_index": p},
            )
            return
        max_price = (
            JustInTimeConsumer.RELATIVE_MAX_PRICE * product.catalog_price
            if product.catalog_price is not None
            else JustInTimeConsumer.MAX_UNIT_PRICE
        )
        cfps = awi.bb_query(
            section="cfps", query={"publisher": self.id, "time": t, "product": p}
        )
        if cfps is not None and len(cfps) > 0:
            for _, cfp in cfps.items():
                if cfp.max_quantity != current_schedule:
                    cfp = CFP(
                        is_buy=True,
                        publisher=self.id,
                        product=p,
                        time=t,
                        unit_price=(0, max_price),
                        quantity=(1, current_schedule),
                    )
                    awi.bb_remove(
                        section="cfps",
                        query={"publisher": self.id, "time": t, "product": p},
                    )
                    awi.register_cfp(cfp)
                    break
        else:
            cfp = CFP(
                is_buy=True,
                publisher=self.id,
                product=p,
                time=t,
                unit_price=(0, max_price),
                quantity=(1, current_schedule),
            )
            awi.register_cfp(cfp)

    def step(self):
        if self.consumption_horizon is None:
            horizon = self.awi.n_steps
        else:
            horizon = min(
                self.awi.current_step + self.consumption_horizon + 1, self.awi.n_steps
            )
        for p, profile in self.profiles.items():
            for t in range(
                self.awi.current_step, horizon
            ):  # + self.transportation_delay
                self.register_product_cfps(p=p, t=t, profile=profile)

    def confirm_contract_execution(self, contract: Contract) -> bool:
        return True

    @staticmethod
    def _qufun(outcome: Dict[str, Any], tau: float, profile: ConsumptionProfile):
        """The ufun value for quantity"""
        q, t = outcome["quantity"], outcome["time"]
        y = profile.schedule_within(t)
        o = profile.overconsumption
        u = profile.underconsumption
        if q == 0 and y != 0:
            return 0.0
        if y <= 0:
            result = -o * ((q - y) ** tau)
        elif q > y:
            result = -o * (((q - y) / y) ** tau)
        elif q < y:
            result = -u * (((y - q) / y) ** tau)
        else:
            result = 1.0
        result = math.exp(result)
        if isinstance(result, complex):
            result = result.real
        if result is None:
            result = float("-inf")
        return result

    def respond_to_negotiation_request(
        self, cfp: "CFP", partner: str
    ) -> Optional[Negotiator]:
        if self.awi.is_bankrupt(partner):
            return None
        profile = self.profiles.get(cfp.product)
        if profile is None:
            return None
        if profile.cv == 0:
            alpha_u, alpha_q = profile.alpha_u, profile.alpha_q
        else:
            alpha_u, alpha_q = tuple(
                dirichlet((profile.alpha_u, profile.alpha_q), size=1)[0]
            )
        beta_u = pos_gauss(profile.beta_u, profile.cv)
        tau_u = pos_gauss(profile.tau_u, profile.cv)
        tau_q = pos_gauss(profile.tau_q, profile.cv)
        ufun = WeightedUtilityFunction(
            ufuns=[
                MappingUtilityFunction(
                    mapping=(lambda x: 1 - x[UNIT_PRICE] ** tau_u / beta_u),
                    issues=cfp.issues,
                ),
                MappingUtilityFunction(
                    mapping=functools.partial(
                        JustInTimeConsumer._qufun, tau=tau_q, profile=profile
                    ),
                    issues=cfp.issues,
                ),
            ],
            weights=[alpha_u, alpha_q],
            name=self.name + "_" + partner[:4],
            issues=cfp.issues,
        )
        negotiator = self.negotiator_type(name=self.name + "*" + partner[:4], ufun=ufun)
        # negotiator.utility_function = ufun
        return negotiator

    def set_renegotiation_agenda(
        self, contract: Contract, breaches: List[Breach]
    ) -> Optional[RenegotiationRequest]:
        """
        Received by partners in ascending order of their total breach levels in order to set the
        renegotiation agenda when contract execution fails

        Args:

            contract: The contract that was breached about which re-negotiation is offered
            breaches: The list of breaches by all parties for the breached contract.

        Returns:

            None if renegotiation is not to be started, otherwise a re-negotiation agenda.

        """
        return None

    def respond_to_renegotiation_request(
        self, contract: Contract, breaches: List[Breach], agenda: RenegotiationRequest
    ) -> Optional[Negotiator]:
        """
        Called to respond to a renegotiation request

        Args:

            agenda: Renegotiation agenda (issues to renegotiate about).
            contract: The contract that was breached
            breaches: All breaches on that contract

        Returns:

            None to refuse to enter the negotiation, otherwise, a negotiator to use for this negotiation.

        """
        return None

    def confirm_loan(self, loan: Loan, bankrupt_if_rejected: bool) -> bool:
        """called by the world manager to confirm a loan if needed by the buyer of a contract that is about to be
        breached"""
        return bankrupt_if_rejected

    def sign_contract(self, contract: Contract) -> Optional[str]:
        if contract is None:
            return None
        cfp: CFP = contract.annotation["cfp"]
        agreement = contract.agreement  # type: ignore
        schedule = self.profiles[cfp.product].schedule_at(agreement["time"])
        if schedule - agreement["quantity"] < 0:
            return None
        return self.id

    def on_contract_signed(self, contract: Contract):
        if contract is None:
            return
        cfp: CFP = contract.annotation["cfp"]
        agreement = contract.agreement  # type: ignore
        self.secured_quantities[cfp.product] += agreement["quantity"]
        old_quantity = self.profiles[cfp.product].schedule_at(agreement["time"])
        new_quantity = old_quantity - agreement["quantity"]
        t = agreement["time"]
        self.profiles[cfp.product].set_schedule_at(
            time=t, value=new_quantity, n_steps=self.awi.n_steps
        )
        if self.immediate_cfp_update and new_quantity != old_quantity:
            self.register_product_cfps(
                p=cfp.product, t=t, profile=self.profiles[cfp.product]
            )
        for negotiation in self._running_negotiations.values():
            self.notify(
                negotiation.negotiator, Notification(type="ufun_modified", data=None)
            )


class ScheduleDrivenConsumer(Consumer):
    """Consumer class"""

    def on_contract_executed(self, contract: Contract) -> None:
        pass

    def on_contract_breached(
        self, contract: Contract, breaches: List[Breach], resolution: Optional[Contract]
    ) -> None:
        pass

    def on_inventory_change(self, product: int, quantity: int, cause: str) -> None:
        pass

    def on_cash_transfer(self, amount: float, cause: str) -> None:
        pass

    def on_new_report(self, report: FinancialReport):
        pass

    def on_neg_request_rejected(self, req_id: str, by: Optional[List[str]]):
        pass

    def on_neg_request_accepted(
        self, req_id: str, mechanism: NegotiatorMechanismInterface
    ):
        pass

    def on_negotiation_failure(
        self,
        partners: List[str],
        annotation: Dict[str, Any],
        mechanism: NegotiatorMechanismInterface,
        state: MechanismState,
    ) -> None:
        pass

    def on_negotiation_success(
        self, contract: Contract, mechanism: NegotiatorMechanismInterface
    ) -> None:
        pass

    def on_contract_cancelled(self, contract: Contract, rejectors: List[str]) -> None:
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
        return True

    def on_remove_cfp(self, cfp: "CFP"):
        pass

    MAX_UNIT_PRICE = 100.0
    RELATIVE_MAX_PRICE = 1.5

    def __init__(
        self,
        profiles: Dict[int, ConsumptionProfile] = None,
        negotiator_type=DEFAULT_NEGOTIATOR,
        consumption_horizon: Optional[int] = 20,
        immediate_cfp_update: bool = True,
        name=None,
    ):
        super().__init__(name=name)
        self.negotiator_type = get_class(negotiator_type, scope=globals())
        self.profiles: Dict[int, ConsumptionProfile] = defaultdict(ConsumptionProfile)
        self.secured_quantities: Dict[int, int] = defaultdict(int)
        if profiles is not None:
            self.set_profiles(profiles=profiles)
        self.consumption_horizon = consumption_horizon
        self.immediate_cfp_update = immediate_cfp_update

    def on_new_cfp(self, cfp: "CFP") -> None:
        pass  # consumers never respond to CFPs

    def init(self):
        if self.consumption_horizon is None:
            self.consumption_horizon = self.awi.n_steps
        self.awi.register_interest(list(self.profiles.keys()))

    def set_profiles(self, profiles: Dict[int, ConsumptionProfile]):
        self.profiles = defaultdict(ConsumptionProfile)
        if profiles is not None:
            for k, v in profiles.items():
                self.profiles[k] = v
        self.secured_quantities = defaultdict(int)
        if profiles is not None:
            for k, v in profiles.items():
                self.secured_quantities[k] = 0

    def register_product_cfps(self, p: int, t: int, profile: ConsumptionProfile):
        current_schedule = profile.schedule_at(t)
        product = self.products[p]
        awi: SCMLAWI = self.awi
        if current_schedule <= 0:
            awi.bb_remove(
                section="cfps",
                query={"publisher": self.id, "time": t, "product_index": p},
            )
            return
        max_price = (
            JustInTimeConsumer.RELATIVE_MAX_PRICE * product.catalog_price
            if product.catalog_price is not None
            else JustInTimeConsumer.MAX_UNIT_PRICE
        )
        cfps = awi.bb_query(
            section="cfps", query={"publisher": self.id, "time": t, "product": p}
        )
        if cfps is not None and len(cfps) > 0:
            for _, cfp in cfps.items():
                if cfp.max_quantity != current_schedule:
                    cfp = CFP(
                        is_buy=True,
                        publisher=self.id,
                        product=p,
                        time=t,
                        unit_price=(0, max_price),
                        quantity=(1, current_schedule),
                    )
                    awi.bb_remove(
                        section="cfps",
                        query={"publisher": self.id, "time": t, "product": p},
                    )
                    awi.register_cfp(cfp)
                    break
        else:
            cfp = CFP(
                is_buy=True,
                publisher=self.id,
                product=p,
                time=t,
                unit_price=(0, max_price),
                quantity=(1, current_schedule),
            )
            awi.register_cfp(cfp)

    def step(self):
        if self.consumption_horizon is None:
            horizon = self.awi.n_steps
        else:
            horizon = min(
                self.awi.current_step + self.consumption_horizon + 1, self.awi.n_steps
            )
        for p, profile in self.profiles.items():
            for t in range(
                self.awi.current_step, horizon
            ):  # + self.transportation_delay
                self.register_product_cfps(p=p, t=t, profile=profile)

    def confirm_contract_execution(self, contract: Contract) -> bool:
        return True

    @staticmethod
    def _qufun(outcome: Dict[str, Any], tau: float, profile: ConsumptionProfile):
        """The ufun value for quantity"""
        q, t = outcome["quantity"], outcome["time"]
        y = profile.schedule_within(t)
        o = profile.overconsumption
        u = profile.underconsumption
        if q == 0 and y != 0:
            return 0.0
        if y <= 0:
            result = -o * ((q - y) ** tau)
        elif q > y:
            result = -o * (((q - y) / y) ** tau)
        elif q < y:
            result = -u * (((y - q) / y) ** tau)
        else:
            result = 1.0
        result = math.exp(result)
        if isinstance(result, complex):
            result = result.real
        if result is None:
            result = float("-inf")
        return result

    def respond_to_negotiation_request(
        self, cfp: "CFP", partner: str
    ) -> Optional[Negotiator]:
        if self.awi.is_bankrupt(partner):
            return None
        profile = self.profiles.get(cfp.product)
        if profile is None:
            return None
        if profile.cv == 0:
            alpha_u, alpha_q = profile.alpha_u, profile.alpha_q
        else:
            alpha_u, alpha_q = tuple(
                dirichlet((profile.alpha_u, profile.alpha_q), size=1)[0]
            )
        beta_u = pos_gauss(profile.beta_u, profile.cv)
        tau_u = pos_gauss(profile.tau_u, profile.cv)
        tau_q = pos_gauss(profile.tau_q, profile.cv)
        ufun = WeightedUtilityFunction(
            ufuns=[
                MappingUtilityFunction(
                    mapping=lambda x: 1 - x[UNIT_PRICE] ** tau_u / beta_u,
                    issues=cfp.issues,
                ),
                MappingUtilityFunction(
                    mapping=functools.partial(
                        JustInTimeConsumer._qufun, tau=tau_q, profile=profile
                    ),
                    issues=cfp.issues,
                ),
            ],
            weights=[alpha_u, alpha_q],
            name=self.name + "_" + partner[:4],
            issues=cfp.issues,
        )
        negotiator = self.negotiator_type(name=self.name + "*" + partner[:4], ufun=ufun)
        # negotiator.utility_function = ufun
        return negotiator

    def set_renegotiation_agenda(
        self, contract: Contract, breaches: List[Breach]
    ) -> Optional[RenegotiationRequest]:
        """
        Received by partners in ascending order of their total breach levels in order to set the
        renegotiation agenda when contract execution fails

        Args:

            contract: The contract that was breached about which re-negotiation is offered
            breaches: The list of breaches by all parties for the breached contract.

        Returns:

            None if renegotiation is not to be started, otherwise a re-negotiation agenda.

        """
        return None

    def respond_to_renegotiation_request(
        self, contract: Contract, breaches: List[Breach], agenda: RenegotiationRequest
    ) -> Optional[Negotiator]:
        """
        Called to respond to a renegotiation request

        Args:

            agenda: Renegotiation agenda (issues to renegotiate about).
            contract: The contract that was breached
            breaches: All breaches on that contract

        Returns:

            None to refuse to enter the negotiation, otherwise, a negotiator to use for this negotiation.

        """
        return None

    def confirm_loan(self, loan: Loan, bankrupt_if_rejected: bool) -> bool:
        """called by the world manager to confirm a loan if needed by the buyer of a contract that is about to be
        breached"""
        return bankrupt_if_rejected

    def sign_contract(self, contract: Contract) -> Optional[str]:
        if contract is None:
            return None
        cfp: CFP = contract.annotation["cfp"]
        agreement = contract.agreement  # type: ignore
        schedule = self.profiles[cfp.product].schedule_at(agreement["time"])
        if schedule - agreement["quantity"] < 0:
            return None
        return self.id

    def on_contract_signed(self, contract: Contract):
        if contract is None:
            return
        cfp: CFP = contract.annotation["cfp"]
        agreement = contract.agreement  # type: ignore
        self.secured_quantities[cfp.product] += agreement["quantity"]
        old_quantity = self.profiles[cfp.product].schedule_at(agreement["time"])
        new_quantity = old_quantity - agreement["quantity"]
        t = agreement["time"]
        self.profiles[cfp.product].set_schedule_at(
            time=t, value=new_quantity, n_steps=self.awi.n_steps
        )
        if self.immediate_cfp_update and new_quantity != old_quantity:
            self.register_product_cfps(
                p=cfp.product, t=t, profile=self.profiles[cfp.product]
            )
        for negotiation in self._running_negotiations.values():
            self.notify(
                negotiation.negotiator, Notification(type="ufun_modified", data=None)
            )
