import warnings
from collections import defaultdict

from scml.scml2019.common import ProductionReport
from scml.scml2019.common import SCMLAgreement, INVALID_UTILITY, ProductionFailure
from scml.scml2019.consumers import ConsumptionProfile
from scml.scml2019.schedulers import Scheduler, ScheduleInfo, GreedyScheduler
from scml.scml2019.simulators import FactorySimulator, FastFactorySimulator
from scml.scml2019.simulators import temporary_transaction
from .builtins import NegotiatorUtility
from .builtins import (
    PessimisticNegotiatorUtility,
    OptimisticNegotiatorUtility,
    AveragingNegotiatorUtility,
)

if True:
    from typing import Dict, Any, Callable, Collection, Type, List, Optional, Union

__all__ = ["RaptFactoryManager"]
import functools
import itertools
from typing import TYPE_CHECKING

from numpy.random import dirichlet

from negmas import AgentMechanismInterface, MechanismState
from scml.scml2019.common import DEFAULT_NEGOTIATOR
from negmas.events import Notification
from negmas.helpers import get_class
from negmas.negotiators import Negotiator
from negmas.situated import Contract, Breach
from negmas.situated import RenegotiationRequest
from negmas.utilities import ComplexWeightedUtilityFunction, MappingUtilityFunction
from scml.scml2019.common import CFP
from scml.scml2019.helpers import pos_gauss

from .builtins import GreedyFactoryManager
import math
import numpy as np

if True:  #
    from typing import Dict, Any, List, Optional, Union
    from scml.scml2019.common import Loan

if TYPE_CHECKING:
    from scml.scml2019.awi import SCMLAWI

from scml.scml2019.consumers import ScheduleDrivenConsumer


class MyScheduleDrivenConsumer(ScheduleDrivenConsumer):
    "MyScheduleDrivenConsumer"

    def __init__(
        self,
        profiles: Dict[int, ConsumptionProfile] = None,
        negotiator_type=DEFAULT_NEGOTIATOR,
        consumption_horizon: Optional[int] = 20,
        immediate_cfp_update: bool = True,
        name=None,
        sp={},
    ):
        super().__init__(name=name)
        self.negotiator_type = get_class(negotiator_type, scope=globals())
        self.profiles: Dict[int, ConsumptionProfile] = dict()
        self.secured_quantities: Dict[int, int] = dict()
        if profiles is not None:
            self.set_profiles(profiles=profiles)
        self.consumption_horizon = consumption_horizon
        self.immediate_cfp_update = immediate_cfp_update
        self.sp = sp

    def on_new_cfp(self, cfp: "CFP") -> None:
        pass  # consumers never respond to CFPs

    def init(self):
        if self.consumption_horizon is None:
            self.consumption_horizon = self.awi.n_steps
        self.awi.register_interest(list(self.profiles.keys()))

    def set_profiles(self, profiles: Dict[int, ConsumptionProfile]):
        self.profiles = profiles if profiles is not None else dict()
        self.secured_quantities = dict(zip(profiles.keys(), itertools.repeat(0)))

    def register_product_cfps(
        self, p: int, t: int, profile: ConsumptionProfile, sp=dict()
    ):
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
            ScheduleDrivenConsumer.RELATIVE_MAX_PRICE * product.catalog_price
            if product.catalog_price is not None
            else ScheduleDrivenConsumer.MAX_UNIT_PRICE
        )
        if sp.get(p) != None and sp[p] > product.catalog_price:
            max_price = ScheduleDrivenConsumer.RELATIVE_MAX_PRICE * sp[p]
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
                self.register_product_cfps(p=p, t=t, profile=profile, sp=self.sp)

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
            result = -1000.0
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
        ufun = ComplexWeightedUtilityFunction(
            ufuns=[
                MappingUtilityFunction(
                    mapping=lambda x: 1 - x["unit_price"] ** tau_u / beta_u
                ),
                MappingUtilityFunction(
                    mapping=functools.partial(
                        ScheduleDrivenConsumer._qufun, tau=tau_q, profile=profile
                    )
                ),
            ],
            weights=[alpha_u, alpha_q],
            name=self.name + "_" + partner,
        )
        ufun.reserved_value = -1500
        # ufun = normalize(, outcomes=cfp.outcomes, infeasible_cutoff=-1500)
        negotiator = self.negotiator_type(name=self.name + "*" + partner, ufun=ufun)
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

            agenda: Renegotiatio agenda (issues to renegotiate about).
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
                p=cfp.product, t=t, profile=self.profiles[cfp.product], sp={}
            )
        for negotiation in self._running_negotiations.values():
            self.notify(
                negotiation.negotiator, Notification(type="ufun_modified", data=None)
            )


class RaptFactoryManager(GreedyFactoryManager):
    """My factory manager"""

    def on_production_failure(self, failures: List[ProductionFailure]) -> None:
        pass

    def on_production_success(self, reports: List[ProductionReport]) -> None:
        pass

    def confirm_loan(self, loan: Loan, bankrupt_if_rejected: bool) -> bool:
        return bankrupt_if_rejected

    def confirm_contract_execution(self, contract: Contract) -> bool:
        return True

    def set_renegotiation_agenda(
        self, contract: Contract, breaches: List[Breach]
    ) -> Optional[RenegotiationRequest]:
        return None

    def respond_to_renegotiation_request(
        self, contract: Contract, breaches: List[Breach], agenda: RenegotiationRequest
    ) -> Optional[Negotiator]:
        return None

    def __init__(
        self,
        name=None,
        simulator_type: Union[str, Type[FactorySimulator]] = FastFactorySimulator,
        scheduler_type: Union[str, Type[Scheduler]] = GreedyScheduler,
        scheduler_params: Optional[Dict[str, Any]] = None,
        optimism: float = 0.0,
        negotiator_type: Union[str, Type[Negotiator]] = DEFAULT_NEGOTIATOR,
        negotiator_params: Optional[Dict[str, Any]] = None,
        n_retrials=5,
        use_consumer=True,
        reactive=False,
        sign_only_guaranteed_contracts=True,
        riskiness=0.0,
        max_insurance_premium: float = 0.1,
        my_reports=[],
        not_fm=[],
        mean_score: float = 1000.0,
        bought={},
        sold={},
    ):
        self.my_reports = []
        self.not_fm = []
        self.mean_score: float = 1000.0
        self.bought = bought
        self.sold = sold
        super().__init__(name=name, simulator_type=simulator_type)
        self.negotiator_type = get_class(negotiator_type, scope=globals())
        self.negotiator_params = (
            negotiator_params if negotiator_params is not None else {}
        )
        self.optimism = optimism
        self.ufun_factory: Union[
            Type[NegotiatorUtility], Callable[[Any, Any], NegotiatorUtility]
        ]
        if optimism < 1e-6:
            self.ufun_factory = PessimisticNegotiatorUtility
        elif optimism > 1 - 1e-6:
            self.ufun_factory = OptimisticNegotiatorUtility
        else:
            self.ufun_factory: NegotiatorUtility = lambda agent, annotation: AveragingNegotiatorUtility(
                agent=agent, annotation=annotation, optimism=self.optimism
            )
        if max_insurance_premium < 0.0:
            warnings.warn(
                f"Negative max insurance ({max_insurance_premium}) is deprecated. Set max_insurance_premium = inf "
                f"for always buying and max_insurance_premium = 0.0 for never buying. Will continue assuming inf"
            )
            max_insurance_premium = float("inf")
        self.max_insurance_premium = max_insurance_premium
        self.n_retrials = n_retrials
        self.n_neg_trials: Dict[str, int] = defaultdict(int)
        self.consumer = None
        self.use_consumer = use_consumer
        self.reactive = reactive
        self.sign_only_guaranteed_contracts = sign_only_guaranteed_contracts
        self.contract_schedules: Dict[str, ScheduleInfo] = {}
        self.riskiness = riskiness
        self.negotiation_margin = int(round(n_retrials * max(0.0, 1.0 - riskiness)))
        self.scheduler_type: Type[Scheduler] = get_class(
            scheduler_type, scope=globals()
        )
        self.scheduler: Scheduler = None
        self.scheduler_params: Dict[
            str, Any
        ] = scheduler_params if scheduler_params is not None else {}

    def total_utility(self, contracts: Collection[Contract] = ()) -> float:
        """Calculates the total utility for the agent of a collection of contracts"""
        if self.scheduler is None:
            raise ValueError("Cannot calculate total utility without a scheduler")
        min_concluded_at = self.awi.current_step
        min_sign_at = min_concluded_at + self.awi.default_signing_delay
        with temporary_transaction(self.scheduler):
            schedule = self.scheduler.schedule(
                contracts=contracts,
                assume_no_further_negotiations=False,
                ensure_storage_for=self.transportation_delay,
                start_at=min_sign_at,
            )
        if not schedule.valid:
            return INVALID_UTILITY
        return schedule.final_balance

    def init(self):
        self.negotiation_margin = max(
            self.negotiation_margin,
            int(round(len(self.products) * max(0.0, 1.0 - self.riskiness))),
        )
        if self.use_consumer:
            # @todo add the parameters of the consumption profile as parameters of the greedy factory manager
            profiles = dict(
                zip(
                    self.consuming.keys(),
                    (
                        ConsumptionProfile(schedule=[_] * self.awi.n_steps)
                        for _ in itertools.repeat(0)
                    ),
                )
            )
            self.consumer: MyScheduleDrivenConsumer = MyScheduleDrivenConsumer(
                profiles=profiles,
                consumption_horizon=self.awi.n_steps,
                immediate_cfp_update=True,
                name=self.name,
            )
            self.consumer.id = self.id
            self.consumer.awi = self.awi
            self.consumer.init_()
        self.scheduler = self.scheduler_type(
            manager_id=self.id,
            awi=self.awi,
            max_insurance_premium=self.max_insurance_premium,
            **self.scheduler_params,
        )
        self.scheduler.init(
            simulator=self.simulator,
            products=self.products,
            processes=self.processes,
            producing=self.producing,
            profiles=self.compiled_profiles,
        )

    def respond_to_negotiation_request(  ###
        self, cfp: "CFP", partner: str
    ) -> Optional[Negotiator]:
        if self.awi.is_bankrupt(partner):
            return None
        if self.use_consumer:  ###

            reports = self.awi.reports_for(cfp.publisher)
            if not reports:
                "report not exist"
            else:
                #                 logger.debug('reports_for')
                #                     logger.debug(reports)
                if reports[-1].cash >= self.mean_score:
                    return None

            return self.consumer.respond_to_negotiation_request(
                cfp=cfp, partner=partner
            )
        else:
            ufun_ = self.ufun_factory(
                self, self._create_annotation(cfp=cfp, partner=partner)
            )
            ufun_.reserved_value = (
                cfp.money_resolution if cfp.money_resolution is not None else 0.1
            )
            neg = self.negotiator_type(
                name=self.name + "*" + partner, **self.negotiator_params, ufun=ufun_
            )
            return neg

    def on_negotiation_success(  ###
        self, contract: Contract, mechanism: AgentMechanismInterface
    ):
        con_cfp = contract.annotation["cfp"]
        con_agr = contract.agreement
        if {con_cfp.is_buy == True and con_cfp.publisher == self.name} or {
            con_cfp.is_buy == False and con_cfp.publisher != self.agent
        }:
            if self.bought.get(con_cfp.product) == None:
                self.bought.setdefault(con_cfp.product, con_agr["unit_price"])
            else:
                self.bought[con_cfp.product] = con_agr["unit_price"]
        if {con_cfp.is_buy == True and con_cfp.publisher != self.name} or {
            con_cfp.is_buy == False and con_cfp.publisher == self.agent
        }:
            if self.sold.get(con_cfp.product) == None:
                self.sold.setdefault(con_cfp.product, con_agr["unit_price"])
            else:
                self.sold[con_cfp.product] = con_agr["unit_price"]

        if self.use_consumer:

            self.consumer.on_negotiation_success(contract, mechanism)

    def on_negotiation_failure(  ###
        self,
        partners: List[str],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
        state: MechanismState,
    ) -> None:
        if self.use_consumer:
            self.consumer.on_negotiation_failure(partners, annotation, mechanism, state)
        cfp = annotation["cfp"]
        thiscfp = self.awi.bb_query(section="cfps", query=cfp.id, query_keys=True)
        if (
            cfp.publisher != self.id
            and thiscfp is not None
            and len(thiscfp) > 0
            and self.n_neg_trials[cfp.id] < self.n_retrials
        ):
            self.awi.logdebug(f"Renegotiating {self.n_neg_trials[cfp.id]} on {cfp}")
            self.n_neg_trials[cfp.id] += 1
            self.on_new_cfp(cfp=annotation["cfp"])

    def _execute_schedule(
        self, schedule: ScheduleInfo, contract: Contract
    ) -> None:  ###contact
        if self.simulator is None:
            raise ValueError("No factory simulator is defined")
        awi: SCMLAWI = self.awi
        total = contract.agreement["unit_price"] * contract.agreement["quantity"]
        product = contract.annotation["cfp"].product
        if contract.annotation["buyer"] == self.id:
            self.simulator.buy(
                product=product,
                quantity=contract.agreement["quantity"],
                price=total,
                t=contract.agreement["time"],
            )
            if total <= 0 or self.max_insurance_premium <= 0.0 or contract is None:
                return
            relative_premium = awi.evaluate_insurance(contract=contract)
            if relative_premium is None:
                return
            premium = relative_premium * total
            if relative_premium <= self.max_insurance_premium:
                self.awi.logdebug(
                    f"{self.name} buys insurance @ {premium:0.02} ({relative_premium:0.02%}) for {str(contract)}"
                )
                awi.buy_insurance(contract=contract)
                self.simulator.pay(premium, self.awi.current_step)
            return
        # I am a seller
        self.simulator.sell(
            product=product,
            quantity=contract.agreement["quantity"],
            price=total,
            t=contract.agreement["time"],
        )
        for job in schedule.jobs:
            if job.action == "run":
                awi.schedule_job(job, contract=contract)
            elif job.action == "stop":
                awi.stop_production(
                    line=job.line,
                    step=job.time,
                    contract=contract,
                    override=job.override,
                )
            else:
                awi.schedule_job(job, contract=contract)
            self.simulator.schedule(job=job, override=False)
        for need in schedule.needs:
            if need.quantity_to_buy <= 0:
                continue
            product_id = need.product
            # self.simulator.reserve(product=product_id, quantity=need.quantity_to_buy, t=need.step)
            if self.use_consumer:
                self.consumer.profiles[product_id].schedule[
                    need.step
                ] += need.quantity_to_buy
                self.consumer.register_product_cfps(
                    p=product_id,
                    t=need.step,
                    profile=self.consumer.profiles[product_id],
                    sp=self.sold,  ###
                )
                continue
            #             logger.debug('seller')
            product = self.products[product_id]
            if product.catalog_price is None:
                if self.sold.get(product_id) == None:
                    price_range = (0.0, 100.0)
                else:
                    price_range = (0.5, 1.5 * self.sold(product_id))
            else:
                #                 logger.debug(product)
                #                 logger.debug(product.catalog_price)
                if self.sold.get(product_id) != None:
                    product_price = max(self.sold(product_id), product.catalog_price)
                price_range = (0.5, 1.5 * product_price)
            # @todo check this. This error is raised sometimes
            if need.step < awi.current_step:
                continue
                # raise ValueError(f'need {need} at {need.step} while running at step {awi.current_step}')
            time = (
                need.step
                if self.max_storage is not None
                else (awi.current_step, need.step)
            )
            cfp = CFP(
                is_buy=True,
                publisher=self.id,
                product=product_id,
                time=time,
                unit_price=price_range,
                quantity=(1, int(1.1 * need.quantity_to_buy)),
            )
            awi.register_cfp(cfp)

    def sign_contract(self, contract: Contract):  ###contact
        if any(self.awi.is_bankrupt(partner) for partner in contract.partners):
            return None
        signature = self.id
        with temporary_transaction(self.scheduler):
            schedule = self.scheduler.schedule(
                assume_no_further_negotiations=False,
                contracts=[contract],
                ensure_storage_for=self.transportation_delay,
                start_at=self.awi.current_step + 1,
            )

        if self.sign_only_guaranteed_contracts and (
            not schedule.valid or len(schedule.needs) > 1
        ):
            self.awi.logdebug(
                f"{self.name} refused to sign contract {contract.id} because it cannot be scheduled"
            )
            return None
        # if schedule.final_balance <= self.simulator.final_balance:
        #     self.awi.logdebug(f'{self.name} refused to sign contract {contract.id} because it is not expected '
        #                       f'to lead to profit')
        #     return None
        if schedule.valid:
            profit = schedule.final_balance - self.simulator.final_balance
            self.awi.logdebug(
                f"{self.name} singing contract {contract.id} expecting "
                f'{-profit if profit < 0 else profit} {"loss" if profit < 0 else "profit"}'
            )
        else:
            self.awi.logdebug(
                f"{self.name} singing contract {contract.id} expecting breach"
            )
            return None

        self.contract_schedules[contract.id] = schedule
        return signature

    def on_contract_signed(self, contract: Contract):  ###contact
        if contract.annotation["buyer"] == self.id and self.use_consumer:
            self.consumer.on_contract_signed(contract)
        schedule = self.contract_schedules[contract.id]
        if schedule is not None and schedule.valid:
            self._execute_schedule(schedule=schedule, contract=contract)
        if contract.annotation["buyer"] != self.id or not self.use_consumer:
            for negotiation in self._running_negotiations.values():
                self.notify(
                    negotiation.negotiator,
                    Notification(type="ufun_modified", data=None),
                )

    def _process_buy_cfp(self, cfp: "CFP") -> None:
        if cfp.publisher == self.id:
            return
        if self.awi.is_bankrupt(cfp.publisher):
            return
        if self.simulator is None or not self.can_expect_agreement(
            cfp=cfp, margin=self.negotiation_margin
        ):
            return
        if not self.can_produce(cfp=cfp):
            return
        neg = self.negotiator_type(
            name=self.name + ">" + cfp.publisher, **self.negotiator_params
        )
        ufun = self.ufun_factory(self, self._create_annotation(cfp=cfp))
        ufun.reserved_value = (
            cfp.money_resolution if cfp.money_resolution is not None else 0.1
        )
        self.request_negotiation(negotiator=neg, cfp=cfp, ufun=ufun)

    def _process_sell_cfp(self, cfp: "CFP"):
        if self.awi.is_bankrupt(cfp.publisher):
            return None
        if self.use_consumer:
            self.consumer.on_new_cfp(cfp=cfp)

    def on_new_cfp(self, cfp: "CFP") -> None:
        if not self.reactive:
            return
        if cfp.satisfies(
            query={"is_buy": True, "products": list(self.producing.keys())}
        ):
            self._process_buy_cfp(cfp)
        if cfp.satisfies(
            query={"is_buy": False, "products": list(self.consuming.keys())}
        ):
            self._process_sell_cfp(cfp)

    def step(self):
        if self.use_consumer:
            self.consumer.step()
        if self.reactive:
            return
        if not self.not_fm:
            reports = self.awi.bb_query(section="reports_agent", query=None)
            if reports:
                for report in reports.values():
                    for rep in report:
                        if rep.cash <= 0.0:
                            self.not_fm.append(rep.agent)
        else:
            cashes = []
            reports = self.awi.bb_query(section="reports_agent", query=None)
            if reports:
                for report in reports.values():
                    for rep in report:
                        if rep.agent not in self.not_fm:
                            cashes.append(rep.cash)
                self.mean_score = np.mean(cashes)

        keys = self.producing.keys()
        if not keys:
            return

        for key in keys:
            key_cfps = self.awi.bb_query(
                section="cfps", query={"products": [key], "is_buy": True}  ###unit_price
            )
            key_cfps = key_cfps.values()
            sort_key_cfp = sorted(
                key_cfps, key=lambda cfp: cfp.unit_price, reverse=True
            )
            #             sort_key_cfp = sorted(key_cfps, key=lambda cfp: cfp.time)
            half_sort_key_cfp = sort_key_cfp[0 : int(len(sort_key_cfp) / 4)]
            #             logger.debug(half_sort_key_cfp)
            for cfp in sort_key_cfp:
                reports = self.awi.reports_for(cfp.publisher)
                if not reports:
                    self._process_buy_cfp(cfp)
                else:
                    #                     logger.debug('reports_for')
                    #                     logger.debug(reports)

                    if reports[-1].cash <= self.mean_score:
                        self._process_buy_cfp(cfp)

        keys = self.consuming.keys()
        if not keys:
            return

        for key in keys:
            key_cfps = self.awi.bb_query(
                section="cfps",
                query={"products": [key], "is_buy": False},  ###unit_price
            )
            key_cfps = key_cfps.values()
            sort_key_cfp = sorted(key_cfps, key=lambda cfp: cfp.unit_price)
            #             sort_key_cfp = sorted(key_cfps, key=lambda cfp: cfp.time)
            half_sort_key_cfp = sort_key_cfp[0 : int(len(sort_key_cfp) / 4)]
            #             logger.debug(half_sort_key_cfp)
            for cfp in sort_key_cfp:
                reports = self.awi.reports_for(cfp.publisher)
                if not reports:
                    self._process_sell_cfp(cfp)
                else:
                    #                     logger.debug('reports_for')
                    #                     logger.debug(reports)

                    if reports[-1].cash <= self.mean_score:
                        self._process_sell_cfp(cfp)

    def can_produce(self, cfp: CFP, assume_no_further_negotiations=False) -> bool:
        """Whether or not we can produce the required item in time"""
        if cfp.product not in self.producing.keys():
            return False
        agreement = SCMLAgreement(
            time=cfp.max_time, unit_price=cfp.max_unit_price, quantity=cfp.min_quantity
        )
        min_concluded_at = self.awi.current_step + 1 - int(self.immediate_negotiations)
        min_sign_at = min_concluded_at + self.awi.default_signing_delay
        if cfp.max_time < min_sign_at + 1:  # 1 is minimum time to produce the product
            return False
        with temporary_transaction(self.scheduler):
            schedule = self.scheduler.schedule(
                contracts=[
                    Contract(
                        partners=[self.id, cfp.publisher],
                        agreement=agreement,
                        annotation=self._create_annotation(cfp=cfp),
                        issues=cfp.issues,
                        signed_at=min_sign_at,
                        concluded_at=min_concluded_at,
                    )
                ],
                ensure_storage_for=self.transportation_delay,
                assume_no_further_negotiations=assume_no_further_negotiations,
                start_at=min_sign_at,
            )
        return schedule.valid and self.can_secure_needs(
            schedule=schedule, step=self.awi.current_step
        )

    def can_secure_needs(self, schedule: ScheduleInfo, step: int):
        """
        Finds if it is possible in principle to arrange these needs at the given time.

        Args:
            schedule:
            step:

        Returns:

        """
        needs = schedule.needs
        if len(needs) < 1:
            return True
        for need in needs:
            if need.quantity_to_buy > 0 and need.step < step + 1 - int(
                self.immediate_negotiations
            ):  # @todo check this
                return False
        return True


TotalUtilityFun = Callable[[Collection[Contract]], float]
