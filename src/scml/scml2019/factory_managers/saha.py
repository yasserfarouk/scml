"""
This module implements a factory manager for the SCM league of ANAC 2019 competition. This advanced version has all
callbacks. Please refer to the [http://www.yasserm.com/scml/scml.pdf](game description)
for all the callbacks.

Your agent can learn about the state of the world and itself by accessing properties in the AWI it has. For example::

self.awi.n_steps  # gives the number of simulation steps

You can access the state of your factory as::

self.awi.state

Your agent can act in the world by calling methods in the AWI it has. For example: >>> self.awi.register_cfp(cfp)  # registers a new CFP

You can access the full list of these capabilities on the documentation.

- For properties/methods available only to SCM agents, check the list here:
https://negmas.readthedocs.io/en/latest/api/scml.scml2019.SCMLAWI.html

- For properties/methods available to all kinds of agents in all kinds of worlds, check the list here:
https://negmas.readthedocs.io/en/latest/api/negmas.situated.AgentWorldInterface.html

The SCML2019Agent class itself has some helper properties/methods that internally call the AWI. These include:

- request_negotiation(): Generates a unique identifier for this negotiation request and passes it to the AWI through a
                         call of awi.request_negotiation(). It is recommended to use this method always to request
                         negotiations. This way, you can access internal lists of requested_negotiations, and
                         running_negotiations. If on the other hand you use awi.request_negotiation(), these internal
                         lists will not be updated and you will have to keep track to requested and running negotiations
                         manually if you need to use them.
- can_expect_agreement(): Checks if it is possible in principle to get an agreement on this CFP by the time it becomes
                          executable
- products, processes: shortcuts to awi.products and awi.processes
"""
import itertools
import math
from collections import defaultdict
from operator import attrgetter

from negmas import (
    Contract,
    Breach,
    RenegotiationRequest,
    Negotiator,
    AgentMechanismInterface,
)
from negmas import MechanismState
from negmas.events import Notification
from negmas.helpers import get_class
from negmas.sao import AspirationNegotiator
from negmas.utilities import normalize
from typing import Dict, Any, Callable, Collection, Type, List, Optional, Union

from scml.scml2019.awi import SCMLAWI
from scml.scml2019.common import (
    SCMLAgreement,
    INVALID_UTILITY,
    CFP,
    Loan,
    ProductionFailure,
)
from scml.scml2019.consumers import ScheduleDrivenConsumer, ConsumptionProfile
from scml.scml2019.schedulers import Scheduler, ScheduleInfo, GreedyScheduler
from scml.scml2019.simulators import (
    FactorySimulator,
    FastFactorySimulator,
    temporary_transaction,
)
from .builtins import (
    DoNothingFactoryManager,
    NegotiatorUtility,
    PessimisticNegotiatorUtility,
    OptimisticNegotiatorUtility,
    AveragingNegotiatorUtility,
)


class ProductData:
    id: int
    stock: int
    minstock: int
    SellThreshold: int
    BuyThreshold: int
    MinPrice: int
    MaxPrice: int
    HistoryMin: []
    HistoryMax: []
    # HistoryPublishers: []
    Asked: int
    retracted: int
    lastSigned: int
    source_of: int
    transformable: bool
    prevMax: int
    prevMin: int

    def __init__(self, id, SellThreshold, BuyThreshold, MinPrice, MaxPrice, minstock=0):
        self.id = id
        self.stock = 0
        self.minstock = minstock
        self.SellThreshold = SellThreshold
        self.BuyThreshold = BuyThreshold
        self.MinPrice = MinPrice
        self.MaxPrice = MaxPrice
        self.HistoryMin = []
        self.HistoryMax = []
        # self.HistoryPublishers = []
        self.Asked = -10
        self.retracted = -10
        self.source_of = -1
        self.transformable = False
        self.prevMax = MaxPrice
        self.prevMin = MinPrice
        self.lastSigned = -1


class SAHAFactoryManager(DoNothingFactoryManager):
    """The default factory manager that will be implemented by the committee of ANAC-SCML 2019"""

    def on_production_failure(self, failures: List[ProductionFailure]) -> None:
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
        negotiator_type: Union[
            str, Type[Negotiator]
        ] = "negmas.sao.AspirationNegotiator",
        negotiator_params: Optional[Dict[str, Any]] = None,
        n_retrials=5,
        use_consumer=True,
        reactive=True,
        sign_only_guaranteed_contracts=False,
        riskiness=0.0,
        max_insurance_premium: float = -1.0,
    ):
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
        self.cfp_records = {}
        self.maxdebt = 0
        self.firstArrival = -1
        self.lastLine = 0

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
        if self.use_consumer:
            # @todo add the parameters of the consumption profile as parameters of the greedy factory manager

            consumer_products = dict(
                zip(
                    self.consuming.keys(),
                    (
                        ConsumptionProfile(schedule=[_] * self.awi.n_steps)
                        for _ in itertools.repeat(0)
                    ),
                )
            )

            a = dict(
                zip(
                    [self.awi.products[0].id],
                    (
                        ConsumptionProfile(schedule=[_] * self.awi.n_steps)
                        for _ in itertools.repeat(0)
                    ),
                )
            )
            consumer_products[self.awi.products[0].id] = a[self.awi.products[0].id]

            self.consumer: ScheduleDrivenConsumer = ScheduleDrivenConsumer(
                profiles=consumer_products,
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

        # print(self.name, "   consume: ", self.consuming.keys(), "     produce: ", self.producing.keys())

        raw_product = min(self.products, key=attrgetter("production_level"))

        if raw_product.id not in self.consuming.keys():
            self.cfp_records[raw_product.id] = ProductData(
                raw_product.id,
                2,
                0.3,
                raw_product.catalog_price,
                raw_product.catalog_price,
            )

        refined_product = max(self.products, key=attrgetter("production_level"))
        if refined_product.id not in self.producing.keys():
            self.cfp_records[refined_product.id] = ProductData(
                refined_product.id,
                2,
                0.3,
                refined_product.catalog_price,
                refined_product.catalog_price,
            )
        for product in self.consuming.keys():
            self.cfp_records[self.products[product].id] = ProductData(
                product,
                2,
                0.3,
                self.products[product].catalog_price,
                self.products[product].catalog_price,
                12,
            )
            self.cfp_records[self.products[product].id].transformable = True

        for product in self.producing.keys():
            self.cfp_records[self.products[product].id] = ProductData(
                product,
                2,
                0.3,
                self.products[product].catalog_price,
                self.products[product].catalog_price,
            )

        for key, product in self.cfp_records.items():
            for item in self.products:
                if (
                    item.production_level
                    == self.products[product.id].production_level + 1
                ):
                    product.source_of = item.id
                    break

    def on_contract_breached(
        self, contract: Contract, breaches: List[Breach], resolution: Optional[Contract]
    ):
        cfp = contract.annotation["cfp"]
        """print("breach! step:", self.awi.current_step, " seller: ", contract.annotation["seller"], " buyer: ",
              contract.annotation["buyer"], " quantity: ", contract.agreement['quantity'], "product: ", cfp.product, "stock: ",
              self.cfp_records[cfp.product].stock, " real stock: ", self.awi.state.storage[cfp.product])
              """
        if contract.annotation["seller"] == self.id:
            self.cfp_records[cfp.product].stock += contract.agreement["quantity"]
        else:
            if self.cfp_records[cfp.product].transformable:
                self.cfp_records[
                    self.cfp_records[cfp.product].source_of
                ].stock -= contract.agreement["quantity"]
            else:
                self.cfp_records[cfp.product].stock -= contract.agreement["quantity"]

    def respond_to_negotiation_request(
        self, cfp: "CFP", partner: str
    ) -> Optional[Negotiator]:

        if self.awi.is_bankrupt(partner):
            return None
        if self.use_consumer:
            return self.consumer.respond_to_negotiation_request(
                cfp=cfp, partner=partner
            )
        else:
            neg = self.negotiator_type(
                name=self.name + "*" + partner, **self.negotiator_params
            )
            ufun = self.ufun_factory(self, self._create_annotation(cfp=cfp))
            neg.utility_function = normalize(
                ufun, outcomes=cfp.outcomes, infeasible_cutoff=0
            )
            return neg

    def on_negotiation_success(
        self, contract: Contract, mechanism: AgentMechanismInterface
    ):
        if self.use_consumer:
            self.consumer.on_negotiation_success(contract, mechanism)

    def on_negotiation_failure(
        self,
        partners: List[str],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
        state: MechanismState,
    ) -> None:

        cfp = annotation["cfp"]
        """print("negotiation fail. Timeout: ", mechanism.state.timedout, " Seller: ", annotation['seller'],
              " buyer: ", annotation['buyer'], " quantity: ", cfp.quantity, " product: ", cfp.product,
        " prices: ", cfp.unit_price, " time: ", cfp.time, " step: ", self.awi.current_step)
"""
        if mechanism.state.timedout:
            return

    def _execute_schedule(self, schedule: ScheduleInfo, contract: Contract) -> None:
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
            if total <= 0 or self.max_insurance_premium < 0.0 or contract is None:
                return
            premium = awi.evaluate_insurance(contract=contract)
            if premium is None:
                return
            relative_premium = premium / total
            if relative_premium <= self.max_insurance_premium:
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
            self.awi.schedule_production(need.product, self.awi.current_step, contract)

    def sign_contract(self, contract: Contract):
        if any(self.awi.is_bankrupt(partner) for partner in contract.partners):
            return None
        product = contract.annotation["cfp"].product
        if self.cfp_records[product].stock < -self.maxdebt:
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

    def on_contract_signed(self, contract: Contract):

        cfp = contract.annotation["cfp"]
        """print("contract signed. Seller: ", contract.annotation['seller'], " buyer: ", contract.annotation['buyer'],
              " quantity: ", contract.agreement['quantity'], " product: ", cfp.product, " price: ", contract.agreement['unit_price'],
              " time: ", contract.agreement['time'], " step: ", self.awi.current_step)
"""
        if contract.annotation["buyer"] == self.id:
            if cfp.product in self.consuming:
                self.cfp_records[
                    self.cfp_records[cfp.product].source_of
                ].stock += contract.agreement["quantity"]
                if self.firstArrival == -1:
                    self.firstArrival = contract.agreement["time"]
            else:
                self.cfp_records[cfp.product].stock += contract.agreement["quantity"]
        else:
            self.cfp_records[cfp.product].stock -= contract.agreement["quantity"]

        self.cfp_records[cfp.product].prevMin = self.cfp_records[cfp.product].MinPrice
        self.cfp_records[cfp.product].prevMax = self.cfp_records[cfp.product].MaxPrice

        if cfp.min_unit_price not in self.cfp_records[cfp.product].HistoryMin:
            self.cfp_records[cfp.product].HistoryMin.append(cfp.min_unit_price)
        if cfp.max_unit_price not in self.cfp_records[cfp.product].HistoryMax:
            self.cfp_records[cfp.product].HistoryMax.append(cfp.max_unit_price)

        self.recalculate_prices(cfp.product)
        self.cfp_records[cfp.product].lastSigned = self.awi.current_step

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
        if self.awi.is_bankrupt(cfp.publisher):
            return None

        """if self.simulator is None or not self.can_expect_agreement(cfp=cfp, margin=self.negotiation_margin):
            return
        if not self.can_produce(cfp=cfp):
            return"""

        if (
            self.awi.n_steps - 3 <= self.awi.current_step
            and self.cfp_records[cfp.product].stock < cfp.min_quantity
        ):
            return None
        if self.cfp_records[cfp.product].stock < -self.maxdebt:
            return None

        if cfp.max_time < 4 + (2 * cfp.product) or self.firstArrival >= cfp.max_time:
            return None

        cfp.unit_price = self.generate_price_ranges(
            self.cfp_records[cfp.product].MaxPrice
            - self.cfp_records[cfp.product].MaxPrice
            * self.cfp_records[cfp.product].BuyThreshold,
            self.cfp_records[cfp.product].MaxPrice
            + self.cfp_records[cfp.product].MaxPrice
            * self.cfp_records[cfp.product].SellThreshold,
        )

        if self.negotiator_type == AspirationNegotiator:
            neg = self.negotiator_type(
                assume_normalized=True, name=self.name + ">" + cfp.publisher
            )
        else:
            neg = self.negotiator_type(name=self.name + ">" + cfp.publisher)
        self.request_negotiation(
            negotiator=neg,
            cfp=cfp,
            ufun=normalize(
                self.ufun_factory(self, self._create_annotation(cfp=cfp)),
                outcomes=cfp.outcomes,
                infeasible_cutoff=-1500,
            ),
        )

    def _process_sell_cfp(self, cfp: "CFP"):
        if self.awi.is_bankrupt(cfp.publisher):
            return None
        if self.awi.n_steps - 4 <= self.awi.current_step:
            return None
        cfp.unit_price = self.generate_price_ranges(
            0,
            self.cfp_records[cfp.product].MinPrice
            + self.cfp_records[cfp.product].MinPrice
            * self.cfp_records[cfp.product].BuyThreshold,
        )

        if self.negotiator_type == AspirationNegotiator:
            neg = self.negotiator_type(
                assume_normalized=True, name=self.name + ">" + cfp.publisher
            )
        else:
            neg = self.negotiator_type(name=self.name + ">" + cfp.publisher)
        self.request_negotiation(
            negotiator=neg,
            cfp=cfp,
            ufun=normalize(
                self.ufun_factory(self, self._create_annotation(cfp=cfp)),
                outcomes=cfp.outcomes,
                infeasible_cutoff=-1500,
            ),
        )

    def on_new_cfp(self, cfp: "CFP") -> None:

        if cfp.product in self.cfp_records:
            # self.cfp_records[cfp.product].HistoryPublishers.append(cfp.publisher)
            if cfp.min_unit_price not in self.cfp_records[cfp.product].HistoryMin:
                self.cfp_records[cfp.product].HistoryMin.append(cfp.min_unit_price)
            if cfp.max_unit_price not in self.cfp_records[cfp.product].HistoryMax:
                self.cfp_records[cfp.product].HistoryMax.append(cfp.max_unit_price)
            self.recalculate_prices(cfp.product)

        want = [value.id for key, value in self.cfp_records.items() if value.stock > 0]
        want += list(self.producing.keys())
        if cfp.satisfies(query={"is_buy": True, "products": want}):
            self._process_buy_cfp(cfp)

        want = [value.id for key, value in self.cfp_records.items()]
        if cfp.satisfies(query={"is_buy": False, "products": want}):
            self._process_sell_cfp(cfp)

    def recalculate_prices(self, product):

        if len(self.cfp_records[product].HistoryMin) > 0:
            self.cfp_records[product].MinPrice = max(
                [math.floor(min(self.cfp_records[product].HistoryMin)), 1]
            )
        if len(self.cfp_records[product].HistoryMax) > 0:
            self.cfp_records[product].MaxPrice = min(
                [math.ceil(max(self.cfp_records[product].HistoryMax)), 1000]
            )

    def generate_price_ranges(self, minPrice, maxPrice):
        stepSize = (maxPrice - minPrice) / 19
        prices = []
        numSteps = int(min([19, 2 * (maxPrice - minPrice)]))
        if numSteps < 10:
            stepSize = 0.5
        for step in range(numSteps + 1):
            prices.append(round(minPrice + step * stepSize, 2))
        if numSteps < 10:
            prices.append(maxPrice)
        return prices

    def on_contract_executed(self, contract: Contract):

        """print("contract executed!!!. Seller: ", contract.annotation['seller'], " buyer: ", contract.annotation['buyer'],
              " quantity: ", contract.agreement['quantity'], " product: ", contract.annotation['cfp'].product,
              " price: ", contract.agreement['unit_price'], " time: ", contract.agreement['time'],
              " step: ", self.awi.current_step)
              """

        if self.cfp_records[contract.annotation["cfp"].product].transformable:
            for i in range(min([contract.agreement["quantity"], 10])):
                self.awi.schedule_production(self.lastLine, self.awi.current_step + 1)
                self.lastLine = (self.lastLine + 1) % 10

    def step(self):
        self.maxdebt = max([0, 30 - 30 * self.awi.current_step / self.awi.n_steps])

        for key, item in self.cfp_records.items():
            if (
                item.lastSigned < self.awi.current_step - 4
                and item.retracted < self.awi.current_step - 4
            ):
                if len(item.HistoryMin) > 1:
                    item.HistoryMin.remove(min(item.HistoryMin))
                    # item.HistoryMin.pop()
                if len(item.HistoryMax) > 1:
                    # item.HistoryMax.pop()
                    item.HistoryMax.remove(max(item.HistoryMax))
                self.recalculate_prices(key)
                item.retracted = self.awi.current_step

        # print("step: ", self.awi.current_step)
        for key, item in self.cfp_records.items():

            if item.transformable:
                stock = self.cfp_records[item.source_of].stock
                if key in self.awi.state.storage:
                    for i in range(min([10, self.awi.state.storage[key]])):
                        self.awi.schedule_production(
                            self.lastLine, self.awi.current_step + 1
                        )
                        self.lastLine = (self.lastLine + 1) % 10
            else:
                stock = item.stock
            if (
                item.id not in self.producing.keys()
                and stock <= item.minstock
                and (item.Asked < self.awi.current_step - 2)
                and self.awi.n_steps - 6 > self.awi.current_step
            ):
                queries = 1
                if item.id in self.consuming.keys():
                    queries = 3

                for i in range(queries):
                    cfp = CFP(
                        is_buy=True,
                        publisher=self.id,
                        product=item.id,
                        time=(
                            self.awi.current_step + 4 + i,
                            min([self.awi.current_step + 6 + i, self.awi.n_steps - 1]),
                        ),
                        unit_price=self.generate_price_ranges(
                            0,
                            self.products[item.id].catalog_price
                            + self.products[item.id].catalog_price
                            * self.cfp_records[item.id].BuyThreshold,
                        ),
                        quantity=3,
                    )

                    self.awi.register_cfp(cfp)
                item.Asked = self.awi.current_step

            if item.stock > 0 and item.id not in self.consuming.keys():
                for unit in range(item.stock):
                    unit_price = self.generate_price_ranges(
                        item.MaxPrice - item.MaxPrice * item.BuyThreshold,
                        item.MaxPrice + item.MaxPrice * item.SellThreshold,
                    )

                    cfp = CFP(
                        is_buy=False,
                        publisher=self.id,
                        product=item.id,
                        time=max([2 + (2 * item.id), self.awi.current_step + 6]),
                        unit_price=unit_price,
                        quantity=1,
                    )
                    self.awi.register_cfp(cfp)
                    item.Asked = self.awi.current_step

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
