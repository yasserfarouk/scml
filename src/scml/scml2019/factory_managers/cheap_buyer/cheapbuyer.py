"""
This module implements a factory manager for the SCM league of ANAC 2019 competition. This advanced version has all
callbacks. Please refer to the [http://www.yasserm.com/scml/scml.pdf](game description)
for all the callbacks.

Your agent can learn about the state of the world and itself by accessing properties in the AWI it has. For example::

self.awi.n_steps  # gives the number of simulation steps

You can access the state of your factory as::

self.awi.state

Your agent can act in the world by calling methods in the AWI it has. For example:

 # registers a new CFP

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
import sys

sys.path.append("/".join(__file__.split("/")[:-1]))
import os
import time
from typing import Any, List, Dict, Union, Type, Optional, Collection

from negmas import Contract, AgentMechanismInterface, MechanismState, Negotiator
from scml.scml2019.common import CFP, SCMLAgreement
from ..builtins import GreedyFactoryManager, Scheduler, temporary_transaction
import random
import shutil

from .MyNegotiator2 import MyNegotiator2
from .NegotiationStepsDataHolder import NegotiationStepsDataHolder
from .SellerUtilityFunction import SellerUtilityFunction
from .myconsumer import MyConsumer
from .myscheduler import MyScheduler
from .myutilityfunction import MyUtilityFunction


class CheapBuyerFactoryManager(GreedyFactoryManager):
    """"""

    # TODO : READ SELL CFPS
    def __init__(
        self,
        name=None,
        scheduler_type: Union[str, Type[Scheduler]] = MyScheduler,
        negotiator_type="my_negotiator.Mynegotiator",
        riskiness=1,
    ):
        if os.path.exists("logs"):
            shutil.rmtree("logs")
        self.name = name
        self.optimism = 0.0
        self.decoyCFPs = []
        self.cumulative_demands = {}
        self.demands_in_this_step = {}
        self.demand_weights = 0
        self.cfps_in_this_step = {}
        self.unweighted_estimated_demands_for_each_step = []
        self.weighted_estimated_demands_for_each_step = []
        self.weighted_sum_of_estimated_demands = {}
        self.unweighted_sum_of_estimated_demands = {}
        self.negotiator_id = 0
        self.number_of_evaluations = 0
        self.number_of_condition_satisfaction = 0
        self.raw_material_type = 0
        self.final_product_type = 0
        self.min_raw_material_purchasable = 5
        self.number_of_negotiation_steps = {}
        self.my_file = open("AVERAGE_STEPS.txt", "a+")
        self.gamma = 0.03
        self.simulataneous_negotiation_limit = 10
        self.amount_sold = 0
        self.total_revenue = 0
        self.successful_buying_negotiations = []
        self.failed_buying_negotiations = []
        self.line_idlenesses = []
        self.process = 0
        self.total_cost = 0
        self.amount_receivable = 0
        self._negotiation_requests = 0
        self.agents_bankrupt = 0
        self.successful_selling_negotiations = []
        self.failed_selling_negotiations = []
        self.negotiators = {}
        self.NEGOTIATOR_ID_FIXED_PART = "NEGOTIATOR_ID_SELLER"
        self.sell_contract_cancellations = 0
        self.buy_contract_cancellations = 0
        super(CheapBuyerFactoryManager, self).__init__(
            scheduler_type=scheduler_type,
            negotiator_type=negotiator_type,
            riskiness=riskiness,
        )
        self.ufun_factory = MyUtilityFunction
        """"""

    def init(self):
        # print("MY ID : "+self.id)
        # print(type(self))
        max_steps = self.awi.n_steps
        number_of_lines = len(self.line_profiles)
        for i in range(max_steps):
            line_idleness = []
            for j in range(number_of_lines):
                line_idleness.append(0)
            self.line_idlenesses.append(line_idleness)
        self.scheduler_params = {"strategy": "earliest_feasible"}
        self.negotiation_margin = max(
            self.negotiation_margin,
            int(round(len(self.products) * max(0.0, 1.0 - self.riskiness))),
        )
        if self.use_consumer:
            # @todo add the parameters of the consumption profile as parameters of the greedy factory manager
            # profiles = dict(
            #     zip(
            #         self.consuming.keys(),
            #         (
            #             ConsumptionProfile(schedule=[_] * self.awi.n_steps)
            #             for _ in itertools.repeat(0)
            #         ),
            #     )
            # )
            self.consumer: MyConsumer = MyConsumer(agent=self, name=self.name)
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

        idsOfProducts = []
        for i in self.products:
            idsOfProducts.append(i.id)
            self.cumulative_demands[i.id] = 0
            self.demands_in_this_step[i.id] = 0

        process = self.line_profiles[0][0].process
        self.process = process
        self.raw_material_type = self.processes[process].inputs[0].product
        self.final_product_type = self.processes[process].outputs[0].product
        # print("MY INPUT PRODUCT : "+str(self.raw_material_type))
        # print("MY OUTPUT PRODUCT : "+str(self.final_product_type))
        # print("NUMBER OF PRODUCT TYPES : "+str(len(self.products)))
        # self.register_all_products()

        # self.weighted_sum_of_estimated_demands = self.initialize_weighted_sum_of_estimated_demands()
        # self.unweighted_sum_of_estimated_demands = self.initialize_unweighted_sum_of_estimated_demands()

        # print("Input : "+str(self.processes[self.line_profiles.get(0)[0].process].inputs) +
        #       " Output : "+str(self.processes[self.line_profiles.get(0)[0].process].outputs))

        # )
        a = 0

    def register_all_products(self):
        products_to_be_registered = []
        for product in range(len(self.products)):
            products_to_be_registered.append(product)
        self.awi.register_interest(products_to_be_registered)

    def step(self):
        super().step()
        balance = str(self.get_balance())
        amount_of_raw_materials = self.get_amount_of_raw_materials()
        amount_of_final_products = self.get_amount_of_final_products()
        successful_buying_negotiations = len(self.successful_buying_negotiations)
        successful_selling_negotiations = len(self.successful_selling_negotiations)
        failed_buying_negotiations = len(self.failed_buying_negotiations)
        failed_selling_negotiations = len(self.failed_selling_negotiations)
        average_buying_price = self.get_average_buying_price()
        _negotiation_requests = self._negotiation_requests
        agents_bankrupt = self.agents_bankrupt
        amount_sold = self.amount_sold
        amount_receivable = self.amount_receivable
        average_selling_price = self.get_average_selling_price()
        sell_contract_cancellations = self.sell_contract_cancellations
        buy_contract_cancellations = self.buy_contract_cancellations
        available_cfps = self.awi.bb_query(
            section="cfps",
            query={"is_buy": True, "products": list(self.producing.keys())},
        )
        # print("STEP : " + str(self.current_step)
        #       +"\nWALLET : " + balance
        #       +"\nRAW MATERIAL : " + str(amount_of_raw_materials)
        #       +"\nFINAL PRODUCT : " + str(amount_of_final_products)
        #       +"\nSUCCESSFUL BUYING NEGOTIATIONS : "+str(successful_buying_negotiations)
        #       +"\nSUCCESSFUL SELLING NEGOTIATIONS : "+str(successful_selling_negotiations)
        #       +"\nFAILED BUYING NEGOTIATIONS : "+str(failed_buying_negotiations)
        #       +"\nFAILED SELLING NEGOTIATIONS : "+str(failed_selling_negotiations)
        #       +"\nAVERAGE BUYING PRICE : "+str(average_buying_price)
        #       +"\nAVERAGE SELLING PRICE : "+str(average_selling_price)
        #       +"\nNEGOTIATION REQUESTS : "+str(_negotiation_requests)
        #       +"\nAGENTS BANKRUPT : "+str(agents_bankrupt)
        #       +"\nAMOUNT SOLD : "+str(amount_sold)
        #       +"\nAMOUNT RECEIVABLE : "+str(amount_receivable)
        #       +"\nSELL CONTRACT CANCELLATIONS : "+str(sell_contract_cancellations)
        #       +"\nBUY CONTRACT CANCELLATIONS : "+str(buy_contract_cancellations)
        #       +"\nAVAILABLE CFPS : "+str(len(available_cfps))
        #       +"\n")
        self._negotiation_requests = 0
        self.post_cfps()
        self.process_raw_materials()

        for cfp in available_cfps.values():
            self.respond_to_cfp(cfp=cfp)

        jobs = []
        # step = self.awi.current_step
        # while amount_of_raw_materials > 0:
        #     line_idleness = self.line_idlenesses[step]
        #     for line in range(len(line_idleness)):
        #         idleness = line_idleness[line]
        #         if idleness == 0:
        #             job = Job(action="run", profile=self.process, line=line, time=step, override=False, contract=None)
        #             self.awi.schedule_job(job=job, contract=None)
        #             amount_of_raw_materials -= 1
        #             line_idleness[line] = 1
        #     step = step + 1
        # print(self.awi.state.jobs)

        # if self.awi.current_step == 15:
        #     a = 0

        # estimated_supply_for_raw_material = 0
        # estimated_demand_for_final_product = 0
        #
        # for key in self.cfps_in_this_step.keys():
        #     cfps = self.cfps_in_this_step.get(key)
        #     for cfp in cfps:
        #         average_quantity = (cfp.quantity[0] + cfp.quantity[1])/2
        #         average_time = ((cfp.max_time + cfp.min_time)/2)-self.current_step + 1
        #         if cfp.product == self.raw_material_type:
        #             estimated_suppy_for_raw_material = estimated_supply_for_raw_material + average_quantity/average_time
        #         elif cfp.product == self.final_product_type:
        #             estimated_demand_for_final_product = estimated_demand_for_final_product + average_quantity/average_time
        # estimated_supply_for_raw_material = math.floor(estimated_supply_for_raw_material)
        # estimated_demand_for_final_product = math.floor(estimated_demand_for_final_product)
        # purchasable = max(int(math.floor(estimated_demand_for_final_product
        #                              - self.simulator.storage_at(self.current_step)[self.raw_material_type])),
        #                        self.min_raw_material_purchasable)
        # print("PURCHASABLE : "+str(purchasable))
        # purchasable = self.min_raw_material_purchasable
        # if purchasable > 0:
        #     for step in range(purchasable+5):
        #         if step+self.current_step < self.awi.n_steps:
        #             my_cfp = CFP(is_buy=True,
        #                  product=self.raw_material_type,
        #                  publisher=self.id,
        #                  quantity=(1, purchasable),
        #                  time=step+self.current_step,
        #                  unit_price=(0, self.products[self.raw_material_type].catalog_price),
        #                  money_resolution=0.1,
        #                  id="NORMAL : " + str(time))
        #             self.awi.register_cfp(my_cfp)
        #             print(str(my_cfp)+" OUR CFP !!")

        for key in self.cfps_in_this_step.keys():
            self.cfps_in_this_step[key] = []

        # self.estimate_unweighted_demand()
        # self.estimate_weighted_demand()
        # self.cfps_in_this_step = {}
        # self.write_to_file(self.weighted_sum_of_estimated_demands, "weighted.txt")
        # self.write_to_file(self.unweighted_sum_of_estimated_demands, "unweighted.txt")

        if self.awi.relative_time == 1:
            # self.write_to_file("ESTIMATED : " + str(self.get_sum_of_weighted_demands()), "weighted.txt")
            # self.write_to_file("ESTIMATED : " + str(self.get_sum_of_unweighted_demands()), "unweighted.txt")
            # print("Condition satisfaction ratio : "+str(self.number_of_condition_satisfaction/self.number_of_evaluations))
            # print(self.simulator.storage_at(self.current_step))

            # weighted_sum_of_estimated_demands = self.initialize_weighted_sum_of_estimated_demands()

            # self.estimate_unweighted_demand()

            # for product in self.products:
            #     weighted_sum_of_estimated_demands[product.id] =\
            #         round(weighted_sum_of_estimated_demands[product.id]/self.current_step, 2)
            # self.write_estimated_average_demands(weighted_sum_of_estimated_demands)
            """"""

    def get_amount_of_final_products(self):
        return self.simulator.storage_at(self.current_step)[self.final_product_type]

    def get_balance(self):
        return self.simulator.wallet_at(self.current_step)

    def get_amount_of_raw_materials(self):
        return self.simulator.storage_at(self.current_step)[self.raw_material_type]

    def post_cfps(self):
        alpha = 16
        for step in range(15):
            my_cfp = CFP(
                is_buy=True,
                product=self.raw_material_type,
                publisher=self.id,
                quantity=(1, step + alpha),
                time=min(step + self.current_step, self.awi.n_steps - 2),
                unit_price=(0.5, self.get_target_price()),
                money_resolution=0.1,
                id="NORMAL : " + str(time),
                penalty=10000,
            )
            self.awi.register_cfp(my_cfp)

    def post_cfps_2(self):
        for step in range(10):
            my_cfp = CFP(
                is_buy=True,
                product=self.raw_material_type,
                publisher=self.id,
                quantity=(1, 1 + random.randint(0, 10)),
                time=random.randint(0, 15) + self.current_step,
                unit_price=(0.5, self.get_target_price()),
                money_resolution=0.1,
                id="NORMAL : " + str(time),
                penalty=10000,
            )
            self.awi.register_cfp(my_cfp)

    def estimate_weighted_demand(self):
        for cfp in self.cfps_in_this_step:
            product = cfp.product
            weight = cfp.max_quantity - cfp.min_quantity + 1
            estimated_average_demand = (cfp.max_quantity + cfp.min_quantity) / 2
            weighted_estimated_average_demand = estimated_average_demand / weight
            self.weighted_sum_of_estimated_demands[product] = round(
                self.weighted_sum_of_estimated_demands[product]
                + weighted_estimated_average_demand,
                2,
            )

            self.unweighted_estimated_demands_for_each_step.append(
                self.weighted_sum_of_estimated_demands
            )

    def estimate_unweighted_demand(self):
        for cfp in self.cfps_in_this_step:
            product = cfp.product
            estimated_average_demand = (cfp.max_quantity + cfp.min_quantity) / 2
            self.unweighted_sum_of_estimated_demands[product] = round(
                self.unweighted_sum_of_estimated_demands[product]
                + estimated_average_demand,
                2,
            )

        self.weighted_estimated_demands_for_each_step.append(
            self.unweighted_sum_of_estimated_demands
        )

    def initialize_weighted_sum_of_estimated_demands(self):
        weighted_sum_of_estimated_demands = {}
        for product in self.products:
            weighted_sum_of_estimated_demands[product.id] = 0
        return weighted_sum_of_estimated_demands

    def initialize_unweighted_sum_of_estimated_demands(self):
        unweighted_sum_of_estimated_demands = {}
        for product in self.products:
            unweighted_sum_of_estimated_demands[product.id] = 0
        return unweighted_sum_of_estimated_demands

    def on_new_cfp(self, cfp: CFP) -> None:
        # self.respond_to_cfp(cfp)
        # print("CFP RECEIVED : "+str(cfp))
        #
        # cfps: list[CFP] = self.cfps_in_this_step.get(cfp.publisher)
        # if cfps is None:
        #     self.cfps_in_this_step[cfp.publisher] = []
        # else:
        #     cfps.append(cfp)
        #     self.cfps_in_this_step[cfp.publisher] = cfps
        #
        """"""

    def respond_to_cfp(self, cfp):
        if self.is_cfp_acceptable_2(cfp):
            self.accept_negotiation(cfp)

    def is_cfp_acceptable_2(self, cfp) -> bool:
        is_agent_bankrupt = self.awi.is_bankrupt(cfp.publisher)
        is_interested = cfp.satisfies(
            query={"is_buy": True, "products": list(self.producing.keys())}
        )
        is_raw_material_bought = self.get_average_buying_price() != float("inf")
        have_final_products = self.get_amount_of_final_products() > 0
        return (
            is_interested
            and not is_agent_bankrupt
            and is_raw_material_bought
            and have_final_products
        )

    def is_cfp_acceptable(self, cfp) -> bool:
        is_agent_bankrupt = self.awi.is_bankrupt(cfp.publisher)
        is_interested = cfp.satisfies(
            query={"is_buy": True, "products": list(self.producing.keys())}
        )
        can_produce = self.can_produce(cfp)
        can_expect_agreement = self.can_expect_agreement(cfp, self.negotiation_margin)
        if is_agent_bankrupt:
            """"""
            # print("AGENT IS BANKRUPT, ", end='')
        if not is_interested:
            """"""
            # print("NOT INTERESTED, ", end='')
        if not can_produce:
            """"""
            # print("CAN'T PRODUCE, ", end='')
        if not can_expect_agreement:
            """"""
            # print("AGREEMENT NOT EXPECTED, ", end='')
        # print()
        return (
            not is_agent_bankrupt
            and is_interested
            and can_produce
            and can_expect_agreement
        )

    def accept_negotiation(self, cfp) -> bool:
        ufun = SellerUtilityFunction(
            unit_cost=self.get_average_buying_price() + self.get_process_cost()
        )
        negotiator_id = self.NEGOTIATOR_ID_FIXED_PART + " : " + str(self.negotiator_id)
        negotiator = MyNegotiator2(
            ufun=ufun,
            name=negotiator_id,
            strategy=MyNegotiator2.STRATEGY_TIME_BASED_CONCESSION,
            reserved_value=1,
        )
        response = self.request_negotiation(negotiator=negotiator, cfp=cfp, ufun=ufun)
        if response:
            self.negotiators[negotiator.name] = negotiator
            self.negotiator_id += 1
        return response

    def accept_negotiation_old(self, cfp):
        negotiation_steps_data_holder: NegotiationStepsDataHolder = self.number_of_negotiation_steps.get(
            cfp.publisher
        )
        average_number_of_steps = 0
        if negotiation_steps_data_holder is not None:
            average_number_of_steps = (
                negotiation_steps_data_holder.get_average_number_of_negotiation_steps()
            )
        alpha_and_beta = self.get_alpha_and_beta(
            average_number_of_steps, gamma=self.gamma
        )
        ufun = self.ufun_factory(
            agent=self,
            annotation=self._create_annotation(cfp=cfp),
            ufun_id=self.negotiator_id,
            alpha=alpha_and_beta[0],
            beta=alpha_and_beta[1],
            average_number_of_steps=average_number_of_steps,
        )
        ufun.reserved_value = (
            cfp.money_resolution if cfp.money_resolution is not None else 0.1
        )
        neg = self.negotiator_type(
            name=self.negotiator_id, ufun=ufun, cfp=cfp, partner_id=cfp.publisher
        )
        self.request_negotiation(negotiator=neg, cfp=cfp, ufun=ufun)

    def write_to_file(self, averaged_estimated_demands, filename):
        import os

        os.makedirs("/".join(__file__.split("/")[:-1]) + "/logs", exist_ok=True)
        with open(
            "/".join(__file__.split("/")[:-1]) + "/logs/" + filename, "a"
        ) as myfile:
            myfile.write(str(averaged_estimated_demands) + "\n")

    def get_unweighted_estimated_average_demands(self):
        averaged_estimated_demands = self.cumulative_demands
        for key in averaged_estimated_demands.keys():
            averaged_estimated_demands[key] = round(
                averaged_estimated_demands[key] / self.current_step, 10
            )
        return averaged_estimated_demands

    def add_to_cumulative_estimated_demands(self, cfp):
        self.cumulative_demands[cfp.product] = self.cumulative_demands[cfp.product] + (
            (cfp.min_quantity + cfp.max_quantity) / 2
        )

    def generateDecoyCFPs(self):
        for i in range(1, 100):
            decoyCFP = CFP(
                is_buy=True,
                product=int(random.uniform(1, len(self.products))),
                publisher=self.id,
                signing_delay=int(random.uniform(1, 9)),
                quantity=1,
                time=int(random.uniform(1, 5)),
                unit_price=(0, 100),
                money_resolution=0.1,
                id="DECOY : " + str(i),
            )
            self.decoyCFPs.append(decoyCFP)
            self.awi.register_cfp(decoyCFP)

    def unweighted_sum_of_estimated_demands(self):
        return self.unweighted_sum_of_estimated_demands

    def get_sum_of_unweighted_demands(self) -> int:
        sum = 0
        for key in self.unweighted_sum_of_estimated_demands.keys():
            sum += self.unweighted_sum_of_estimated_demands.get(key)
        return sum

    def get_sum_of_weighted_demands(self) -> int:
        sum = 0
        for key in self.weighted_sum_of_estimated_demands.keys():
            sum += self.weighted_sum_of_estimated_demands.get(key)
        return sum

    def confirm_contract_execution(self, contract: Contract) -> bool:
        super().confirm_contract_execution(contract)

    def on_neg_request_accepted(self, req_id: str, mechanism: AgentMechanismInterface):
        # print("NEGOTIATION REQUEST ACCEPTED MY: "+str(self.negotiator_id))
        self.negotiator_id = self.negotiator_id + 1
        super().on_neg_request_accepted(req_id=req_id, mechanism=mechanism)

    def on_neg_request_rejected(self, req_id: str, by: Optional[List[str]]):
        """"""

    def on_negotiation_failure(
        self,
        partners: List[str],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
        state: MechanismState,
    ) -> None:
        negotiator_id = None
        if annotation.get("buyer") == self.id:

            for participant in mechanism.participants:
                if self.consumer.NEGOTIATOR_ID_FIXED_PART in participant.id:
                    negotiator_id = participant.name
                    break
            negotiator = self.consumer.get_negotiator(negotiator_id)
            self.failed_buying_negotiations.append(negotiator)
            self.consumer.on_negotiation_failure(
                partners=partners,
                annotation=annotation,
                mechanism=mechanism,
                state=state,
            )
        else:
            for participant in mechanism.participants:
                if self.NEGOTIATOR_ID_FIXED_PART in participant.id:
                    negotiator_id = participant.name
                    break
            negotiator = self.negotiators.get(negotiator_id)
            self.failed_selling_negotiations.append(negotiator)

        super().on_negotiation_failure(
            partners=partners, annotation=annotation, mechanism=mechanism, state=state
        )

    def can_produce(self, cfp: CFP, assume_no_further_negotiations=False) -> bool:
        """Whether or not we can produce the required item in time"""
        self.number_of_evaluations = self.number_of_evaluations + 1
        if cfp.product not in self.producing.keys():
            self.number_of_condition_satisfaction = (
                self.number_of_condition_satisfaction + 1
            )
            return False
        agreement = SCMLAgreement(
            time=cfp.max_time, unit_price=cfp.max_unit_price, quantity=cfp.min_quantity
        )
        min_concluded_at = self.awi.current_step + 1 - int(self.immediate_negotiations)
        min_sign_at = min_concluded_at + self.awi.default_signing_delay
        if cfp.max_time < min_sign_at + 1:  # 1 is minimum time to produce the product
            self.number_of_condition_satisfaction = (
                self.number_of_condition_satisfaction + 1
            )
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
        result = schedule.valid and self.can_secure_needs(
            schedule=schedule, step=self.awi.current_step
        )

        if not result:
            self.number_of_condition_satisfaction = (
                self.number_of_condition_satisfaction + 1
            )
        return result

    def respond_to_negotiation_request(
        self, cfp: "CFP", partner: str
    ) -> Optional[Negotiator]:
        self._negotiation_requests += 1
        # print("NEGOTIATION REQUESTED : "+str(cfp))

        if partner == self.id or self.awi.is_bankrupt(partner):
            return None

        return self.consumer.respond_to_negotiation_request(cfp=cfp, partner=partner)

    def on_negotiation_success(
        self, contract: Contract, mechanism: AgentMechanismInterface
    ) -> None:
        negotiator_id = None

        if contract.annotation.get("buyer") == self.id:
            negotiator_id_fixed_part = self.consumer.NEGOTIATOR_ID_FIXED_PART

            for participant in mechanism.participants:
                if self.consumer.NEGOTIATOR_ID_FIXED_PART in participant.id:
                    negotiator_id = participant.name
                    break

            negotiator = self.consumer.get_negotiator(negotiator_id)
            self.successful_buying_negotiations.append(negotiator)
        else:
            negotiator_id_fixed_part = self.NEGOTIATOR_ID_FIXED_PART

            for participant in mechanism.participants:
                if self.NEGOTIATOR_ID_FIXED_PART in participant.id:
                    negotiator_id = participant.name
                    break
            self.successful_selling_negotiations.append(self.negotiators[negotiator_id])
        self.consumer.on_negotiation_success(contract=contract, mechanism=mechanism)

        for participant in mechanism.participants:
            if negotiator_id_fixed_part in participant.id:
                negotiator_id = participant.name
                break

    # def on_inventory_change(self, product: int, quantity: int, cause: str) -> None:
    #     if cause == "transport":
    #         self.amount_receivable -= quantity

    # print("NEGOTIATION SUCCESS : "+str(mechanism.state))

    def add_negotiation_step_data(self, mechanism, partner_id):
        data_holder: NegotiationStepsDataHolder = self.number_of_negotiation_steps.get(
            partner_id
        )
        if data_holder is None:
            data_holder = NegotiationStepsDataHolder()
        data_holder.add_new_data(number_of_negotiation_steps=mechanism.state.step)
        self.number_of_negotiation_steps[partner_id] = data_holder

    def get_alpha_and_beta(self, number_of_negotiations, gamma):
        alpha = max(0, 1 - number_of_negotiations * gamma)
        beta = 1 - alpha
        return alpha, beta

    def on_contract_signed(self, contract: Contract):
        if contract.annotation.get("seller") == self.id:
            self.amount_sold += contract.agreement.get("quantity")
            self.total_revenue += contract.agreement.get(
                "unit_price"
            ) * contract.agreement.get("quantity")
        else:
            self.amount_receivable += contract.agreement.get("quantity")
            self.total_cost += contract.agreement.get(
                "quantity"
            ) * contract.agreement.get("unit_price")
        # print("AVERAGE SELLING PRICE ", self.get_average_selling_price())
        # super().on_contract_signed(contract=contract)

    def get_average_selling_price(self):
        if self.amount_sold > 0:
            return self.total_revenue / self.amount_sold
        else:
            return float("inf")

    def get_average_buying_price(self):
        if self.amount_receivable > 0:
            return self.total_cost / self.amount_receivable
        else:
            return float("inf")

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
            return float("-inf")
        return schedule.final_balance

    def get_total_profit(self, contracts: Collection[Contract] = ()) -> float:
        """Calculates the total utility for the agent of a collection of contracts"""
        total = 0
        for contract in contracts:
            total += (
                contract.agreement.get("unit_price") - self.get_average_selling_price()
            ) * contract.agreement.get("quantity")
        return total

    def get_target_price(self):
        raw_material_type = self.raw_material_type
        return self.products[raw_material_type].catalog_price

    def on_agent_bankrupt(self, agent_id: str) -> None:
        self.agents_bankrupt += 1

    def sign_contract(self, contract: Contract):
        if contract.annotation.get("buyer") == self.id:
            cost = contract.agreement.get("quantity") * contract.agreement.get(
                "unit_price"
            )
            if self.get_balance() >= cost:
                # self.awi.hide_funds(amount=cost)
                return self.id
        else:
            product_type = contract.annotation.get("cfp").get("product")
            available_product_quantity = self.simulator.storage_at(self.current_step)[
                product_type
            ]
            required_production_quantity = contract.agreement.get("quantity")
            if available_product_quantity >= required_production_quantity:
                # self.awi.hide_inventory(product=product_type, quantity=required_production_quantity)
                return self.id
        return None

    def process_raw_materials(self, quantity=10):
        for profile in range(int(min(quantity, self.get_amount_of_raw_materials()))):
            self.awi.schedule_production(
                profile=profile,
                contract=None,
                step=self.current_step - 1,
                override=False,
            )

    def get_process_cost(self):
        return self.line_profiles[0][0].cost

    def on_contract_cancelled(self, contract: Contract, rejectors: List[str]) -> None:
        if contract.annotation.get("seller") == self.id and rejectors[0] != self.id:
            self.sell_contract_cancellations += 1
        elif contract.annotation.get("buyer") == self.id and rejectors[0] != self.id:
            self.buy_contract_cancellations += 1


# def main(competition='std', reveal_names=True, n_steps=200, n_configs=1, max_n_worlds_per_config=1, n_runs_per_world=1):
#     """
#     **Not needed for submission.** You can use this function to test your agent.
#
#     Args:
#         competition: The competition type to run (possibilities are std, collusion and sabotage).
#         reveal_names: If true, agent names will reveal their types. This will be false in the actual competition
#         n_steps: The number of simulation steps.
#         n_configs: Number of different world configurations to try. Different world configurations will correspond to
#                    different number of factories, profiles, production graphs etc
#         max_n_worlds_per_config: How many manager-factory assignments are allowed for each generated configuration. The
#                                  system will sample this number of worlds (at most) from all possible permutations of
#                                  manager-factory assignments. Notice that if your competition is set to 'std', only two
#                                  worlds will be generated per configuration at most
#         n_runs_per_world: How many times will each world be run.
#
#     Returns:
#         None
#
#     Remarks:
#
#         - This function will take several minutes to run.
#         - To speed it up, use a smaller `n_step` value
#         - Please notice that the greedy factory manager (the default agent that always exists in the world), will lose
#           in the beginning of the simulation (in most cases). To get a better understanding of your agent's performance
#           set `n_step` to a value of at least 100. The actual league will use 200 steps.
#
#     """
#     start = time.perf_counter()
#     if competition == 'std':
#         results = anac2019_std(competitors=[GreedyFactoryManager, GreedyFactoryManager], agent_names_reveal_type=reveal_names
#                                , verbose=True, n_steps=n_steps, n_configs=n_configs
#                                , max_worlds_per_config=max_n_worlds_per_config, n_runs_per_world=n_runs_per_world,
#                                parallelism="serial")
#         # with open('logs/tournaments/') as csv_file:
#         #     csv_reader = csv.reader(csv_file, delimiter=',')
#     elif competition == 'collusion':
#         results = anac2019_collusion(competitors=[CheapBuyerFactoryManager, GreedyFactoryManager],
#                                      agent_names_reveal_type=reveal_names
#                                      , verbose=True, n_steps=n_steps, n_configs=n_configs
#                                      , max_worlds_per_config=max_n_worlds_per_config, n_runs_per_world=n_runs_per_world)
#     elif competition == 'sabotage':
#         print('The sabotage competition will be run by comparing the score of all other agents with and without '
#               'the agent being tested. The exact way this is to be done is left for the organization committee to '
#               'decide')
#         return
#     else:
#         raise ValueError(f'Unknown competition type {competition}')
#     print(tabulate(results.total_scores, headers='keys', tablefmt='psql'))
#     print(f'Finished in {humanize_time(time.perf_counter() - start)}')
#
#
# """if __name__ == '__main__':
#     Will be called if you run this file directly"""
# main()
# signed_contracts_path = glob.glob("logs/tournaments/*/*/signed_contracts.csv")[0]


# def get_real_demand() -> int:
#     total_quantity_in_signed_contracts = 0
#     with open(signed_contracts_path) as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=' ')
#         line_count = 0
#
#         for row in csv_reader:
#             data = row[0].split(',')[7]
#             if data.isdigit():
#                 total_quantity_in_signed_contracts += int(data)
#     csv_file.close()
#     return total_quantity_in_signed_contracts

# for line in fileinput.input("logs/weighted.txt", inplace=True):
#     if "ESTIMATED" in line:
#         #print(line.rstrip() + " REAL : " + str(get_real_demand()))
#         """"""
# for line in fileinput.input("logs/unweighted.txt", inplace=True):
#     if "ESTIMATED" in line:
#         print(line.rstrip() + " REAL : " + str(get_real_demand()))
