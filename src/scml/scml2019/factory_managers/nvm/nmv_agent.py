import sys

sys.path.append("/".join(__file__.split("/")[:-1]))
import math
import os
import random
import string
from typing import Optional, List, Dict, Any, Union, Type

import matplotlib.pyplot as plt
from negmas import (
    Contract,
    Negotiator,
    AgentMechanismInterface,
    MappingUtilityFunction,
    INVALID_UTILITY,
)
from negmas import MechanismState
from scml.scml2019.common import CFP, Job, ProductionFailure
from scml.scml2019.simulators import FactorySimulator, FastFactorySimulator
from scml.scml2019.factory_managers.builtins import DoNothingFactoryManager
from negmas.sao import AspirationNegotiator
from prettytable import PrettyTable

from .agent_brain import AgentBrain
import sys

sys.path.append("/".join(__file__.split("/")[:-1]))


class NVMFactoryManager(DoNothingFactoryManager):
    """
    This agent implements a multi-period news-vendor model (MPNVM) -based strategy.
    """

    def __init__(
        self,
        name=None,
        simulator_type: Union[str, Type[FactorySimulator]] = FastFactorySimulator,
        parameters: Optional[Dict] = None,
    ):
        """
        Constructor. Received some parameters for the agent.
        :param parameters:
        :param name:
        :param simulator_type:
        """
        super().__init__(name, simulator_type)
        self.num_intermediate_products: int = None
        self.production_cost: float = None
        self.input_index: int = None
        self.output_index: int = None
        self.signed_contracts_for_factory: dict = None
        self.contracted_buys_at_t: dict = {}
        self.contracted_sales_at_t: dict = {}
        self.executed_buys_at_t: dict = {}
        self.executed_sales_at_t: dict = {}
        self.agent_brain: AgentBrain = None
        self.input_negotiator_ufun = None
        self.output_negotiator_ufun = None
        self.wallet_history: list = []
        self.storage_history: dict = {}
        self.expected_catalog_prices: dict = {}
        self.middle_man_products: dict = {}
        self.verbose: bool = False
        self.hyper_parameter_optimization: bool = False

        # Parameters of the agent
        self.marginal_calculation_on_sign_contract: bool = False
        self.middle_man_active: bool = True
        self.fixed_number_of_inputs: int = 1
        self.limit_post_cfps: int = None
        self.start_sell_negotiation_bound: int = 5
        self.limit_sign_time_input_product: int = None
        self.limit_sign_storage: int = 10  # HPO
        self.limit_number_sales_contracts_at_t: int = 5  # HPO
        self.limit_number_buys_contracts_at_t: int = 5  # HPO
        self.cfp_qtty_range_width: int = 5  # HPO
        self.cfp_time_lower_bound: int = 5
        self.cfp_time_upper_bound: int = 15
        self.agent_aspiration_type: str = "boulware"  # boulware, conceder (with conceder, we almost use all money), linear (so far, not so different from boulware.

        if parameters is not None:
            self.marginal_calculation_on_sign_contract = parameters[
                "marginal_calculation_on_sign_contract"
            ]
            # self.limit_sign_storage = parameters['limit_sign_storage']
            # self.limit_number_sales_contracts_at_t = parameters['limit_number_sales_contracts_at_t']
            # self.limit_number_buys_contracts_at_t = parameters['limit_number_buys_contracts_at_t']
            # self.cfp_qtty_range_width = parameters['cfp_qtty_range_width']

    # =====================
    # Time-Driven Callbacks
    # =====================

    def init(self):
        """Called once after the agent-world interface is initialized"""

        # We assume the agent only takes one kind of input.
        if len(self.consuming.keys()) != 1:
            raise Exception("The agent is design to consume only one input")
        self.input_index = list(self.consuming.keys())[0]

        # We assume the agent only produced one kind of output.
        if len(self.producing.keys()) != 1:
            raise Exception("The agent is design to produce only one output")
        self.output_index = list(self.producing.keys())[0]

        self.production_cost = self.line_profiles[0][0].cost
        self.num_intermediate_products = len(self.awi.processes) - 1

        # Compute Expected Catalog Prices
        self.expected_catalog_prices[0] = 1.0
        for p in range(1, self.num_intermediate_products + 2):
            self.expected_catalog_prices[p] = 1.15 * (
                self.expected_catalog_prices[p - 1] + 2.5
            )

        # Initialize book keeping structures.
        for p in range(0, self.num_intermediate_products + 2):
            self.storage_history[p] = []
            (
                self.contracted_buys_at_t[p],
                self.contracted_sales_at_t[p],
                self.executed_buys_at_t[p],
                self.executed_sales_at_t[p],
            ) = (
                {},
                {},
                {},
                {},
            )
            self.signed_contracts_for_factory = {
                t: [] for t in range(0, self.awi.n_steps)
            }
            for t in range(0, self.awi.n_steps):
                self.contracted_buys_at_t[p][t] = 0
                self.contracted_sales_at_t[p][t] = 0
                self.executed_buys_at_t[p][t] = (0, 0)
                self.executed_sales_at_t[p][t] = (0, 0)

        # Initialize the negotiator that will negotiate for inputs
        self.input_negotiator_ufun = MappingUtilityFunction(
            mapping=lambda outcome: 1 - outcome["unit_price"],
            reserved_value=INVALID_UTILITY,
        )

        # Initialize the negotiator that will negotiate for outputs
        self.output_negotiator_ufun = MappingUtilityFunction(
            mapping=lambda outcome: (math.exp(outcome["unit_price"]) - 1.5)
            * outcome["quantity"]
            if outcome["unit_price"] > 0.0
            else INVALID_UTILITY
        )

        # Set the time limit for posting CFPs.
        self.limit_post_cfps = self.awi.n_steps - 16

        # Set the time limit to sign CFPs to buy input
        self.limit_sign_time_input_product = self.awi.n_steps - 10

        # Initialize the brain of the agent. The brain takes in the input product, the output product, cost of production and num_intm_products.
        self.agent_brain = AgentBrain(
            game_length=self.awi.n_steps,
            input_product_index=self.input_index,
            output_product_index=self.output_index,
            production_cost=self.production_cost,
            num_intermediate_products=self.num_intermediate_products,
            verbose=self.verbose,
        )

        # Register interest in all products.
        self.awi.unregister_interest([self.input_index, self.output_index])
        self.awi.register_interest(
            [p for p in range(0, self.num_intermediate_products + 2)]
        )

        if self.verbose:
            # Print some init info for informational purposes
            print(
                f"\n+++++++++++++++++++++++++++++++++++++++\n"
                f"Starting game with a total of {self.awi.n_steps} steps\n"
                f"\t Expected catalog prices = {self.expected_catalog_prices}\n"
                f"\t There are {self.num_intermediate_products} intermediate products. \n"
                f"\t SCML2020World processes = {self.awi.processes}\n"
                f"\t My Cost = {self.line_profiles[0][0].cost}\n"
                f"\t I, {self.id}, consume {self.input_index} and produce {self.output_index}\n"
                f"\t Is the middle man active? {self.middle_man_active}\n"
                f"+++++++++++++++++++++++++++++++++++++++\n"
            )

            print(
                f"Parameters: "
                f"\n\t hyper_parameter_optimization = {self.hyper_parameter_optimization}"
                f"\n\t agent_aspiration_type = {self.agent_aspiration_type}"
                f"\n\t limit_sign_storage = {self.limit_sign_storage}"
                f"\n\t limit_number_sales_contracts_at_t = {self.limit_number_sales_contracts_at_t}"
                f"\n\t limit_number_buys_contracts_at_t = {self.limit_number_buys_contracts_at_t}"
                f"\n\t cfp_qtty_range_width = {self.cfp_qtty_range_width}"
                f"\n ------------------------------------"
            )

        # Which products will the middle man buy and sell?
        self.middle_man_products = {
            p
            for p in range(0, self.num_intermediate_products + 2)
            if p != self.input_index and p != self.output_index
        }
        if self.verbose:
            print(f"The middle man can buy and sell {self.middle_man_products}")

    def step(self):
        """Called at every production step by the world"""

        # Book keeping
        for p in range(0, self.num_intermediate_products + 2):
            self.storage_history[p].append(self.awi.state.storage[p])
        self.wallet_history.append(self.awi.state.wallet)

        # ---------------- MPNVM STUFF --------
        # Plan how many inputs to go for. We ask the brain for the number of inputs. If the brain could not find data, we go for a fixed number.
        plan_for_inputs = (
            self.agent_brain.get_plan_for_inputs(self.current_step + 1, self.verbose)
            if self.agent_brain.there_is_data
            else [self.fixed_number_of_inputs]
        )

        # Post a call for proposal to buy inputs using the plan computed before. We limit the call for buy stuff up to 15 steps before the end of game.
        # @todo There could be a further optimization problem here, as in, how many CFPs to post? At the moment we just post one.
        if (
            self.current_step <= self.limit_post_cfps
            and self.storage_history[self.output_index][-1] <= self.limit_sign_storage
        ):
            self.awi.register_cfp(
                CFP(
                    is_buy=True,
                    publisher=self.name,
                    product=self.input_index,
                    # @todo The time of negotiation matters a lot for longer chains. For chain of size 4, +5, +15 worked.
                    time=(
                        self.current_step + self.cfp_time_lower_bound,
                        self.current_step + self.cfp_time_upper_bound,
                    ),
                    unit_price=(0.0, self.expected_catalog_prices[self.input_index]),
                    quantity=(
                        max(1, plan_for_inputs[0] - self.cfp_qtty_range_width),
                        plan_for_inputs[0] + self.cfp_qtty_range_width,
                    ),
                )
            )

        # Send all inputs to production to get outputs. We always turn every input into output. We don't hold on to inputs.
        schedule_for_production = 0
        for l in range(0, 10):
            if (
                self.storage_history[self.input_index][-1] > 0
                and schedule_for_production
                <= self.storage_history[self.input_index][-1]
                and self.awi.current_step < self.awi.n_steps - 1
            ):
                self.awi.schedule_production(l, self.awi.current_step)
                self.simulator.schedule(
                    Job(
                        profile=l,
                        time=self.awi.current_step,
                        line=-1,
                        action="run",
                        contract=None,
                        override=False,
                    )
                )
                schedule_for_production += 1

        # Read all the CFPs and engage in negotiations with agents that want to buy our output.
        the_cfps = self.awi.bb_query("cfps", None)
        if the_cfps:
            for i, c in the_cfps.items():
                c: CFP
                # Negotiate about outputs. It is highly unlikely the agent can have any output product ready before parameter self.start_sell_negotiation_bound
                if (
                    c.publisher != self.id
                    and c.is_buy
                    and c.product == self.output_index
                    and c.min_time >= self.start_sell_negotiation_bound
                ):
                    self.request_negotiation(
                        cfp=c,
                        negotiator=AspirationNegotiator(
                            ufun=self.output_negotiator_ufun,
                            aspiration_type=self.agent_aspiration_type,
                        ),
                    )
                # @todo Negotiate about inputs. This is currently not active, as in the case of inputs our agent is proactive. Should we activate it?
                # elif not c.is_buy and c.product == self.input_index:
                #    self.request_negotiation(cfp=c, negotiator=AspirationNegotiator(name="my-goog-buyer", ufun=self.input_negotiator_ufun))

        # ---------------- MIDDLE MAN STUFF --------
        # @todo Add parameters for the ranges over which we post CFPs for the middle man.
        if self.middle_man_active:
            # First, post CFP to buy and sell stuff, up to time given by parameter self.limit_post_cfps.
            if self.current_step <= self.limit_post_cfps:
                for p in self.middle_man_products:
                    # Post a CFP to buy stuff to be later resold
                    self.awi.register_cfp(
                        CFP(
                            is_buy=True,
                            publisher=self.name,
                            product=p,
                            time=(
                                self.current_step + self.cfp_time_lower_bound,
                                self.current_step + self.cfp_time_upper_bound,
                            ),
                            unit_price=(0.0, self.expected_catalog_prices[p]),
                            quantity=(1, 5),
                        )
                    )
                    # Post a CFP to sell stuff
                    self.awi.register_cfp(
                        CFP(
                            is_buy=False,
                            publisher=self.name,
                            product=p,
                            time=(
                                self.current_step + self.cfp_time_lower_bound,
                                self.current_step + self.cfp_time_upper_bound,
                            ),
                            unit_price=(3.5, 25.5),
                            quantity=(1, 5),
                        )
                    )
            if the_cfps:
                for i, c in the_cfps.items():
                    c: CFP
                    # Make sure we don't respond to ourselves.
                    if c.publisher != self.id:
                        # Respond to CFPs when we try to buy stuff for the middle man
                        if c.product in self.middle_man_products and not c.is_buy:
                            # print(f'Responding to a cfp from {c.publisher} to buy {c.product}. Here it is: CFP = {c}')
                            self.request_negotiation(
                                cfp=c,
                                negotiator=AspirationNegotiator(
                                    ufun=self.input_negotiator_ufun,
                                    aspiration_type=self.agent_aspiration_type,
                                ),
                            )
                        # Respond to CFPs when we try to sell stuff for the middle man
                        if c.product in self.middle_man_products and c.is_buy:
                            # print(f'Responding to a cfp to sell CFP for product {c.product} by {c.publisher},  CFP = {c}')
                            self.request_negotiation(
                                cfp=c,
                                negotiator=AspirationNegotiator(
                                    ufun=self.output_negotiator_ufun,
                                    aspiration_type=self.agent_aspiration_type,
                                    max_aspiration=0.95,
                                ),
                            )

        # DEBUG INFO
        if self.verbose:
            # Print some debug info for development purposes.
            self.print_debug_info(
                plan_for_inputs,
                self.storage_history[self.input_index][-1],
                self.storage_history[self.output_index][-1],
            )

        # Save results of the hyper-parameter optimization
        if self.current_step + 1 == self.awi.n_steps:
            if not os.path.exists("my_results"):
                os.makedirs("my_results")
            results_file_name = "".join(
                random.choices(string.ascii_uppercase + string.digits, k=10)
            )
            with open(f"my_results/{results_file_name}.dat", "w") as file:
                file.write(
                    f"{self.marginal_calculation_on_sign_contract},"
                    f"{self.limit_sign_storage},"
                    f"{self.limit_number_buys_contracts_at_t},"
                    f"{self.limit_number_sales_contracts_at_t},"
                    f"{self.cfp_qtty_range_width},"
                    f"{self.input_index},{self.output_index},"
                    f"{self.num_intermediate_products},"
                    f"{self.wallet_history[-1]}"
                )

    def on_production_failure(self, failures: List[ProductionFailure]) -> None:
        """Will be called whenever a failure happens in one of the agent's factory's production lines"""
        if self.verbose:
            print(
                f'A failure on production occur at {self.awi.current_step}!!! = {[str(fail)  + "**" for fail in failures]}'
            )

    # ================================
    # Negotiation Control and Feedback
    # ================================

    def respond_to_negotiation_request(
        self, cfp: "CFP", partner: str
    ) -> Optional[Negotiator]:
        """Called whenever someone (partner) is requesting a negotiation with the agent about a Call-For-Proposals
        (cfp) that was earlier published by this agent to the bulletin-board

        Returning `None` means rejecting to enter this negotiation

        """
        if cfp.publisher == self.id and cfp.is_buy:
            neg_ufun = self.input_negotiator_ufun
        elif cfp.publisher == self.id and not cfp.is_buy:
            neg_ufun = self.output_negotiator_ufun
        else:
            if self.verbose:
                print(f"--- WARNING!!! WARNING!!! Rejecting a negotiation --- ")
            return None
        return AspirationNegotiator(
            ufun=neg_ufun, aspiration_type=self.agent_aspiration_type
        )

    def on_negotiation_failure(
        self,
        partners: List[str],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
        state: MechanismState,
    ) -> None:
        """Called whenever a negotiation ends without agreement"""
        if self.verbose and annotation["cfp"]["product"] == self.output_index:
            print(
                f"\tFailed to negotiate about output: q = {annotation['cfp']['quantity']}, p = {annotation['cfp']['unit_price']}, t = {annotation['cfp']['time']}"
            )
        if self.verbose and annotation["cfp"]["product"] == self.input_index:
            print(
                f"\tFailed to negotiate about input: q = {annotation['cfp']['quantity']}, p = {annotation['cfp']['unit_price']}, t = {annotation['cfp']['time']}"
            )
        if self.verbose and annotation["cfp"]["product"] in self.middle_man_products:
            print(
                f"\tMiddle Man Failed to negotiate about a cfp to buy? {annotation['cfp']['is_buy']} for product {annotation['cfp']['product']}. "
                f"Am I the publisher? {annotation['cfp']['publisher'] == self.id}"
            )

    # =============================
    # Contract Control and Feedback
    # =============================

    def sign_contract(self, contract: Contract) -> Optional[str]:
        """Called after the signing delay from contract conclusion to sign the contract. Contracts become binding only after they are signed. """
        sign = False

        # We don't sign contracts at the end or beyond.
        if contract.agreement["time"] >= self.awi.n_steps:
            return None

        # print(f' Sign this contract? Product: {contract}, product:', contract.annotation['cfp'].product, ' buyer: ', contract.annotation['buyer'] == self.id)

        # ---- STUFF ABOUT THE FACTORY
        if (
            contract.annotation["cfp"].product == self.input_index
            or contract.annotation["cfp"].product == self.output_index
        ):
            if self.marginal_calculation_on_sign_contract:
                value = self.agent_brain.marginal_value_contract(
                    current_time=self.current_step,
                    total_game_time=self.awi.n_steps,
                    contracts=self.signed_contracts_for_factory,
                    contract=contract,
                    agent_is_buy=contract.annotation["buyer"] == self.id,
                )
                """value = self.agent_brain.get_value_of_contract(current_time=self.current_step,
                                                               total_game_time=self.awi.n_steps,
                                                               contracts=self.signed_contracts_for_factory,
                                                               contract=contract,
                                                               agent_is_buy=contract.annotation['buyer'] == self.id)"""
            else:
                value = 1.0

            # Sign only if the value of this contract is good
            if (
                value >= 0
                and contract.annotation["buyer"] == self.id
                and contract.annotation["cfp"].product == self.input_index
                and self.contracted_buys_at_t[self.input_index][
                    contract.agreement["time"]
                ]
                <= self.limit_number_buys_contracts_at_t
                and self.awi.state.storage[self.output_index] <= self.limit_sign_storage
                and contract.agreement["time"] <= self.limit_sign_time_input_product
            ):
                sign = True
            elif (
                value >= 0
                and contract.annotation["buyer"] != self.id
                and contract.annotation["cfp"].product == self.output_index
                and self.contracted_sales_at_t[self.output_index][
                    contract.agreement["time"]
                ]
                <= self.limit_number_sales_contracts_at_t
                and sum(
                    [
                        self.contracted_buys_at_t[self.input_index][t]
                        for t in range(0, contract.agreement["time"] - 1)
                    ]
                )
                >= contract.agreement["quantity"]
            ):
                sign = True
            if sign:
                # print(f'CONTRACT REGISTERED AS SIGNED!: {contract}')
                self.signed_contracts_for_factory[contract.agreement["time"]] += [
                    (
                        contract.annotation["buyer"] == self.id,
                        (
                            contract.agreement["unit_price"],
                            contract.agreement["quantity"],
                            contract.agreement["time"],
                        ),
                    )
                ]
        # ---- MIDDLE MAN STUFF
        else:
            if (
                self.middle_man_active
                and contract.annotation["cfp"].product in self.middle_man_products
                and contract.annotation["buyer"] == self.id
                and self.contracted_buys_at_t[contract.annotation["cfp"].product][
                    contract.agreement["time"]
                ]
                <= self.limit_number_buys_contracts_at_t
                and self.awi.state.storage[contract.annotation["cfp"].product]
                <= self.limit_sign_storage
                and sum(
                    [
                        self.contracted_buys_at_t[self.input_index][
                            t
                        ]  # @todo warning. I am trying to limit the amount of stuff we buy, sometimes we get too much stuff.
                        for t in range(
                            max(0, contract.agreement["time"] - 10),
                            contract.agreement["time"] - 1,
                        )
                    ]
                )
                <= self.limit_sign_storage
            ):
                sign = True
            elif (
                self.middle_man_active
                and contract.annotation["cfp"].product in self.middle_man_products
                and contract.annotation["buyer"] != self.id
                and self.awi.state.storage[contract.annotation["cfp"].product]
                >= contract.agreement["quantity"]
            ):
                sign = True
        return True if sign else None

    def on_contract_signed(self, contract: Contract) -> None:
        """Called whenever a contract is signed by all partners"""
        if contract.annotation["buyer"] == self.id:
            if self.verbose:
                print(
                    f"\t\tSigned a buy contract for product {contract.annotation['cfp'].product}: "
                    f"q = {contract.agreement['quantity']}, p = {contract.agreement['unit_price']}, t = {contract.agreement['time']}"
                )
            self.contracted_buys_at_t[contract.annotation["cfp"].product][
                contract.agreement["time"]
            ] += contract.agreement["quantity"]
        elif contract.annotation["buyer"] != self.id:
            if self.verbose:
                print(
                    f"\t\tSigned a sell contract for product {contract.annotation['cfp'].product}: "
                    f"q = {contract.agreement['quantity']}, p = {contract.agreement['unit_price']}, t = {contract.agreement['time']}"
                )
            self.contracted_sales_at_t[contract.annotation["cfp"].product][
                contract.agreement["time"]
            ] += contract.agreement["quantity"]
        # @todo If we wanted, here we would evaluate the insurance and decide whether or not to buy it.
        # print(f'evaluating contract = {self.awi.evaluate_insurance(contract)}')
        # self.awi.buy_insurance(contract)

    def confirm_contract_execution(self, contract: Contract):
        """On contract execution, we keep track of some statistics."""
        if contract.annotation["buyer"] == self.id:
            self.executed_buys_at_t[contract.annotation["cfp"].product][
                self.awi.current_step
            ] = (
                self.executed_buys_at_t[contract.annotation["cfp"].product][
                    self.awi.current_step
                ][0]
                + contract.agreement["quantity"],
                self.executed_buys_at_t[contract.annotation["cfp"].product][
                    self.awi.current_step
                ][1]
                + contract.agreement["unit_price"],
            )
        else:
            self.executed_sales_at_t[contract.annotation["cfp"].product][
                self.awi.current_step
            ] = (
                self.executed_sales_at_t[contract.annotation["cfp"].product][
                    self.awi.current_step
                ][0]
                + contract.agreement["quantity"],
                self.executed_sales_at_t[contract.annotation["cfp"].product][
                    self.awi.current_step
                ][1]
                + contract.agreement["unit_price"],
            )
        return True

    def on_new_cfp(self, cfp: "CFP"):
        """Call whenever a CFP is posted. """
        if cfp.publisher != self.id:
            self.request_negotiation(
                cfp=cfp,
                negotiator=AspirationNegotiator(
                    ufun=self.output_negotiator_ufun
                    if cfp.is_buy
                    else self.input_negotiator_ufun,
                    aspiration_type=self.agent_aspiration_type,
                ),
            )

    # =============================
    # Helpers
    # =============================

    def get_num_my_breaches(self):
        """
        Read the breaches list and compute the number of breaches of the agent. This is for informational purposes only.
        :return:
        """
        the_breaches = self.awi.bb_query("breaches", None)
        num_my_breaches = 0
        if the_breaches:
            for i, b in the_breaches.items():
                if b["perpetrator"] == self.id:
                    num_my_breaches += 1
        return num_my_breaches

    def print_debug_info(self, plan_for_inputs, number_of_inputs, number_of_outputs):
        """
        A helper function to debug game play
        :param plan_for_inputs:
        :param number_of_inputs:
        :param number_of_outputs:
        :return:
        """
        # Print some information to learn about the agent's state
        print(
            f"\n************************************************* t = {self.awi.current_step} ************************************************* \n"
            f"\t Plan inp. \t= {plan_for_inputs} \n"
            f"\t Inputs  \t= {number_of_inputs} \n"
            f"\t Outputs \t= {number_of_outputs} \n"
            f"\t Money     \t= {round(self.awi.state.wallet, 2)} \n"
            f"\t Breaches \t= {self.get_num_my_breaches()} \n"
        )

        print_span = 8
        start = max(self.awi.current_step - print_span, 0)
        end = self.awi.current_step + 1
        table = PrettyTable(["-"] + [str(t) for t in range(start, end)])
        table.add_row(
            ["W"] + [round(money, 2) for money in self.wallet_history[start:]]
        )
        for p in range(0, self.num_intermediate_products + 2):
            table.add_row(
                [
                    f"ST_{p}"
                    + (
                        str("*")
                        if p == self.input_index or p == self.output_index
                        else str("_")
                    )
                ]
                + self.storage_history[p][start:]
            )
        table.add_row(["-----"] + ["-----" for t in range(start, end)])
        for p in range(0, self.num_intermediate_products + 2):
            table.add_row(
                [
                    f"CO_{p}"
                    + (
                        str("*")
                        if p == self.input_index or p == self.output_index
                        else str("_")
                    )
                ]
                + [
                    (self.contracted_buys_at_t[p][t], self.contracted_sales_at_t[p][t])
                    for t in range(start, end)
                ]
            )
            # table.add_row([f'CS_{p}'] + [self.contracted_sales_at_t[p][t] for t in range(start, end)])
            # table.add_row([f'EB_{p}'] + [self.executed_buys_at_t[p][t] for t in range(start, end)])
            # table.add_row([f'ES_{p}'] + [self.executed_sales_at_t[p][t] for t in range(start, end)])
        print(table)

        # Do stuff at the end of the game. Right now we are plotting.
        if self.awi.current_step + 1 == self.awi.n_steps:
            self.plot_results()

    def plot_results(self):
        """
        Function to plot results of a game
        :return:
        """
        plt.subplot(self.num_intermediate_products + 3, 1, 1)
        ax = plt.subplot(str(self.num_intermediate_products + 3) + "11")
        ax.set_ylabel("Wallet Balance")
        plt.plot(self.wallet_history)
        counter = 2
        for p in range(0, self.num_intermediate_products + 2):
            plt.subplot(self.num_intermediate_products + 3, 1, counter)
            ax = plt.subplot(
                str(self.num_intermediate_products + 3) + "1" + str(counter)
            )
            ax.set_ylabel(
                "P"
                + str(p)
                + ("*" if p == self.input_index or p == self.output_index else "")
            )
            if p != self.num_intermediate_products:
                ax.get_xaxis().set_visible(False)
            plt.plot(self.storage_history[p])
            counter += 1
        plt.show()


class MarginalCalculator(NVMFactoryManager):
    pass
