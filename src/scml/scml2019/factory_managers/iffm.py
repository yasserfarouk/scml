"""
This module implements a factory manager for the SCM league of ANAC 2019 competition. This basic version has what we
consider as the most important callbacks. Please refer to the [http://www.yasserm.com/scml/scml.pdf](game description)
for all the callbacks.

Your agent can learn about the state of the world and itself by accessing properties in the AWI it has. For example::

self.awi.n_steps  # gives the number of simulation steps

You can access the state of your factory as::

self.awi.state

Your agent can act in the world by calling methods in the AWI it has. For example:

>> self.awi.register_cfp(cfp)  # registers a new CFP

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
import math
import random

import negmas
from negmas import (
    Contract,
    Breach,
    MechanismState,
    AgentMechanismInterface,
    RenegotiationRequest,
)
from negmas.helpers import get_class
from negmas.negotiators import Negotiator, Controller
from negmas.outcomes import Outcome, ResponseType
from typing import Dict, Any, Callable, Collection, Type, List, Optional, Union

from scml.scml2019.awi import SCMLAWI
from scml.scml2019.common import (
    ProductionReport,
    SCMLAgreement,
    CFP,
    Loan,
    ProductionFailure,
    FinancialReport,
)
from scml.scml2019.schedulers import Scheduler, ScheduleInfo, GreedyScheduler
from scml.scml2019.simulators import (
    FactorySimulator,
    FastFactorySimulator,
    temporary_transaction,
)
from .builtins import (
    PessimisticNegotiatorUtility,
    NegotiatorUtility,
    OptimisticNegotiatorUtility,
    DoNothingFactoryManager,
)


class PrintingFactoryManager(DoNothingFactoryManager):
    def __init__(
        self,
        isPrint=True,
        printDepth=3,
        simulator_type: Union[str, Type[FactorySimulator]] = FastFactorySimulator,
        scheduler_type: Union[str, Type[Scheduler]] = GreedyScheduler,
        scheduler_params: Optional[Dict[str, Any]] = None,
        optimism: float = 0.0,
        negotiator_type: Union[str, Type[Negotiator]] = negmas.sao.AspirationNegotiator,
        negotiator_params: Optional[Dict[str, Any]] = None,
        n_retrials=5,
        use_consumer=True,
        reactive=True,
        sign_only_guaranteed_contracts=False,
        riskiness=0.0,
        max_insurance_premium: float = 0.0,
        reserved_value: float = -float("inf"),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._is_print = isPrint
        if not self._is_print:
            printDepth = 0
        self._print_depth = printDepth

        self.negotiation_margin = int(round(n_retrials * max(0.0, 1.0 - riskiness)))
        self.max_insurance_premium = max_insurance_premium
        self.contract_schedules: Dict[str, ScheduleInfo] = {}
        self.scheduler_type: Type[Scheduler] = get_class(
            scheduler_type, scope=globals()
        )
        self.scheduler: Scheduler = None
        self.scheduler_params: Dict[
            str, Any
        ] = scheduler_params if scheduler_params is not None else {}

    def step(self, *args, **kwargs):
        super().step(*args, **kwargs)
        self._dump_data()

    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)

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

        self._step_first_material_came = -1
        self._total_production = 0
        self._partner_credit = {}
        self._partner_credit_history = [{} for i in range(self.awi.n_steps)]
        self._process_stat = [
            [0 for j in range(self.awi.n_steps)] for i in range(self.scheduler.n_lines)
        ]
        self._process_continue_stat = [
            [False for j in range(self.awi.n_steps)]
            for i in range(self.scheduler.n_lines)
        ]
        self._line_list = [
            i.line for i in sorted(self.awi.state.profiles, key=lambda x: x.cost)
        ]

        self.buy = {}
        for i in self.consuming.keys():
            self.buy[i] = [0 for j in range(self.awi.n_steps)]
        self.sell = {}
        for i in self.producing.keys():
            self.sell[i] = [0 for j in range(self.awi.n_steps)]
        if self._is_print and self._print_depth >= 1:
            print("--------")
            print(str(self.name) + "'s dump data initialization is done.")

        self._agent_layer_min = min(self.consuming.keys())
        self._agent_layer_max = min(self.producing.keys()) - 1
        self._max_layer = max([i.id for i in self.products])
        self._running_last_step = max(
            self.awi.n_steps - 1 - (self._max_layer - self._agent_layer_max), 0
        )
        return None

    def _del_scheduled_schedule(self):
        for _line in range(self.scheduler.n_lines):
            for i in range(self.awi.n_steps):
                if (
                    self._process_stat[_line][i] != 0
                    and self._process_stat[_line][i][3:] == "S"
                ):
                    self._process_stat[_line][i] = 0
                    self._process_continue_stat[_line][i] = False

    def _process_schedule_for_dump(self):
        self._del_scheduled_schedule()
        # if self._is_print:
        #    print(self.contract_schedules)
        _strage_estimated = {}
        _sell_estimated = {}
        _first = True
        for k, v in self.contract_schedules.items():
            _start = v.start
            _end = v.end
            for _need in v.needs:
                _product_id = _need.product
                _need_quantity = _need.quantity_to_buy
                _need_at = _need.step
            for job in v.jobs:
                if job.action == "run":
                    # 厳密には違う
                    _profile = job.contract.annotation["cfp"].product - 1
                    _time = job.time
                    _line = job.line
                    _quantity = job.contract.agreement["quantity"]
                    _p_name = "P" + ("0" + str(_profile))[-2:]
                    # if _time + _quantity > self.awi.n_steps:
                    #    if _first:
                    #        _first = False
                    #        print(self.contract_schedules)
                    #    print(job)
                    #    continue
                    for i in range(_time, min(_time + _quantity, self.awi.n_steps)):
                        if self._process_stat[_line][i] == 0 or (
                            self._process_stat[_line][i][3:] != "F"
                            and self._process_stat[_line][i][3:] != "D"
                        ):
                            self._process_stat[_line][i] = _p_name + "S"
                    for i in range(_time + 1, min(_time + _quantity, self.awi.n_steps)):
                        if self._process_stat[_line][i] == 0 or (
                            self._process_stat[_line][i][3:] != "F"
                            and self._process_stat[_line][i][3:] != "D"
                        ):
                            self._process_continue_stat[_line][i] = True
                else:
                    if _first:
                        _first = False
                        if self._is_print:
                            print(self.contract_schedules)
                    if self._is_print:
                        print(job)

    def _print_schedule(self, start=None, end=None):
        if start == None:
            _range_start = max(0, self.awi.current_step - 10)
        else:
            _range_start = start
        if end == None:
            _range_end = min(self.awi.n_steps, self.awi.current_step + 10)
        else:
            _range_end = end
        _print_str = "TimeStep|"
        for i in range(_range_start, _range_end):
            if i == self.awi.current_step:
                _print_str += "\033[1m"
            _print_str += ("000" + str(i))[-4:]
            if i == self.awi.current_step:
                _print_str += "\033[0m"
            _print_str += "|"
        print(_print_str)
        # sell/buy
        for k, v in self.buy.items():
            _print_str = "BUY  " + self.products[k].name + " |"
            for i in range(_range_start, _range_end):
                _print_str += ("0000" + str(v[i]))[-4:] + "|"
        print(_print_str)
        for k, v in self.sell.items():
            _print_str = "SELL " + self.products[k].name + " |"
            for i in range(_range_start, _range_end):
                _print_str += ("0000" + str(v[i]))[-4:] + "|"
        print(_print_str)
        # schedule
        for i in range(self.scheduler.n_lines):
            _print_str = "Line " + ("00" + str(i))[-3:]
            for j in range(_range_start, _range_end):
                # 新規の場合 |P01
                # 予定の場合 |P01S
                # 終了の場合 |P01D
                # 失敗の場合 |P01F
                # 継続の場合  =>01F
                # なしの場合 |
                if self._process_continue_stat[i][j]:
                    if self._process_stat[i][j] != 0:
                        _print_str += (
                            "=>" + (str(self._process_stat[i][j][1:]) + " ")[:2]
                        )
                        if (
                            str(self._process_stat[i][j][3]) == "F"
                            or str(self._process_stat[i][j][3]) == "D"
                        ):
                            _print_str += (
                                "\033[31m"
                                + (str(self._process_stat[i][j][1:]) + " ")[2]
                                + "\033[0m"
                            )
                        else:
                            _print_str += (str(self._process_stat[i][j][1:]) + " ")[2]
                    else:
                        _print_str += "=>   "
                else:
                    _print_str += "|"
                    if self._process_stat[i][j] != 0:
                        _print_str += (str(self._process_stat[i][j]) + " ")[:3]
                        if (
                            str(self._process_stat[i][j][3]) == "F"
                            or str(self._process_stat[i][j][3]) == "D"
                        ):
                            _print_str += (
                                "\033[31m"
                                + (str(self._process_stat[i][j]) + " ")[3]
                                + "\033[0m"
                            )
                        else:
                            _print_str += (str(self._process_stat[i][j]) + " ")[3]
                    else:
                        _print_str += "    "
            _print_str += "|"
            print(_print_str)
        return None

    def _dump_data(self, additional_print=None):
        self._process_schedule_for_dump()
        if self._is_print and self._print_depth >= 1:
            print("--------")
            if additional_print:
                print(additional_print)
            # stepを表示
            print(
                "time:\t\t"
                + str(self.awi.current_step)
                + " steps / "
                + str(self.awi.n_steps)
                + " steps ("
                + str(self.awi.relative_time)
                + ")"
            )
            # 自分の名前を表示
            print("name:\t\t" + str(self.name))
            print("id:\t\t" + str(self.id))
            # products 概要
            print(
                "produce:\t"
                + str([self.products[i].name for i in self.consuming.keys()])
                + "=>"
                + str([self.products[i].name for i in self.producing.keys()])
            )
            for process in self.awi.processes:
                _print_str = "process" + ("0" + str(process.id))[-2:] + ":\t"
                _is_first = True
                for _input in process.inputs:
                    if not _is_first:
                        _print_str += " + "
                    _print_str += (
                        str(self.products[_input.product].name)
                        + ":"
                        + str(_input.quantity)
                        + "@"
                        + str(int(_input.step))
                    )
                _print_str += " ==> "
                for _output in process.outputs:
                    _print_str += (
                        str(self.products[_output.product].name)
                        + ":"
                        + str(_output.quantity)
                        + "@"
                        + str(int(_output.step))
                    )
                print(_print_str)
            # lines
            print("lines:\t\t" + str(self.awi.state.n_lines) + " lines")
            print(
                "line usage:\t"
                + str(sum(i == 1 for i in self.awi.state.line_schedules))
                + "/"
                + str(len(self.awi.state.line_schedules))
            )
            # storage
            _print_str = "storage:\t"
            _is_stock = False
            for _p_id, _n_stocks in self.awi.state.storage.items():
                if _n_stocks == 0:
                    continue
                _is_stock = True
                _print_str += (
                    str(self.products[_p_id].name) + ":" + str(_n_stocks) + ", "
                )
            if not _is_stock:
                _print_str += "---"
            print(_print_str)
            # products catalog price
            _print_str = "products catalog price:\t"
            for _product in self.awi.products:
                _print_str += (
                    str(_product.name) + ":" + str(_product.catalog_price) + ", "
                )
            print(_print_str)
        # commandのプリント
        # if self._is_print:
        #    print(self.awi.state.commands)
        # process_statの修正
        for _command in self.awi.state.commands:
            if _command.profile == None:
                continue
            else:
                _line = _command.profile.line
                _cost = _command.profile.cost
                _process = _command.profile.process
                _p_name = "P" + ("0" + str(_process.id))[-2:]
                _start = _command.beg
                _end = _command.end
                for i in range(_start, _end):
                    if self._process_stat[_line][i] == _p_name + "S":
                        if _end <= self.awi.current_step:
                            self._process_stat[_line][i] = _p_name + "D"
                        else:
                            self._process_stat[_line][i] = _p_name
                    elif self._process_stat[_line][i] == _p_name + "F":
                        continue
                    else:
                        if _end <= self.awi.current_step:
                            self._process_stat[_line][i] = _p_name + "D"
                        else:
                            self._process_stat[_line][i] = _p_name
                for i in range(_start + 1, _end):
                    self._process_continue_stat[_line][i] = True
                self._process_continue_stat[_line][_start] = False

        self._running_steps_left = self._running_last_step - self.awi.current_step + 1
        self._total_buy_steps = max(self.awi.current_step - self._agent_layer_min, 0)
        self._total_sell_steps = max(
            self.awi.current_step - self._agent_layer_max - 1, 0
        )
        self._total_production = sum(
            [
                sum(
                    [
                        self._process_stat[i][j] != 0
                        and self._process_stat[i][j][3:] == "D"
                        for j in range(self.awi.current_step)
                    ]
                )
                for i in range(self.scheduler.n_lines)
            ]
        )
        self._buy_offer_before = sum(
            [sum(self.buy[i][: self.awi.current_step]) for i in self.consuming.keys()]
        )
        self._buy_offer_before_10 = sum(
            [
                sum(self.buy[i][(self.awi.current_step - 10) : self.awi.current_step])
                for i in self.consuming.keys()
            ]
        )
        self._material_bought = self._total_production + sum(
            [self.awi.state.storage[i] for i in self.consuming.keys()]
        )
        if self._material_bought != 0 and self._step_first_material_came == 0:
            self._step_first_material_came = self.awi.current_step
        self._buy_order_excuted_ratio = (
            self._material_bought * 1.0 / (self._buy_offer_before + 1e-6)
        )
        self._buy_offer_after = sum(
            [sum(self.buy[i][self.awi.current_step :]) for i in self.consuming.keys()]
        )
        self._buy_offer_after_10 = sum(
            [
                sum(self.buy[i][self.awi.current_step : self.awi.current_step + 10])
                for i in self.consuming.keys()
            ]
        )
        self._buy_offer_before_ave = (
            self._buy_offer_before * 1.0 / (self._total_buy_steps + 1e-6)
        )
        self._buy_offer_before_ave_10 = self._buy_offer_before_10 * 0.1
        self._buy_offer_after_ave = (
            self._buy_offer_after * 1.0 / (self._running_steps_left + 1e-6)
        )
        self._buy_offer_after_ave_10 = self._buy_offer_after_10 * 0.1
        self._buy_offer_excution_est_min = (
            self._buy_offer_after * self._buy_order_excuted_ratio
        )
        self._buy_offer_excution_est_max = (
            max(
                self._buy_offer_after_ave,
                self._buy_offer_after_ave,
                self._buy_offer_after_ave_10,
                self._buy_offer_before_ave_10,
            )
            * self._running_steps_left
        )
        self._sell_offer_before = sum(
            [sum(self.sell[i][: self.awi.current_step]) for i in self.producing.keys()]
        )
        self._sell_offer_after = sum(
            [sum(self.sell[i][self.awi.current_step :]) for i in self.producing.keys()]
        )
        self._products_sold = self._total_production - sum(
            [self.awi.state.storage[i] for i in self.producing.keys()]
        )
        if self._is_print and self._print_depth >= 1:
            print("Money:\t\t" + str(self.awi.state.wallet))
            print(
                "Agent layer:\t\t"
                + str(self._agent_layer_min)
                + "-"
                + str(self._agent_layer_max)
            )
            print("Max layer:\t\t" + str(self._max_layer))
            print("The last step factory runs:\t" + str(self._running_last_step))
            print("Running steps last:\t\t" + str(self._running_steps_left))
            print("Total buy setps:\t\t" + str(self._total_buy_steps))
            print("Total sell setps:\t\t" + str(self._total_sell_steps))
            print("Total Production:\t\t" + str(self._total_production))
            print("Buy offer Before:\t\t" + str(self._buy_offer_before))
            print("Buy offer Before10:\t\t" + str(self._buy_offer_before_10))
            print("Buy offer Before (Ave.):\t" + str(self._buy_offer_before_ave))
            print("Buy offer Before10 (Ave.):\t" + str(self._buy_offer_before_ave_10))
            print("Buy Offer After:\t\t" + str(self._buy_offer_after))
            print("Buy Offer After10:\t\t" + str(self._buy_offer_after_10))
            print("Buy Offer After (Ave.):\t\t" + str(self._buy_offer_after_ave))
            print("Buy Offer After10 (Ave.):\t" + str(self._buy_offer_after_ave_10))
            print("Material bought:\t\t" + str(self._material_bought))
            print(
                "When material was bought at first:\t"
                + str(self._step_first_material_came)
            )
            print("Buy Order Excuted ratio:\t" + str(self._buy_order_excuted_ratio))
            print(
                "Buy Order Excution EST.(min):\t"
                + str(self._buy_offer_excution_est_min)
            )
            print(
                "Buy Order Excution EST.(max):\t"
                + str(self._buy_offer_excution_est_max)
            )
            print("Sell Offer Before:\t\t" + str(self._sell_offer_before))
            print("Sell Offer After:\t\t" + str(self._sell_offer_after))
            print("Sold products:\t\t" + str(self._products_sold))
            print("Other agents data:")
            for partnerid, data in sorted(self._partner_credit.items()):
                print(str(partnerid) + ":\t\t" + str(data))
        self._partner_credit_history[self.awi.current_step] = self._partner_credit
        # process_stat print
        if self._is_print and self._print_depth >= 1:
            if self.awi.current_step < self.scheduler.n_steps - 1:
                self._print_schedule()
            else:
                print("Final Schedule:")
                for i in range(int((self.awi.n_steps - 1) / 20) + 1):
                    self._print_schedule(
                        start=i * 20, end=min((i + 1) * 20, self.awi.n_steps)
                    )
            # print(self.scheduler.simulator.line_schedules_at(self.awi.current_step))
            # print(self.scheduler.simulator.line_schedules_to(10))

    def _dump_failure(self, failures: List[ProductionFailure]) -> None:
        if self._is_print and self._print_depth >= 2:
            print("--------")
            print(str(self.name) + "'s production failures.")
            print(failures)
        for _fail in failures:
            _command = _fail.command
            _line = _command.profile.line
            _cost = _command.profile.cost
            _process = _command.profile.process
            _p_name = "P" + ("0" + str(_process.id))[-2:] + "F"
            _start = _command.beg
            _end = _command.end
            for i in range(_start, _end):
                self._process_stat[_line][i] = _p_name
            for i in range(_start + 1, _end):
                self._process_continue_stat[_line][i] = True
        # print("--------")
        pass

    def _split_contract(self, contract: Contract) -> Any:
        _cfp = contract.annotation["cfp"]
        _seller_id = contract.annotation["seller"]
        _buyer_id = contract.annotation["buyer"]
        _time = contract.agreement["time"]
        _quantity = contract.agreement["quantity"]
        _unit_price = contract.agreement["unit_price"]
        _product_id = _cfp.product
        return (_cfp, _seller_id, _buyer_id, _time, _quantity, _unit_price, _product_id)

    def _dump_contract(self, contract: Contract, addPrint="", isSigned=False) -> None:
        if self._is_print and self._print_depth >= 2:
            print("--------")
            print(str(self.name) + "'s contract " + str(addPrint))
        (
            _cfp,
            _seller_id,
            _buyer_id,
            _time,
            _quantity,
            _unit_price,
            _product_id,
        ) = self._split_contract(contract=contract)
        if self.id == _buyer_id:
            _is_buy = True
        elif self.id == _seller_id:
            _is_buy = False
        else:
            _is_buy = None
            print("\033[31mBUY/SELL ERROR\033[0m")
            pass
        if isSigned:
            if _is_buy:
                if _product_id in self.buy:
                    self.buy[_product_id][_time] += _quantity
            else:
                if _product_id in self.sell:
                    self.sell[_product_id][_time] += _quantity
        if self._is_print and self._print_depth >= 2:
            _print_str = "Buy: " if _is_buy else "Sell: "
            _print_str += str(self.products[_product_id].name)
            _print_str += (
                ":"
                + str(_quantity)
                + "@"
                + str(_time)
                + " price:"
                + str(_unit_price)
                + "x"
                + str(_quantity)
            )
            print(_print_str)
            print("w/ " + _seller_id if _is_buy else _buyer_id)
            print("insurance premier:\t" + str(self.awi.evaluate_insurance(contract)))
        pass

    def on_contract_signed(self, contract: Contract) -> None:
        super().on_contract_signed(contract)
        self._dump_contract(contract, addPrint="SIGNED", isSigned=True)
        pass

    def on_negotiation_success(
        self, contract: Contract, mechanism: AgentMechanismInterface
    ) -> None:
        super().on_negotiation_success(contract=contract, mechanism=mechanism)
        self._dump_contract(contract, addPrint="AGREED")
        if self._is_print and self._print_depth >= 2:
            print("To be signed at " + str(contract.to_be_signed_at))
        pass

    def on_production_failure(self, failures: List[ProductionFailure]) -> None:
        super().on_production_failure(failures)
        self._dump_failure(failures)
        pass

    def on_production_success(self, reports: List[ProductionReport]) -> None:
        super().on_production_success(reports)
        if self._is_print and self._print_depth >= 3:
            print("--------")
            print(self.name)
            print("success", reports)

    def on_inventory_change(self, product: int, quantity: int, cause: str) -> None:
        super().on_inventory_change(product, quantity, cause)
        if self._is_print and self._print_depth >= 3:
            print("--------")
            print(self.name)
            print("change", product, quantity, cause)

    def on_cash_transfer(self, amount: float, cause: str) -> None:
        super().on_cash_transfer(amount, cause)
        if self._is_print and self._print_depth >= 2:
            print("--------")
            print(self.name)
            print("cash", amount, cause)

    def on_new_report(self, report: FinancialReport):
        super().on_new_report(report)
        if self._is_print and self._print_depth >= 3:
            print("--------")
            print(self.name)
            print("report", report)

    def on_contract_executed(self, contract: Contract) -> None:
        super().on_contract_executed(contract)
        if self._is_print and self._print_depth >= 3:
            print("--------")
            print("Contract was excuted")
            print(contract)
        (
            _cfp,
            _seller_id,
            _buyer_id,
            _time,
            _quantity,
            _unit_price,
            _product_id,
        ) = self._split_contract(contract=contract)
        if self.id == _buyer_id:
            _is_buy = True
        elif self.id == _seller_id:
            _is_buy = False
        else:
            _is_buy = None
            print("\033[31mBUY/SELL ERROR\033[0m")
            pass
        if _is_buy:
            if self._is_print and self._print_depth >= 2:
                print(_seller_id)
            if _seller_id not in self._partner_credit:
                self._partner_credit[_seller_id] = {
                    "breach": 0,
                    "excuted": 0,
                    "cancelled": 0,
                }
            self._partner_credit[_seller_id]["excuted"] += 1
        pass

    def on_contract_breached(
        self, contract: Contract, breaches: List[Breach], resolution: Optional[Contract]
    ) -> None:
        super().on_contract_breached(contract, breaches, resolution)
        if self._is_print and self._print_depth >= 3:
            print("--------")
            print("\033[31mContract was breached\033[0m")
            print(contract)
            print(breaches)
        for breach in breaches:
            if self.id == breach.perpetrator:
                continue
            if self._is_print and self._print_depth >= 2:
                print(breach.perpetrator)
            if breach.perpetrator not in self._partner_credit:
                self._partner_credit[breach.perpetrator] = {
                    "breach": 0,
                    "excuted": 0,
                    "cancelled": 0,
                }
            self._partner_credit[breach.perpetrator]["breach"] += 1
        pass


class AspirationNego2(negmas.sao.AspirationNegotiator):
    def aspiration(self, t: float) -> float:
        """
        The aspiration level

        Args:
            t: relative time (a number between zero and one)

        Returns:
            aspiration level
        """
        if t is None:
            raise ValueError(
                f"Aspiration negotiators cannot be used in negotiations with no time or #steps limit!!"
            )
        if self.exponent < 1e-7:
            return 0.0
        return self.max_aspiration * ((1.0 - math.pow(t, 10.0)) * 0.5 + 0.5)


class NonDemandDrivenAgent(PrintingFactoryManager):
    """
    This is the only class you *need* to implement. The current skeleton has a basic do-nothing implementation.
    You can modify any parts of it as you need. You can act in the world by calling methods in the agent-world-interface
    instantiated as `self.awi` in your agent. See the documentation for more details

    """

    def __init__(
        self,
        optimism: float = 0.0,
        negotiator_type: Union[
            str, Type[Negotiator]
        ] = AspirationNego2,  # negmas.sao.AspirationNegotiator,
        negotiator_params: Optional[Dict[str, Any]] = None,
        sign_only_guaranteed_contracts=False,
        *args,
        **kwargs,
    ):
        super().__init__(
            isPrint=False,
            printDepth=4,
            negotiator_type=negotiator_type,
            max_insurance_premium=0.0,
            *args,
            **kwargs,
        )

        self.negotiator_type = get_class(negotiator_type, scope=globals())
        self.sign_only_guaranteed_contracts = sign_only_guaranteed_contracts
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
        self.negotiator_type = get_class(negotiator_type, scope=globals())
        self.negotiator_params = (
            negotiator_params if negotiator_params is not None else {}
        )

    # =====================
    # Time-Driven Callbacks
    # =====================

    def init(self):
        """Called once after the agent-world interface is initialized"""
        super().init()
        self.__reserved_value = -float("inf")

        self.add_i = 0
        pass

    def step(self):
        """Called at every production step by the world"""
        super().step()
        # 0. remove all my CFPs
        self.awi.bb_remove(section="cfps", query={"publisher": self})

        ###
        # production
        ##
        # j = 0
        # for _p_id, _n_stocks in self.awi.state.storage.items():
        #    if _p_id in self.consuming.keys():
        #        for i in range(min(int(self.scheduler.n_lines),int(_n_stocks + self.buy[_p_id][self.awi.current_step]))):
        #            _line = self._line_list[j]
        #            j += 1
        #            if self.awi.current_step < self.awi.n_steps:
        #                #if self._process_stat[_line][self.awi.current_step] == 0:
        #                    job = Job(
        #                        profile=_line, time=self.awi.current_step, line=_line, action='run',
        #                        contract=None,
        #                        override=True
        #                        )
        #                    self.awi.schedule_job(job,contract=None)

        ###
        # Order
        ###
        f = self._buy_order_excuted_ratio
        ci = self._buy_offer_after_ave_10
        clambda = f * ci

        co = (
            sum(
                [
                    sum(
                        self.sell[i][self.awi.current_step : self.awi.current_step + 10]
                    )
                    for i in self.producing.keys()
                ]
            )
            * 0.1
        )
        storage = sum(
            [_n_stocks for _p_id, _n_stocks in self.awi.state.storage.items()]
        )
        mu = max(co - storage * 0.1, 0)

        hatL = 10

        tlambda1 = hatL / (1 + hatL) * mu
        tlambda2 = (-hatL + math.sqrt(hatL ** 2 + 4 * hatL)) * 0.5 * mu

        # 切り替え可能
        tlambda = tlambda1
        if f != 0:
            ti = tlambda2 / f
        else:
            ti = 10

        di = ti - ci

        if di > 0:
            # s = 0.09
            s = 0.09 * (self._agent_layer_min + 1) / self._max_layer
        else:
            s = 1.0

        self.add_i += s * di

        if self.add_i < 0:
            self.add_i = 0

        self._needs = self.add_i

        if self._is_print and self._print_depth >= 2:
            print("Controll:")
            print(
                "fail rate: {}, current input: {}, current lambdda: {}, current onuput: {}, mu: {}, target lambda: {},\
target input: {}, delta input: {}".format(
                    f, ci, clambda, co, mu, tlambda, ti, di
                )
            )
            print(self.add_i, s * di)

        ####
        # To Do:
        #    最後の打ち切り
        ###
        if co != 0:
            step_cut = hatL / co * 2
        else:
            step_cut = hatL

        num_cfp = 1
        if (
            self.awi.current_step <= self._running_last_step - step_cut - 10
            and self._needs > 0
        ):
            for product_id in self.consuming.keys():
                product = self.products[product_id]
                if product.catalog_price is None:
                    price_range = (0.0, 100.0)
                else:
                    price_range = (0, 1.5 * product.catalog_price)
                _quantity_max = max(int(self.add_i * 0.3), 1)
                for i in range(num_cfp):
                    time = min(
                        self.awi.current_step + int(random.randint(1, 10)),
                        self._running_last_step,
                    )
                    cfp = CFP(
                        is_buy=True,
                        publisher=self.id,
                        product=product_id,
                        time=time,
                        unit_price=price_range,
                        quantity=(1, int(_quantity_max)),
                    )
                    if self._is_print:
                        print(cfp)
                    self.awi.register_cfp(cfp)

    # ==========================
    # Important Events Callbacks
    # ==========================

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
            name=self.name + ">" + cfp.publisher[:4], **self.negotiator_params
        )
        ufun = self.ufun_factory(self, self._create_annotation(cfp=cfp))
        ufun.reserved_value = self.__reserved_value
        self.request_negotiation(negotiator=neg, cfp=cfp, ufun=ufun)
        # normalize(, outcomes=cfp.outcomes, infeasible_cutoff=-1)

    def _process_sell_cfp(self, cfp: "CFP"):
        if self.awi.is_bankrupt(cfp.publisher):
            return None

    def on_new_cfp(self, cfp: CFP) -> None:
        """Called when a new CFP for a product for which the agent registered interest is published"""
        if cfp.satisfies(
            query={"is_buy": True, "products": list(self.producing.keys())}
        ):
            self._process_buy_cfp(cfp)
        if cfp.satisfies(
            query={"is_buy": False, "products": list(self.consuming.keys())}
        ):
            self._process_sell_cfp(cfp)

    def on_remove_cfp(self, cfp: CFP) -> None:
        """Called when a new CFP for a product for which the agent registered interest is removed"""
        super().on_remove_cfp(cfp)

    # ====================
    # Production Callbacks
    # ====================

    def on_production_failure(self, failures: List[ProductionFailure]) -> None:
        """Will be called whenever a failure happens in one of the agent's factory's production lines"""
        super().on_production_failure(failures)

    def on_production_success(self, reports: List[ProductionReport]) -> None:
        """Will be called whenever some production succeeds in the factory owned by the manager"""
        super().on_production_success(reports)

    def on_inventory_change(self, product: int, quantity: int, cause: str) -> None:
        """Will be called whenever there is a change in inventory for a cause other than production (e.g. contract
        execution)."""
        super().on_inventory_change(product, quantity, cause)

    def on_cash_transfer(self, amount: float, cause: str) -> None:
        """Called whenever there is a cash transfer to or from the agent"""
        super().on_cash_transfer(amount, cause)

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
        if self.awi.is_bankrupt(partner):
            return None
        ufun_ = self.ufun_factory(
            self, self._create_annotation(cfp=cfp, partner=partner)
        )
        ufun_.reserved_value = self.__reserved_value
        neg = self.negotiator_type(
            name=self.name + "*" + partner[:4], **self.negotiator_params, ufun=ufun_
        )
        return neg

    def on_neg_request_rejected(self, req_id: str, by: Optional[List[str]]):
        """Called when a negotiation request sent by this agent is rejected. The ``req_id`` is a unique identifier
        for this negotiation request.

        Remarks:

            - You **MUST** call super() here before doing anything else.

        """
        super().on_neg_request_rejected(req_id, by)

    def on_neg_request_accepted(self, req_id: str, mechanism: AgentMechanismInterface):
        """Called when a requested negotiation is accepted. The ``req_id`` is a unique identifier for this negotiation
        request."""
        super().on_neg_request_accepted(req_id=req_id, mechanism=mechanism)

    def on_negotiation_failure(
        self,
        partners: List[str],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
        state: MechanismState,
    ) -> None:
        """Called whenever a negotiation ends without agreement"""
        super().on_negotiation_failure(partners, annotation, mechanism, state)

    def on_negotiation_success(
        self, contract: Contract, mechanism: AgentMechanismInterface
    ) -> None:
        """Called whenever a negotiation ends with agreement"""
        super().on_negotiation_success(contract, mechanism)

    # =============================
    # Contract Control and Feedback
    # =============================

    def sign_contract(self, contract: Contract) -> Optional[str]:
        """Called after the signing delay from contract conclusion to sign the contract. Contracts become binding
        only after they are signed.

        Remarks:

            - Return `None` if you decided not to sign the contract. Return your ID (self.id) otherwise.

        """
        (
            _cfp,
            _seller_id,
            _buyer_id,
            _time,
            _quantity,
            _unit_price,
            _product_id,
        ) = self._split_contract(contract=contract)
        if self.id == _buyer_id:
            _is_buy = True
        elif self.id == _seller_id:
            _is_buy = False
        else:
            _is_buy = None
            print("\033[31mBUY/SELL ERROR\033[0m")

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

        if _is_buy:
            if self._is_print and self._print_depth >= 2:
                print("--------")
                print(self.name)
                print("Signing Contract")
                print(str(self._needs) + " needs left")
            if self._needs < _quantity:
                if self._is_print and self._print_depth >= 2:
                    print("Deny to sign")
                return None
            else:
                self._needs -= _quantity

        self.contract_schedules[contract.id] = schedule
        return signature

    def on_contract_signed(self, contract: Contract) -> None:
        """Called whenever a contract is signed by all partners"""
        # super().on_contract_signed(contract)
        # print("保険")
        # print(contract)
        # self.awi.buy_insurance(contract)
        super().on_contract_signed(contract)
        schedule = self.contract_schedules[contract.id]
        if schedule is not None and schedule.valid:
            self._execute_schedule(schedule=schedule, contract=contract)

    def on_contract_cancelled(self, contract: Contract, rejectors: List[str]) -> None:
        """Called whenever at least a partner did not sign the contract"""
        super().on_contract_cancelled(contract, rejectors)

    def on_contract_nullified(
        self, contract: Contract, bankrupt_partner: str, compensation: float
    ) -> None:
        """Will be called whenever a contract the agent is involved in is nullified because another partner went
        bankrupt"""
        super().on_contract_nullified(contract, bankrupt_partner, compensation)

    def on_contract_executed(self, contract: Contract) -> None:
        """Called whenever a contract is fully executed without any breaches"""
        super().on_contract_executed(contract)

    def on_contract_breached(
        self, contract: Contract, breaches: List[Breach], resolution: Optional[Contract]
    ) -> None:
        """Called after full processing of contracts that were breached.

        Args:

            contract: The contract breached
            breaches: A list of all breaches committed
            resolution: If not None, the resolution contract resulting from re-negotiation (if any).

        Remarks:

            - Even if renegotiation resulted in an agreement, this callback will be called.

        """
        super().on_contract_breached(contract, breaches, resolution)

    def confirm_contract_execution(self, contract: Contract) -> bool:
        """Called at the delivery time specified in the contract to confirm that the agent wants to execute it.

        Returning False is equivalent to committing a `refusal-to-execute` breach of maximum level (1.0).

        """
        return super().confirm_contract_execution(contract)

    def confirm_partial_execution(
        self, contract: Contract, breaches: List[Breach]
    ) -> bool:
        """Will be called whenever a contract cannot be fully executed due to breaches by the other partner.

        Will not be called if both partners committed breaches.
        """
        return super().confirm_partial_execution(contract, breaches)

    # ====================================
    # Re-negotiations when breaches happen
    # ====================================

    def set_renegotiation_agenda(
        self, contract: Contract, breaches: List[Breach]
    ) -> Optional[RenegotiationRequest]:
        """Will be called when a contract fails to be concluded due to any kind of breach to allow partners to start
        a re-negotiation that may lead to a new contract that nullifies these breaches. It is always called on agents
        in descending order of their total breach levels on this contract.

        Returning `None` will mean that you pass your opportunity to set the renegotiation agenda.

        """
        return super().set_renegotiation_agenda(contract, breaches)

    def respond_to_renegotiation_request(
        self, contract: Contract, breaches: List[Breach], agenda: RenegotiationRequest
    ) -> Optional[Negotiator]:
        """Will be called whenever a renegotiation agenda is set by one agent after a breach asking the other agent to
        join the re-negotiation.

        Returning None means that you refuse to renegotiate.

        """
        return super().respond_to_negotiation_request(contract, breaches, agenda)

    # ===================================
    # Loans, Financial issues and Banking
    # ===================================

    def confirm_loan(self, loan: Loan, bankrupt_if_rejected: bool) -> bool:
        """Will be called whenever the agent needs to pay for something (e.g. loan interest, products it bought) but
        does not have enough money in its wallet.

        Args:

            loan: The loan information
            bankrupt_if_rejected: If this is true, rejecting the loan will declare the agent bankrupt.

        Remarks:

            - will NEVER be called in ANAC 2019 League. The bank is disabled and no loans are allowed.

        """
        return super().confirm_loan(loan, bankrupt_if_rejected)

    def on_agent_bankrupt(self, agent_id: str) -> None:
        """
        Will be called whenever any agent goes bankrupt

        Args:

            agent_id: The ID of the agent that went bankrupt

        Remarks:

            - Agents can go bankrupt in two cases:

                1. Failing to pay one installments of a loan they bought and refusing (or being unable to) get another
                   loan to pay it.
                2. Failing to pay a penalty on a sell contract they failed to honor (and refusing or being unable to get
                   a loan to pay for it).

            - The first bankruptcy case above *will never happen* in ANAC 2019 league as the bank is disabled.
            - The second bankruptcy case above *may still happen* in ANAC 2019 league.
            - All built-in agents ignore this call and they use the bankruptcy list ONLY to decide whether or not to
              negotiate in their `on_new_cfp` and `respond_to_negotiation_request` callbacks by pulling the
              bulletin-board using the helper function `is_bankrupt` of their AWI.
        """
        super().on_agent_bankrupt(agent_id)

    def on_new_report(self, report: FinancialReport):
        """Called whenever a financial report is published.

        Args:

            report: The financial report giving details of the standing of an agent at some time (see `FinancialReport`)

        Remarks:

            - Agents must opt-in to receive these calls by calling `receive_financial_reports` on their AWI
        """
        super().on_new_report(report)

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
            product = self.products[product_id]
            if product.catalog_price is None:
                price_range = (0.0, 100.0)
            else:
                price_range = (0, 1.2 * product.catalog_price)
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


class InsuranceFraudFactoryManager(NonDemandDrivenAgent):
    def _hideMoney(self, amount):
        if amount < 0:
            amount = self.awi.state.wallet + amount
        if amount > self.awi.state.wallet:
            return False
        self._hiddenModey += amount
        self.awi.hide_funds(amount)
        return True

    def _unhideMoney(self, amount):
        if amount < 0:
            amount = self._hiddenModey + amount
        if amount > self._hiddenModey:
            return False
        self._hiddenModey -= amount
        self.awi.unhide_funds(amount)
        return True

    def _hideAllMoney(self):
        self._hideMoney(self.awi.state.wallet)

    def _unhideAllMoney(self):
        self._unhideMoney(self._hiddenModey)

    def _adjustMoney(self, amount):
        self._unhideAllMoney()
        self._hideMoney(-amount)

    def init(self):
        """Called once after the agent-world interface is initialized"""
        super().init()
        # import random
        # self._collusion_para1 = random.randint(50,300)#123
        # self._collusion_para2 = random.randint(50,300)*1000#12300
        self._collusion_para1 = 123
        self._collusion_para2 = 12300
        # print(self._collusion_para1,self._collusion_para2)
        self._hiddenModey = 0

        self._adjustMoney(100)
        pass

    def _process_buy_cfp(self, cfp: CFP) -> None:
        if cfp.max_quantity == self._collusion_para1:
            print(cfp.publisher)
            if cfp.publisher not in self._collusion_target:
                self._collusion_target.append(cfp.publisher)
        else:
            super()._process_buy_cfp(cfp)

    def step(self):
        """Called at every production step by the world"""
        self.awi.bb_remove(section="cfps", query={"publisher": self})
        super().step()
        # print("Hiddeen Money:\t\t "+str(self._hiddenModey))
        # storage = sum([_n_stocks for _p_id, _n_stocks in self.awi.state.storage.items()])
        if self.awi.current_step <= 5:
            for product_id in self.consuming.keys():
                product = self.products[product_id]
                if product.catalog_price is None:
                    price_range = (0.0, 100.0)
                else:
                    price_range = (0.0, 1.5 * product.catalog_price)
                cfp = CFP(
                    is_buy=True,
                    publisher=self.id,
                    product=product_id,
                    time=self.awi.current_step + 2,
                    unit_price=0.0,
                    quantity=(1, self._collusion_para1),
                )
                self.awi.register_cfp(cfp)
            for product_id in self.producing.keys():
                product = self.products[product_id]
                if product.catalog_price is None:
                    price_range = (0.0, 100.0)
                else:
                    price_range = (0.0, 1.5 * product.catalog_price)
                cfp = CFP(
                    is_buy=True,
                    publisher=self.id,
                    product=product_id,
                    time=self.awi.current_step + 2,
                    unit_price=0.0,
                    quantity=(1, self._collusion_para1),
                )
                self.awi.register_cfp(cfp)

        if (
            self.awi.current_step < self.awi.n_steps - 2
        ):  # and (self._agent_layer_min %2 == self.awi.current_step %2):
            for product_id in self.consuming.keys():
                cfp = CFP(
                    is_buy=False,
                    publisher=self.id,
                    product=product_id,
                    time=self.awi.current_step + 2,
                    unit_price=self._collusion_para2,
                    quantity=1,
                )
                self.awi.register_cfp(cfp)

            for product_id in self.producing.keys():
                cfp = CFP(
                    is_buy=False,
                    publisher=self.id,
                    product=product_id,
                    time=self.awi.current_step + 2,
                    unit_price=self._collusion_para2,
                    quantity=1,
                )
                self.awi.register_cfp(cfp)
            if self.awi.current_step > 15:
                price = self._collusion_para2 * 10
                for product_id in self.consuming.keys():
                    cfp = CFP(
                        is_buy=False,
                        publisher=self.id,
                        product=product_id,
                        time=self.awi.current_step + 2,
                        unit_price=price,
                        quantity=1,
                    )
                    self.awi.register_cfp(cfp)

                for product_id in self.producing.keys():
                    cfp = CFP(
                        is_buy=False,
                        publisher=self.id,
                        product=product_id,
                        time=self.awi.current_step + 2,
                        unit_price=price,
                        quantity=1,
                    )
                    self.awi.register_cfp(cfp)
            if self.awi.current_step > 30:
                price = self._collusion_para2 * 100
                for product_id in self.consuming.keys():
                    cfp = CFP(
                        is_buy=False,
                        publisher=self.id,
                        product=product_id,
                        time=self.awi.current_step + 2,
                        unit_price=price,
                        quantity=1,
                    )
                    self.awi.register_cfp(cfp)

                for product_id in self.producing.keys():
                    cfp = CFP(
                        is_buy=False,
                        publisher=self.id,
                        product=product_id,
                        time=self.awi.current_step + 2,
                        unit_price=price,
                        quantity=1,
                    )
                    self.awi.register_cfp(cfp)
            if self.awi.current_step > 45:
                price = self._collusion_para2 * 1000
                for product_id in self.consuming.keys():
                    cfp = CFP(
                        is_buy=False,
                        publisher=self.id,
                        product=product_id,
                        time=self.awi.current_step + 2,
                        unit_price=price,
                        quantity=1,
                    )
                    self.awi.register_cfp(cfp)

                for product_id in self.producing.keys():
                    cfp = CFP(
                        is_buy=False,
                        publisher=self.id,
                        product=product_id,
                        time=self.awi.current_step + 2,
                        unit_price=price,
                        quantity=1,
                    )
                    self.awi.register_cfp(cfp)
            if self.awi.current_step > 60:
                price = self._collusion_para2 * 10000
                for product_id in self.consuming.keys():
                    cfp = CFP(
                        is_buy=False,
                        publisher=self.id,
                        product=product_id,
                        time=self.awi.current_step + 2,
                        unit_price=price,
                        quantity=1,
                    )
                    self.awi.register_cfp(cfp)

                for product_id in self.producing.keys():
                    cfp = CFP(
                        is_buy=False,
                        publisher=self.id,
                        product=product_id,
                        time=self.awi.current_step + 2,
                        unit_price=price,
                        quantity=1,
                    )
                    self.awi.register_cfp(cfp)
        self._adjustMoney(100)
        if self.awi.current_step == self.awi.n_steps - 1:
            self._unhideAllMoney()

    # ==========================
    # Important Events Callbacks
    # ==========================

    def on_new_cfp(self, cfp: CFP) -> None:
        """Called when a new CFP for a product for which the agent registered interest is published"""
        if cfp.max_quantity == self._collusion_para1:
            # print("publisher: "+ str(cfp.publisher)+" me:" + self.id)
            # print("find_collusion")
            # neg2 = AspirationNegotiator()
            ufun_ = PessimisticNegotiatorUtility(self, self._create_annotation(cfp=cfp))
            neg2 = InsuranceFraudNegotiator(agent=self, cfp=cfp)
            self.request_negotiation(negotiator=neg2, cfp=cfp, ufun=ufun_)
        elif cfp.max_unit_price in [
            self._collusion_para2 * i for i in [10 ** i for i in range(4)]
        ]:
            # print("publisher: "+ str(cfp.publisher)+" me:" + self.id)
            # print("find_collusion")
            # neg2 = AspirationNegotiator()
            ufun_ = PessimisticNegotiatorUtility(self, self._create_annotation(cfp=cfp))
            neg2 = InsuranceFraudNegotiator(agent=self, cfp=cfp)
            self.request_negotiation(negotiator=neg2, cfp=cfp, ufun=ufun_)
        else:
            super().on_new_cfp(cfp)

    def on_cash_transfer(self, amount: float, cause: str) -> None:
        """Called whenever there is a cash transfer to or from the agent"""
        super().on_cash_transfer(amount, cause)
        self._adjustMoney(100)
        if self.awi.current_step == self.awi.n_steps - 1:
            self._unhideAllMoney()

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
        if cfp.publisher == self.id:
            pass
        if cfp.max_quantity == self._collusion_para1:
            # print("respond_collusion to " + partner + " by "+ self.name)
            # neg2 = AspirationNegotiator()
            ufun_ = PessimisticNegotiatorUtility(
                self, self._create_annotation(cfp=cfp, partner=partner)
            )
            neg2 = InsuranceFraudNegotiator(agent=self, cfp=cfp)
            return neg2
        elif cfp.max_unit_price in [
            self._collusion_para2 * i for i in [10 ** i for i in range(4)]
        ]:
            ufun_ = PessimisticNegotiatorUtility(
                self, self._create_annotation(cfp=cfp, partner=partner)
            )
            neg2 = InsuranceFraudNegotiator(agent=self, cfp=cfp)
            return neg2
        else:
            # return None
            return super().respond_to_negotiation_request(cfp, partner)

    def sign_contract(self, contract: Contract) -> Optional[str]:
        """Called after the signing delay from contract conclusion to sign the contract. Contracts become binding
        only after they are signed.

        Remarks:

            - Return `None` if you decided not to sign the contract. Return your ID (self.id) otherwise.

        """
        signature = self.id
        # self.awi.buy_insurance(contract=contract)
        (
            _cfp,
            _seller_id,
            _buyer_id,
            _time,
            _quantity,
            _unit_price,
            _product_id,
        ) = self._split_contract(contract=contract)
        if self.id == _buyer_id:
            _is_buy = True
        elif self.id == _seller_id:
            _is_buy = False
        else:
            _is_buy = None
            print("\033[31mBUY/SELL ERROR\033[0m")
            pass

        if contract.agreement["quantity"] == self._collusion_para1:
            # print(">>>>>>>>>>>>>> 詐欺するよ")
            # print(contract)
            # print(contract.agreement["quantity"])
            return signature
        elif contract.agreement["unit_price"] in [
            self._collusion_para2 * i for i in [10 ** i for i in range(4)]
        ]:
            # print(">>>>>>>>>>>>>> 詐欺するよ2")
            # print(contract)
            # print(contract.agreement["unit_price"])
            return signature
        # if _is_buy and self.awi.current_step <=7:
        #    return None
        return super().sign_contract(contract)

    def on_contract_signed(self, contract: Contract) -> None:
        """Called whenever a contract is signed by all partners"""

        (
            _cfp,
            _seller_id,
            _buyer_id,
            _time,
            _quantity,
            _unit_price,
            _product_id,
        ) = self._split_contract(contract=contract)
        if self.id == _buyer_id:
            _is_buy = True
        elif self.id == _seller_id:
            _is_buy = False
        else:
            _is_buy = None
            print("\033[31mBUY/SELL ERROR\033[0m")
            pass

        if contract.agreement["quantity"] == self._collusion_para1:
            if not _is_buy:
                return None
            # 保険かう
            # print(">>>>>>>>>>>>>>>> 詐欺用保険かうよ")
            # print("insurance premier:\t"+str(self.awi.evaluate_insurance(contract)))
            self._unhideAllMoney()
            self.awi.buy_insurance(contract=contract)
            self._adjustMoney(100)
            return None
        elif contract.agreement["unit_price"] in [
            self._collusion_para2 * i for i in [10 ** i for i in range(4)]
        ]:
            if _is_buy:
                return None
            # print(">>>>>>>>>>>>>>>> 詐欺用保険かうよ2")
            # print("insurance premier:\t"+str(self.awi.evaluate_insurance(contract)))
            if (
                self.awi.evaluate_insurance(contract) is None
                or self.awi.evaluate_insurance(contract) > 0.8
            ):
                # if self.awi.evaluate_insurance(contract) > 0.8:
                return None
            self._unhideAllMoney()
            self.awi.buy_insurance(contract=contract)
            self._adjustMoney(100)
            return None
        super().on_contract_signed(contract)

    def on_contract_executed(self, contract: Contract) -> None:
        """Called whenever a contract is fully executed without any breaches"""
        super().on_contract_executed(contract)
        self._adjustMoney(100)
        if self.awi.current_step == self.awi.n_steps - 1:
            self._unhideAllMoney()
        # print(self.awi.state.wallet)

    # def total_utility(self, contracts: Collection[Contract] = ()) -> float:
    #    """Calculates the total utility for the agent of a collection of contracts"""
    #    return 100

    def _split_contract(self, contract: Contract) -> Any:
        _cfp = contract.annotation["cfp"]
        _seller_id = contract.annotation["seller"]
        _buyer_id = contract.annotation["buyer"]
        _time = contract.agreement["time"]
        _quantity = contract.agreement["quantity"]
        _unit_price = contract.agreement["unit_price"]
        _product_id = _cfp.product
        return (_cfp, _seller_id, _buyer_id, _time, _quantity, _unit_price, _product_id)


class InsuranceFraudNegotiator(negmas.sao.AspirationNegotiator):
    def __init__(
        self,
        name=None,
        ufun=None,
        parent: Controller = None,
        dynamic_ufun=True,
        randomize_offer=False,
        can_propose=True,
        assume_normalized=False,
        ### aspiration init
        max_aspiration=0.95,
        aspiration_type="boulware",
        above_reserved_value=False,
        agent=None,
        cfp=None,
    ):
        super().__init__(
            name=name,
            assume_normalized=assume_normalized,
            parent=parent,
            ufun=ufun,
            dynamic_ufun=dynamic_ufun,
            randomize_offer=randomize_offer,
            can_propose=can_propose,
        )

        self.rational_proposal = False
        self.can_propose = False

        self.partner = cfp.publisher
        self.agent = agent
        self._collusion_para1 = agent._collusion_para1
        self._collusion_para2 = agent._collusion_para2
        self._collusion_type = 0

        self.cfp = cfp
        self.outcomes = cfp.outcomes

    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        # print("!!!!!!!!!!!!!!!")
        # if self._collusion_type == 2:
        #    print(">>>>>>>>>>>>>>>")
        # print(offer)
        return ResponseType.ACCEPT_OFFER

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        # print("!!!!!outcome propose")
        # if self._collusion_type == 2:
        #    print(">>>>>>>>>>>>>>>")
        # print(self.outcomes[-1])
        return self.outcomes[-1]
