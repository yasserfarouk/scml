import math

from negmas import Contract
from typing import Collection, Optional, List, Callable, Dict, Any

from scml.scml2019.common import (
    NO_PRODUCTION,
    Job,
    ProductionNeed,
    ManufacturingProfileCompiled,
    ProductManufacturingInfo,
    Process,
    Product,
)
from scml.scml2019.factory_managers.builtins import (
    ScheduleInfo,
    FactorySimulator,
    SCMLAgreement,
)
from scml.scml2019.schedulers import Scheduler
from scml.scml2019.schedulers import transaction

INVALID_UTILITY = float("-inf")


class MyScheduler(Scheduler):
    """Default scheduler used by the DefaultFactoryManager"""

    def __getstate__(self):
        result = self.__dict__.copy()
        if "fields" in result.keys():
            result.pop("fields", None)

    def __setstate__(self, state):
        self.__dict__ = state
        self.fields = [
            self.total_unit_cost,
            self.unit_time,
            self.production_unit_cost,
            self.input_unit_cost,
        ]

    def __init__(
        self,
        manager_id: str,
        awi: "SCMLAWI",
        max_insurance_premium: float,
        horizon: Optional[int] = None,
        add_catalog_prices=True,
        strategy: str = "latest",
        profile_sorter: str = "total-cost>time",
    ):
        """

        Args:

            manager_id: ID of the factory manager using this scheduler.
            awi: Agent-world interface (used to access insurance calculations and `n_steps`).
            max_insurance_premium: Maximum insurance premium over which the factory maanger will not buy insuracne
            horizon: Scheduling horizon (by default it is the number of simulation step in the AWI)
            add_catalog_prices: Whether to add total catalog price costs to costs of production
            strategy: How to schedule production. Possible values are earliest, latest, shortest, longest
            profile_sorter: The method used to sort profiles that can produce the same product

        Remarks:

            The following `production_strategy` values are supported:

            - earliest: Try to produce things as early as possible. Useful for infinite storage
            - latest: Try to produce things as late as possible. Useful for finite storage
            - shortest: Schedule in the time/line that has the shortest empty slot that is enough for production
            - longest: Schedule in the time/line that has the longest empty slot that is enough for production

            The `profile_sorter` string consists of one or more of the following sections separated by ``>`` characters
            to indicate sorting order. Costs are sorted ascendingly and times descendingly. Costs and times refer to
            unit cost/time (total divided by quantity generated):

            - time, t: profile production time per unit
            - input-cost, ic, icost: Input cost per unit only using catalog prices
            - production-cost, pc, pcost: Production cost as specified in the profile per unit
            - total-cost, tc, tcost: Total cost per unit including input cost


        """
        super().__init__(
            manager_id=manager_id,
            horizon=horizon,
            awi=awi,
            max_insurance_premium=max_insurance_premium,
        )
        self.add_catalog_prices = add_catalog_prices
        self.strategy = strategy
        self.fields: List[Callable[[ProductManufacturingInfo], float]] = [
            self.total_unit_cost,
            self.unit_time,
            self.production_unit_cost,
            self.input_unit_cost,
        ]
        mapper = {"tc": 0, "t": 1, "pc": 2, "ic": 3}
        self.field_order: List[int] = []
        sort_fields = profile_sorter.split(">")
        self.producing: Dict[int, List[ProductManufacturingInfo]] = {}
        for field_name in sort_fields:
            if field_name in ("time", "t"):
                self.field_order.append(mapper["t"])
            elif field_name in ("total-cost", "tc", "tcost"):
                self.field_order.append(mapper["tc"])
            elif field_name in ("production-cost", "pc", "pcost"):
                self.field_order.append(mapper["pc"])
            elif field_name in ("input-cost", "ic", "icost"):
                self.field_order.append(mapper["ic"])

    def init(
        self,
        simulator: FactorySimulator,
        products: List[Product],
        processes: List[Process],
        profiles: List[ManufacturingProfileCompiled],
        producing: Dict[int, List[ProductManufacturingInfo]],
    ):
        super().init(
            simulator=simulator,
            products=products,
            processes=processes,
            producing=producing,
            profiles=profiles,
        )
        self.producing = {
            k: sorted(v, key=self._profile_sorter) for k, v in self.producing.items()
        }

    def _profile_sorter(self, info: ProductManufacturingInfo) -> Any:
        vals = [field(info) for field in self.fields]
        profile = self.profiles[info.profile]
        return tuple(
            [vals[indx] for indx in self.field_order] + [profile.line, profile.process]
        )

    def unit_time(self, info: ProductManufacturingInfo) -> float:
        profile = self.profiles[info.profile]
        return profile.n_steps / info.quantity

    def total_cost(self, info: ProductManufacturingInfo) -> float:
        products = self.products
        profile = self.profiles[info.profile]
        process = self.processes[profile.process]
        production_cost = profile.cost

        def safe(x):
            return 0.0 if x is None else x

        inputs_cost = sum(
            safe(products[inp.product].catalog_price) * inp.quantity
            for inp in process.inputs
        )
        return production_cost + inputs_cost

    def total_unit_cost(self, info: ProductManufacturingInfo) -> float:
        return self.total_cost(info=info) / info.quantity

    def production_cost(self, info: ProductManufacturingInfo) -> float:
        profile = self.profiles[info.profile]
        return profile.cost

    def production_unit_cost(self, info: ProductManufacturingInfo) -> float:
        return self.production_cost(info=info) / info.quantity

    def input_cost(self, info: ProductManufacturingInfo):
        products = self.products
        profile = self.profiles[info.profile]
        process = self.processes[profile.process]

        def safe(x):
            return 0.0 if x is None else x

        return sum(
            safe(products[inp.product].catalog_price) * inp.quantity
            for inp in process.inputs
        )

    def input_unit_cost(self, info: ProductManufacturingInfo) -> float:
        return self.input_cost(info=info) / info.quantity

    # noinspection PyUnusedLocal
    def schedule_contract(
        self,
        contract: Contract,
        assume_no_further_negotiations=False,
        end: int = None,
        ensure_storage_for: int = 0,
        start_at: int = 0,
    ) -> ScheduleInfo:
        """
        Schedules this contract if possible and returns information about the resulting schedule

        Args:

            contract: The contract being scheduled
            assume_no_further_negotiations: If true no further negotiations will be assumed possible
            end: The scheduling horizon (None for the default).
            ensure_storage_for: The number of steps all needs must be in storage before they are consumed in production
            start_at: No jobs will be scheduled before that time.

        Returns:

            Full schedule information including validity, line schedulers, production needs, etc (see `SchedulerInfo`).

        """
        ignore_failures = not assume_no_further_negotiations
        simulator: FactorySimulator = self.simulator
        start = max(simulator.fixed_before, start_at)

        if end is None:
            end = simulator.n_steps
        if contract.agreement is None:
            return ScheduleInfo(
                end=end, final_balance=self.simulator.balance_at(end - 1)
            )
        agreement: SCMLAgreement
        if isinstance(contract.agreement, dict):
            agreement = SCMLAgreement(**contract.agreement)
        else:
            agreement = contract.agreement  # type: ignore
        t = agreement["time"]
        if t < start:
            return ScheduleInfo(
                end=end,
                final_balance=INVALID_UTILITY,
                valid=False,
                ignored_contracts=[contract],
            )
        q, u = int(agreement["quantity"]), agreement["unit_price"]
        p = u * q
        pid: int = contract.annotation["cfp"].product
        if contract.annotation["buyer"] == self.manager_id:
            # I am a buyer
            # We do not ignore money shortage for buying. This means that the agent will not buy if the money it needs
            # may partially come from a sell contract that is not considered yet
            if not simulator.buy(
                product=pid,
                quantity=q,
                price=p,
                t=t,
                ignore_space_shortage=ignore_failures,
                ignore_money_shortage=ignore_failures,
            ):
                return ScheduleInfo(
                    end=end,
                    valid=False,
                    failed_contracts=[contract],
                    final_balance=INVALID_UTILITY,
                )
            if p <= 0:
                return ScheduleInfo(
                    valid=True,
                    end=end,
                    final_balance=self.simulator.balance_at(end - 1),
                )
            insurance = self.awi.evaluate_insurance(
                contract=contract, t=self.awi.current_step
            )
            if insurance is not None and insurance / p < self.max_insurance_premium:
                # if it is not possible to buy the insurance, the factory manager will not try to buy it. This is still
                # a valid schedule
                simulator.pay(insurance, t=t)
            return ScheduleInfo(
                valid=True, end=end, final_balance=self.simulator.balance_at(end - 1)
            )
        elif contract.annotation["seller"] == self.manager_id:
            # I am a seller

            # if enough is available in storage and not reserved, just sell it
            q_needed = q - simulator.available_storage_at(t)[pid]
            if q_needed <= 0:
                if simulator.sell(
                    product=pid,
                    quantity=q,
                    price=p,
                    t=t,
                    ignore_money_shortage=ignore_failures,
                    ignore_inventory_shortage=ignore_failures,
                ):
                    return ScheduleInfo(
                        end=end, final_balance=self.simulator.balance_at(end - 1)
                    )
                else:
                    return ScheduleInfo(
                        end=end,
                        valid=False,
                        failed_contracts=[contract],
                        final_balance=INVALID_UTILITY,
                    )
            jobs: List[Job] = []
            needs: List[ProductionNeed] = []
            current_schedule = simulator.line_schedules_to(t - ensure_storage_for - 1)
            if self.strategy == "earliest_feasible":
                number_of_rows = current_schedule.shape[0]
                number_of_columns = current_schedule.shape[1]
                empty_slots = []
                # bookmark = simulator.bookmark()
                with transaction(simulator) as bookmark:
                    for time in range(number_of_columns):
                        for line in range(number_of_rows):
                            if current_schedule[line][time] == NO_PRODUCTION:
                                info = self.producing[pid][line]
                                job = Job(
                                    line=line,
                                    action="run",
                                    time=max(time, start),
                                    profile=info.profile,
                                    contract=contract,
                                    override=False,
                                )
                                if not simulator.schedule(
                                    job,
                                    override=False,
                                    ignore_inventory_shortage=ignore_failures,
                                    ignore_money_shortage=ignore_failures,
                                    ignore_space_shortage=ignore_failures,
                                ):
                                    continue
                                else:
                                    jobs.append(job)
                                    profile = self.profiles[info.profile]
                                    process_index = profile.process
                                    process = self.processes[process_index]
                                    length = profile.n_steps
                                    q_produced = info.quantity
                                    for i in process.inputs:
                                        pind, quantity = i.product, i.quantity
                                        # I need the input to be available the step before production
                                        step = min(
                                            start
                                            + time
                                            + int(math.floor(i.step * length))
                                            - 1,
                                            self.awi.n_steps - 1,
                                        )
                                        if step < 0:
                                            break
                                        available = max(
                                            0,
                                            self.simulator.available_storage_at(step)[
                                                pind
                                            ]
                                            - quantity,
                                        )
                                        if available >= quantity:
                                            instore, tobuy = quantity, 0
                                        else:
                                            instore, tobuy = (
                                                available,
                                                quantity - available,
                                            )
                                        if tobuy > 0 or instore > 0:
                                            if step < start:
                                                break
                                            needs.append(
                                                ProductionNeed(
                                                    product=pind,
                                                    needed_for=contract,
                                                    quantity_in_storage=instore,
                                                    quantity_to_buy=tobuy,
                                                    step=step,
                                                )
                                            )
                                    else:
                                        q_needed -= q_produced
                                        if q_needed <= 0:
                                            break
                        else:
                            continue
                        break

                    if q_needed <= 0:
                        if not simulator.sell(
                            product=pid,
                            quantity=q,
                            price=p,
                            t=t,
                            ignore_money_shortage=ignore_failures,
                            ignore_inventory_shortage=ignore_failures,
                        ):
                            simulator.rollback(bookmark)
                            return ScheduleInfo(
                                end=end,
                                valid=False,
                                failed_contracts=[contract],
                                final_balance=INVALID_UTILITY,
                            )

                        for need in needs:
                            product_index = need.product
                            product = self.products[product_index]
                            catalog_price = product.catalog_price
                            if catalog_price == 0 or need.quantity_to_buy <= 0:
                                continue
                            price = need.quantity_to_buy * catalog_price
                            simulator.pay(price, t=need.step)
                        schedule = ScheduleInfo(
                            jobs=jobs,
                            end=end,
                            needs=needs,
                            failed_contracts=[],
                            final_balance=self.simulator.balance_at(end - 1),
                        )
                        # print(str(current_schedule)+"\n")
                        return schedule
                    simulator.rollback(bookmark)
                    return ScheduleInfo(
                        valid=False,
                        failed_contracts=[contract],
                        end=end,
                        final_balance=self.simulator.balance_at(end - 1),
                    )
        raise ValueError(
            f"{self.manager_id} Not a seller of a buyer in Contract: {contract} with "
            f"annotation: {contract.annotation}"
        )

    def schedule_contracts(
        self,
        contracts: Collection[Contract],
        end: int = None,
        assume_no_further_negotiations=False,
        ensure_storage_for: int = 0,
        start_at: int = 0,
    ) -> ScheduleInfo:
        """
        Schedules a set of contracts and returns the `ScheduleInfo`.

        Args:

            contracts: Contracts to schedule
            assume_no_further_negotiations: If true, no further negotiations will be assumed to be possible
            end: The end of the simulation for the schedule (exclusive)
            ensure_storage_for: Ensure that the outcome will be at the storage for at least this time
            start_at: The timestep at which to start scheduling

        Returns:

            ScheduleInfo giving the schedule after these contracts is included. `valid` member can be used to check
            whether this is a valid contract

        """
        simulator = self.simulator
        if end is None:
            end = simulator.n_steps
        result = ScheduleInfo(
            valid=True, end=end, final_balance=self.simulator.final_balance
        )
        contracts = sorted(contracts, key=lambda x: x.agreement["time"])
        for contract in contracts:
            new_schedule = self.schedule_contract(
                contract,
                end=end,
                ensure_storage_for=ensure_storage_for,
                assume_no_further_negotiations=assume_no_further_negotiations,
                start_at=start_at,
            )
            result.combine(new_schedule)
            if new_schedule.valid:
                result.final_balance = self.simulator.final_balance
            else:
                result.final_balance = INVALID_UTILITY
        return result

    def find_schedule(
        self,
        contracts: Collection[Contract],
        start: int,
        end: int,
        assume_no_further_negotiations=False,
        ensure_storage_for: int = 0,
        start_at: int = 0,
    ):

        # Now, schedule the contracts
        schedule = self.schedule_contracts(
            contracts=contracts,
            end=end,
            ensure_storage_for=ensure_storage_for,
            assume_no_further_negotiations=assume_no_further_negotiations,
            start_at=start_at,
        )

        # Mark the schedule as invalid if it has any production needs and we assume_no_further_negotiations
        if (
            assume_no_further_negotiations
            and schedule.needs is not None
            and len(schedule.needs) > 0
        ):
            schedule.valid = False
            return schedule

        return schedule
