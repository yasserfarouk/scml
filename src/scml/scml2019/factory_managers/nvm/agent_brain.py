from typing import Optional
import sys

MY_PATH = "/".join(__file__.split("/")[:-1])
sys.path.append(MY_PATH)
from negmas import Contract

from . import pydata
from . import mp_nvp
from . import mp_nvp_lp
from . import newsvendor
import os
import time


class AgentBrain:
    def __init__(
        self,
        game_length: int,
        input_product_index: int,
        output_product_index: int,
        production_cost: float,
        num_intermediate_products: int,
        verbose: bool = False,
    ):
        """
        Initializes the brain of the agent. The constructor reads all the static data once and prepares it for the game.
        :param game_length
        :param input_product_index:
        :param output_product_index:
        :param production_cost:
        :param num_intermediate_products:
        """
        # Set verbosity and game length
        self.verbose = verbose
        self.game_length = game_length

        # Save stuff about the agent
        self.input_product_index = input_product_index
        self.output_product_index = output_product_index
        self.mpnvp_production_cost = production_cost
        self.num_intermediate_products = num_intermediate_products

        # Initialize stuff about the MPNVP that is going to be reused each time
        self.mpnvp_number_of_periods = 4
        self.mpnvp_quantities_domain_size = 20
        self.mpnvp_feasible_sols = mp_nvp.read_qtty_feasible_domain(
            self.mpnvp_number_of_periods, self.mpnvp_quantities_domain_size
        )

        # Initialize stuff about the MPNVP-LP
        self.mpnvplp_horizon = 10

        # Check if the data exists. If so, load it and set self.there_is_data flag to True.
        if self.check_if_data_exists():
            self.there_is_data = True
            # Get the data for quantities
            q_uncertainty_model = pydata.get_json_dict(
                f"{MY_PATH}/data/dict_qtty_num_intermediate_products"
                f"_{self.num_intermediate_products}.json"
            )
            self.q_inn_uncertainty_model = q_uncertainty_model[
                "p" + str(self.input_product_index)
            ]
            self.q_out_uncertainty_model = q_uncertainty_model[
                "p" + str(self.output_product_index)
            ]
            self.expectations_q_min_inn = {
                t: newsvendor.compute_min_expectation(
                    self.q_inn_uncertainty_model[str(t + 1)],
                    self.mpnvp_quantities_domain_size,
                )
                for t in range(game_length - 1)
            }
            self.expectations_q_min_out = {
                t: newsvendor.compute_min_expectation(
                    self.q_out_uncertainty_model[str(t + 1)],
                    self.mpnvp_quantities_domain_size,
                )
                for t in range(game_length - 1)
            }

            # Get the data for prices
            prices = pydata.get_json_dict(
                f"{MY_PATH}/data/dict_price_num_intermediate_products"
                f"_{self.num_intermediate_products}.json"
            )
            self.prices_inn = prices["p" + str(self.input_product_index)]
            self.prices_out = prices["p" + str(self.output_product_index)]

            # Compute the expected quantities
            self.q_inn_expected = {
                t: sum(
                    [
                        int(i) * p
                        for i, p in self.q_inn_uncertainty_model[str(t + 1)].items()
                    ]
                )
                for t in range(0, game_length - 1)
            }
            self.q_out_expected = {
                t: sum(
                    [
                        int(i) * p
                        for i, p in self.q_out_uncertainty_model[str(t + 1)].items()
                    ]
                )
                for t in range(0, game_length - 1)
            }
        else:
            self.there_is_data = False

        if self.verbose:
            print(
                "Loaded Brain with data"
                if self.there_is_data
                else "Warning! Loaded Brain with no data"
            )

    def get_complete_plan(
        self, current_time: int, verbose: bool = False
    ) -> Optional[list]:
        """
        Given a time of the simulation, solves for a plan.
        :param current_time:
        :param verbose
        :return:
        """
        assert self.there_is_data

        end = current_time + self.mpnvp_number_of_periods
        if end >= self.game_length:
            return None
        slice_expectations_q_min_out = {
            t - current_time: self.expectations_q_min_out[t]
            for t in range(current_time, end)
        }
        slice_expectations_q_min_inn = {
            t - current_time: self.expectations_q_min_inn[t]
            for t in range(current_time, end)
        }
        slice_prices_inn = {
            t - current_time: self.prices_inn[str(t)]
            if str(t) in self.prices_inn
            else 0.0
            for t in range(current_time, end)
        }
        slice_prices_out = {
            t - current_time: self.prices_out[str(t)]
            if str(t) in self.prices_out
            else 0.0
            for t in range(current_time, end)
        }

        if verbose:
            print(
                f"\n\t Solving MPNVP for number of periods = {self.mpnvp_number_of_periods}, and domain size = {self.mpnvp_quantities_domain_size}"
            )
            t0 = time.time()
        optimal_sol_value, optimal_sol, positive_solutions = mp_nvp.solve_mpnvm(
            self.mpnvp_feasible_sols,
            slice_expectations_q_min_out,
            slice_expectations_q_min_inn,
            slice_prices_out,
            slice_prices_inn,
            self.mpnvp_production_cost,
        )
        if verbose:
            print(f"\t\t Done solving MPNVP. Took {time.time() - t0} sec. ")

        return optimal_sol

    def get_plan_for_inputs(self, current_time: int, verbose: bool = False) -> list:
        """
        Returns a plan only for inputs
        :param current_time:
        :param verbose
        :return:
        """
        complete_plan = mp_nvp.pandas_tuple_to_list_of_tuple(
            self.get_complete_plan(current_time, verbose), self.mpnvp_number_of_periods
        )
        return (
            [x for x, y, z in complete_plan]
            if complete_plan is not None
            else [0] * self.mpnvp_number_of_periods
        )

    def get_value_contracts(
        self, current_time: int, total_game_time: int, contracts: dict
    ) -> float:
        """
        Given a set of buy and sell contracts, returns the value of planning assuming the buy and sell contracts will actually happen.
        :param current_time
        :param total_game_time
        :param contracts:
        :return:
        """
        # If there is no data, we cannot solve th MPNVM-LP. Hence, we just return an infinity value and hope for the best.
        if not self.there_is_data:
            return float("inf")

        end = min(current_time + self.mpnvplp_horizon, total_game_time - 1)
        dict_of_buy_contracts = {
            t - current_time: [c for buy, c in contracts[t] if buy]
            for t in range(current_time, end)
        }
        dict_of_sell_contracts = {
            t - current_time: [c for buy, c in contracts[t] if not buy]
            for t in range(current_time, end)
        }
        slice_q_out_expected = {
            t - current_time: self.q_out_expected[t] for t in range(current_time, end)
        }
        slice_q_inn_expected = {
            t - current_time: self.q_inn_expected[t] for t in range(current_time, end)
        }
        slice_prices_inn = {
            t - current_time: self.prices_inn[str(t)]
            if str(t) in self.prices_inn
            else 0.0
            for t in range(current_time, end)
        }
        slice_prices_out = {
            t - current_time: self.prices_out[str(t)]
            if str(t) in self.prices_out
            else 0.0
            for t in range(current_time, end)
        }

        if self.verbose and False:
            print(f"dict_of_buy_contracts = {dict_of_buy_contracts}")
            print(f"dict_of_sell_contracts = {dict_of_sell_contracts}")
            print(f"slice_q_out_expected = {slice_q_out_expected}")
            print(f"slice_q_inn_expected = {slice_q_inn_expected}")
            print(f"slice_prices_inn = {slice_prices_inn}")
            print(f"slice_prices_out = {slice_prices_out}")

        return mp_nvp_lp.solve_mp_nvp_lp(
            p_out=slice_prices_out,
            p_inn=slice_prices_inn,
            q_out=slice_q_out_expected,
            q_inn=slice_q_inn_expected,
            production_capacity=10,  # This is fixed for the 2019 competition.
            production_cost=self.mpnvp_production_cost,
            input_offers=dict_of_buy_contracts,
            output_offers=dict_of_sell_contracts,
            time_horizon=min(self.mpnvplp_horizon, total_game_time - current_time - 1),
            verbose=self.verbose and False,
        )

    def get_value_of_contract(
        self,
        current_time: int,
        total_game_time: int,
        contracts: dict,
        contract: Optional[Contract],
        agent_is_buy: bool,
    ):
        """
        Computes the value of a contract.
        :param current_time:
        :param total_game_time:
        :param contracts:
        :param contract:
        :param agent_is_buy:
        :return:
        """
        # The horizon of the calculation should account for the time of the contract under consideration.
        horizon = 0 if contract is None else contract.agreement["time"]
        # If there is data, then we solve the program. First, we filter the contracts of interest.
        contracts_filtered = {
            t: [c for c in contracts[t]]
            for t in range(
                current_time - 1,
                max(
                    horizon,
                    min(current_time + self.mpnvplp_horizon, total_game_time - 1),
                )
                + 1,
            )
        }
        # @ todo check this weird error here that I am going to guard against.
        if (
            contract is not None
            and contract.agreement["time"] not in contracts_filtered
        ):
            return float("-inf")

        # If a contract is received, add it.
        if contract is not None:
            contracts_filtered[contract.agreement["time"]] += [
                (
                    agent_is_buy,
                    (
                        contract.agreement["unit_price"],
                        contract.agreement["quantity"],
                        contract.agreement["time"],
                    ),
                )
            ]
        return self.get_value_contracts(
            current_time=current_time,
            total_game_time=total_game_time,
            contracts=contracts_filtered,
        )

    def marginal_value_contract(
        self,
        current_time: int,
        total_game_time: int,
        contracts: dict,
        contract: Contract,
        agent_is_buy: bool,
    ):
        """
        Compute the marginal value of a contract.
        :param current_time:
        :param total_game_time:
        :param contracts:
        :param contract:
        :param agent_is_buy:
        :return:
        """
        value_with___ = self.get_value_of_contract(
            current_time=current_time,
            total_game_time=total_game_time,
            contracts=contracts,
            contract=contract,
            agent_is_buy=agent_is_buy,
        )
        value_without = self.get_value_of_contract(
            current_time=current_time,
            total_game_time=total_game_time,
            contracts=contracts,
            contract=None,
            agent_is_buy=agent_is_buy,
        )
        if self.verbose and False:
            print(f"\t ** ")
            print(f"\t Value with = {value_with___}")
            print(f"\t Value without = {value_without}")
            print(f"\t value = {value_with___ - value_without}")
        if value_with___ == float("-inf"):
            ret = float("-inf")
        elif value_with___ == float("inf"):
            ret = float("inf")
        elif value_without == float("inf") and value_with___ < float("inf"):
            ret = float("-inf")
        elif value_without == float("-inf") and value_with___ > float("-inf"):
            ret = value_with___
        else:
            ret = value_with___ - value_without
        return ret

    def check_if_data_exists(self) -> bool:
        """
        Check if the uncertainty model exists, both for prices and quantity
        :return:
        """
        return os.path.isfile(
            f"data/dict_qtty_num_intermediate_products_{self.num_intermediate_products}.json"
        ) and os.path.isfile(
            f"data/dict_price_num_intermediate_products_{self.num_intermediate_products}.json"
        )
