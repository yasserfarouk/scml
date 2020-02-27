import sys

sys.path.append("/".join(__file__.split("/")[:-1]))
import pandas as pd
import time


def read_qtty_feasible_domain(
    the_number_of_periods: int, the_quantities_domain_size: int
) -> pd.DataFrame:
    """
    Reads the variables associated with a domain for a number of time periods and a size for each of the quantities variables.
    :param the_number_of_periods:
    :param the_quantities_domain_size:
    :return:
    """
    return pd.read_csv(
        "/".join(__file__.split("/")[:-1])
        + "/qtty_domain/qtty_domain_t_"
        + str(the_number_of_periods)
        + "_d_"
        + str(the_quantities_domain_size)
        + ".zip",
        compression="gzip",
        sep=",",
    )


def pandas_tuple_to_list_of_tuple(sol: pd, number_of_periods: int) -> list:
    """
    Converts a pandas tuple to a list of regular tuples
    :param sol:
    :param number_of_periods:
    :return:
    """
    return (
        [
            tuple((sol[t * 3], sol[t * 3 + 1], sol[t * 3 + 2]))
            for t in range(number_of_periods)
        ]
        if sol is not None
        else None
    )


def solve_mpnvm(
    the_feasible_sols: pd.DataFrame,
    the_expectations_q_min_out: dict,
    the_expectations_q_min_in: dict,
    the_prices_out: dict,
    the_prices_in: dict,
    the_production_cost: float,
    verbose: bool = False,
):
    """
    Solves the stochastic Multi-Step NewsVendor Problem.
    :param the_feasible_sols: a DataFrame with all the solutions to be checked. The number of columns must be a multiple of 3
    :param the_expectations_q_min_out:
    :param the_expectations_q_min_in:
    :param the_prices_out:
    :param the_prices_in:
    :param the_production_cost:
    :param verbose
    :return:
    """
    assert len(the_feasible_sols.columns) % 3 == 0
    optimal_sol = None
    optimal_sol_revenue = 0.0
    # Solve the MPNVM for each feasible solution.
    positive_solutions = []
    # The time horizon is implicit in the number of columns of the solutions' DataFrame.
    T = int(len(the_feasible_sols.columns) / 3)
    # Loop through each row of the feasible solutions DataFrame. Use itertuples for faster looping.
    t0 = time.time()
    for row in the_feasible_sols.itertuples(index=False):
        # Compute the objective value
        candidate_sol_value = sum(
            [
                the_prices_out[t]
                * the_expectations_q_min_out[t]["min_" + str(row[(t * 3) + 1])]
                - the_prices_in[t]
                * the_expectations_q_min_in[t]["min_" + str(row[t * 3])]
                - the_production_cost * row[(t * 3) + 2]
                for t in range(T)
            ]
        )
        # Keep track in case this solution improves on the optimal so far
        if candidate_sol_value > optimal_sol_revenue:
            optimal_sol_revenue = candidate_sol_value
            optimal_sol = row
            if verbose:
                print(
                    f"it took "
                    + format(time.time() - t0, ".4f")
                    + f" seconds to find a better solution: {pandas_tuple_to_list_of_tuple(optimal_sol, T)}, "
                    f"revenue = " + format(optimal_sol_revenue, ".4f")
                )

        # For debugging purposes only, keep track of all solutions with positive objective value.
        if candidate_sol_value > 0:
            positive_solutions.append((row, candidate_sol_value))
    return optimal_sol_revenue, optimal_sol, positive_solutions
