from pulp import *
from prettytable import PrettyTable


def solve_mp_nvp_lp(
    p_out: dict,
    p_inn: dict,
    q_out: dict,
    q_inn: dict,
    time_horizon: int,
    production_capacity: int,
    production_cost: float,
    input_offers: dict,
    output_offers: dict,
    verbose: bool = False,
):
    """
    Formulates and solve the utility calculation LP.
    :param p_out:
    :param p_inn:
    :param q_out:
    :param q_inn:
    :param time_horizon:
    :param production_capacity:
    :param production_cost:
    :param input_offers:
    :param output_offers:
    :param verbose:
    :return:
    """
    mpnvp_lp = LpProblem(name="mpnvp-lp", sense=LpMaximize)

    x = LpVariable.dicts("x", range(0, time_horizon), 0)
    y = LpVariable.dicts("y", range(0, time_horizon), 0)
    z = LpVariable.dicts("z", range(0, time_horizon), 0)

    # The objective is composed of stuff that is being decided plus the input/output contracts.
    mpnvp_lp += sum(
        [
            p_out[t] * y[t] - p_inn[t] * x[t] - production_cost * z[t]
            for t in range(0, time_horizon)
        ]
        + [
            c_q * c_p
            for contracts in output_offers.values()
            for c_p, c_q, c_t in contracts
        ]
        + [
            -c_q * c_p
            for contracts in input_offers.values()
            for c_p, c_q, c_t in contracts
        ]
    )

    # Add all the constraints
    for t in range(0, time_horizon):
        # Sell constraints
        mpnvp_lp += y[t] <= sum([z[k] - y[k] for k in range(0, t)])
        # Production Capacity constraints
        mpnvp_lp += z[t] <= sum([x[k] - z[k] for k in range(0, t)])
        # Output storage capacity
        mpnvp_lp += z[t] <= production_capacity

        # Input offers.
        if t in input_offers:
            mpnvp_lp += sum(q for _, q, _ in input_offers[t]) <= x[t]
            mpnvp_lp += x[t] <= sum(q for _, q, _ in input_offers[t]) + q_inn[t]
        else:
            mpnvp_lp += x[t] <= q_inn[t]
        # Output offers - ensure inputs.
        if t in output_offers:
            mpnvp_lp += sum(q for _, q, _ in output_offers[t]) <= y[t]
            mpnvp_lp += y[t] <= sum(q for _, q, _ in output_offers[t]) + q_out[t]
        else:
            mpnvp_lp += y[t] <= q_out[t]

    # Solve the LP
    mpnvp_lp.solve()

    # DEBUG print the LP.
    if verbose:
        # print(mpnvp_lp)
        print(f"Status: {pulp.LpStatus[mpnvp_lp.status]}")

    # Get the status of the LP. If the LP is infeasible, the value is -infty. Otherwise, is the value of the objective function.
    if pulp.LpStatus[mpnvp_lp.status] == "Infeasible":
        return float("-inf")
    else:
        if verbose:
            table = PrettyTable(["t", "x", "y", "z"])
            for t in range(0, time_horizon):
                table.add_row([t, x[t].value(), y[t].value(), z[t].value()])
            print(table)
        return pulp.value(mpnvp_lp.objective)
