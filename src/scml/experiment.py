import itertools
import random
import sys
import time
import traceback
from collections import namedtuple
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple
from typing import Union

import click
import pandas as pd
from joblib import Parallel
from joblib import delayed
from negmas.helpers import unique_name
from tqdm import tqdm

from scml import DecentralizingAgent
from scml import SCML2020World

N_STEPS = 10
N_WORLDS = 2
COMPACT = True
NOLOGS = True

Constraint = namedtuple(
    "Constraint",
    ["condition_vars", "condition_values", "conditioned_var", "feasible_values"],
)


def bankruptcy_level(world: SCML2020World):
    return world.bankruptcy_rate


def relative_productivity(world: SCML2020World):
    return world.relative_productivity


def n_bankrupt(world: SCML2020World):
    return world.num_bankrupt


def relative_welfare_all(world: SCML2020World):
    return world.relative_welfare(include_bankrupt=True)


def relative_welfare_non_bankrupt(world: SCML2020World):
    return world.relative_welfare(include_bankrupt=False)


def welfare_all(world: SCML2020World):
    return world.welfare(include_bankrupt=True)


def welfare_non_bankrupt(world: SCML2020World):
    return world.welfare(include_bankrupt=False)


def productivity(world: SCML2020World):
    return world.relative_productivity


def bankruptcy_rate(world: SCML2020World):
    return world.bankruptcy_rate


def contract_execution(world: SCML2020World):
    return world.contract_execution_fraction


def breach_rate(world: SCML2020World):
    return world.breach_rate


dep_vars = {
    "relative_welfare_all": relative_welfare_all,
    "relative_welfare_non_bankrupt": relative_welfare_non_bankrupt,
    "welfare_all": welfare_all,
    "welfare_non_bankrupt": welfare_non_bankrupt,
    "productivity": productivity,
    "relative_productivity": relative_productivity,
    "bankruptcy_rate": bankruptcy_rate,
    "contract_execution": contract_execution,
    "breach_rate": breach_rate,
}

fixed_vars = {
    "initial_balance": None,
    "cost_increases_with_level": True,
    "bankruptcy_limit": 1.0,
    "n_steps": N_STEPS,
    "neg_n_steps": 20,
    "negotiation_speed": 21,
    "cash_availability": (0.8, 1.0),
    "max_productivity": (0.8, 1.0),
    "compact": COMPACT,
    "no_logs": NOLOGS,
    "agent_types": [DecentralizingAgent],
}


def jobs(n_jobs: Union[float, int]) -> int:
    if n_jobs <= 0:
        return cpu_count()
    if n_jobs == 1 and isinstance(n_jobs, int):
        return 1
    return int(0.5 + n_jobs * cpu_count())


def get_var_vals(var: str, val: Any) -> Dict[str, Any]:
    """Extracts variable name and value allowing for multiple names separated by semicolon"""
    if ";" not in var:
        return {var: val}
    return dict(zip(var.split(";"), val))


def run_config(world_config: Dict[str, Any], funcs: List[str]):
    """Runs a single configuration and returns values of all functions for that configuration"""
    world = SCML2020World(**SCML2020World.generate(**world_config))
    results = {}
    results["log_folder"] = world.log_folder
    try:
        _start = time.perf_counter()
        world.run()
        _end = time.perf_counter()
        results.update({func: dep_vars[func](world) for func in funcs})
        results["time"] = _end - _start
        results["time_per_step"] = (_end - _start) / world.n_steps
        results["failed_run"] = False
        results["exception"] = None
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print(f"Exception occurred: {str(e)}\n{traceback.format_tb(exc_traceback)}")
        results.update({func: float("nan") for func in funcs})
        results["time"] = float("nan")
        results["time_per_step"] = float("nan")
        results["failed_run"] = True
        results["exception"] = str(e)

    results.update(world_config)
    results.update(world.info)
    return results


def satisfied(config: Dict[str, Any], constraints: Iterable[Constraint]) -> bool:
    """
    Tests whether the constraints are all satisfied in the config or not

    Args:
        config:
        constraints:

    Returns:

    """
    for c in constraints:
        if not all(
            config[var] in val for var, val in zip(c.condition_vars, c.condition_values)
        ):
            continue
        if not config[c.conditioned_var] in c.feasible_values:
            return False
    return True


def generate_configs_factorial(
    ind_vars: Dict[str, list],
    fixed_vars: Dict[str, Any],
    n_worlds_per_condition: 5,
    constraints: Tuple[Constraint] = tuple(),
) -> List[Dict[str, Any]]:
    """
    Generates all configs for an experiment with a factorial design

    Args:
        ind_vars: Independent variables and their list of values
        fixed_vars: Fixed variables to be passed directly to the world generator
        n_worlds_per_condition: Number of simulations for each config
        constraints: List of constraints that must be satisfied by all configs tested

    Returns:
        List of configs

    """
    combinations = []
    for vlist in ind_vars.values():
        combinations.append([])
        for v in vlist:
            combinations[-1].append(v)
    combinations = itertools.product(*combinations)
    configs = []
    for c in combinations:
        d = {}
        for key, value in zip(ind_vars.keys(), c):
            d.update(get_var_vals(key, value))
        config = dict(**fixed_vars)
        config.update(d)
        if not satisfied(config, constraints):
            continue
        configs += [config] * n_worlds_per_condition
    return configs


def generate_configs_single(
    ind_var: str,
    ind_values: List[Any],
    fixed_vars: Dict[str, Any],
    n_worlds_per_condition: 5,
) -> List[Dict[str, Any]]:
    """
        Generates all configs for an experiment with a factorial design

        Args:
            ind_var: The independent variable. Note that multiple variables can be passed by separating them with
                     semicolons
            ind_values: The values to try for that independent variable. Note that multiple values can be passed in a
                        tuple
            fixed_vars: Fixed variables to be passed directly to the world generator
            n_worlds_per_condition: Number of simulations for each config

        Returns:
            List of configs

        """
    configs = []
    for val in ind_values:
        vars_ = get_var_vals(ind_var, val)
        vars_.update(fixed_vars)
        configs += [vars_] * n_worlds_per_condition
    return configs


def run_configs(configs: Iterable[Dict[str, Any]], n_jobs: int) -> pd.DataFrame:
    configs = list(configs)
    if n_jobs == 1:
        results = []
        for world_config, funcs in configs:
            results.append(run_config(world_config, funcs))
        return pd.DataFrame(results)

    results = Parallel(n_jobs=n_jobs)(
        delayed(run_config)(configs[i], list(dep_vars.keys()))
        for i in tqdm(range(len(configs)))
    )
    return pd.DataFrame(results)


def run(
    ind_vars: Dict[str, list],
    fixed_vars: Dict[str, Any],
    n_worlds_per_condition: 5,
    factorial: bool = True,
    constraints: Tuple[Constraint] = tuple(),
    n_jobs: Union[float, int] = 0,
) -> pd.DataFrame:
    """
    Runs an experiment
    Args:
        ind_vars: The independent variables
        fixed_vars: Variables not tested but passed to the world generator
        n_worlds_per_condition: Number of simulations to run for each condition
        factorial: If true run all possibilities, otherwise test each ind_var in an experiment (much faster)
        constraints: List of constraints for world generation each specifying the feasible values of an independent
                     variable given that a condition is met
        n_jobs: Number of jobs to use. If 1, processing will be serial. If zero all processes will be used. If a
                a fraction between zero and one, this fraction of CPU count will be used
    Returns:
        A dataframe with the results
    """
    n_jobs = jobs(n_jobs)
    if factorial:
        configs = generate_configs_factorial(
            ind_vars, fixed_vars, n_worlds_per_condition, constraints
        )
    else:
        configs = []
        for k, v in ind_vars.items():
            configs += generate_configs_single(k, v, fixed_vars, n_worlds_per_condition)
    print(
        f"Will run a total of {len(configs)} configs using {n_jobs} core{'s' if n_jobs > 1 else ''}"
    )
    return run_configs(configs, n_jobs)


@click.command("Runs an experiment")
@click.option(
    "-w",
    "--worlds",
    type=int,
    default=N_WORLDS,
    help="Number of world simulation per condition",
)
@click.option(
    "-s",
    "--steps",
    type=int,
    default=N_STEPS,
    help="Number of simulation steps per world",
)
@click.option(
    "--factorial/--independent",
    default=False,
    help="Factorial experiments will try all combinations of "
    "all independent variables. Independent experiments"
    " will run a set of simulations for each independent variable"
    " independently. Factorial experiments will need exponentially "
    "more simulations",
)
@click.option(
    "-v",
    "--variables",
    type=str,
    default="all",
    help="A semicolon separated list of independent variable names to try. The"
    " special value 'all' will use all independent variables",
)
@click.option(
    "--compact/--debug",
    default=COMPACT,
    help="Compact fast run or slower run with more extensive logs for debugging",
)
@click.option(
    "--log/--nolog",
    default=not NOLOGS,
    help="Whether to keep or not keep logs. Should not use --nolog except with "
    "--compact.",
)
@click.option("-n", "--name", type=str, default=None, help="Experiment Name")
@click.option(
    "-j",
    "--jobs",
    type=int,
    default=0,
    help="Number of parallel jobs to use. 0 means use all cores.",
)
def main(worlds, factorial, variables, name, steps, compact, log, jobs):
    fixed_vars["n_steps"] = steps
    fixed_vars["compact"] = compact
    fixed_vars["no_logs"] = not log
    ind_vars = {
        "borrow_on_breach": [True, False],
        "buy_missing_products": [True, False],
        "production_buy_missing": [True, False],
        "exogenous_buy_missing": [True, False],
        "production_no_borrow": [True, False],
        "exogenous_no_borrow": [True, False],
        "exogenous_force_max": [True, False],
        "breach_penalty;production_penalty;exogenous_penalty": [
            (0.15, 0.15, 0.15),
            (0.25, 0.25, 0.25),
        ],
        "interest_rate": [0.04, 0.08],
        "signing_delay": [0, 1],
    }

    if variables is not None and variables != "all":
        variables = ";".split(variables)
        ind_vars = {k: v for k, v in ind_vars.items() if k in variables}

    untested = list(set(ind_vars.keys()) - set(variables))
    for v in untested:
        if ";" in v:
            vs = v.split(";")
            vals = random.sample(ind_vars[v], 1)[0]
            fixed_vars.update(dict(zip(vs, vals)))
            continue
        fixed_vars[v] = random.sample(ind_vars[v], 1)[0]

    constraints = (
        Constraint(
            condition_vars=["borrow_on_breach"],
            condition_values=[[False]],
            conditioned_var="production_no_borrow",
            feasible_values=[True],
        ),
        Constraint(
            condition_vars=["borrow_on_breach"],
            condition_values=[[False]],
            conditioned_var="exogenous_no_borrow",
            feasible_values=[True],
        ),
        Constraint(
            condition_vars=["exogenous_force_max"],
            condition_values=[[False]],
            conditioned_var="production_no_borrow",
            feasible_values=[False],
        ),
        Constraint(
            condition_vars=["exogenous_force_max"],
            condition_values=[[True]],
            conditioned_var="exogenous_buy_missing",
            feasible_values=[True],
        ),
    )
    print(
        f"Running experiment:\n"
        f"Independent Variables {list(ind_vars.keys())}\nDependent Variables: {list(dep_vars.keys())}\n"
        f"Untested Variables: {list(fixed_vars.keys())}\n"
        f"n. runs per world: {worlds} ({'factorial' if factorial else 'non-factorial'})"
    )
    path = (
        Path.home()
        / "negmas"
        / "experiments"
        / unique_name("E" if name is None else name, add_time=True, sep="")
    )
    path.mkdir(parents=True, exist_ok=True)
    run(ind_vars, fixed_vars, worlds, factorial, constraints, n_jobs=jobs).to_csv(
        path / "results.csv"
    )


if __name__ == "__main__":
    main()
