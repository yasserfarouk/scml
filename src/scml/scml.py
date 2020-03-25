#!/usr/bin/env python
"""The SCML universal command line tool"""
from collections import defaultdict
import math
import os
import sys
import traceback
from functools import partial
from pathlib import Path
from pprint import pformat, pprint
from time import perf_counter
import numpy as np
from typing import List
import click
import click_config_file
import pandas as pd
import progressbar
import yaml
from tabulate import tabulate

try:
    from scml.vendor.quick.quick import gui_option
except:

    def gui_option(x):
        return x


import negmas
from negmas import save_stats

import scml
from scml.scml2019 import SCML2019World, FactoryManager
from scml.scml2019.utils import (
    anac2019_std,
    anac2019_sabotage,
    anac2019_collusion,
    DefaultGreedyManager,
)
from negmas.helpers import humanize_time, unique_name, load
from negmas.java import jnegmas_bridge_is_running

from scml.scml2020.utils import anac2020_std, anac2020_collusion
from scml.scml2020 import SCML2020Agent, SCML2020World

try:
    # disable a warning in yaml 1b1 version
    yaml.warnings({"YAMLLoadWarning": False})
except:
    pass

n_completed = 0
n_total = 0


def get_range(x, x_min, x_max):
    """Gets a range with possibly overriding it with a single value"""
    if x is not None:
        return x, x
    return x_min, x_max


def default_log_path():
    """Default location for all logs"""
    return Path.home() / "negmas" / "logs"


def default_tournament_path():
    """The default path to store tournament run info"""
    return default_log_path() / "tournaments"


def default_world_path():
    """The default path to store world run info"""
    return default_log_path() / "scml"


def print_progress(_, i, n) -> None:
    """Prints the progress of a tournament"""
    global n_completed, n_total
    n_completed = i + 1
    n_total = n
    print(
        f"{n_completed:04} of {n:04} worlds completed ({n_completed / n:0.2%})",
        flush=True,
    )


def print_world_progress(world) -> None:
    """Prints the progress of a world"""
    step = world.current_step + 1
    s = (
        f"SCML2020World# {n_completed:04}: {step:04}  of {world.n_steps:04} "
        f"steps completed ({step / world.n_steps:0.2f}) "
    )
    if n_total > 0:
        s += f"TOTAL: ({n_completed + step / world.n_steps / n_total:0.2f})"
    print(s, flush=True)


def shortest_unique_names(strs: List[str], sep="."):
    """
    Finds the shortest unique strings starting from the end of each input
    string based on the separator.

    The final strings will only be unique if the inputs are unique.

    Example:
        given ["a.b.c", "d.e.f", "a.d.c"] it will generate ["b.c", "f", "d.c"]
    """
    lsts = [_.split(sep) for _ in strs]
    names = [_[-1] for _ in lsts]
    if len(names) == len(set(names)):
        return names
    locs = defaultdict(list)
    for i, s in enumerate(names):
        locs[s].append(i)
    mapping = {"": ""}
    for s, l in locs.items():
        if len(s) < 1:
            continue
        if len(l) == 1:
            mapping[strs[l[0]]] = s
            continue
        strs_new = [sep.join(lsts[_][:-1]) for _ in l]
        prefixes = shortest_unique_names(strs_new, sep)
        for loc, prefix in zip(l, prefixes):
            x = sep.join([prefix, s])
            if x.startswith(sep):
                x = x[len(sep) :]
            mapping[strs[loc]] = x
    return [mapping[_] for _ in strs]


def nCr(n, r):
    return math.factorial(n) / math.factorial(r) / math.factorial(n - r)


click.option = partial(click.option, show_default=True)


@gui_option
@click.group()
def main():
    pass


@main.command(help="Runs an SCML2019 tournament")
@click.option(
    "--parallel/--serial",
    default=True,
    help="Run a parallel/serial tournament on a single machine",
)
@click.option(
    "--name",
    "-n",
    default="random",
    help='The name of the tournament. The special value "random" will result in a random name',
)
@click.option(
    "--steps",
    "-s",
    default=10,
    type=int,
    help="Number of steps. If passed then --steps-min and --steps-max are " "ignored",
)
@click.option(
    "--ttype",
    "--tournament-type",
    "--tournament",
    default="std",
    type=click.Choice(["collusion", "std", "sabotage"]),
    help="The config to use. It can be collusion, std or sabotage",
)
@click.option(
    "--timeout",
    "-t",
    default=-1,
    type=int,
    help="Timeout the whole tournament after the given number of seconds (0 for infinite)",
)
@click.option(
    "--configs",
    default=5,
    type=int,
    help="Number of unique configurations to generate.",
)
@click.option("--runs", default=2, help="Number of runs for each configuration")
@click.option(
    "--max-runs",
    default=-1,
    type=int,
    help="Maximum total number of runs. Zero or negative numbers mean no limit",
)
@click.option(
    "--competitors",
    default="GreedyFactoryManager;DoNothingFactoryManager",
    help="A semicolon (;) separated list of agent types to use for the competition. You"
    " can also pass the special value default for the default builtin"
    " agents",
)
@click.option(
    "--jcompetitors",
    "--java-competitors",
    default="",
    help="A semicolon (;) separated list of agent types to use for the competition.",
)
@click.option(
    "--non-competitors",
    default="",
    help="A semicolon (;) separated list of agent types to exist in the worlds as non-competitors "
    "(their scores will not be calculated).",
)
@click.option(
    "--log",
    "-l",
    type=click.Path(dir_okay=True, file_okay=False),
    default=default_tournament_path(),
    help="Default location to save logs (A folder will be created under it)",
)
@click.option(
    "--world-config",
    type=click.Path(dir_okay=False, file_okay=True),
    default=None,
    multiple=False,
    help="A file to load extra configuration parameters for world simulations from.",
)
@click.option(
    "--verbosity",
    default=1,
    type=int,
    help="verbosity level (from 0 == silent to 1 == world progress)",
)
@click.option(
    "--log-ufuns/--no-ufun-logs",
    default=False,
    help="Log ufuns into their own CSV file. Only effective if --debug is given",
)
@click.option(
    "--log-negs/--no-neg-logs",
    default=False,
    help="Log all negotiations. Only effective if --debug is given",
)
@click.option(
    "--compact/--debug",
    default=True,
    help="If True, effort is exerted to reduce the memory footprint which"
    "includes reducing logs dramatically.",
)
@click.option(
    "--raise-exceptions/--ignore-exceptions",
    default=True,
    help="Whether to ignore agent exceptions",
)
@click.option(
    "--path",
    default="",
    help="A path to be added to PYTHONPATH in which all competitors are stored. You can path a : separated list of "
    "paths on linux/mac and a ; separated list in windows",
)
@click.option(
    "--cw",
    default=3,
    type=int,
    help="Number of competitors to run at every world simulation. It must "
    "either be left at default or be a number > 1 and < the number "
    "of competitors passed using --competitors",
)
@click_config_file.configuration_option()
def tournament2019(
    parallel,
    name,
    steps,
    ttype,
    timeout,
    configs,
    runs,
    max_runs,
    competitors,
    jcompetitors,
    non_competitors,
    log,
    world_config,
    verbosity,
    log_ufuns,
    log_negs,
    compact,
    raise_exceptions,
    path,
    cw,
):
    kwargs = {}
    if world_config is not None and len(world_config) > 0:
        for wc in world_config:
            kwargs.update(load(wc))
    log = _path(log)
    if len(path) > 0:
        sys.path.append(path)
    warning_n_runs = 2000
    if timeout <= 0:
        timeout = None
    if name == "random":
        name = unique_name(base="", rand_digits=0)
    if max_runs <= 0:
        max_runs = None
    if compact:
        log_ufuns = False

    reveal_names = True
    if not compact:
        verbosity = max(1, verbosity)

    worlds_per_config = (
        None if max_runs is None else int(round(max_runs / (configs * runs)))
    )

    all_competitors = competitors.split(";")
    for i, cp in enumerate(all_competitors):
        if "." not in cp:
            all_competitors[i] = ("scml.scml2019.factory_managers.") + cp
    all_competitors_params = [dict() for _ in range(len(all_competitors))]
    if jcompetitors is not None and len(jcompetitors) > 0:
        jcompetitor_params = [{"java_class_name": _} for _ in jcompetitors.split(";")]
        for jp in jcompetitor_params:
            if "." not in jp["java_class_name"]:
                jp["java_class_name"] = (
                    "jnegmas.apps.scml.factory_managers." + jp["java_class_name"]
                )
        jcompetitors = ["scml.scml2019.JavaFactoryManager"] * len(jcompetitor_params)
        all_competitors += jcompetitors
        all_competitors_params += jcompetitor_params
        print("You are using some Java agents. The tournament MUST run serially")
        if not jnegmas_bridge_is_running():
            print(
                "Error: You are using java competitors but jnegmas bridge is not running\n\nTo correct this issue"
                " run the following command IN A DIFFERENT TERMINAL because it will block:\n\n"
                "$ negmas jnegmas"
            )
            exit(1)

    permutation_size = len(all_competitors) if "sabotage" not in ttype else 1
    if cw > len(all_competitors):
        cw = len(all_competitors)
    recommended = runs * configs * permutation_size
    if worlds_per_config is not None and worlds_per_config < 1:
        print(
            f"You need at least {(configs * runs)} runs even with a single permutation of managers."
            f".\n\nSet --max-runs to at least {(configs * runs)} (Recommended {recommended})"
        )
        return

    if max_runs is not None and max_runs < recommended:
        print(
            f"You are running {max_runs} worlds only but it is recommended to set {max_runs} to at least "
            f"{recommended}. Will continue"
        )

    if ttype == "std":
        agents = 1

    if worlds_per_config is None:
        n_worlds = (
            permutation_size
            * runs
            * configs
            * (nCr(len(all_competitors), cw) if "sabotage" not in ttype else 1)
        )
        if n_worlds > warning_n_runs:
            print(
                f"You are running the maximum possible number of permutations for each configuration. This is roughly"
                f" {n_worlds} simulations (each for {steps} steps). That will take a VERY long time."
                f"\n\nYou can reduce the number of simulations by setting --configs>=1 (currently {configs}) or "
                f"--runs>= 1 (currently {runs}) to a lower value. "
                f"\nFinally, you can limit the maximum number of worlds to run by setting --max-runs=integer."
            )
            max_runs = int(
                input(
                    f"Input the maximum number of simulations to run. Zero to run all of the {n_worlds} "
                    f"simulations. ^C or a negative number to exit [0 : {n_worlds}]:"
                )
            )
            if max_runs == 0:
                max_runs = None
            if max_runs is not None and max_runs < 0:
                exit(0)
            worlds_per_config = (
                None if max_runs is None else int(round(max_runs / (configs * runs)))
            )

    if len(jcompetitors) > 0:
        print("You are using java-competitors. The tournament will be run serially")
        parallelism = "serial"

    non_competitor_params = None
    if len(non_competitors) < 1:
        non_competitors = None
    else:
        non_competitors = non_competitors.split(";")
        for i, cp in enumerate(non_competitors):
            if "." not in cp:
                non_competitors[i] = (
                    "scml.scml2019.factory_managers."
                    # if "2019" in ttype
                    # else "scml.scml2020.agents."
                ) + cp
    if non_competitors is None:
        non_competitors = (DefaultGreedyManager,)
        non_competitor_params = ({},)
    # breakpoint()
    print(f"Tournament will be run between {len(all_competitors)} agents: ")
    pprint(all_competitors)
    print("Non-competitors are: ")
    pprint(non_competitors)
    runner = (
        anac2019_std
        if ttype == "std"
        else anac2019_collusion
        if ttype == "collusion"
        else anac2019_sabotage
    )
    start = perf_counter()
    results = runner(
        competitors=all_competitors,
        competitor_params=all_competitors_params,
        non_competitors=non_competitors,
        non_competitor_params=non_competitor_params,
        agent_names_reveal_type=reveal_names,
        n_competitors_per_world=cw,
        n_configs=configs,
        n_runs_per_world=runs,
        max_worlds_per_config=worlds_per_config,
        tournament_path=log,
        total_timeout=timeout,
        name=name,
        verbose=verbosity > 0,
        compact=compact,
        n_steps=steps,
        log_ufuns=log_ufuns,
        log_negotiations=log_negs,
        ignore_agent_exceptions=not raise_exceptions,
        ignore_contract_execution_exceptions=not raise_exceptions,
        **kwargs,
    )
    end_time = humanize_time(perf_counter() - start)
    display_results(results, "median")
    print(f"Finished in {end_time}")


@main.command(help="Runs an SCML2020 tournament")
@click.option(
    "--name",
    "-n",
    default="random",
    help='The name of the tournament. The special value "random" will result in a random name',
)
@click.option(
    "--steps",
    "-s",
    default=10,
    type=int,
    help="Number of steps. If passed then --steps-min and --steps-max are " "ignored",
)
@click.option(
    "--ttype",
    "--tournament-type",
    "--tournament",
    default="std",
    type=click.Choice(["collusion", "std"]),
    help="The config to use. It can be collusion or std",
)
@click.option(
    "--timeout",
    "-t",
    default=-1,
    type=int,
    help="Timeout the whole tournament after the given number of seconds (0 for infinite)",
)
@click.option(
    "--configs",
    default=5,
    type=int,
    help="Number of unique configurations to generate.",
)
@click.option("--runs", default=2, help="Number of runs for each configuration")
@click.option(
    "--max-runs",
    default=-1,
    type=int,
    help="Maximum total number of runs. Zero or negative numbers mean no limit",
)
@click.option(
    "--competitors",
    default="DecentralizingAgent;BuyCheapSellExpensiveAgent;RandomAgent",
    help="A semicolon (;) separated list of agent types to use for the competition. You"
    " can also pass the special value default for the default builtin"
    " agents",
)
@click.option(
    "--non-competitors",
    default="",
    help="A semicolon (;) separated list of agent types to exist in the worlds as non-competitors "
    "(their scores will not be calculated).",
)
@click.option(
    "--log",
    "-l",
    type=click.Path(dir_okay=True, file_okay=False),
    default=default_tournament_path(),
    help="Default location to save logs (A folder will be created under it)",
)
@click.option(
    "--world-config",
    type=click.Path(dir_okay=False, file_okay=True),
    default=None,
    multiple=False,
    help="A file to load extra configuration parameters for world simulations from.",
)
@click.option(
    "--verbosity",
    default=1,
    type=int,
    help="verbosity level (from 0 == silent to 1 == world progress)",
)
@click.option(
    "--log-ufuns/--no-ufun-logs",
    default=False,
    help="Log ufuns into their own CSV file. Only effective if --debug is given",
)
@click.option(
    "--log-negs/--no-neg-logs",
    default=False,
    help="Log all negotiations. Only effective if --debug is given",
)
@click.option(
    "--compact/--debug",
    default=True,
    help="If True, effort is exerted to reduce the memory footprint which"
    "includes reducing logs dramatically.",
)
@click.option(
    "--raise-exceptions/--ignore-exceptions",
    default=True,
    help="Whether to ignore agent exceptions",
)
@click.option(
    "--path",
    default="",
    help="A path to be added to PYTHONPATH in which all competitors are stored. You can path a : separated list of "
    "paths on linux/mac and a ; separated list in windows",
)
@click.option(
    "--cw",
    default=3,
    type=int,
    help="Number of competitors to run at every world simulation. It must "
    "either be left at default or be a number > 1 and < the number "
    "of competitors passed using --competitors",
)
@click.option(
    "--parallel/--serial",
    default=True,
    help="Run a parallel/serial tournament on a single machine",
)
@click_config_file.configuration_option()
def tournament2020(
    name,
    steps,
    timeout,
    ttype,
    log,
    verbosity,
    runs,
    configs,
    max_runs,
    competitors,
    world_config,
    non_competitors,
    compact,
    log_ufuns,
    log_negs,
    raise_exceptions,
    path,
    cw,
    parallel,
):
    kwargs = {}
    if world_config is not None and len(world_config) > 0:
        for wc in world_config:
            kwargs.update(load(wc))
    log = _path(log)
    if len(path) > 0:
        sys.path.append(path)
    warning_n_runs = 2000
    if timeout <= 0:
        timeout = None
    if name == "random":
        name = unique_name(base="", rand_digits=0)
    if max_runs <= 0:
        max_runs = None
    if compact:
        log_ufuns = False

    reveal_names = True
    if not compact:
        verbosity = max(1, verbosity)

    worlds_per_config = (
        None if max_runs is None else int(round(max_runs / (configs * runs)))
    )

    all_competitors = competitors.split(";")
    for i, cp in enumerate(all_competitors):
        if "." not in cp:
            all_competitors[i] = ("scml.scml2020.agents.") + cp
    all_competitors_params = [dict() for _ in range(len(all_competitors))]

    permutation_size = len(all_competitors)
    if cw > len(all_competitors):
        cw = len(all_competitors)
    recommended = runs * configs * permutation_size
    if worlds_per_config is not None and worlds_per_config < 1:
        print(
            f"You need at least {(configs * runs)} runs even with a single permutation of managers."
            f".\n\nSet --max-runs to at least {(configs * runs)} (Recommended {recommended})"
        )
        return

    if max_runs is not None and max_runs < recommended:
        print(
            f"You are running {max_runs} worlds only but it is recommended to set {max_runs} to at least "
            f"{recommended}. Will continue"
        )

    if ttype == "std":
        agents = 1

    if worlds_per_config is None:
        n_worlds = permutation_size * runs * configs * nCr(len(all_competitors), cw)
        if n_worlds > warning_n_runs:
            print(
                f"You are running the maximum possible number of permutations for each configuration. This is roughly"
                f" {n_worlds} simulations (each for {steps} steps). That will take a VERY long time."
                f"\n\nYou can reduce the number of simulations by setting --configs>=1 (currently {configs}) or "
                f"--runs>= 1 (currently {runs}) to a lower value. "
                f"\nFinally, you can limit the maximum number of worlds to run by setting --max-runs=integer."
            )
            max_runs = int(
                input(
                    f"Input the maximum number of simulations to run. Zero to run all of the {n_worlds} "
                    f"simulations. ^C or a negative number to exit [0 : {n_worlds}]:"
                )
            )
            if max_runs == 0:
                max_runs = None
            if max_runs is not None and max_runs < 0:
                exit(0)
            worlds_per_config = (
                None if max_runs is None else int(round(max_runs / (configs * runs)))
            )

    non_competitor_params = None
    if len(non_competitors) < 1:
        non_competitors = None
    else:
        non_competitors = non_competitors.split(";")
        for i, cp in enumerate(non_competitors):
            if "." not in cp:
                non_competitors[i] = ("scml.scml2020.agents.") + cp

    if non_competitors is None:
        non_competitors = scml.scml2020.utils.DefaultAgents
        non_competitor_params = tuple({} for _ in range(len(non_competitors)))
    # breakpoint()
    print(f"Tournament will be run between {len(all_competitors)} agents: ")
    pprint(all_competitors)
    print("Non-competitors are: ")
    pprint(non_competitors)
    runner = anac2020_std if ttype == "std" else anac2020_collusion
    parallelism = "parallel" if parallel else "serial"
    prog_callback = print_world_progress if verbosity > 1 else None
    start = perf_counter()
    results = runner(
        competitors=all_competitors,
        competitor_params=all_competitors_params,
        non_competitors=non_competitors,
        non_competitor_params=non_competitor_params,
        agent_names_reveal_type=reveal_names,
        n_competitors_per_world=cw,
        n_configs=configs,
        n_runs_per_world=runs,
        max_worlds_per_config=worlds_per_config,
        tournament_path=log,
        total_timeout=timeout,
        name=name,
        verbose=verbosity > 0,
        compact=compact,
        n_steps=steps,
        parallelism=parallelism,
        world_progress_callback=prog_callback,
        log_ufuns=log_ufuns,
        log_negotiations=log_negs,
        ignore_agent_exceptions=not raise_exceptions,
        ignore_contract_execution_exceptions=not raise_exceptions,
        **kwargs,
    )
    end_time = humanize_time(perf_counter() - start)
    display_results(results, "median")
    print(f"Finished in {end_time}")


def display_results(results, metric):
    viewmetric = ["50%" if metric == "median" else metric]
    strs = results.score_stats["agent_type"].values.tolist()
    short_names = shortest_unique_names(strs)
    mapping = dict(zip(strs, short_names))
    results.score_stats["agent_type"] = short_names
    print(
        tabulate(
            results.score_stats.sort_values(by=viewmetric, ascending=False),
            headers="keys",
            tablefmt="psql",
        )
    )
    if metric in ("mean", "sum"):
        results.ttest["a"] = [mapping[_] for _ in results.ttest["a"]]
        results.ttest["b"] = [mapping[_] for _ in results.ttest["b"]]
        print(tabulate(results.ttest, headers="keys", tablefmt="psql"))
    else:
        results.kstest["a"] = [mapping[_] for _ in results.kstest["a"]]
        results.kstest["b"] = [mapping[_] for _ in results.kstest["b"]]
        print(tabulate(results.kstest, headers="keys", tablefmt="psql"))

    agg_stats = results.agg_stats.loc[
        :,
        [
            "n_negotiations_sum",
            "n_contracts_concluded_sum",
            "n_contracts_signed_sum",
            "n_contracts_executed_sum",
            "activity_level_sum",
        ],
    ]
    agg_stats.columns = ["negotiated", "concluded", "signed", "executed", "business"]
    print(tabulate(agg_stats.describe(), headers="keys", tablefmt="psql"))


def _path(path) -> Path:
    """Creates an absolute path from given path which can be a string"""
    if isinstance(path, Path):
        return path.absolute()
    path.replace("/", os.sep)
    if isinstance(path, str):
        if path.startswith("~"):
            path = Path.home() / (os.sep.join(path.split(os.sep)[1:]))
    return Path(path).absolute()


@main.command(help="Run an SCML2019 world simulation")
@click.option("--steps", default=100, type=int, help="Number of steps.")
@click.option(
    "--levels",
    default=3,
    type=int,
    help="Number of intermediate production levels (processes). "
    "-1 means a single product and no factories.",
)
@click.option(
    "--competitors",
    default="GreedyFactoryManager",
    help="A semicolon (;) separated list of agent types to use for the competition.",
)
@click.option(
    "--jcompetitors",
    "--java-competitors",
    default="",
    help="A semicolon (;) separated list of agent types to use for the competition.",
)
@click.option(
    "--log",
    type=click.Path(file_okay=False, dir_okay=True),
    default=default_log_path(),
    help="Default location to save logs (A folder will be created under it)",
)
@click.option(
    "--log-ufuns/--no-ufun-logs",
    default=False,
    help="Log ufuns into their own CSV file. Only effective if --debug is given",
)
@click.option(
    "--log-negs/--no-neg-logs",
    default=False,
    help="Log all negotiations. Only effective if --debug is given",
)
@click.option(
    "--compact/--debug",
    default=False,
    help="If True, effort is exerted to reduce the memory footprint which"
    "includes reducing logs dramatically.",
)
@click.option(
    "--raise-exceptions/--ignore-exceptions",
    default=True,
    help="Whether to ignore agent exceptions",
)
@click.option(
    "--path",
    default="",
    help="A path to be added to PYTHONPATH in which all competitors are stored. You can path a : separated list of "
    "paths on linux/mac and a ; separated list in windows",
)
@click.option(
    "--world-config",
    type=click.Path(dir_okay=False, file_okay=True),
    default=None,
    multiple=False,
    help="A file to load extra configuration parameters for world simulations from.",
)
@click_config_file.configuration_option()
def run2019(
    steps,
    levels,
    competitors,
    jcompetitors,
    log,
    compact,
    log_ufuns,
    log_negs,
    raise_exceptions,
    path,
    world_config,
):
    kwargs = dict(
        no_bank=True,
        no_insurance=False,
        prevent_cfp_tampering=True,
        ignore_negotiated_penalties=False,
        neg_step_time_limit=10,
        breach_penalty_society=0.02,
        premium=0.03,
        premium_time_increment=0.1,
        premium_breach_increment=0.001,
        max_allowed_breach_level=None,
        breach_penalty_society_min=0.0,
        breach_penalty_victim=0.0,
        breach_move_max_product=True,
        transfer_delay=0,
        start_negotiations_immediately=False,
        catalog_profit=0.15,
        financial_reports_period=10,
        default_price_for_products_without_one=1,
        compensation_fraction=0.5,
    )
    if world_config is not None and len(world_config) > 0:
        for wc in world_config:
            kwargs.update(load(wc))
    if len(path) > 0:
        sys.path.append(path)

    params = {"steps": steps, "levels": levels}
    if compact:
        log_ufuns = False
        log_negs = False
    log_dir = _path(log)
    world_name = unique_name(base="scml", add_time=True, rand_digits=0)
    log_dir = log_dir / world_name
    log_dir = log_dir.absolute()
    os.makedirs(log_dir, exist_ok=True)

    factory_kwargs = {}
    exception = None

    def _no_default(s):
        return not (
            s.startswith("scml.scml2019") and s.endswith("GreedyFactoryManager")
        )

    all_competitors = competitors.split(";")
    for i, cp in enumerate(all_competitors):
        if "." not in cp:
            all_competitors[i] = "scml.scml2019.factory_managers." + cp
    all_competitors_params = [
        dict() if _no_default(_) else factory_kwargs for _ in all_competitors
    ]
    if jcompetitors is not None and len(jcompetitors) > 0:
        jcompetitor_params = [{"java_class_name": _} for _ in jcompetitors.split(";")]
        for jp in jcompetitor_params:
            if "." not in jp["java_class_name"]:
                jp["java_class_name"] = (
                    "jnegmas.apps.scml.factory_managers." + jp["java_class_name"]
                )
        jcompetitors = ["scml.scml2019.JavaFactoryManager"] * len(jcompetitor_params)
        all_competitors += jcompetitors
        all_competitors_params += jcompetitor_params
        print("You are using some Java agents. The tournament MUST run serially")
        parallelism = "serial"
        if not jnegmas_bridge_is_running():
            print(
                "Error: You are using java competitors but jnegmas bridge is not running\n\nTo correct this issue"
                " run the following command IN A DIFFERENT TERMINAL because it will block:\n\n"
                "$ negmas jnegmas"
            )
            exit(1)

    world = SCML2019World.chain_world(
        n_steps=steps,
        n_intermediate_levels=levels,
        default_manager_params=factory_kwargs,
        compact=compact,
        agent_names_reveal_type=True,
        log_ufuns=log_ufuns,
        manager_types=all_competitors,
        manager_params=all_competitors_params,
        log_negotiations=log_negs,
        log_folder=log_dir,
        name=world_name,
        ignore_agent_exceptions=not raise_exceptions,
        ignore_contract_execution_exceptions=not raise_exceptions,
        **kwargs,
    )
    failed = False
    strt = perf_counter()
    try:
        for i in progressbar.progressbar(range(world.n_steps), max_value=world.n_steps):
            elapsed = perf_counter() - strt
            if world.time_limit is not None and elapsed >= world.time_limit:
                break
            if not world.step():
                break
    except Exception:
        exception = traceback.format_exc()
        failed = True
    elapsed = perf_counter() - strt

    def print_and_log(s):
        world.logdebug(s)
        print(s)

    world.logdebug(f"{pformat(world.stats, compact=True)}")
    world.logdebug(
        f"=================================================\n"
        f"steps: {steps}, levels: {levels}\n"
        f"=================================================="
    )

    save_stats(world=world, log_dir=log_dir, params=params)

    if len(world.saved_contracts) > 0:
        data = pd.DataFrame(world.saved_contracts)
        data = data.sort_values(["delivery_time"])
        data = data.loc[
            data.signed_at >= 0,
            [
                "seller_type",
                "buyer_type",
                "seller_name",
                "buyer_name",
                "delivery_time",
                "unit_price",
                "quantity",
                "product_name",
                "n_neg_steps",
                "signed_at",
            ],
        ]
        data.columns = [
            "seller_type",
            "buyer_type",
            "seller",
            "buyer",
            "t",
            "price",
            "q",
            "product",
            "steps",
            "signed",
        ]
        print_and_log(tabulate(data, headers="keys", tablefmt="psql"))

        data["product_id"] = np.array([_.id for _ in data["product"].values])
        d2 = (
            data.loc[(~(data["signed"].isnull())) & (data["signed"] > -1), :]
            .groupby(["product_id"])
            .apply(
                lambda x: pd.DataFrame(
                    [
                        {
                            "uprice": np.sum(x["price"] * x["q"]) / np.sum(x["q"]),
                            "quantity": np.sum(x["q"]),
                        }
                    ]
                )
            )
        )
        d2 = d2.reset_index().sort_values(["product_id"])
        products = dict(zip([_.id for _ in world.products], world.products))
        d2["Product"] = np.array([products[_] for _ in d2["product_id"].values])
        d2 = d2.loc[:, ["Product", "uprice", "quantity"]]
        d2.columns = ["Product", "Avg. Unit Price", "Total Quantity"]
        print_and_log(tabulate(d2, headers="keys", tablefmt="psql"))

        n_executed = sum(world.stats["n_contracts_executed"])
        n_negs = sum(world.stats["n_negotiations"])
        n_contracts = len(world.saved_contracts)
        try:
            agent_scores = sorted(
                [
                    [_.name, world.a2f[_.id].total_balance]
                    for _ in world.agents.values()
                    if isinstance(_, FactoryManager)
                ],
                key=lambda x: x[1],
                reverse=True,
            )
            agent_scores = pd.DataFrame(
                data=np.array(agent_scores), columns=["Agent", "Final Balance"]
            )
            print_and_log(tabulate(agent_scores, headers="keys", tablefmt="psql"))
        except:
            pass
        winners = [
            f"{_.name} gaining {world.a2f[_.id].total_balance / world.a2f[_.id].initial_balance - 1.0:0.0%}"
            for _ in world.winners
        ]
        print_and_log(
            f"{n_contracts} contracts :-) [N. Negotiations: {n_negs}, Agreement Rate: "
            f"{world.agreement_rate:0.0%}]"
            f" (rounds/successful negotiation: {world.n_negotiation_rounds_successful:5.2f}, "
            f"rounds/broken negotiation: {world.n_negotiation_rounds_failed:5.2f})"
        )
        total = (
            world.contract_dropping_fraction
            + world.contract_nullification_fraction
            + world.contract_err_fraction
            + world.breach_fraction
            + world.contract_execution_fraction
        )
        n_cancelled = int(round(n_contracts * world.cancellation_rate))
        n_signed = n_contracts - n_cancelled
        n_dropped = int(round(n_signed * world.contract_dropping_fraction))
        n_nullified = int(round(n_signed * world.contract_nullification_fraction))
        n_erred = int(round(n_signed * world.contract_err_fraction))
        n_breached = int(round(n_signed * world.breach_fraction))
        n_executed = int(round(n_signed * world.contract_execution_fraction))

        print_and_log(
            f"Cancelled: {world.cancellation_rate:0.1%}, "
            f"Executed: {world.contract_execution_fraction:0.1%}"
            f", Breached: {world.breach_fraction:0.1%}"
            f", Erred: {world.contract_err_fraction:0.1%}"
            f", Nullified: {world.contract_nullification_fraction:0.1%}"
            f", Dropped: {world.contract_dropping_fraction:0.1%}"
            f" (Sum:{total: 0.2%})\n"
            f"Negotiated: {n_negs} Concluded: {n_contracts} Signed: {n_signed} Dropped: {n_dropped}  "
            f"Nullified: {n_nullified} "
            f"Erred {n_erred} Breached {n_breached} (b. level {world.breach_level:0.1%}) => Executed: {n_executed}\n"
            f"Business size: "
            f"{world.business_size}\n"
            f"Winners: {winners}\n"
            f"Running Time {humanize_time(elapsed)}"
        )
    else:
        print_and_log("No contracts! :-(")
        print_and_log(f"Running Time {humanize_time(elapsed)}")

    if failed:
        print(exception)
        world.logdebug(exception)
        print(f"FAILED at step {world.current_step} of {world.n_steps}\n")


@main.command(help="Run an SCML2020 world simulation")
@click.option("--steps", default=10, type=int, help="Number of steps.")
@click.option(
    "--competitors",
    default="RandomAgent;BuyCheapSellExpensiveAgent;DecentralizingAgent;DoNothingAgent",
    help="A semicolon (;) separated list of agent types to use for the competition.",
)
@click.option(
    "--log",
    type=click.Path(file_okay=False, dir_okay=True),
    default=default_world_path(),
    help="Default location to save logs (A folder will be created under it)",
)
@click.option(
    "--time", "-t", default=-1, type=int, help="Allowed time for the simulation"
)
@click.option(
    "--log-ufuns/--no-ufun-logs",
    default=False,
    help="Log ufuns into their own CSV file. Only effective if --debug is given",
)
@click.option(
    "--log-negs/--no-neg-logs",
    default=False,
    help="Log all negotiations. Only effective if --debug is given",
)
@click.option(
    "--compact/--debug",
    default=False,
    help="If True, effort is exerted to reduce the memory footprint which"
    "includes reducing logs dramatically.",
)
@click.option(
    "--show-contracts/--no-contracts",
    default=True,
    help="Show or do not show all signed contracts",
)
@click.option(
    "--raise-exceptions/--ignore-exceptions",
    default=True,
    help="Whether to ignore agent exceptions",
)
@click.option(
    "--path",
    default="",
    help="A path to be added to PYTHONPATH in which all competitors are stored. You can path a : separated list of "
    "paths on linux/mac and a ; separated list in windows",
)
@click.option(
    "--world-config",
    type=click.Path(dir_okay=False, file_okay=True),
    default=None,
    multiple=False,
    help="A file to load extra configuration parameters for world simulations from.",
)
@click_config_file.configuration_option()
def run2020(
    steps,
    time,
    competitors,
    log,
    compact,
    log_ufuns,
    log_negs,
    raise_exceptions,
    path,
    world_config,
    show_contracts,
):
    if time <= 0:
        time = None
    kwargs = {"n_steps": steps}
    if world_config is not None and len(world_config) > 0:
        for wc in world_config:
            kwargs.update(load(wc))
    if len(path) > 0:
        sys.path.append(path)

    params = kwargs

    if compact:
        log_ufuns = False
        log_negs = False

    log_dir = _path(log)

    world_name = unique_name(base="scml2020", add_time=True, rand_digits=0)
    log_dir = log_dir / world_name
    log_dir = log_dir.absolute()
    os.makedirs(log_dir, exist_ok=True)

    exception = None

    def _no_default(s):
        return not (s.startswith("scml.2020.agents."))

    all_competitors = competitors.split(";")
    for i, cp in enumerate(all_competitors):
        if "." not in cp:
            all_competitors[i] = "scml.scml2020.agents." + cp
    all_competitors_params = [dict() for _ in all_competitors]
    world = SCML2020World(
        **SCML2020World.generate(
            time_limit=time,
            compact=compact,
            log_ufuns=log_ufuns,
            agent_types=all_competitors,
            agent_params=all_competitors_params,
            log_negotiations=log_negs,
            log_folder=log_dir,
            name=world_name,
            ignore_agent_exceptions=not raise_exceptions,
            ignore_contract_execution_exceptions=not raise_exceptions,
            **kwargs,
        )
    )
    failed = False
    strt = perf_counter()
    try:
        for i in progressbar.progressbar(range(world.n_steps), max_value=world.n_steps):
            elapsed = perf_counter() - strt
            if world.time_limit is not None and elapsed >= world.time_limit:
                break
            if not world.step():
                break
    except Exception:
        exception = traceback.format_exc()
        failed = True
    elapsed = perf_counter() - strt

    def print_and_log(s):
        world.logdebug(s)
        print(s)

    world.logdebug(f"{pformat(world.stats, compact=True)}")
    world.logdebug(
        f"=================================================\n"
        f"steps: {steps}, time: {time}\n"
        f"=================================================="
    )

    save_stats(world=world, log_dir=log_dir, params=params)

    if len(world.saved_contracts) > 0:
        data = pd.DataFrame(world.saved_contracts)
        data = data.sort_values(["delivery_time"])
        data = data.loc[
            data.signed_at >= 0,
            [
                "seller_name",
                "buyer_name",
                "delivery_time",
                "unit_price",
                "quantity",
                "product_name",
                "n_neg_steps",
                "signed_at",
                "executed_at",
            ],
        ]
        data.columns = [
            "seller",
            "buyer",
            "t",
            "price",
            "q",
            "product",
            "steps",
            "signed",
            "executed",
        ]
        if show_contracts:
            print_and_log(tabulate(data, headers="keys", tablefmt="psql"))

        d2 = (
            data.loc[(~(data["executed"].isnull())) & (data["executed"] > -1), :]
            .groupby(["product"])
            .apply(
                lambda x: pd.DataFrame(
                    [
                        {
                            "uprice": np.sum(x["price"] * x["q"]) / np.sum(x["q"]),
                            "quantity": np.sum(x["q"]),
                        }
                    ]
                )
            )
        )
        d2 = d2.reset_index().sort_values(["product"])
        d2["Catalog"] = world.catalog_prices[
            d2["product"].str.slice(start=-1).astype(int).values
        ]
        d2["Trading"] = world.trading_prices[
            d2["product"].str.slice(start=-1).astype(int).values
        ]
        d2["Product"] = d2["product"]
        d2 = d2.loc[:, ["Product", "quantity", "uprice", "Catalog", "Trading"]]

        d2.columns = ["Product", "Quantity", "Avg. Price", "Catalog", "Trading"]
        print_and_log(tabulate(d2, headers="keys", tablefmt="psql"))

        n_executed = sum(world.stats["n_contracts_executed"])
        n_negs = sum(world.stats["n_negotiations"])
        n_contracts = len(world.saved_contracts)
        try:
            agent_scores = sorted(
                [
                    [_.name, world.a2f[_.id].total_balance]
                    for _ in world.agents.values()
                    if isinstance(_, SCML2020Agent)
                ],
                key=lambda x: x[1],
                reverse=True,
            )
            agent_scores = pd.DataFrame(
                data=np.array(agent_scores), columns=["Agent", "Final Balance"]
            )
            print_and_log(tabulate(agent_scores, headers="keys", tablefmt="psql"))
        except:
            pass
        winners = [
            f"{_.name} gaining {world.a2f[_.id].current_balance / world.a2f[_.id].initial_balance - 1.0:0.0%}"
            for _ in world.winners
        ]
        print_and_log(
            f"{n_contracts} contracts :-) [N. Negotiations: {n_negs}, Agreement Rate: "
            f"{world.agreement_fraction:0.0%}]"
            f" (rounds/successful negotiation: {world.n_negotiation_rounds_successful:5.2f}, "
            f"rounds/broken negotiation: {world.n_negotiation_rounds_failed:5.2f})"
        )
        total = (
            world.contract_dropping_fraction
            + world.contract_nullification_fraction
            + world.contract_err_fraction
            + world.breach_fraction
            + world.contract_execution_fraction
        )
        n_cancelled = (
            int(round(n_contracts * world.cancellation_rate)) if n_negs > 0 else 0
        )
        n_signed = n_contracts - n_cancelled
        n_dropped = int(round(n_signed * world.contract_dropping_fraction))
        n_nullified = int(round(n_signed * world.contract_nullification_fraction))
        n_erred = int(round(n_signed * world.contract_err_fraction))
        n_breached = int(round(n_signed * world.breach_fraction))
        n_executed = int(round(n_signed * world.contract_execution_fraction))
        exogenous = [_ for _ in world.saved_contracts if not _["issues"]]
        negotiated = [_ for _ in world.saved_contracts if _["issues"]]
        n_exogenous = len(exogenous)
        n_negotiated = len(negotiated)
        n_exogenous_signed = len([_ for _ in exogenous if _["signed_at"] >= 0])
        n_negotiated_signed = len([_ for _ in negotiated if _["signed_at"] >= 0])
        print_and_log(
            f"Exogenous Contracts : {n_exogenous} of which {n_exogenous_signed} "
            f" were signed ({n_exogenous_signed/n_exogenous if n_exogenous!=0 else 0: 0.1%})"
        )
        print_and_log(
            f"Negotiated Contracts: {n_negotiated} of which {n_negotiated_signed} "
            f" were signed ({n_negotiated_signed/n_negotiated if n_negotiated!=0 else 0: 0.1%})"
        )
        print_and_log(
            f"All Contracts       : {n_exogenous + n_negotiated} of which {n_exogenous_signed + n_negotiated_signed} "
            f" were signed ({1-world.cancellation_rate:0.1%})"
        )
        print_and_log(
            f"Executed: {world.contract_execution_fraction:0.1%}"
            f", Breached: {world.breach_fraction:0.1%}"
            f", Erred: {world.contract_err_fraction:0.1%}"
            f", Nullified: {world.contract_nullification_fraction:0.1%}"
            f", Dropped: {world.contract_dropping_fraction:0.1%}"
            f" (Sum:{total: 0.0%})\n"
            f"Negotiated: {n_negs} Concluded: {n_contracts} Signed: {n_signed} Dropped: {n_dropped}  "
            f"Nullified: {n_nullified} "
            f"Erred {n_erred} Breached {n_breached} (b. level {world.breach_level:0.1%}) => Executed: {n_executed}\n"
            f"Business size: "
            f"{world.business_size}\n"
            f"Welfare (Excluding Bankrupt): {world.welfare(False)} ({world.relative_welfare(False):5.03%}), "
            f"Welfare (Including Bankrupt): {world.welfare(True)} ({world.relative_welfare(True):5.03%})\n"
            f"Productivity: {world.productivity:5.03%} ({world.relative_productivity:4.03%}), "
            f"Bankrupted Agents: {world.num_bankrupt} ({world.bankruptcy_rate:5.03%})\n"
            f"Winners: {winners}\n"
            f"Running Time {humanize_time(elapsed)}"
        )
    else:
        print_and_log("No contracts! :-(")
        print_and_log(f"Running Time {humanize_time(elapsed)}")

    if failed:
        print(exception)
        world.logdebug(exception)
        print(f"FAILED at step {world.current_step} of {world.n_steps}\n")


@main.command(help="Prints SCML version and NegMAS version")
def version():
    print(f"SCML: {scml.__version__} (NegMAS: {negmas.__version__})")


if __name__ == "__main__":
    main()
