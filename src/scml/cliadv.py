#!/usr/bin/env python
"""The SCML universal command line tool"""
import os
import pathlib
import sys
import traceback
import warnings
from functools import partial
from pathlib import Path
from pprint import pformat
from pprint import pprint
from time import perf_counter

import click
import click_config_file
import negmas
import numpy as np
import pandas as pd
import progressbar
import yaml
from negmas import save_stats
from negmas.helpers import camel_case
from negmas.helpers import humanize_time
from negmas.helpers import load
from negmas.helpers import unique_name
from negmas.java import init_jnegmas_bridge
from negmas.java import jnegmas_bridge_is_running
from negmas.tournaments import combine_tournament_stats
from negmas.tournaments import combine_tournaments
from negmas.tournaments import create_tournament
from negmas.tournaments import evaluate_tournament
from negmas.tournaments import run_tournament
from tabulate import tabulate

import scml
from scml.scml2019.common import DEFAULT_NEGOTIATOR
from scml.scml2019.utils import anac2019_assigner
from scml.scml2019.utils import anac2019_config_generator
from scml.scml2019.utils import anac2019_sabotage
from scml.scml2019.utils import anac2019_sabotage_assigner
from scml.scml2019.utils import anac2019_sabotage_config_generator
from scml.scml2019.utils import anac2019_world_generator
from scml.scml2019.utils import sabotage_effectiveness
from scml.scml2020.utils import anac2020_assigner
from scml.scml2020.utils import anac2020_config_generator
from scml.scml2020.utils import anac2020_world_generator

try:
    from .vendor.quick.quick import gui_option
except:

    def gui_option(x):
        return x


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


click.option = partial(click.option, show_default=True)


@gui_option
@click.group()
def cli():
    pass


@cli.group(chain=True, invoke_without_command=True)
@click.pass_context
@click.option(
    "--ignore-warnings/--show-warnings",
    default=False,
    help="Ignore/show runtime warnings",
)
def tournament(ctx, ignore_warnings):
    if ignore_warnings:
        import warnings

        warnings.filterwarnings("ignore")
    ctx.obj = {}


@tournament.command(help="Creates a tournament")
@click.option(
    "--name",
    "-n",
    default="random",
    help='The name of the tournament. The special value "random" will result in a random name',
)
@click.option(
    "--steps",
    "-s",
    default=None,
    type=int,
    help="Number of steps. If passed then --steps-min and --steps-max are " "ignored",
)
@click.option(
    "--steps-min",
    default=50,
    type=int,
    help="Minimum number of steps (only used if --steps was not passed",
)
@click.option(
    "--steps-max",
    default=100,
    type=int,
    help="Maximum number of steps (only used if --steps was not passed",
)
@click.option(
    "--ttype",
    "--tournament-type",
    "--tournament",
    default="anac2020std",
    type=click.Choice(
        [
            "scml2019collusion",
            "scml2019std",
            "scml2019sabotage",
            "scml2020std",
            "scml2020collusion",
            "anac2019collusion",
            "anac2019std",
            "anac2019sabotage",
            "anac2020std",
            "anac2020collusion",
        ]
    ),
    help="The config to use. Default is ANAC 2019. Options supported are anac2019std, anac2019collusion, "
    "anac2019sabotage, anac2020scml, anac2020collusion. You can replace anac with scml",
)
@click.option(
    "--timeout",
    "-t",
    default=0,
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
    "--agents",
    default=3,
    type=int,
    help="Number of agents per competitor (not used for anac2019std in which this is preset to 1).",
)
@click.option(
    "--factories", default=2, type=int, help="Numbers of factories to have per level."
)
@click.option(
    "--competitors",
    default="default",
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
    default=tuple(),
    multiple=True,
    help="A file to load extra configuration parameters for world simulations from.",
)
@click.option(
    "--verbosity",
    default=1,
    type=int,
    help="verbosity level (from 0 == silent to 1 == world progress)",
)
@click.option(
    "--reveal-names/--hidden-names",
    default=True,
    help="Reveal agent names (should be used only for " "debugging)",
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
    default=None,
    type=int,
    help="Number of competitors to run at every world simulation. It must "
    "either be left at default or be a number > 1 and < the number "
    "of competitors passed using --competitors",
)
@click.option(
    "--factories-min",
    default=2,
    type=int,
    help="SCML2020: Minimum number of agents per production level",
)
@click.option(
    "--factories-max",
    default=6,
    type=int,
    help="SCML2020: Maximum number of agents per production level",
)
@click.option(
    "--inputs",
    type=int,
    default=1,
    help="SCML2020: The number inputs to each production process",
)
@click.option(
    "--inputs-min",
    type=int,
    default=1,
    help="SCML2020: The minimum number inputs to each production process",
)
@click.option(
    "--inputs-max",
    type=int,
    default=1,
    help="SCML2020: The maximum number inputs to each production process",
)
@click.option(
    "--outputs",
    type=int,
    default=1,
    help="SCML2020: The number outputs to each production process",
)
@click.option(
    "--outputs-min",
    type=int,
    default=1,
    help="SCML2020: The minimum number outputs to each production process",
)
@click.option(
    "--outputs-max",
    type=int,
    default=1,
    help="The maximum number outputs to each production process",
)
@click.option(
    "--costs",
    type=int,
    default=None,
    help="SCML2020: The production cost (see --increasing-costs, --costs-min, --costs-max)",
)
@click.option(
    "--costs-min", type=int, default=1, help="SCML2020: The minimum production cost"
)
@click.option(
    "--costs-max", type=int, default=10, help="SCML2020: The maximum production cost"
)
@click.option(
    "--productivity",
    type=float,
    default=None,
    help="SCML2020: The fraction of production slots (lines/steps) that can be occupied with production given the "
    "exogenous contracts",
)
@click.option(
    "--productivity-min",
    type=float,
    default=0.8,
    help="SCML2020: The minimum fraction of production slots (lines/steps) that can be occupied with production given the "
    "exogenous contracts",
)
@click.option(
    "--productivity-max",
    type=float,
    default=1.0,
    help="SCML2020: The maximum fraction of production slots (lines/steps) that can be occupied with production given the "
    "exogenous contracts",
)
@click.option(
    "--cash-availability",
    type=float,
    default=None,
    help="SCML2020: The availability of cash which is a nubmer between zero and one specifying how much of the total"
    " production requirements of agents is available as initial balance. It is only effective"
    " when --balance < 0",
)
@click.option(
    "--cash-availability-min",
    type=float,
    default=0.8,
    help="SCML2020: The availability of cash which is a nubmer between zero and one specifying how much of the total"
    " production requirements of agents is available as initial balance. It is only effective"
    " when --balance < 0",
)
@click.option(
    "--cash-availability-max",
    type=float,
    default=1.0,
    help="SCML2020: The availability of cash which is a nubmer between zero and one specifying how much of the total"
    " production requirements of agents is available as initial balance. It is only effective"
    " when --balance < 0",
)
@click.option(
    "--profit-mean",
    type=float,
    default=0.15,
    help="SCML2020: The mean of the expected profit used to select catalog prices for the world",
)
@click.option(
    "--profit-std",
    type=float,
    default=0.05,
    help="SCML2020: The std. dev. of the expected profit used to select catalog prices for the world",
)
@click.option(
    "--increasing-costs/--fixed-costs",
    default=True,
    help="SCML2020: Whether to increase the costs with production level (i.e. processes near the end"
    " of the chain become more costly to run than those near the raw material)",
)
@click.option(
    "--equal-exogenous-supplies/--variable-exogenous-supplies",
    default=False,
    help="SCML2020: Whether exogenous supplies are equal for all agents executing process 0",
)
@click.option(
    "--equal-exogenous-sales/--variable-exogenous-sales",
    default=False,
    help="SCML2020: Whether exogenous sales are equal for all agents executing the last prodess",
)
@click.option(
    "--buy-missing/--inventory-breach",
    default=True,
    help="SCML2020: If true, missing products will be assumed to be bought from an exogenous source at a price"
    " higher than both the catalog price and the unit price in the contract.",
)
@click.option(
    "--borrow/--fine",
    default=True,
    help="SCML2020: If --borrow, agents will borrow when they commit a breach and will pay the breach penalty as part of "
    "this borrowing process. If --fine, agents that commit a breach will be fined and the contract will not "
    "execute fully.",
)
@click.option(
    "--borrow-to-produce/--fail-to-produce",
    default=True,
    help="SCML2020: If a production command cannot be executed, should the agent borrow to do the production"
    " (i.e. buying missing inputs at higher than catalog price and borrowing to cover production "
    "costs) or not.",
)
@click.option(
    "--bankruptcy-limit",
    type=float,
    default=0.0,
    help="SCML2020: Fraction of the average initial balance that agents are allowed to borrow before going bankrupt",
)
@click.option(
    "--penalty",
    type=float,
    default=0.2,
    help="SCML2020: The penalty relative to the breach committed.",
)
@click.option(
    "--reports",
    type=int,
    default=5,
    help="SCML2020: The period for financial report publication",
)
@click.option(
    "--interest",
    type=float,
    default=0.05,
    help="SCML2020: Interest rate for negative balances (borrowed money)",
)
@click.option(
    "--force-exogenous/--sign-exogenous",
    default=False,
    help="SCML2020: Whether the exogenous contracts are forced to their full quantity"
    " or agents can choose to sign or not sign them.",
)
@click.option(
    "--balance",
    default=-1,
    type=int,
    help="SCML2020: Initial balance of all factories. A negative number will make the balance"
    " automatically calculated by the system. It will go up with process level",
)
@click.option(
    "--horizon",
    default=0.1,
    type=float,
    help="SCML2020: The exogenous contract revelation horizon as a fraction of the number of steps.",
)
@click.option(
    "--processes",
    default=3,
    type=int,
    help="SCML2020: Number of processes. Should never be less than 2",
)
@click_config_file.configuration_option()
@click.pass_context
def create(
    ctx,
    name,
    steps,
    ttype,
    timeout,
    log,
    verbosity,
    reveal_names,
    runs,
    configs,
    max_runs,
    competitors,
    world_config,
    jcompetitors,
    non_competitors,
    compact,
    factories,
    agents,
    log_ufuns,
    log_negs,
    raise_exceptions,
    steps_min,
    steps_max,
    path,
    cw,
    inputs,
    inputs_min,
    inputs_max,
    outputs,
    outputs_min,
    outputs_max,
    costs,
    costs_min,
    costs_max,
    increasing_costs,
    equal_exogenous_supplies,
    equal_exogenous_sales,
    profit_mean,
    profit_std,
    productivity,
    productivity_min,
    productivity_max,
    cash_availability,
    cash_availability_min,
    cash_availability_max,
    buy_missing,
    borrow,
    bankruptcy_limit,
    penalty,
    reports,
    interest,
    force_exogenous,
    borrow_to_produce,
    balance,
    processes,
    factories_min,
    factories_max,
    horizon,
):
    if balance < 0:
        balance = None
    productivity = get_range(productivity, productivity_min, productivity_max)
    cash_availability = get_range(
        cash_availability, cash_availability_min, cash_availability_max
    )
    inputs = get_range(inputs, inputs_min, inputs_max)
    if "2020" in ttype:
        factories = (min(factories, factories_min), max(factories, factories_max))
    outputs = get_range(outputs, outputs_min, outputs_max)
    costs = get_range(costs, costs_min, costs_max)
    kwargs = {}
    if world_config is not None and len(world_config) > 0:
        for wc in world_config:
            kwargs.update(load(wc))
    if "2020" in ttype:
        kwargs.update(
            {
                "n_processes": processes,
                "process_inputs": inputs,
                "process_outputs": outputs,
                "production_costs": costs,
                "cost_increases_with_level": increasing_costs,
                "equal_exogenous_supply": equal_exogenous_supplies,
                "equal_exogenous_sales": equal_exogenous_sales,
                "max_productivity": productivity,
                "cash_availability": cash_availability,
                "profit_means": profit_mean,
                "profit_stddevs": profit_std,
                "buy_missing_products": buy_missing,
                "borrow_on_breach": borrow,
                "bankruptcy_limit": bankruptcy_limit,
                "spot_market_global_loss": penalty,
                "production_penalty": penalty,
                "financial_report_period": reports,
                "interest_rate": interest,
                "exogenous_force_max": force_exogenous,
                "production_no_borrow": not borrow_to_produce,
                "production_no_bankruptcy": not borrow_to_produce,
                "initial_balance": balance,
            }
        )

    log = _path(log)
    if competitors == "default":
        if "2020" in ttype:
            competitors = "RandomAgent;BuyCheapSellExpensiveAgent;DoNothingAgent;DecentralizingAgent"
        else:
            competitors = "DefaultFactoryManager;DoNothingFactoryManager"
    if jcompetitors is not None and len(jcompetitors) > 0 and "2020" in ttype:
        raise ValueError("Java competitors are not supported in SCML2020")
    if "2020" in ttype and "sabotage" in ttype.lower():
        raise ValueError("There is no sabotage track in SCML2020")

    if len(path) > 0:
        sys.path.append(path)
    warning_n_runs = 2000
    if timeout <= 0:
        timeout = None
    if name == "random":
        name = unique_name(base="", rand_digits=0)
    ctx.obj["tournament_name"] = name
    if max_runs <= 0:
        max_runs = None
    if compact:
        log_ufuns = False

    if not compact:
        if not reveal_names:
            print(
                "You are running the tournament with --debug. Will reveal agent types in their names"
            )
        reveal_names = True
        verbosity = max(1, verbosity)

    worlds_per_config = (
        None if max_runs is None else int(round(max_runs / (configs * runs)))
    )

    all_competitors = competitors.split(";")
    for i, cp in enumerate(all_competitors):
        if "." not in cp:
            all_competitors[i] = (
                "scml.scml2019.factory_managers."
                if "2019" in ttype
                else "scml.scml2020.agents."
            ) + cp
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

    # if ttype.lower() == "anac2019std":
    #     if (
    #         "scml.scml2019.factory_managers.GreedyFactoryManager"
    #         not in all_competitors
    #     ):
    #         all_competitors.append(
    #             "scml.scml2019.factory_managers.GreedyFactoryManager"
    #         )
    #         all_competitors_params.append({})

    permutation_size = len(all_competitors) if "sabotage" not in ttype else 1
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

    if ttype == "anac2019std":
        agents = 1

    if steps is None:
        steps = (steps_min, steps_max)

    if worlds_per_config is None:
        n_comp = len(all_competitors) if ttype != "anac2019sabotage" else 2
        n_worlds = permutation_size * runs * configs
        if n_worlds > warning_n_runs:
            print(
                f"You are running the maximum possible number of permutations for each configuration. This is roughly"
                f" {n_worlds} simulations (each for {steps} steps). That will take a VERY long time."
                f"\n\nYou can reduce the number of simulations by setting --configs>=1 (currently {configs}) or "
                f"--runs>= 1 (currently {runs}) to a lower value. "
                f"\nFinally, you can limit the maximum number of worlds to run by setting --max-runs=integer."
            )
            # if (
            #     not input(f"Are you sure you want to run {n_worlds} simulations?")
            #     .lower()
            #     .startswith("y")
            # ):
            #     exit(0)
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
                    if "2019" in ttype
                    else "scml.scml2020.agents."
                ) + cp

    if ttype.lower().startswith("scml"):
        ttype = ttype.lower().replace("scml", "anac")

    if ttype.lower() == "anac2019std":
        if non_competitors is None:
            non_competitors = (DefaultGreedyManager,)
            non_competitor_params = ({},)
        print(f"Tournament will be run between {len(all_competitors)} agents: ")
        pprint(all_competitors)
        print("Non-competitors are: ")
        pprint(non_competitors)
        results = create_tournament(
            competitors=all_competitors,
            competitor_params=all_competitors_params,
            non_competitors=non_competitors,
            non_competitor_params=non_competitor_params,
            agent_names_reveal_type=reveal_names,
            n_competitors_per_world=cw,
            n_configs=configs,
            n_runs_per_world=runs,
            max_worlds_per_config=worlds_per_config,
            base_tournament_path=log,
            total_timeout=timeout,
            name=name,
            verbose=verbosity > 0,
            n_agents_per_competitor=1,
            world_generator=anac2019_world_generator,
            config_generator=anac2019_config_generator,
            config_assigner=anac2019_assigner,
            score_calculator=balance_calculator,
            min_factories_per_level=factories,
            compact=compact,
            n_steps=steps,
            log_ufuns=log_ufuns,
            log_negotiations=log_negs,
            ignore_agent_exceptions=not raise_exceptions,
            ignore_contract_execution_exceptions=not raise_exceptions,
            **kwargs,
        )
    elif ttype.lower() == "anac2020std":
        if non_competitors is None:
            non_competitors = scml.scml2020.utils.DefaultAgents
            non_competitor_params = tuple({} for _ in range(len(non_competitors)))
        print(f"Tournament will be run between {len(all_competitors)} agents: ")
        pprint(all_competitors)
        print("Non-competitors are: ")
        pprint(non_competitors)
        results = create_tournament(
            competitors=all_competitors,
            competitor_params=all_competitors_params,
            non_competitors=non_competitors,
            non_competitor_params=non_competitor_params,
            n_competitors_per_world=cw,
            n_configs=configs,
            n_runs_per_world=runs,
            max_worlds_per_config=worlds_per_config,
            base_tournament_path=log,
            total_timeout=timeout,
            name=name,
            verbose=verbosity > 0,
            n_agents_per_competitor=1,
            world_generator=anac2020_world_generator,
            config_generator=anac2020_config_generator,
            config_assigner=anac2020_assigner,
            score_calculator=scml.scml2020.utils.balance_calculator2020,
            min_factories_per_level=factories[0],
            max_factories_per_level=factories[1],
            compact=compact,
            n_steps=steps,
            log_ufuns=log_ufuns,
            log_negotiations=log_negs,
            ignore_agent_exceptions=not raise_exceptions,
            ignore_contract_execution_exceptions=not raise_exceptions,
            horizon=horizon,
            **kwargs,
        )
    elif ttype.lower() in ("anac2019collusion", "anac2019"):
        print(f"Tournament will be run between {len(all_competitors)} agents: ")
        pprint(all_competitors)
        print("Non-competitors are: ")
        pprint(non_competitors)
        results = create_tournament(
            competitors=all_competitors,
            competitor_params=all_competitors_params,
            non_competitors=non_competitors,
            non_competitor_params=non_competitor_params,
            agent_names_reveal_type=reveal_names,
            n_competitors_per_world=cw,
            n_configs=configs,
            n_runs_per_world=runs,
            max_worlds_per_config=worlds_per_config,
            base_tournament_path=log,
            total_timeout=timeout,
            name=name,
            verbose=verbosity > 0,
            n_agents_per_competitor=agents,
            world_generator=anac2019_world_generator,
            config_generator=anac2019_config_generator,
            config_assigner=anac2019_assigner,
            score_calculator=balance_calculator,
            min_factories_per_level=factories,
            compact=compact,
            n_steps=steps,
            log_ufuns=log_ufuns,
            log_negotiations=log_negs,
            ignore_agent_exceptions=not raise_exceptions,
            ignore_contract_execution_exceptions=not raise_exceptions,
            **kwargs,
        )
    elif ttype.lower() in ("anac2020collusion", "anac2020"):
        if non_competitors is None:
            non_competitors = scml.scml2020.utils.DefaultAgents
            non_competitor_params = tuple({} for _ in range(len(non_competitors)))
        print(f"Tournament will be run between {len(all_competitors)} agents: ")
        pprint(all_competitors)
        print("Non-competitors are: ")
        pprint(non_competitors)
        results = create_tournament(
            competitors=all_competitors,
            competitor_params=all_competitors_params,
            non_competitors=non_competitors,
            non_competitor_params=non_competitor_params,
            n_competitors_per_world=cw,
            n_configs=configs,
            n_runs_per_world=runs,
            max_worlds_per_config=worlds_per_config,
            base_tournament_path=log,
            total_timeout=timeout,
            name=name,
            verbose=verbosity > 0,
            n_agents_per_competitor=agents,
            world_generator=anac2019_world_generator,
            config_generator=anac2019_config_generator,
            config_assigner=anac2019_assigner,
            score_calculator=scml.scml2020.utils.balance_calculator2020,
            min_factories_per_level=factories[0],
            max_factories_per_level=factories[1],
            compact=compact,
            n_steps=steps,
            log_ufuns=log_ufuns,
            log_negotiations=log_negs,
            ignore_agent_exceptions=not raise_exceptions,
            ignore_contract_execution_exceptions=not raise_exceptions,
            horizon=horizon,
            **kwargs,
        )
    elif ttype.lower() == "anac2019sabotage":
        print(f"Tournament will be run between {len(all_competitors)} agents: ")
        pprint(all_competitors)
        print("Non-competitors are: ")
        pprint(non_competitors)
        results = create_tournament(
            competitors=all_competitors,
            competitor_params=all_competitors_params,
            agent_names_reveal_type=reveal_names,
            n_agents_per_competitor=agents,
            base_tournament_path=log,
            total_timeout=timeout,
            name=name,
            verbose=verbosity > 0,
            n_runs_per_world=runs,
            n_configs=configs,
            max_worlds_per_config=worlds_per_config,
            non_competitors=non_competitors,
            min_factories_per_level=factories,
            n_steps=steps,
            compact=compact,
            log_ufuns=log_ufuns,
            log_negotiations=log_negs,
            ignore_agent_exceptions=not raise_exceptions,
            ignore_contract_execution_exceptions=not raise_exceptions,
            non_competitor_params=non_competitor_params,
            world_generator=anac2019_world_generator,
            config_generator=anac2019_sabotage_config_generator,
            config_assigner=anac2019_sabotage_assigner,
            score_calculator=sabotage_effectiveness,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown tournament type {ttype}")
    ctx.obj["tournament_name"] = results.name
    ctx.obj["tournament_log_folder"] = log
    ctx.obj["compact"] = compact
    print(f"Saved all configs to {str(results)}\nTournament name is {results.name}")


@tournament.command(help="Runs/continues a tournament")
@click.option(
    "--name",
    "-n",
    default="",
    help="The name of the tournament. When invoked after create, there is no need to pass it",
)
@click.option(
    "--log",
    "-l",
    type=click.Path(dir_okay=True, file_okay=False),
    default=default_tournament_path(),
    help="Default location to save logs",
)
@click.option(
    "--verbosity",
    default=1,
    type=int,
    help="verbosity level (from 0 == silent to 1 == world progress)",
)
@click.option(
    "--parallel/--serial",
    default=True,
    help="Run a parallel/serial tournament on a single machine",
)
@click.option(
    "--distributed/--single-machine",
    default=False,
    help="Run a distributed tournament using dask",
)
@click.option(
    "--ip",
    default="127.0.0.1",
    help="The IP address for a dask scheduler to run the distributed tournament."
    " Effective only if --distributed",
)
@click.option(
    "--port",
    default=8786,
    type=int,
    help="The IP port number a dask scheduler to run the distributed tournament."
    " Effective only if --distributed",
)
@click.option(
    "--compact/--debug",
    default=True,
    help="If True, effort is exerted to reduce the memory footprint which"
    "includes reducing logs dramatically.",
)
@click.option(
    "--path",
    default="",
    help="A path to be added to PYTHONPATH in which all competitors are stored. You can path a : separated list of "
    "paths on linux/mac and a ; separated list in windows",
)
@click.option(
    "--metric",
    default="median",
    type=str,
    help="The statistical metric used for choosing the winners. Possibilities are mean, median, std, var, sum",
)
@click_config_file.configuration_option()
@click.pass_context
def run(
    ctx, name, verbosity, parallel, distributed, ip, port, compact, path, log, metric
):
    if len(name) == 0:
        name = ctx.obj.get("tournament_name", "")
    if len(name) == 0:
        print(
            "Name is not given to run command and was not stored during a create command call"
        )
        exit(1)
    if len(path) > 0:
        sys.path.append(path)

    saved_log_folder = ctx.obj.get("tournament_log_folder", None)
    if saved_log_folder is not None:
        log = saved_log_folder
    parallelism = "distributed" if distributed else "parallel" if parallel else "serial"
    prog_callback = print_world_progress if verbosity > 1 and not distributed else None
    tpath = str(pathlib.Path(log) / name)
    start = perf_counter()
    run_tournament(
        tournament_path=tpath,
        verbose=verbosity > 0,
        compact=compact,
        world_progress_callback=prog_callback,
        parallelism=parallelism,
        scheduler_ip=ip,
        scheduler_port=port,
        print_exceptions=verbosity > 1,
    )
    end_time = humanize_time(perf_counter() - start)
    results = evaluate_tournament(
        tournament_path=tpath, verbose=verbosity > 0, metric=metric, recursive=False
    )
    display_results(results, metric)
    print(f"Finished in {end_time}")


def display_results(results, metric):
    viewmetric = ["50%" if metric == "median" else metric]
    print(
        tabulate(
            results.score_stats.sort_values(by=viewmetric, ascending=False),
            headers="keys",
            tablefmt="psql",
        )
    )
    if metric in ("mean", "sum"):
        print(tabulate(results.ttest, headers="keys", tablefmt="psql"))
    else:
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


@cli.command(help="Run an SCML2019 world simulation")
@click.option("--steps", default=100, type=int, help="Number of steps.")
@click.option(
    "--levels",
    default=3,
    type=int,
    help="Number of intermediate production levels (processes). "
    "-1 means a single product and no factories.",
)
@click.option("--neg-speedup", default=21, help="Negotiation Speedup.")
@click.option(
    "--negotiator",
    default=DEFAULT_NEGOTIATOR,
    help="Negotiator type to use for builtin agents.",
)
@click.option(
    "--min-consumption",
    default=3,
    type=int,
    help="The minimum number of units consumed by each consumer at every " "time-step.",
)
@click.option(
    "--max-consumption",
    default=5,
    type=int,
    help="The maximum number of units consumed by each consumer at every " "time-step.",
)
@click.option(
    "--agents",
    default=5,
    type=int,
    help="Number of agents (miners/negmas.consumers) per production level",
)
@click.option("--horizon", default=15, type=int, help="Consumption horizon.")
@click.option("--transport", default=0, type=int, help="Transportation Delay.")
@click.option("--time", default=7200, type=int, help="Total time limit.")
@click.option(
    "--neg-time", default=120, type=int, help="Time limit per single negotiation"
)
@click.option(
    "--neg-steps", default=20, type=int, help="Number of rounds per single negotiation"
)
@click.option(
    "--sign",
    default=1,
    type=int,
    help="The default delay between contract conclusion and signing",
)
@click.option(
    "--guaranteed",
    default=False,
    help="Whether to only sign contracts that are guaranteed not to cause " "breaches",
)
@click.option("--lines", default=10, help="The number of lines per factory")
@click.option(
    "--retrials",
    default=2,
    type=int,
    help="The number of times an agent re-tries on failed negotiations",
)
@click.option(
    "--use-consumer/--no-consumer",
    default=True,
    help="Use internal consumer object in factory managers",
)
@click.option(
    "--max-insurance",
    default="inf",
    type=float,
    help="Use insurance against partner in factory managers up to this premium. Pass zero for never buying insurance"
    " and a 'inf' (without quotes) for infinity.",
)
@click.option(
    "--riskiness", default=0.0, help="How risky is the default factory manager"
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
    "--shared-profile/--multi-profile",
    default=True,
    help="If True, all lines in the same factory will have the same cost.",
)
@click.option(
    "--reserved-value",
    default="-inf",
    type=float,
    help="The reserved value used by GreedyFactoryManager",
)
@click.option(
    "--raise-exceptions/--ignore-exceptions",
    default=True,
    help="Whether to ignore agent exceptions",
)
@click.option(
    "--balance",
    default=1000,
    type=float,
    help="Initial balance of all factories (see --increasing-balance)",
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
    default=tuple(),
    multiple=True,
    help="A file to load extra configuration parameters for world simulations from.",
)
@click_config_file.configuration_option()
def run2019(
    steps,
    levels,
    neg_speedup,
    negotiator,
    agents,
    horizon,
    min_consumption,
    max_consumption,
    transport,
    time,
    neg_time,
    neg_steps,
    sign,
    guaranteed,
    lines,
    retrials,
    use_consumer,
    max_insurance,
    riskiness,
    competitors,
    jcompetitors,
    log,
    compact,
    log_ufuns,
    log_negs,
    reserved_value,
    balance,
    shared_profile,
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
    if max_insurance < 0:
        warnings.warn(
            f"Negative max insurance ({max_insurance}) is deprecated. Set --max-insurance=inf for always "
            f"buying and --max-insurance=0.0 for never buying. Will continue assuming --max-insurance=inf"
        )
        max_insurance = float("inf")

    if "." not in negotiator:
        negotiator = "negmas.sao." + negotiator

    params = {
        "steps": steps,
        "levels": levels,
        "neg_speedup": neg_speedup,
        "negotiator": negotiator,
        "agents": agents,
        "horizon": horizon,
        "min_consumption": min_consumption,
        "max_consumption": max_consumption,
        "transport": transport,
        "time": time,
        "neg_time": neg_time,
        "neg_steps": neg_steps,
        "sign": sign,
        "guaranteed": guaranteed,
        "lines": lines,
        "retrials": retrials,
        "use_consumer": use_consumer,
        "max_insurance": max_insurance,
        "riskiness": riskiness,
    }
    if compact:
        log_ufuns = False
        log_negs = False
    neg_speedup = neg_speedup if neg_speedup is not None and neg_speedup > 0 else None
    if min_consumption == max_consumption:
        consumption = min_consumption
    else:
        consumption = (min_consumption, max_consumption)
    customer_kwargs = {"negotiator_type": negotiator, "consumption_horizon": horizon}
    miner_kwargs = {"negotiator_type": negotiator, "n_retrials": retrials}
    factory_kwargs = {
        "negotiator_type": negotiator,
        "n_retrials": retrials,
        "sign_only_guaranteed_contracts": guaranteed,
        "use_consumer": use_consumer,
        "riskiness": riskiness,
        "max_insurance_premium": max_insurance,
        "reserved_value": reserved_value,
    }
    log_dir = _path(log)
    world_name = unique_name(base="scml", add_time=True, rand_digits=0)
    log_dir = log_dir / world_name
    log_dir = log_dir.absolute()
    os.makedirs(log_dir, exist_ok=True)

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
        negotiation_speed=neg_speedup,
        n_intermediate_levels=levels,
        n_miners=agents,
        n_consumers=agents,
        n_factories_per_level=agents,
        consumption=consumption,
        consumer_kwargs=customer_kwargs,
        miner_kwargs=miner_kwargs,
        default_manager_params=factory_kwargs,
        transportation_delay=transport,
        time_limit=time,
        neg_time_limit=neg_time,
        neg_n_steps=neg_steps,
        default_signing_delay=sign,
        n_lines_per_factory=lines,
        compact=compact,
        agent_names_reveal_type=True,
        log_ufuns=log_ufuns,
        manager_types=all_competitors,
        manager_params=all_competitors_params,
        log_negotiations=log_negs,
        log_folder=log_dir,
        name=world_name,
        shared_profile_per_factory=shared_profile,
        initial_wallet_balances=balance,
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
        f"steps: {steps}, horizon: {horizon}, time: {time}, levels: {levels}, agents_per_level: "
        f"{agents}, lines: {lines}, guaranteed: {guaranteed}, negotiator: {negotiator}\n"
        f"consumption: {consumption}"
        f", transport_to: {transport}, sign: {sign}, speedup: {neg_speedup}, neg_steps: {neg_steps}"
        f", retrials: {retrials}"
        f", neg_time: {neg_time}\n"
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


@cli.command(help="Run an SCML2020 world simulation")
@click.option(
    "--force-signing/--confirm-signing", default=False, help="Whether to force signing"
)
@click.option(
    "--batch-signing/--individual-signing",
    default=True,
    help="Whether to sign contracts in batch by a call to sign_all_contracts or individually"
    " by a call to sign_contract (only effectivec if --confirm-signing)",
)
@click.option("--steps", default=10, type=int, help="Number of steps.")
@click.option(
    "--processes",
    default=3,
    type=int,
    help="Number of processes. Should never be less than 2",
)
@click.option("--neg-speedup", default=21, help="Negotiation Speedup.")
@click.option(
    "--factories", default=None, type=int, help="Number of agents per production level"
)
@click.option(
    "--factories-min",
    default=2,
    type=int,
    help="Minimum number of agents per production level",
)
@click.option(
    "--factories-max",
    default=10,
    type=int,
    help="Maximum number of agents per production level",
)
@click.option("--horizon", default=15, type=int, help="Exogenous contracts horizon.")
@click.option("--time", default=7200, type=int, help="Total time limit.")
@click.option(
    "--neg-time", default=120, type=int, help="Time limit per single negotiation"
)
@click.option(
    "--neg-steps", default=20, type=int, help="Number of rounds per single negotiation"
)
@click.option(
    "--lines",
    type=int,
    default=10,
    help="The number of lines per factory. Overrides " "--lines-min, --lines-max",
)
@click.option(
    "--inputs", type=int, default=1, help="The number inputs to each production process"
)
@click.option(
    "--inputs-min",
    type=int,
    default=1,
    help="The minimum number inputs to each production process",
)
@click.option(
    "--inputs-max",
    type=int,
    default=1,
    help="The maximum number inputs to each production process",
)
@click.option(
    "--outputs",
    type=int,
    default=1,
    help="The number outputs to each production process",
)
@click.option(
    "--outputs-min",
    type=int,
    default=1,
    help="The minimum number outputs to each production process",
)
@click.option(
    "--outputs-max",
    type=int,
    default=1,
    help="The maximum number outputs to each production process",
)
@click.option(
    "--costs",
    type=int,
    default=None,
    help="The production cost (see --increasing-costs, --costs-min, --costs-max)",
)
@click.option("--costs-min", type=int, default=1, help="The minimum production cost")
@click.option("--costs-max", type=int, default=10, help="The maximum production cost")
@click.option(
    "--productivity",
    type=float,
    default=None,
    help="The fraction of production slots (lines/steps) that can be occupied with production given the "
    "exogenous contracts",
)
@click.option(
    "--productivity-min",
    type=float,
    default=0.8,
    help="The minimum fraction of production slots (lines/steps) that can be occupied with production given the "
    "exogenous contracts",
)
@click.option(
    "--productivity-max",
    type=float,
    default=1.0,
    help="The maximum fraction of production slots (lines/steps) that can be occupied with production given the "
    "exogenous contracts",
)
@click.option(
    "--cash-availability",
    type=float,
    default=None,
    help="The availability of cash which is a nubmer between zero and one specifying how much of the total"
    " production requirements of agents is available as initial balance. It is only effective"
    " when --balance < 0",
)
@click.option(
    "--cash-availability-min",
    type=float,
    default=0.8,
    help="The availability of cash which is a nubmer between zero and one specifying how much of the total"
    " production requirements of agents is available as initial balance. It is only effective"
    " when --balance < 0",
)
@click.option(
    "--cash-availability-max",
    type=float,
    default=1.0,
    help="The availability of cash which is a nubmer between zero and one specifying how much of the total"
    " production requirements of agents is available as initial balance. It is only effective"
    " when --balance < 0",
)
@click.option(
    "--profit-mean",
    type=float,
    default=0.15,
    help="The mean of the expected profit used to select catalog prices for the world",
)
@click.option(
    "--profit-std",
    type=float,
    default=0.05,
    help="The std. dev. of the expected profit used to select catalog prices for the world",
)
@click.option(
    "--increasing-costs/--fixed-costs",
    default=True,
    help="Whether to increase the costs with production level (i.e. processes near the end"
    " of the chain become more costly to run than those near the raw material)",
)
@click.option(
    "--equal-exogenous-supplies/--variable-exogenous-supplies",
    default=False,
    help="Whether exogenous supplies are equal for all agents executing process 0",
)
@click.option(
    "--equal-exogenous-sales/--variable-exogenous-sales",
    default=False,
    help="Whether exogenous sales are equal for all agents executing the last prodess",
)
@click.option(
    "--buy-missing/--inventory-breach",
    default=True,
    help="If true, missing products will be assumed to be bought from an exogenous source at a price"
    " higher than both the catalog price and the unit price in the contract.",
)
@click.option(
    "--borrow/--fine",
    default=True,
    help="If --borrow, agents will borrow when they commit a breach and will pay the breach penalty as part of "
    "this borrowing process. If --fine, agents that commit a breach will be fined and the contract will not "
    "execute fully.",
)
@click.option(
    "--borrow-to-produce/--fail-to-produce",
    default=True,
    help="If a production command cannot be executed, should the agent borrow to do the production"
    " (i.e. buying missing inputs at higher than catalog price and borrowing to cover production "
    "costs) or not.",
)
@click.option(
    "--bankruptcy-limit",
    type=float,
    default=0.0,
    help="Fraction of the average initial balance that agents are allowed to borrow before going bankrupt",
)
@click.option(
    "--penalty",
    type=float,
    default=0.2,
    help="The penalty relative to the breach committed.",
)
@click.option(
    "--reports", type=int, default=5, help="The period for financial report publication"
)
@click.option(
    "--interest",
    type=float,
    default=0.05,
    help="Interest rate for negative balances (borrowed money)",
)
@click.option(
    "--force-exogenous/--sign-exogenous",
    default=False,
    help="Whether the exogenous contracts are forced to their full quantity"
    " or agents can choose to sign or not sign some of them.",
)
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
    "--balance",
    default=-1,
    type=int,
    help="Initial balance of all factories. A negative number will make the balance"
    " automatically calculated by the system. It will go up with process level",
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
    default=tuple(),
    multiple=True,
    help="A file to load extra configuration parameters for world simulations from.",
)
@click_config_file.configuration_option()
def run2020(
    force_signing,
    batch_signing,
    steps,
    processes,
    neg_speedup,
    factories,
    factories_min,
    factories_max,
    horizon,
    time,
    neg_time,
    neg_steps,
    lines,
    competitors,
    log,
    compact,
    log_ufuns,
    log_negs,
    balance,
    raise_exceptions,
    path,
    world_config,
    inputs,
    inputs_min,
    inputs_max,
    outputs,
    outputs_min,
    outputs_max,
    costs,
    costs_min,
    costs_max,
    increasing_costs,
    equal_exogenous_supplies,
    equal_exogenous_sales,
    profit_mean,
    profit_std,
    productivity,
    productivity_min,
    productivity_max,
    cash_availability,
    cash_availability_min,
    cash_availability_max,
    buy_missing,
    borrow,
    bankruptcy_limit,
    penalty,
    reports,
    interest,
    force_exogenous,
    borrow_to_produce,
    show_contracts,
):
    if balance < 0:
        balance = None
    productivity = get_range(productivity, productivity_min, productivity_max)
    cash_availability = get_range(
        cash_availability, cash_availability_min, cash_availability_max
    )
    factories = get_range(factories, factories_min, factories_max)
    inputs = get_range(inputs, inputs_min, inputs_max)
    outputs = get_range(outputs, outputs_min, outputs_max)
    costs = get_range(costs, costs_min, costs_max)
    kwargs = {}
    if world_config is not None and len(world_config) > 0:
        for wc in world_config:
            kwargs.update(load(wc))
    if len(path) > 0:
        sys.path.append(path)

    kwargs.update(
        {
            "force_signing": force_signing,
            "batch_signing": batch_signing,
            "n_steps": steps,
            "n_processes": processes,
            "negotiation_speed": neg_speedup,
            "process_inputs": inputs,
            "process_outputs": outputs,
            "production_costs": costs,
            "cost_increases_with_level": increasing_costs,
            "equal_exogenous_supply": equal_exogenous_supplies,
            "equal_exogenous_sales": equal_exogenous_sales,
            "max_productivity": productivity,
            "cash_availability": cash_availability,
            "profit_means": profit_mean,
            "profit_stddevs": profit_std,
            "buy_missing_products": buy_missing,
            "borrow_on_breach": borrow,
            "bankruptcy_limit": bankruptcy_limit,
            "spot_market_global_loss": penalty,
            "production_penalty": penalty,
            "financial_report_period": reports,
            "interest_rate": interest,
            "exogenous_force_max": force_exogenous,
            "production_no_borrow": not borrow_to_produce,
            "production_no_bankruptcy": not borrow_to_produce,
            "n_agents_per_process": factories,
            "initial_balance": balance,
        }
    )

    params = kwargs

    if compact:
        log_ufuns = False
        log_negs = False
    neg_speedup = neg_speedup if neg_speedup is not None and neg_speedup > 0 else None

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
    world = scml.SCML2020World(
        **scml.SCML2020World.generate(
            time_limit=time,
            neg_time_limit=neg_time,
            neg_n_steps=neg_steps,
            n_lines=lines,
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
        f"steps: {steps}, horizon: {horizon}, time: {time}, processes: {processes}, agents_per_process: "
        f"{factories}, lines: {lines}, speedup: {neg_speedup}, neg_steps: {neg_steps}"
        f", neg_time: {neg_time}\n"
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


@cli.command(help="Prints SCML version and NegMAS version")
def version():
    print(f"SCML: {scml.__version__} (NegMAS: {negmas.__version__})")


if __name__ == "__main__":
    cli()
