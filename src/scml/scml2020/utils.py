import copy
from collections import defaultdict
from os import PathLike
from random import randint
from random import random
from random import shuffle
from typing import Optional
from typing import Tuple

import numpy as np
from negmas import Agent
from negmas.helpers import get_full_type_name
from negmas.serialization import serialize, deserialize
from negmas.helpers import unique_name
from negmas.tournaments import TournamentResults
from negmas.tournaments import WorldRunResults
from negmas.tournaments import tournament
from scipy.stats import tmean

from scml.oneshot.agents import RandomOneShotAgent
from scml.oneshot.agents import SyncRandomOneShotAgent
from scml.oneshot.world import SCML2020OneShotWorld
from scml.scml2020.agents import BuyCheapSellExpensiveAgent
from scml.scml2020.agents import DecentralizingAgent
from scml.scml2020.agents import MarketAwareDecentralizingAgent
from scml.scml2020.agents import MarketAwareIndDecentralizingAgent
from scml.scml2020.agents import MarketAwareMovingRangeAgent
from scml.scml2020.world import SCML2020World, SCML2021World
from scml.scml2020.world import is_system_agent

if True:
    from typing import Any
    from typing import Callable
    from typing import Dict
    from typing import Iterable
    from typing import List
    from typing import Optional
    from typing import Sequence
    from typing import Tuple
    from typing import Type
    from typing import Union

    from .world import SCML2020Agent

__all__ = [
    "anac2020_config_generator",
    "anac2020_assigner",
    "anac2020_world_generator",
    "anac2020_tournament",
    "anac2020_collusion",
    "anac2020_std",
    "balance_calculator2020",
    "balance_calculator2021",
    "balance_calculator2021oneshot",
    "DefaultAgents",
    "DefaultAgents2021",
    "DefaultAgentsOneShot",
]


FORCED_LOGS_FRACTION = 1.0


ROUND_ROBIN = True


DefaultAgents = [DecentralizingAgent, BuyCheapSellExpensiveAgent]


DefaultAgents2021 = [
    DecentralizingAgent,
    # MarketAwareDecentralizingAgent,
    # MarketAwareIndDecentralizingAgent,
    # RandomOneShotAgent,
]


DefaultAgentsOneShot = [
    RandomOneShotAgent,
    # SyncRandomOneShotAgent,
]


def integer_cut(
    n: int,
    l: int,
    l_m: Union[int, List[int]],
    l_max: Union[int, List[int]] = float("inf"),
) -> List[int]:
    """
    Generates l random integers that sum to n where each of them is at least l_m
    Args:
        n: total
        l: number of levels
        l_m: minimum per level
        l_max: maximum per level. Can be set to infinity

    Returns:

    """
    if not isinstance(l_m, Iterable):
        l_m = [l_m] * l
    if not isinstance(l_max, Iterable):
        l_max = [l_max] * l
    sizes = np.asarray(l_m)
    if n < sizes.sum():
        raise ValueError(
            f"Cannot generate {l} numbers summing to {n}  with a minimum summing to {sizes.sum()}"
        )
    maxs = np.asarray(l_max)
    if n > maxs.sum():
        raise ValueError(
            f"Cannot generate {l} numbers summing to {n}  with a maximum summing to {maxs.sum()}"
        )
    # TODO  That is most likely the most stupid way to do it. We just try blindly. There MUST be a better way
    while sizes.sum() < n:
        indx = randint(0, l - 1)
        if sizes[indx] >= l_max[indx]:
            continue
        sizes[indx] += 1
    return list(sizes.tolist())


def integer_cut_dynamic(
    n: int, l_min: int, l_max: int, min_levels: int = 0
) -> List[int]:
    """
    Generates a list random integers that sum to n where each of them is between l_m and l_max

    Args:
        n: total
        l_min: minimum per level
        l_max: maximum per level. Can be set to infinity
        min_levels: THe minimum number of levels to use

    Returns:

    """
    if n < min_levels * l_min:
        raise ValueError(
            f"Cannot cut {n} into at least {min_levels} numbers each is at least {l_min}"
        )

    sizes = [l_min] * min_levels

    for i in range(len(sizes)):
        sizes[i] += randint(0, l_max - l_min)

    while sum(sizes) < n:
        i = randint(l_min, l_max)
        sizes.append(i)

    to_remove = sum(sizes) - n
    if to_remove == 0:
        return sizes

    sizes = sorted(sizes, reverse=True)
    for i, x in enumerate(sizes):
        can_remove = x - l_min
        removed = min(can_remove, to_remove)
        sizes[i] -= removed
        to_remove -= removed
        if sum(sizes) == n:
            break
    sizes = [_ for _ in sizes if _ > 0]
    shuffle(sizes)
    assert sum(sizes) == n, f"n={n}\nsizes={sizes}"
    return sizes


def _realin(rng: Union[Tuple[float, float], float]) -> float:
    """
    Selects a random number within a range if given or the input if it was a float

    Args:
        rng: Range or single value

    Returns:

        the real within the given range
    """
    if isinstance(rng, float):
        return rng
    if abs(rng[1] - rng[0]) < 1e-8:
        return rng[0]
    return rng[0] + random() * (rng[1] - rng[0])


def _intin(rng: Union[Tuple[int, int], int]) -> int:
    """
    Selects a random number within a range if given or the input if it was an int

    Args:
        rng: Range or single value

    Returns:

        the int within the given range
    """
    if not isinstance(rng, Iterable):
        return int(rng)
    if rng[0] == rng[1]:
        return rng[0]
    return randint(rng[0], rng[1])


def anac2020_config_generator(
    n_competitors: int,
    n_agents_per_competitor: int,
    agent_names_reveal_type: bool = False,
    non_competitors: Optional[Tuple[Union[str, SCML2020Agent]]] = None,
    non_competitor_params: Optional[Tuple[Dict[str, Any]]] = None,
    compact: bool = False,
    *,
    n_steps: Union[int, Tuple[int, int]] = (50, 200),
    n_processes: Tuple[int, int] = (
        2,
        4,
    ),  # minimum is strictly guarantee but maximum is only guaranteed if select_n_levels_first
    min_factories_per_level: int = 2,  # strictly guaranteed
    max_factories_per_level: int = 6,  # not strictly guaranteed except if select_n_levels_first is False
    n_lines: int = 10,
    select_n_levels_first=True,
    oneshot_world: bool = False,
    **kwargs,
) -> List[Dict[str, Any]]:
    if non_competitors is None:
        non_competitors = DefaultAgents
        non_competitor_params = [dict() for _ in non_competitors]
    if isinstance(n_processes, Iterable):
        n_processes = tuple(n_processes)
    else:
        n_processes = [n_processes, n_processes]

    n_steps = _intin(n_steps)

    if select_n_levels_first:
        n_processes = randint(*n_processes)
        n_agents = n_agents_per_competitor * n_competitors
        n_default_managers = max(0, n_processes * min_factories_per_level)
        n_defaults = integer_cut(n_default_managers, n_processes, 0)
        n_a_list = integer_cut(n_agents, n_processes, 0)
        for i, n_a in enumerate(n_a_list):
            if n_a + n_defaults[i] < min_factories_per_level:
                n_defaults[i] = min_factories_per_level - n_a
            if n_a + n_defaults[i] > max_factories_per_level and n_defaults[i] > 1:
                n_defaults[i] = max(1, min_factories_per_level - n_a)
        n_f_list = [a + b for a, b in zip(n_defaults, n_a_list)]
    else:
        min_n_processes = randint(*n_processes)
        n_agents = n_agents_per_competitor * n_competitors
        n_default_managers = max(
            0, min_n_processes * min_factories_per_level - n_agents
        )
        n_f_list = integer_cut_dynamic(
            n_agents + n_default_managers,
            min_factories_per_level,
            max_factories_per_level,
            min_n_processes,
        )
        n_processes = len(n_f_list)
        n_defaults = [0] * n_processes
        while n_default_managers > 0:
            indx = randint(0, n_processes - 1)
            if n_f_list[indx] <= n_defaults[indx]:
                continue
            n_defaults[indx] += 1
            n_default_managers -= 1

    n_factories = sum(n_f_list)

    if non_competitor_params is None:
        non_competitor_params = [{}] * len(non_competitors)

    non_competitors = [get_full_type_name(_) for _ in non_competitors]

    max_def_agents = len(non_competitors) - 1
    agent_types = [None] * n_factories
    manager_params = [None] * n_factories
    first_in_level = 0
    for level in range(n_processes):
        n_d = n_defaults[level]
        n_f = n_f_list[level]
        assert (
            n_d <= n_f
        ), f"Got {n_f} total factories at level {level} out of which {n_d} are default!!"
        for j in range(n_f):
            if j >= n_f - n_d:  # default managers are last managers in the list
                def_indx = randint(0, max_def_agents)
                agent_types[first_in_level + j] = non_competitors[def_indx]
                params_ = copy.deepcopy(non_competitor_params[def_indx])
                if agent_names_reveal_type:
                    params_["name"] = f"_df_{level}_{j}"
                else:
                    params_[
                        "name"
                    ] = f"_df_{level}_{j}"  # because I use name to know that this is a default agent in evaluate.
                    # @todo do not use name to identify default agents in evaluation
                manager_params[first_in_level + j] = params_
        first_in_level += n_f

    world_name = unique_name("", add_time=True, rand_digits=4)
    agent_types = [
        get_full_type_name(_) if isinstance(_, SCML2020Agent) else _
        for _ in agent_types
    ]
    no_logs = compact
    if oneshot_world:
        world_params = dict(
            name=world_name,
            agent_types=agent_types,
            agent_params=manager_params,
            time_limit=7200 + 3600,
            neg_time_limit=120,
            neg_n_steps=20,
            neg_step_time_limit=10,
            negotiation_speed=21,
            start_negotiations_immediately=False,
            n_agents_per_process=n_f_list,
            n_processes=n_processes,
            n_steps=n_steps,
            n_lines=n_lines,
            compact=compact,
            no_logs=no_logs,
        )
    else:
        world_params = dict(
            name=world_name,
            agent_types=agent_types,
            agent_params=manager_params,
            time_limit=7200 + 3600,
            neg_time_limit=120,
            neg_n_steps=20,
            neg_step_time_limit=10,
            negotiation_speed=21,
            spot_market_global_loss=0.2,
            interest_rate=0.08,
            bankruptcy_limit=1.0,
            initial_balance=None,
            start_negotiations_immediately=False,
            n_agents_per_process=n_f_list,
            n_processes=n_processes,
            n_steps=n_steps,
            n_lines=n_lines,
            compact=compact,
            no_logs=no_logs,
        )
    world_params.update(kwargs)
    # _agent_types = copy.deepcopy(world_params.pop("agent_types"))
    # _agent_params = copy.deepcopy(world_params.pop("agent_params"))
    if oneshot_world:
        generated_world_params = SCML2020OneShotWorld.generate( **world_params)
    else:
        generated_world_params = SCML2021World.generate( **world_params)
    # world_params["agent_types"] = _agent_types
    # world_params["agent_params"] = _agent_params
    for k in ("agent_types", "agent_params"):
        if k in generated_world_params.keys():
            del generated_world_params[k]
    if oneshot_world:
        for _p in generated_world_params["profiles"]:
            _p.cost = int(_p.cost)
    else:
        for _p in generated_world_params["profiles"]:
            _p.costs = _p.costs.tolist()
    world_params["__exact_params"] = serialize(generated_world_params, deep=True, ignore_lambda=True)
    config = {
        "world_params": world_params,
        "compact": compact,
        "scoring_context": {},
        "non_competitors": non_competitors,
        "non_competitor_params": non_competitor_params,
        "agent_types": agent_types,
        "agent_params": manager_params,
    }
    config.update(kwargs)
    return [config]


def anac2020_assigner(
    config: List[Dict[str, Any]],
    max_n_worlds: int,
    n_agents_per_competitor: int = 1,
    fair: bool = True,
    competitors: Sequence[Type[Agent]] = (),
    params: Sequence[Dict[str, Any]] = (),
    dynamic_non_competitors: Optional[List[Type[Agent]]] = None,
    dynamic_non_competitor_params: Optional[List[Dict[str, Any]]] = None,
    exclude_competitors_from_reassignment: bool = True,
) -> List[List[Dict[str, Any]]]:
    config = config[0]
    competitors = list(
        get_full_type_name(_) if not isinstance(_, str) and _ is not None else _
        for _ in competitors
    )
    n_competitors = len(competitors)
    params = (
        list(params) if params is not None else [dict() for _ in range(n_competitors)]
    )

    n_permutations = n_competitors

    agent_types = config["agent_types"]
    is_default = [_ is not None for _ in agent_types]
    # assign non-competitor factories to extra-non-competitors
    if dynamic_non_competitors is not None:
        n_extra = len(dynamic_non_competitors)
        dynamic_non_competitors = list(
            get_full_type_name(_) if not isinstance(_, str) and _ is not None else _
            for _ in dynamic_non_competitors
        )
        if dynamic_non_competitor_params is None:
            dynamic_non_competitor_params = [dict() for _ in range(n_extra)]
        # removing the competitors from the dynamic competitors
        if exclude_competitors_from_reassignment:
            # TODO May be use a better way to hash the a parameters than just conversion to str
            # Note that None and and empty dict() will both become ""
            compset = set(zip(competitors, (str(_) if _ else "" for _ in params)))
            dynset = list(
                zip(
                    dynamic_non_competitors,
                    (str(_) if _ else "" for _ in dynamic_non_competitor_params),
                )
            )
            dynamic_non_competitor_indices = [
                i for i, _ in enumerate(dynset) if _ not in compset
            ]
            dynamic_non_competitors = [
                dynamic_non_competitors[i] for i in dynamic_non_competitor_indices
            ]
            dynamic_non_competitor_params = [
                dynamic_non_competitor_params[i] for i in dynamic_non_competitor_indices
            ]
            n_extra = len(dynamic_non_competitors)
        if n_extra:
            for i, isd in enumerate(is_default):
                if not isd:
                    continue
                extra_indx = randint(0, n_extra - 1)
                config["agent_types"][i] = dynamic_non_competitors[extra_indx]
                config["agent_params"][i] = dynamic_non_competitor_params[extra_indx]
    assignable_factories = [i for i, mtype in enumerate(agent_types) if mtype is None]
    shuffle(assignable_factories)
    assignable_factories = (
        np.asarray(assignable_factories)
        .reshape((n_competitors, n_agents_per_competitor))
        .tolist()
    )

    configs = []

    def _copy_config(perm_, c, indx):
        new_config = copy.deepcopy(c)
        # new_config["world_params"]["name"] += f".{indx:02d}"
        new_config["is_default"] = is_default
        for (a, p_), assignable in zip(perm_, assignable_factories):
            for factory in assignable:
                new_config["agent_types"][factory] = a
                new_config["agent_params"][factory] = copy.deepcopy(p_)
        return [new_config]

    if n_permutations is not None and max_n_worlds is None:
        permutation = list(zip(competitors, params))
        assert len(permutation) == len(assignable_factories)
        shuffle(permutation)
        perm = permutation
        for k in range(n_permutations):
            perm = copy.deepcopy(perm)
            perm = perm[-1:] + perm[:-1]
            configs.append(_copy_config(perm, config, k))
    elif max_n_worlds is None:
        raise ValueError(f"Did not give max_n_worlds and cannot find n_permutations.")
    else:
        permutation = list(zip(competitors, params))
        assert len(permutation) == len(assignable_factories)
        if fair:
            n_min = len(assignable_factories)
            n_rounds = int(max_n_worlds // n_min)
            if n_rounds < 1:
                raise ValueError(
                    f"Cannot guarantee fair assignment: n. competitors {len(assignable_factories)}, at least"
                    f" {n_min} runs are needed for fair assignment"
                )
            max_n_worlds = n_rounds * n_min
            k = 0
            for _ in range(n_rounds):
                shuffle(permutation)
                for __ in range(n_min):
                    k += 1
                    perm = copy.deepcopy(permutation)
                    perm = perm[-1:] + perm[:-1]
                    configs.append(_copy_config(perm, config, k))
        else:
            for k in range(max_n_worlds):
                perm = copy.deepcopy(permutation)
                shuffle(perm)
                configs.append(_copy_config(perm, config, k))

    return configs


def anac2020_world_generator(**kwargs):
    assert sum(kwargs["world_params"]["n_agents_per_process"]) == len(
        kwargs["world_params"]["agent_types"]
    )
    cnfg = kwargs["world_params"].pop("__exact_params")
    cnfg = deserialize(cnfg)
    cnfg2 = SCML2021World.generate(**kwargs["world_params"])
    for k in ("agent_types", "agent_params"):
        cnfg[k] = cnfg2[k]
    for _p in cnfg["profiles"]:
        _p.costs = np.asarray(_p.costs)
    if "info" not in cnfg.keys():
        cnfg["info"] = dict()
    cnfg["info"]["is_default"] = kwargs["is_default"]
    world = SCML2021World(**cnfg)
    return world


def anac2020oneshot_world_generator(**kwargs):
    assert sum(kwargs["world_params"]["n_agents_per_process"]) == len(
        kwargs["world_params"]["agent_types"]
    )
    # cnfg = SCML2020OneShotWorld.generate(**kwargs["world_params"])
    # for k in ("n_agents_per_process","n_processes"):
    #     del kwargs["world_params"][k]
    cnfg = kwargs["world_params"].pop("__exact_params")
    cnfg = deserialize(cnfg)
    cnfg2 = SCML2020OneShotWorld.generate(**kwargs["world_params"])
    for k in ("agent_types", "agent_params", "name"):
        cnfg[k] = cnfg2[k]
    if "info" not in cnfg.keys():
        cnfg["info"] = dict()
    cnfg["info"]["is_default"] = kwargs["is_default"]
    world = SCML2020OneShotWorld(**cnfg)
    return world


def balance_calculator2020(
    worlds: List[SCML2020World],
    scoring_context: Dict[str, Any],
    dry_run: bool,
    ignore_default=True,
    inventory_catalog_price_weight=0.0,
    inventory_trading_average_weight=0.5,
    consolidated=False,
) -> WorldRunResults:
    """A scoring function that scores factory managers' performance by the final balance only ignoring whatever still
    in their inventory.

    Args:

        worlds: The world which is assumed to be run up to the point at which the scores are to be calculated.
        scoring_context:  A dict of context parameters passed by the world generator or assigner.
        dry_run: A boolean specifying whether this is a dry_run. For dry runs, only names and types are expected in
                 the returned `WorldRunResults`
        ignore_default: Whether to ignore non-competitors (default agents)
        inventory_catalog_price_weight: The weight assigned to catalog price
        inventory_trading_average_weight: The weight assigned to trading price average
        consolidated: If true, the score of an agent type will be based on a consolidated statement of
                     all the factories it controlled

    Returns:
        WorldRunResults giving the names, scores, and types of factory managers.

    """
    if scoring_context is not None:
        inventory_catalog_price_weight = scoring_context.get(
            "inventory_catalog_price_weight", inventory_catalog_price_weight
        )
        inventory_trading_average_weight = scoring_context.get(
            "inventory_trading_average_weight", inventory_trading_average_weight
        )
        consolidated = scoring_context.get("consolidated", consolidated)
    assert len(worlds) == 1
    world = worlds[0]
    if world.inventory_valuation_trading is not None:
        inventory_trading_average_weight = world.inventory_valuation_trading
    if world.inventory_valuation_catalog is not None:
        inventory_catalog_price_weight = world.inventory_valuation_catalog
    result = WorldRunResults(
        world_names=[world.name], log_file_names=[world.log_file_name]
    )
    initial_balances = []
    is_default = world.info["is_default"]
    factories = [_ for _ in world.factories if not is_system_agent(_.agent_id)]
    agents = [world.agents[f.agent_id] for f in factories]
    agent_types = [
        _ for _ in world.agent_unique_types if not _.startswith("system_agent")
    ]
    if len(set(agent_types)) == len(set(world.agent_types)):
        agent_types = [_ for _ in world.agent_types if not _.startswith("system_agent")]
    for i, factory in enumerate(factories):
        if is_default[i] and ignore_default:
            continue
        initial_balances.append(factory.initial_balance)
    normalize = all(_ != 0 for _ in initial_balances)
    consolidated_scores = defaultdict(float)
    individual_scores = list()
    initial_sums = defaultdict(float)
    for default, factory, manager, agent_type in zip(
        is_default, factories, agents, agent_types
    ):
        if default and ignore_default:
            continue
        result.names.append(manager.name)
        result.ids.append(manager.id)
        result.types.append(agent_type)
        if dry_run:
            result.scores.append(None)
            continue
        final_balance = factory.current_balance
        if inventory_catalog_price_weight != 0.0:
            final_balance += np.sum(
                inventory_catalog_price_weight
                * factory.current_inventory
                * world.catalog_prices
            )
        if inventory_trading_average_weight != 0.0:
            final_balance += np.sum(
                inventory_trading_average_weight
                * factory.current_inventory
                * world.trading_prices
            )
        profit = final_balance - factory.initial_balance
        individual_scores.append(
            profit / factory.initial_balance if normalize else profit
        )
        consolidated_scores[agent_type] += profit
        initial_sums[agent_type] += factory.initial_balance
    if normalize:
        for k in consolidated_scores.keys():
            consolidated_scores[k] /= initial_sums[k]
    extra = []
    for k, v in consolidated_scores.items():
        extra.append(dict(type=k, score=v))
    result.extra_scores["combined_scores"] = extra
    result.extra_scores["consolidated_scores"] = extra

    if consolidated:
        for indx, type_ in enumerate(result.types):
            result.scores.append(consolidated_scores[type_])
    else:
        result.scores = individual_scores

    return result


def balance_calculator2021oneshot(
    worlds: List[SCML2020World],
    scoring_context: Dict[str, Any],
    dry_run: bool,
    ignore_default=True,
    consolidated=True,
    **kwargs,
) -> WorldRunResults:
    """A scoring function that scores factory managers' performance by the final balance only ignoring whatever still
    in their inventory.

    Args:

        worlds: The world which is assumed to be run up to the point at which the scores are to be calculated.
        scoring_context:  A dict of context parameters passed by the world generator or assigner.
        dry_run: A boolean specifying whether this is a dry_run. For dry runs, only names and types are expected in
                 the returned `WorldRunResults`
        ignore_default: Whether to ignore non-competitors (default agents)
        consolidated: If true, the score of an agent type will be based on a consolidated statement of
                     all the factories it controlled

    Returns:
        WorldRunResults giving the names, scores, and types of factory managers.

    """
    if scoring_context is not None:
        consolidated = scoring_context.get("consolidated", consolidated)
    assert len(worlds) == 1
    world = worlds[0]
    result = WorldRunResults(
        world_names=[world.name], log_file_names=[world.log_file_name]
    )
    is_default = world.info["is_default"]
    agents = list(world.agents.values())
    agent_types = [
        _ for _ in world.agent_unique_types if not _.startswith("system_agent")
    ]
    if len(set(agent_types)) == len(set(world.agent_types)):
        agent_types = [_ for _ in world.agent_types if not _.startswith("system_agent")]
    consolidated_scores = defaultdict(float)
    individual_scores = list()
    scores = world.scores()
    for default, manager, agent_type in zip(is_default, agents, agent_types):
        if default and ignore_default:
            continue
        result.names.append(manager.name)
        result.ids.append(manager.id)
        result.types.append(agent_type)
        if dry_run:
            result.scores.append(None)
            continue
        profit = scores[manager.id]
        individual_scores.append(profit)
        consolidated_scores[agent_type] += profit
    extra = []
    for k, v in consolidated_scores.items():
        extra.append(dict(type=k, score=v))
    result.extra_scores["combined_scores"] = extra
    result.extra_scores["consolidated_scores"] = extra

    if consolidated:
        for indx, type_ in enumerate(result.types):
            result.scores.append(consolidated_scores[type_])
    else:
        result.scores = individual_scores

    return result


def balance_calculator2021(
    worlds: List[SCML2020World],
    scoring_context: Dict[str, Any],
    dry_run: bool,
    ignore_default=True,
    inventory_catalog_price_weight=0.0,
    inventory_trading_average_weight=0.5,
) -> WorldRunResults:
    """A scoring function that scores factory managers' performance by the
    final balance only ignoring whatever still in their inventory after
    consolidating all factories in the simulation that belong to the same
    agent type.

    Args:

        worlds: The world which is assumed to be run up to the point at which the scores are to be calculated.
        scoring_context:  A dict of context parameters passed by the world generator or assigner.
        dry_run: A boolean specifying whether this is a dry_run. For dry runs, only names and types are expected in
                 the returned `WorldRunResults`
        ignore_default: Whether to ignore non-competitors (default agents)
        inventory_catalog_price_weight: The weight assigned to catalog price
        inventory_trading_average_weight: The weight assigned to trading price average

    Returns:
        WorldRunResults giving the names, scores, and types of factory managers.

    Remarks:

        - If multiple agents belonged to the same agent_type, the score of
          all of these agents will be set to the same value which is the
          consolidated profit of the group. This means that agent types that
          have more instantiations will tend to have higher scores at the end.
          When using this balance calculator, it is recommended to have the
          same number of instantiations of all agent types in each simulation
          to make sure that scores of different agent types are comparable in
          each and every simulation.

    """
    return balance_calculator2020(
        worlds,
        scoring_context,
        dry_run,
        ignore_default,
        inventory_catalog_price_weight,
        inventory_trading_average_weight,
        consolidated=True,
    )


def anac2020_tournament(
    competitors: Sequence[Union[str, Type[SCML2020Agent]]],
    agent_names_reveal_type=False,
    n_configs: int = 5,
    max_worlds_per_config: Optional[int] = None,
    n_runs_per_world: int = 2,
    n_agents_per_competitor: int = 3,
    min_factories_per_level: int = 2,
    tournament_path: str = None,
    total_timeout: Optional[int] = None,
    parallelism="parallel",
    scheduler_ip: Optional[str] = None,
    scheduler_port: Optional[str] = None,
    tournament_progress_callback: Callable[[Optional[WorldRunResults]], None] = None,
    world_progress_callback: Callable[[Optional[SCML2020World]], None] = None,
    name: str = None,
    verbose: bool = False,
    configs_only=False,
    compact=False,
    **kwargs,
) -> Union[TournamentResults, PathLike]:
    """
    The function used to run ANAC 2020 SCML tournament (collusion track).

    Args:

        name: Tournament name
        competitors: A list of class names for the competitors
        agent_names_reveal_type: If true then the type of an agent should be readable in its name (most likely at its
                                 beginning).
        n_configs: The number of different world configs (up to competitor assignment) to be generated.
        max_worlds_per_config: The maximum number of worlds to run per config. If None, then all possible assignments
                             of competitors within each config will be tried (by rotating agents over factories).
        n_runs_per_world: Number of runs per world. All of these world runs will have identical competitor assignment
                          and identical world configuration.
        n_agents_per_competitor: Number of agents per competitor
        min_factories_per_level: Minimum number of factories for each production level
        total_timeout: Total timeout for the complete process
        tournament_path: Path at which to store all results. A scores.csv file will keep the scores and logs folder will
                         keep detailed logs
        parallelism: Type of parallelism. Can be 'serial' for serial, 'parallel' for parallel and 'distributed' for distributed
        scheduler_port: Port of the dask scheduler if parallelism is dask, dist, or distributed
        scheduler_ip: IP Address of the dask scheduler if parallelism is dask, dist, or distributed
        world_progress_callback: A function to be called after everystep of every world run (only allowed for serial
                                 evaluation and should be used with cautious).
        tournament_progress_callback: A function to be called with `WorldRunResults` after each world finished
                                      processing
        verbose: Verbosity
        configs_only: If true, a config file for each
        compact: If true, effort will be made to reduce memory footprint including disableing most logs
        kwargs: Arguments to pass to the `world_generator` function

    Returns:

        `TournamentResults` The results of the tournament or a `PathLike` giving the location where configs were saved

    Remarks:

        Default parameters will be used in the league with the exception of `parallelism` which may use distributed
        processing

    """
    return anac2020_collusion(
        competitors=competitors,
        agent_names_reveal_type=agent_names_reveal_type,
        n_configs=n_configs,
        max_worlds_per_config=max_worlds_per_config,
        n_runs_per_world=n_runs_per_world,
        n_agents_per_competitor=n_agents_per_competitor,
        tournament_path=tournament_path,
        total_timeout=total_timeout,
        parallelism=parallelism,
        scheduler_ip=scheduler_ip,
        scheduler_port=scheduler_port,
        min_factories_per_level=min_factories_per_level,
        tournament_progress_callback=tournament_progress_callback,
        world_progress_callback=world_progress_callback,
        name=name,
        verbose=verbose,
        compact=compact,
        configs_only=configs_only,
        non_competitors=None,
        non_competitor_params=None,
        **kwargs,
    )


def anac2020_std(
    competitors: Sequence[Union[str, Type[SCML2020Agent]]],
    competitor_params: Optional[Sequence[Dict[str, Any]]] = None,
    agent_names_reveal_type=False,
    n_configs: int = 5,
    max_worlds_per_config: Optional[int] = None,
    n_runs_per_world: int = 1,
    min_factories_per_level: int = 2,
    tournament_path: str = None,
    total_timeout: Optional[int] = None,
    parallelism="parallel",
    scheduler_ip: Optional[str] = None,
    scheduler_port: Optional[str] = None,
    tournament_progress_callback: Callable[[Optional[WorldRunResults]], None] = None,
    world_progress_callback: Callable[[Optional[SCML2020World]], None] = None,
    non_competitors: Optional[Sequence[Union[str, Type[SCML2020Agent]]]] = None,
    non_competitor_params: Optional[Sequence[Union[str, Type[SCML2020Agent]]]] = None,
    dynamic_non_competitors: Optional[List[Type[Agent]]] = None,
    dynamic_non_competitor_params: Optional[List[Dict[str, Any]]] = None,
    exclude_competitors_from_reassignment: bool = True,
    name: str = None,
    verbose: bool = False,
    configs_only=False,
    compact=False,
    n_competitors_per_world=None,
    forced_logs_fraction: float = FORCED_LOGS_FRACTION,
    **kwargs,
) -> Union[TournamentResults, PathLike]:
    """
    The function used to run ANAC 2020 SCML tournament (standard track).

    Args:

        name: Tournament name
        competitors: A list of class names for the competitors
        competitor_params: A list of competitor parameters (used to initialize the competitors).
        agent_names_reveal_type: If true then the type of an agent should be readable in its name (most likely at its
                                 beginning).
        n_configs: The number of different world configs (up to competitor assignment) to be generated.
        max_worlds_per_config: The maximum number of worlds to run per config. If None, then all possible assignments
                             of competitors within each config will be tried (all permutations).
        n_runs_per_world: Number of runs per world. All of these world runs will have identical competitor assignment
                          and identical world configuration.
        min_factories_per_level: Minimum number of factories for each production level
        total_timeout: Total timeout for the complete process
        tournament_path: Path at which to store all results. A scores.csv file will keep the scores and logs folder will
                         keep detailed logs
        parallelism: Type of parallelism. Can be 'serial' for serial, 'parallel' for parallel and 'distributed' for
                     distributed
        scheduler_port: Port of the dask scheduler if parallelism is dask, dist, or distributed
        scheduler_ip: IP Address of the dask scheduler if parallelism is dask, dist, or distributed
        world_progress_callback: A function to be called after everystep of every world run (only allowed for serial
                                 evaluation and should be used with cautious).
        tournament_progress_callback: A function to be called with `WorldRunResults` after each world finished
                                      processing
        non_competitors: A list of agent types that will not be competing in the sabotage competition but will exist
                         in the world
        non_competitor_params: parameters of non competitor agents
        dynamic_non_competitors: A list of non-competing agents that are assigned to the simulation dynamically during
                                 the creation of the final assignment instead when the configuration is created
        dynamic_non_competitor_params: paramters of dynamic non competitor agents
        exclude_competitors_from_reassignment: If true, competitors are excluded from the dyanamic non-competitors
        verbose: Verbosity
        configs_only: If true, a config file for each
        compact: If true, compact logs will be created and effort will be made to reduce the memory footprint
        n_competitors_per_world: Number of competitors in every simulation. If not given it will be a random number
                                 between 2 and min(2, n), where n is the number of competitors
        forced_logs_fraction: Fraction of simulations for which logs are always saved (including negotiations)
        kwargs: Arguments to pass to the `world_generator` function

    Returns:

        `TournamentResults` The results of the tournament or a `PathLike` giving the location where configs were saved

    Remarks:

        Default parameters will be used in the league with the exception of `parallelism` which may use distributed
        processing

    """
    if n_competitors_per_world is None:
        n_competitors_per_world = kwargs.get(
            "n_competitors_per_world", randint(2, min(4, len(competitors)))
        )
    kwargs.pop("n_competitors_per_world", None)
    if non_competitors is None:
        non_competitors = DefaultAgents
        non_competitor_params = [dict() for _ in non_competitors]
    kwargs["round_robin"] = kwargs.get("round_robin", ROUND_ROBIN)
    return tournament(
        competitors=competitors,
        competitor_params=competitor_params,
        non_competitors=non_competitors,
        non_competitor_params=non_competitor_params,
        agent_names_reveal_type=agent_names_reveal_type,
        n_configs=n_configs,
        n_runs_per_world=n_runs_per_world,
        max_worlds_per_config=max_worlds_per_config,
        tournament_path=tournament_path,
        total_timeout=total_timeout,
        parallelism=parallelism,
        scheduler_ip=scheduler_ip,
        scheduler_port=scheduler_port,
        tournament_progress_callback=tournament_progress_callback,
        world_progress_callback=world_progress_callback,
        name=name,
        verbose=verbose,
        configs_only=configs_only,
        n_agents_per_competitor=1,
        world_generator=anac2020_world_generator,
        config_generator=anac2020_config_generator,
        config_assigner=anac2020_assigner,
        score_calculator=balance_calculator2020,
        min_factories_per_level=min_factories_per_level,
        compact=compact,
        metric="median",
        n_competitors_per_world=n_competitors_per_world,
        dynamic_non_competitors=dynamic_non_competitors,
        dynamic_non_competitor_params=dynamic_non_competitor_params,
        exclude_competitors_from_reassignment=exclude_competitors_from_reassignment,
        save_video_fraction=0.0,
        forced_logs_fraction=forced_logs_fraction,
        **kwargs,
    )


def anac2020_collusion(
    competitors: Sequence[Union[str, Type]],
    competitor_params: Optional[Sequence[Dict[str, Any]]] = None,
    agent_names_reveal_type=False,
    n_configs: int = 5,
    max_worlds_per_config: Optional[int] = None,
    n_runs_per_world: int = 1,
    n_agents_per_competitor: int = 3,
    min_factories_per_level: int = 2,
    tournament_path: str = None,
    total_timeout: Optional[int] = None,
    parallelism="parallel",
    scheduler_ip: Optional[str] = None,
    scheduler_port: Optional[str] = None,
    tournament_progress_callback: Callable[[Optional[WorldRunResults]], None] = None,
    world_progress_callback: Callable[[Optional[SCML2020World]], None] = None,
    non_competitors: Optional[Sequence[Union[str, Type[SCML2020Agent]]]] = None,
    non_competitor_params: Optional[Sequence[Union[str, Type[SCML2020Agent]]]] = None,
    dynamic_non_competitors: Optional[List[Type[Agent]]] = None,
    dynamic_non_competitor_params: Optional[List[Dict[str, Any]]] = None,
    exclude_competitors_from_reassignment: bool = True,
    name: str = None,
    verbose: bool = False,
    configs_only=False,
    compact=False,
    n_competitors_per_world=None,
    forced_logs_fraction: float = FORCED_LOGS_FRACTION,
    **kwargs,
) -> Union[TournamentResults, PathLike]:
    """
    The function used to run ANAC 2020 SCML tournament (collusion track).

    Args:

        name: Tournament name
        competitors: A list of class names for the competitors
        competitor_params: A list of competitor parameters (used to initialize the competitors).
        agent_names_reveal_type: If true then the type of an agent should be readable in its name (most likely at its
                                 beginning).
        n_configs: The number of different world configs (up to competitor assignment) to be generated.
        max_worlds_per_config: The maximum number of worlds to run per config. If None, then all possible assignments
                             of competitors within each config will be tried (all permutations).
        n_runs_per_world: Number of runs per world. All of these world runs will have identical competitor assignment
                          and identical world configuration.
        n_agents_per_competitor: Number of agents per competitor
        min_factories_per_level: Minimum number of factories for each production level
        total_timeout: Total timeout for the complete process
        tournament_path: Path at which to store all results. A scores.csv file will keep the scores and logs folder will
                         keep detailed logs
        parallelism: Type of parallelism. Can be 'serial' for serial, 'parallel' for parallel and 'distributed' for
                     distributed
        scheduler_port: Port of the dask scheduler if parallelism is dask, dist, or distributed
        scheduler_ip: IP Address of the dask scheduler if parallelism is dask, dist, or distributed
        world_progress_callback: A function to be called after everystep of every world run (only allowed for serial
                                 evaluation and should be used with cautious).
        tournament_progress_callback: A function to be called with `WorldRunResults` after each world finished
                                      processing
        non_competitors: A list of agent types that will not be competing in the sabotage competition but will exist
                         in the world
        non_competitor_params: parameters of non competitor agents
        dynamic_non_competitors: A list of non-competing agents that are assigned to the simulation dynamically during
                                 the creation of the final assignment instead when the configuration is created
        dynamic_non_competitor_params: paramters of dynamic non competitor agents
        exclude_competitors_from_reassignment: If true, competitors are excluded from the dyanamic non-competitors
        n_competitors_per_world: Number of competitors in every simulation. If not given it will be a random number
                                 between 2 and min(2, n), where n is the number of competitors
        verbose: Verbosity
        configs_only: If true, a config file for each
        compact: If true, compact logs will be created and effort will be made to reduce the memory footprint
        forced_logs_fraction: Fraction of simulations for which logs are always saved (including negotiations)
        kwargs: Arguments to pass to the `world_generator` function

    Returns:

        `TournamentResults` The results of the tournament or a `PathLike` giving the location where configs were saved

    Remarks:

        Default parameters will be used in the league with the exception of `parallelism` which may use distributed
        processing

    """
    if n_competitors_per_world is None:
        n_competitors_per_world = kwargs.get(
            "n_competitors_per_world", randint(2, min(4, len(competitors)))
        )
    kwargs.pop("n_competitors_per_world", None)
    if non_competitors is None:
        non_competitors = DefaultAgents
        non_competitor_params = [dict() for _ in non_competitors]
    kwargs["round_robin"] = kwargs.get("round_robin", ROUND_ROBIN)
    return tournament(
        competitors=competitors,
        competitor_params=competitor_params,
        non_competitors=non_competitors,
        non_competitor_params=non_competitor_params,
        agent_names_reveal_type=agent_names_reveal_type,
        n_configs=n_configs,
        n_runs_per_world=n_runs_per_world,
        max_worlds_per_config=max_worlds_per_config,
        tournament_path=tournament_path,
        total_timeout=total_timeout,
        n_agents_per_competitor=n_agents_per_competitor,
        parallelism=parallelism,
        scheduler_ip=scheduler_ip,
        scheduler_port=scheduler_port,
        tournament_progress_callback=tournament_progress_callback,
        world_progress_callback=world_progress_callback,
        name=name,
        verbose=verbose,
        configs_only=configs_only,
        world_generator=anac2020_world_generator,
        config_generator=anac2020_config_generator,
        config_assigner=anac2020_assigner,
        score_calculator=balance_calculator2020,
        min_factories_per_level=min_factories_per_level,
        compact=compact,
        metric="median",
        n_competitors_per_world=n_competitors_per_world,
        dynamic_non_competitors=dynamic_non_competitors,
        dynamic_non_competitor_params=dynamic_non_competitor_params,
        exclude_competitors_from_reassignment=exclude_competitors_from_reassignment,
        save_video_fraction=0.0,
        forced_logs_fraction=forced_logs_fraction,
        **kwargs,
    )


def truncated_mean(
    scores: np.ndarray,
    limits: Optional[Tuple[float, float]] = None,
    top_limit=2.0,
    bottom_limit=float("inf"),
    base="tukey",
    return_limits=False,
) -> Union[float, Tuple[float, Optional[Tuple[float, float]]]]:
    """
    Calculates the truncated mean

    Args:
        scores: A list of scores for which to calculate the truncated mean
        limits: The limits to use for trimming the scores. If not given, they will
                be calculated based on `top_fraction`, `bottom_fraction` and `base.`
                You can pass the special value "mean" as a string to disable limits and
                calcualte the mean. You can pass the special value "median" to calculate
                the median (which is the same as passing top_fraction==bottom_fraction=0.5
                and base == "scores").
        top_limit: top limit on scores to use for truncated mean calculation. See `base`
        bottom_limit: bottom limit on scores to use for truncated mean calculation. See `base`
        base: The base for calculating the limits used to apply the `top_limit` and `bottom_limit`.
              Possible values are:
              - zscore: the number of sigmas to remove above/below. A good default choice is 3. Pass inf to disable a side.
              - tukey: the fraction of IQR to remove above/below. A good default choice is 1.5 or 3 (we use 2). Pass inf to disable a side.
              - iqr : same as tukey
              - iqr_fraction: the fraction is interpreted as the fraction of scores above/below the 1st/3rd qauntile
              - scores: the fraction is interpreted as fraction of highest and lowest scores
              - fraction: the fraction is interpreted as literal fraction of the values (i.e. given 10 values and 0.1, removes 1 value)
              - mean: simply returns the mean (limits ignored)
              - median: simply returns the median (limits ignored)
        return_limits: If true, the method will also return the limiting scores used in its mean calculation.
    """

    scores = np.asarray(scores)
    scores = scores[~np.isnan(scores)]

    if isinstance(limits, str) and limits.lower() == "mean":
        return tmean(scores, None) if not return_limits else (tmean(scores, None), None)
    if isinstance(limits, str) and limits.lower() == "median":
        return np.median(scores) if not return_limits else (np.median(scores), None)
    if limits is not None:
        return np.mean(scores) if not return_limits else (np.mean(scores), None)

    if base == "zscore":
        m, s = np.nanmean(scores), np.nanstd(scores)
        limits = (m - s * bottom_limit, m + s * top_limit)
    elif base in ("tukey", "iqr"):
        q1, q3 = np.quantile(scores, 0.25), np.quantile(scores, 0.75)
        iqr = q3 - q1
        limits = (
            q1 - (bottom_limit * iqr if not np.isinf(bottom_limit) else bottom_limit),
            q3 + (top_limit * iqr if not np.isinf(top_limit) else top_limit),
        )
    elif base == "iqr_fraction":
        bottom_limit = min(1, max(0, bottom_limit))
        top_limit = min(1, max(0, top_limit))
        limits = (np.quantile(scores, 0.25), np.quantile(scores, 0.75))
        high = np.sort(scores[scores > limits[1]])
        low = np.sort(scores[scores < limits[0]])
        limits = (
            low[int((len(low) - 1) * bottom_limit)] if len(low) > 0 else None,
            high[int((len(high) - 1) * (1 - top_limit))] if len(high) > 0 else None,
        )
    elif base == "fraction":
        bottom_limit = min(1, max(0, bottom_limit))
        top_limit = min(1, max(0, top_limit))
        scores = np.sort(scores)
        top_indx = int((len(scores) - 1) * (1 - top_limit))
        bottom_indx = int((len(scores) - 1) * bottom_limit)
        if top_indx < bottom_indx:
            return float("nan") if not return_limits else (float("nan"), limits)
        m = np.mean(scores[bottom_indx : top_indx + 1])
        return m if not return_limits else (m, (scores[bottom_indx], scores[top_indx]))
    elif base == "scores":
        bottom_limit = min(1, max(0, bottom_limit))
        top_limit = min(1, max(0, top_limit))
        limits = (
            np.quantile(scores, bottom_limit),
            np.quantile(scores, 1 - top_limit),
        )
        if limits[0] > limits[1]:
            return float("nan") if not return_limits else (float("nan"), limits)
    elif base == "mean":
        return np.mean(scores) if not return_limits else (np.mean(scores), None)
    elif base == "median":
        return np.median(scores) if not return_limits else (np.median(scores), None)
    else:
        raise ValueError(f"Unknown base for truncated_mean ({base})")
    if len(scores) == 0 or limits[1] < limits[0]:
        return float("nan") if not return_limits else (float("nan"), limits)
    try:
        # this is an inclusive trimmed mean
        # tm = tmean(scores, limits)
        scores = scores[scores >= limits[0]]
        scores = scores[scores <= limits[1]]
        if len(scores) == 0:
            return float("nan") if not return_limits else (float("nan"), limits)
        tm = np.mean(scores)
        return tm if not return_limits else (tm, limits)
    except ValueError:
        return float("nan") if not return_limits else (float("nan"), limits)


def anac2021_tournament(
    competitors: Sequence[Union[str, Type[SCML2020Agent]]],
    agent_names_reveal_type=False,
    n_configs: int = 5,
    max_worlds_per_config: Optional[int] = None,
    n_runs_per_world: int = 2,
    n_agents_per_competitor: int = 3,
    min_factories_per_level: int = 2,
    tournament_path: str = None,
    total_timeout: Optional[int] = None,
    parallelism="parallel",
    scheduler_ip: Optional[str] = None,
    scheduler_port: Optional[str] = None,
    tournament_progress_callback: Callable[[Optional[WorldRunResults]], None] = None,
    world_progress_callback: Callable[[Optional[SCML2020World]], None] = None,
    name: str = None,
    verbose: bool = False,
    configs_only=False,
    compact=False,
    **kwargs,
) -> Union[TournamentResults, PathLike]:
    """
    The function used to run ANAC 2020 SCML tournament (collusion track).

    Args:

        name: Tournament name
        competitors: A list of class names for the competitors
        agent_names_reveal_type: If true then the type of an agent should be readable in its name (most likely at its
                                 beginning).
        n_configs: The number of different world configs (up to competitor assignment) to be generated.
        max_worlds_per_config: The maximum number of worlds to run per config. If None, then all possible assignments
                             of competitors within each config will be tried (by rotating agents over factories).
        n_runs_per_world: Number of runs per world. All of these world runs will have identical competitor assignment
                          and identical world configuration.
        n_agents_per_competitor: Number of agents per competitor
        min_factories_per_level: Minimum number of factories for each production level
        total_timeout: Total timeout for the complete process
        tournament_path: Path at which to store all results. A scores.csv file will keep the scores and logs folder will
                         keep detailed logs
        parallelism: Type of parallelism. Can be 'serial' for serial, 'parallel' for parallel and 'distributed' for distributed
        scheduler_port: Port of the dask scheduler if parallelism is dask, dist, or distributed
        scheduler_ip: IP Address of the dask scheduler if parallelism is dask, dist, or distributed
        world_progress_callback: A function to be called after everystep of every world run (only allowed for serial
                                 evaluation and should be used with cautious).
        tournament_progress_callback: A function to be called with `WorldRunResults` after each world finished
                                      processing
        verbose: Verbosity
        configs_only: If true, a config file for each
        compact: If true, effort will be made to reduce memory footprint including disableing most logs
        kwargs: Arguments to pass to the `world_generator` function

    Returns:

        `TournamentResults` The results of the tournament or a `PathLike` giving the location where configs were saved

    Remarks:

        Default parameters will be used in the league with the exception of `parallelism` which may use distributed
        processing

    """
    return anac2021_collusion(
        competitors=competitors,
        agent_names_reveal_type=agent_names_reveal_type,
        n_configs=n_configs,
        max_worlds_per_config=max_worlds_per_config,
        n_runs_per_world=n_runs_per_world,
        n_agents_per_competitor=n_agents_per_competitor,
        tournament_path=tournament_path,
        total_timeout=total_timeout,
        parallelism=parallelism,
        scheduler_ip=scheduler_ip,
        scheduler_port=scheduler_port,
        min_factories_per_level=min_factories_per_level,
        tournament_progress_callback=tournament_progress_callback,
        world_progress_callback=world_progress_callback,
        name=name,
        verbose=verbose,
        compact=compact,
        configs_only=configs_only,
        non_competitors=None,
        non_competitor_params=None,
        **kwargs,
    )


def anac2021_std(
    competitors: Sequence[Union[str, Type[SCML2020Agent]]],
    competitor_params: Optional[Sequence[Dict[str, Any]]] = None,
    agent_names_reveal_type=False,
    n_configs: int = 5,
    max_worlds_per_config: Optional[int] = None,
    n_runs_per_world: int = 1,
    min_factories_per_level: int = 2,
    tournament_path: str = None,
    total_timeout: Optional[int] = None,
    parallelism="parallel",
    scheduler_ip: Optional[str] = None,
    scheduler_port: Optional[str] = None,
    tournament_progress_callback: Callable[[Optional[WorldRunResults]], None] = None,
    world_progress_callback: Callable[[Optional[SCML2020World]], None] = None,
    non_competitors: Optional[Sequence[Union[str, Type[SCML2020Agent]]]] = None,
    non_competitor_params: Optional[Sequence[Union[str, Type[SCML2020Agent]]]] = None,
    dynamic_non_competitors: Optional[List[Type[Agent]]] = None,
    dynamic_non_competitor_params: Optional[List[Dict[str, Any]]] = None,
    exclude_competitors_from_reassignment: bool = True,
    name: str = None,
    verbose: bool = False,
    configs_only=False,
    compact=False,
    n_competitors_per_world=None,
    forced_logs_fraction: float = FORCED_LOGS_FRACTION,
    **kwargs,
) -> Union[TournamentResults, PathLike]:
    """
    The function used to run ANAC 2020 SCML tournament (standard track).

    Args:

        name: Tournament name
        competitors: A list of class names for the competitors
        competitor_params: A list of competitor parameters (used to initialize the competitors).
        agent_names_reveal_type: If true then the type of an agent should be readable in its name (most likely at its
                                 beginning).
        n_configs: The number of different world configs (up to competitor assignment) to be generated.
        max_worlds_per_config: The maximum number of worlds to run per config. If None, then all possible assignments
                             of competitors within each config will be tried (all permutations).
        n_runs_per_world: Number of runs per world. All of these world runs will have identical competitor assignment
                          and identical world configuration.
        min_factories_per_level: Minimum number of factories for each production level
        total_timeout: Total timeout for the complete process
        tournament_path: Path at which to store all results. A scores.csv file will keep the scores and logs folder will
                         keep detailed logs
        parallelism: Type of parallelism. Can be 'serial' for serial, 'parallel' for parallel and 'distributed' for
                     distributed
        scheduler_port: Port of the dask scheduler if parallelism is dask, dist, or distributed
        scheduler_ip: IP Address of the dask scheduler if parallelism is dask, dist, or distributed
        world_progress_callback: A function to be called after everystep of every world run (only allowed for serial
                                 evaluation and should be used with cautious).
        tournament_progress_callback: A function to be called with `WorldRunResults` after each world finished
                                      processing
        non_competitors: A list of agent types that will not be competing in the sabotage competition but will exist
                         in the world
        non_competitor_params: parameters of non competitor agents
        dynamic_non_competitors: A list of non-competing agents that are assigned to the simulation dynamically during
                                 the creation of the final assignment instead when the configuration is created
        dynamic_non_competitor_params: paramters of dynamic non competitor agents
        exclude_competitors_from_reassignment: If true, competitors are excluded from the dyanamic non-competitors
        verbose: Verbosity
        configs_only: If true, a config file for each
        compact: If true, compact logs will be created and effort will be made to reduce the memory footprint
        n_competitors_per_world: Number of competitors in every simulation. If not given it will be a random number
                                 between 2 and min(2, n), where n is the number of competitors
        forced_logs_fraction: Fraction of simulations for which logs are always saved (including negotiations)
        kwargs: Arguments to pass to the `world_generator` function

    Returns:

        `TournamentResults` The results of the tournament or a `PathLike` giving the location where configs were saved

    Remarks:

        Default parameters will be used in the league with the exception of `parallelism` which may use distributed
        processing

    """
    if n_competitors_per_world is None:
        n_competitors_per_world = kwargs.get(
            "n_competitors_per_world", randint(2, min(4, len(competitors)))
        )
    kwargs.pop("n_competitors_per_world", None)
    if non_competitors is None:
        non_competitors = DefaultAgents
        non_competitor_params = [dict() for _ in non_competitors]
    kwargs["round_robin"] = kwargs.get("round_robin", ROUND_ROBIN)
    return tournament(
        competitors=competitors,
        competitor_params=competitor_params,
        non_competitors=non_competitors,
        non_competitor_params=non_competitor_params,
        agent_names_reveal_type=agent_names_reveal_type,
        n_configs=n_configs,
        n_runs_per_world=n_runs_per_world,
        max_worlds_per_config=max_worlds_per_config,
        tournament_path=tournament_path,
        total_timeout=total_timeout,
        parallelism=parallelism,
        scheduler_ip=scheduler_ip,
        scheduler_port=scheduler_port,
        tournament_progress_callback=tournament_progress_callback,
        world_progress_callback=world_progress_callback,
        name=name,
        verbose=verbose,
        configs_only=configs_only,
        n_agents_per_competitor=1,
        world_generator=anac2020_world_generator,
        config_generator=anac2020_config_generator,
        config_assigner=anac2020_assigner,
        score_calculator=balance_calculator2021,
        min_factories_per_level=min_factories_per_level,
        compact=compact,
        metric=truncated_mean,
        n_competitors_per_world=n_competitors_per_world,
        dynamic_non_competitors=dynamic_non_competitors,
        dynamic_non_competitor_params=dynamic_non_competitor_params,
        exclude_competitors_from_reassignment=exclude_competitors_from_reassignment,
        save_video_fraction=0.0,
        forced_logs_fraction=forced_logs_fraction,
        publish_exogenous_summary=True,
        publish_trading_prices=True,
        **kwargs,
    )


def anac2021_collusion(
    competitors: Sequence[Union[str, Type]],
    competitor_params: Optional[Sequence[Dict[str, Any]]] = None,
    agent_names_reveal_type=False,
    n_configs: int = 5,
    max_worlds_per_config: Optional[int] = None,
    n_runs_per_world: int = 1,
    n_agents_per_competitor: int = 3,
    min_factories_per_level: int = 2,
    tournament_path: str = None,
    total_timeout: Optional[int] = None,
    parallelism="parallel",
    scheduler_ip: Optional[str] = None,
    scheduler_port: Optional[str] = None,
    tournament_progress_callback: Callable[[Optional[WorldRunResults]], None] = None,
    world_progress_callback: Callable[[Optional[SCML2020World]], None] = None,
    non_competitors: Optional[Sequence[Union[str, Type[SCML2020Agent]]]] = None,
    non_competitor_params: Optional[Sequence[Union[str, Type[SCML2020Agent]]]] = None,
    dynamic_non_competitors: Optional[List[Type[Agent]]] = None,
    dynamic_non_competitor_params: Optional[List[Dict[str, Any]]] = None,
    exclude_competitors_from_reassignment: bool = True,
    name: str = None,
    verbose: bool = False,
    configs_only=False,
    compact=False,
    n_competitors_per_world=1,
    forced_logs_fraction: float = FORCED_LOGS_FRACTION,
    **kwargs,
) -> Union[TournamentResults, PathLike]:
    """
    The function used to run ANAC 2020 SCML tournament (collusion track).

    Args:

        name: Tournament name
        competitors: A list of class names for the competitors
        competitor_params: A list of competitor parameters (used to initialize the competitors).
        agent_names_reveal_type: If true then the type of an agent should be readable in its name (most likely at its
                                 beginning).
        n_configs: The number of different world configs (up to competitor assignment) to be generated.
        max_worlds_per_config: The maximum number of worlds to run per config. If None, then all possible assignments
                             of competitors within each config will be tried (all permutations).
        n_runs_per_world: Number of runs per world. All of these world runs will have identical competitor assignment
                          and identical world configuration.
        n_agents_per_competitor: Number of agents per competitor
        min_factories_per_level: Minimum number of factories for each production level
        total_timeout: Total timeout for the complete process
        tournament_path: Path at which to store all results. A scores.csv file will keep the scores and logs folder will
                         keep detailed logs
        parallelism: Type of parallelism. Can be 'serial' for serial, 'parallel' for parallel and 'distributed' for
                     distributed
        scheduler_port: Port of the dask scheduler if parallelism is dask, dist, or distributed
        scheduler_ip: IP Address of the dask scheduler if parallelism is dask, dist, or distributed
        world_progress_callback: A function to be called after everystep of every world run (only allowed for serial
                                 evaluation and should be used with cautious).
        tournament_progress_callback: A function to be called with `WorldRunResults` after each world finished
                                      processing
        non_competitors: A list of agent types that will not be competing in the sabotage competition but will exist
                         in the world
        non_competitor_params: parameters of non competitor agents
        dynamic_non_competitors: A list of non-competing agents that are assigned to the simulation dynamically during
                                 the creation of the final assignment instead when the configuration is created
        dynamic_non_competitor_params: paramters of dynamic non competitor agents
        exclude_competitors_from_reassignment: If true, competitors are excluded from the dyanamic non-competitors
        n_competitors_per_world: Number of competitors in every simulation. If not given it will be a random number
                                 between 2 and min(2, n), where n is the number of competitors. This value will
                                 always be set to 1 in SCML2021
        verbose: Verbosity
        configs_only: If true, a config file for each
        compact: If true, compact logs will be created and effort will be made to reduce the memory footprint
        forced_logs_fraction: Fraction of simulations for which logs are always saved (including negotiations)
        kwargs: Arguments to pass to the `world_generator` function

    Returns:

        `TournamentResults` The results of the tournament or a `PathLike` giving the location where configs were saved

    Remarks:

        Default parameters will be used in the league with the exception of `parallelism` which may use distributed
        processing

    """
    n_competitors_per_world = 1
    kwargs.pop("n_competitors_per_world", None)
    if non_competitors is None:
        non_competitors = DefaultAgents
        non_competitor_params = [dict() for _ in non_competitors]
    kwargs["round_robin"] = kwargs.get("round_robin", ROUND_ROBIN)
    return tournament(
        competitors=competitors,
        competitor_params=competitor_params,
        non_competitors=non_competitors,
        non_competitor_params=non_competitor_params,
        agent_names_reveal_type=agent_names_reveal_type,
        n_configs=n_configs,
        n_runs_per_world=n_runs_per_world,
        max_worlds_per_config=max_worlds_per_config,
        tournament_path=tournament_path,
        total_timeout=total_timeout,
        n_agents_per_competitor=n_agents_per_competitor,
        parallelism=parallelism,
        scheduler_ip=scheduler_ip,
        scheduler_port=scheduler_port,
        tournament_progress_callback=tournament_progress_callback,
        world_progress_callback=world_progress_callback,
        name=name,
        verbose=verbose,
        configs_only=configs_only,
        world_generator=anac2020_world_generator,
        config_generator=anac2020_config_generator,
        config_assigner=anac2020_assigner,
        score_calculator=balance_calculator2021,
        min_factories_per_level=min_factories_per_level,
        compact=compact,
        metric=truncated_mean,
        n_competitors_per_world=n_competitors_per_world,
        dynamic_non_competitors=dynamic_non_competitors,
        dynamic_non_competitor_params=dynamic_non_competitor_params,
        exclude_competitors_from_reassignment=exclude_competitors_from_reassignment,
        save_video_fraction=0.0,
        forced_logs_fraction=forced_logs_fraction,
        publish_exogenous_summary=True,
        publish_trading_prices=True,
        **kwargs,
    )


def anac2021_oneshot(
    competitors: Sequence[Union[str, Type[SCML2020Agent]]],
    competitor_params: Optional[Sequence[Dict[str, Any]]] = None,
    agent_names_reveal_type=False,
    n_configs: int = 5,
    max_worlds_per_config: Optional[int] = None,
    n_runs_per_world: int = 1,
    min_factories_per_level: int = 4,
    tournament_path: str = None,
    total_timeout: Optional[int] = None,
    parallelism="parallel",
    scheduler_ip: Optional[str] = None,
    scheduler_port: Optional[str] = None,
    tournament_progress_callback: Callable[[Optional[WorldRunResults]], None] = None,
    world_progress_callback: Callable[[Optional[SCML2020World]], None] = None,
    non_competitors: Optional[Sequence[Union[str, Type[SCML2020Agent]]]] = None,
    non_competitor_params: Optional[Sequence[Union[str, Type[SCML2020Agent]]]] = None,
    dynamic_non_competitors: Optional[List[Type[Agent]]] = None,
    dynamic_non_competitor_params: Optional[List[Dict[str, Any]]] = None,
    exclude_competitors_from_reassignment: bool = True,
    name: str = None,
    verbose: bool = False,
    configs_only=False,
    compact=False,
    n_competitors_per_world=None,
    n_steps: Union[Tuple[int, int], int] = (50, 200),
    n_processes: Union[Tuple[int, int], int] = 2,
    n_lines: Union[np.ndarray, Tuple[int, int], int] = 10,
    process_inputs: Union[np.ndarray, Tuple[int, int], int] = 1,
    process_outputs: Union[np.ndarray, Tuple[int, int], int] = 1,
    production_costs: Union[np.ndarray, Tuple[int, int], int] = (1, 10),
    profit_means: Union[np.ndarray, Tuple[float, float], float] = (0.1, 0.2),
    profit_stddevs: Union[np.ndarray, Tuple[float, float], float] = 0.05,
    max_productivity: Union[np.ndarray, Tuple[float, float], float] = (0.8, 1.0),
    initial_balance: Optional[Union[np.ndarray, Tuple[int, int], int]] = None,
    cost_increases_with_level=True,
    equal_exogenous_supply=False,
    equal_exogenous_sales=False,
    exogenous_supply_predictability: Union[Tuple[float, float], float] = (0.6, 0.9),
    exogenous_sales_predictability: Union[Tuple[float, float], float] = (0.6, 0.9),
    exogenous_control: Union[Tuple[float, float], float] = -1,
    cash_availability: Union[Tuple[float, float], float] = (1.5, 2.5),
    force_signing=True,
    profit_basis=np.mean,
    disposal_cost: Union[np.ndarray, Tuple[float, float], float] = (0.0, 0.2),
    shortfall_penalty: Union[np.ndarray, Tuple[float, float], float] = (0.2, 1.0),
    disposal_cost_dev: Union[np.ndarray, Tuple[float, float], float] = (0.0, 0.02),
    shortfall_penalty_dev: Union[np.ndarray, Tuple[float, float], float] = (
        0.0,
        0.1,
    ),
    exogenous_price_dev: Union[np.ndarray, Tuple[float, float], float] = (0.1, 0.2),
    price_multiplier: Union[np.ndarray, Tuple[float, float], float] = (1.5, 2.0),
    random_agent_types: bool = False,
    penalties_scale: Union[str, List[str]] = "trading",
    forced_logs_fraction: float = FORCED_LOGS_FRACTION,
    **kwargs,
) -> Union[TournamentResults, PathLike]:
    """
    The function used to run ANAC 2021 SCML tournament (oneshot track).

    Args:

        name: Tournament name
        competitors: A list of class names for the competitors
        competitor_params: A list of competitor parameters (used to initialize the competitors).
        agent_names_reveal_type: If true then the type of an agent should be readable in its name (most likely at its
                                 beginning).
        n_configs: The number of different world configs (up to competitor assignment) to be generated.
        max_worlds_per_config: The maximum number of worlds to run per config. If None, then all possible assignments
                             of competitors within each config will be tried (all permutations).
        n_runs_per_world: Number of runs per world. All of these world runs will have identical competitor assignment
                          and identical world configuration.
        min_factories_per_level: Minimum number of factories for each production level
        total_timeout: Total timeout for the complete process
        tournament_path: Path at which to store all results. A scores.csv file will keep the scores and logs folder will
                         keep detailed logs
        parallelism: Type of parallelism. Can be 'serial' for serial, 'parallel' for parallel and 'distributed' for
                     distributed
        scheduler_port: Port of the dask scheduler if parallelism is dask, dist, or distributed
        scheduler_ip: IP Address of the dask scheduler if parallelism is dask, dist, or distributed
        world_progress_callback: A function to be called after everystep of every world run (only allowed for serial
                                 evaluation and should be used with cautious).
        tournament_progress_callback: A function to be called with `WorldRunResults` after each world finished
                                      processing
        non_competitors: A list of agent types that will not be competing in the sabotage competition but will exist
                         in the world
        non_competitor_params: parameters of non competitor agents
        dynamic_non_competitors: A list of non-competing agents that are assigned to the simulation dynamically during
                                 the creation of the final assignment instead when the configuration is created
        dynamic_non_competitor_params: paramters of dynamic non competitor agents
        exclude_competitors_from_reassignment: If true, competitors are excluded from the dyanamic non-competitors
        verbose: Verbosity
        configs_only: If true, a config file for each
        compact: If true, compact logs will be created and effort will be made to reduce the memory footprint
        n_competitors_per_world: Number of competitors in every simulation. If not given it will be a random number
                                 between 2 and min(2, n), where n is the number of competitors
        forced_logs_fraction: Fraction of simulations for which logs are always saved (including negotiations)
        kwargs: Arguments to pass to the `world_generator` function

    Returns:

        `TournamentResults` The results of the tournament or a `PathLike` giving the location where configs were saved

    Remarks:

        Default parameters will be used in the league with the exception of `parallelism` which may use distributed
        processing

    """
    # if competitor_params is None:
    #     competitor_params = [dict() for _ in range(len(competitors))]
    # for t, p in zip(competitors, competitor_params):
    #     p["controller_type"] = get_full_type_name(t)
    # competitors = ["scml.oneshot.world.DefaultOneShotAdapter"] * len(competitors)
    if n_competitors_per_world is None:
        n_competitors_per_world = kwargs.get(
            "n_competitors_per_world", randint(2, min(4, len(competitors)))
        )
    kwargs.pop("n_competitors_per_world", None)
    if non_competitors is None:
        non_competitors = ["scml.oneshot.agents.RandomOneShotAgent"]
        non_competitor_params = [dict() for _ in non_competitors]
    kwargs["round_robin"] = kwargs.get("round_robin", ROUND_ROBIN)
    kwargs["oneshot_world"] = True
    return tournament(
        competitors=competitors,
        competitor_params=competitor_params,
        non_competitors=non_competitors,
        non_competitor_params=non_competitor_params,
        agent_names_reveal_type=agent_names_reveal_type,
        n_configs=n_configs,
        n_runs_per_world=n_runs_per_world,
        max_worlds_per_config=max_worlds_per_config,
        tournament_path=tournament_path,
        total_timeout=total_timeout,
        parallelism=parallelism,
        scheduler_ip=scheduler_ip,
        scheduler_port=scheduler_port,
        tournament_progress_callback=tournament_progress_callback,
        world_progress_callback=world_progress_callback,
        name=name,
        verbose=verbose,
        configs_only=configs_only,
        n_agents_per_competitor=1,
        world_generator=anac2020oneshot_world_generator,
        config_generator=anac2020_config_generator,
        config_assigner=anac2020_assigner,
        score_calculator=balance_calculator2021oneshot,
        min_factories_per_level=min_factories_per_level,
        compact=compact,
        metric=truncated_mean,
        n_competitors_per_world=n_competitors_per_world,
        dynamic_non_competitors=dynamic_non_competitors,
        dynamic_non_competitor_params=dynamic_non_competitor_params,
        exclude_competitors_from_reassignment=exclude_competitors_from_reassignment,
        save_video_fraction=0.0,
        forced_logs_fraction=forced_logs_fraction,
        publish_exogenous_summary=True,
        publish_trading_prices=True,
        n_steps=n_steps,
        n_processes=n_processes,
        n_lines=n_lines,
        process_inputs=process_inputs,
        process_outputs=process_outputs,
        production_costs=production_costs,
        profit_means=profit_means,
        profit_stddevs=profit_stddevs,
        max_productivity=max_productivity,
        initial_balance=initial_balance,
        cost_increases_with_level=cost_increases_with_level,
        equal_exogenous_supply=equal_exogenous_supply,
        equal_exogenous_sales=equal_exogenous_sales,
        exogenous_supply_predictability=exogenous_supply_predictability,
        exogenous_sales_predictability=exogenous_sales_predictability,
        exogenous_control=exogenous_control,
        cash_availability=cash_availability,
        force_signing=force_signing,
        profit_basis=profit_basis,
        disposal_cost=disposal_cost,
        shortfall_penalty=shortfall_penalty,
        disposal_cost_dev=disposal_cost_dev,
        shortfall_penalty_dev=shortfall_penalty_dev,
        exogenous_price_dev=exogenous_price_dev,
        price_multiplier=price_multiplier,
        random_agent_types=random_agent_types,
        penalties_scale=penalties_scale,
        **kwargs,
    )
