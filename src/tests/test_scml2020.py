import numpy as np
from negmas import save_stats
from negmas.helpers import unique_name
from pytest import mark
from hypothesis import given, settings
import hypothesis.strategies as st
from scml.scml2020 import (
    SCML2020World,
    DoNothingAgent,
    FactoryProfile,
    RandomAgent,
    BuyCheapSellExpensiveAgent,
    INFINTE_COST,
)
import random
from scml.scml2020.agents.satisfiser import SatisfiserAgent

# agent types to be tested
types = [DoNothingAgent, RandomAgent, BuyCheapSellExpensiveAgent, SatisfiserAgent]
active_types = [_ for _ in types if _ != DoNothingAgent]


def generate_world(
    agent_types,
    n_processes=2,
    n_steps=15,
    n_agents_per_level=3,
    n_lines=10,
    initial_balance=10_000_000,
    **kwargs,
):
    profiles = []
    catalog = 20 * np.arange(1, n_processes + 2, dtype=int)
    for process in range(n_processes):
        supply = np.zeros((n_steps, n_processes + 1), dtype=int)
        sales = np.zeros((n_steps, n_processes + 1), dtype=int)
        supply_prices = np.random.randint(
            catalog[0] // 2, catalog[0] + 1, size=(n_steps, n_processes + 1)
        )
        sales_prices = np.random.randint(
            catalog[-1], catalog[-1] * 2 + 1, size=(n_steps, n_processes + 1)
        )
        if process == 0:
            supply[:, process] = np.random.randint(1, 10, size=n_steps)
        elif process == n_processes - 1:
            sales[:, process] = np.random.randint(1, 10, size=n_steps)
        for a in range(n_agents_per_level):
            costs = INFINTE_COST * np.ones((n_lines, n_processes), dtype=int)
            costs[:, process] = random.randint(1, 6)
            profiles.append(
                FactoryProfile(
                    costs=costs,
                    guaranteed_sale_prices=sales_prices,
                    guaranteed_sales=sales,
                    guaranteed_supplies=supply,
                    guaranteed_supply_prices=supply_prices,
                )
            )

    agent_types_final = [
        random.sample(agent_types, 1)[0]
        for _ in range(n_agents_per_level * n_processes)
    ]
    assert len(agent_types_final) == len(profiles)
    world = SCML2020World(
        process_inputs=np.ones(n_processes, dtype=int),
        process_outputs=np.ones(n_processes, dtype=int),
        catalog_prices=catalog,
        agent_types=agent_types_final,
        profiles=profiles,
        n_steps=n_steps,
        initial_balance=initial_balance,
        **kwargs,
    )
    for s1, s2 in zip(world.suppliers[:-1], world.suppliers[1:]):
        assert len(set(s1).intersection(set(s2))) == 0
    for s1, s2 in zip(world.consumers[:-1], world.consumers[1:]):
        assert len(set(s1).intersection(set(s2))) == 0
    for p in range(n_processes):
        assert len(world.suppliers[p + 1]) == n_agents_per_level
        assert len(world.consumers[p]) == n_agents_per_level
    for a in world.agents.keys():
        assert len(world.agent_inputs[a]) == 1
        assert len(world.agent_outputs[a]) == 1
        assert len(world.agent_processes[a]) == 1
        assert len(world.agent_suppliers[a]) == (
            n_agents_per_level if world.agent_inputs[a][0] != 0 else 0
        )
        assert len(world.agent_consumers[a]) == (
            n_agents_per_level if world.agent_outputs[a][0] != n_processes else 0
        )
    return world


@mark.parametrize("agent_type", types)
def test_can_run_with_a_single_agent_type(agent_type):
    world = generate_world([agent_type])
    world.run()
    save_stats(world, world.log_folder)


@given(
    agent_types=st.lists(
        st.sampled_from(active_types),
        min_size=1,
        max_size=len(active_types),
        unique=True,
    )
)
def test_can_run_with_a_multiple_agent_types(agent_types):
    world = generate_world(agent_types, compact=True)
    world.run()


# def test_can_run_with_a_bcse_agent():
#     world = generate_world([BuyCheapSellExpensiveAgent])
#     world.run()
#     save_stats(world, world.log_folder)
#
#
# def test_can_run_with_a_random_agent():
#     world = generate_world([RandomAgent])
#     world.run()
#     save_stats(world, world.log_folder)
#
#
# def test_can_run_with_do_nothing():
#     world = generate_world([DoNothingAgent])
#     world.run()
#     save_stats(world, world.log_folder)
