import numpy as np
from negmas import save_stats
from negmas.helpers import unique_name
from pytest import mark
from hypothesis import given, settings
import hypothesis.strategies as st
from scml.scml2020 import (
    World,
    DoNothingAgent,
    FactoryProfile,
    RandomAgent,
    BuyCheapSellExpensiveAgent,
    INFINITE_COST,
)
import random
random.seed(0)
from scml.scml2020.agents.satisfiser import SatisfiserAgent

# agent types to be tested
types = [DoNothingAgent, RandomAgent, BuyCheapSellExpensiveAgent, SatisfiserAgent]
active_types = [_ for _ in types if _ != DoNothingAgent]


def generate_world(
    agent_types,
    n_processes=3,
    n_steps=20,
    n_agents_per_level=2,
    n_lines=10,
    initial_balance=10_000,
    random_supply_demand=False,
    buy_missing_products=True,
    **kwargs,
):
    profiles = []
    catalog = 20 * np.arange(1, n_processes + 2, dtype=int)
    agent_params = []
    agent_types_final = random.choices(agent_types, k=n_agents_per_level * n_processes)

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
            supply[:-n_processes, process] = (
                np.random.randint(1, n_lines * 2, size=n_steps-n_processes)
                if random_supply_demand
                else n_lines
            )
        elif process == n_processes - 1:
            sales[n_processes:, process + 1] = (
                np.random.randint(1, n_lines * 2, size=n_steps-n_processes)
                if random_supply_demand
                else n_lines
            )
        for a in range(n_agents_per_level):
            agent_params.append({"name": f"a{process}_{a}@"
                                         f"{agent_types_final[n_agents_per_level * process + a].__name__[:3]}"})
            costs = INFINITE_COST * np.ones((n_lines, n_processes), dtype=int)
            costs[:, process] = random.randint(1, 6)
            profiles.append(
                FactoryProfile(
                    costs=costs,
                    external_sale_prices=sales_prices,
                    external_sales=sales,
                    external_supplies=supply,
                    external_supply_prices=supply_prices,
                )
            )

    assert len(agent_types_final) == len(profiles)

    world = World(
        process_inputs=np.ones(n_processes, dtype=int),
        process_outputs=np.ones(n_processes, dtype=int),
        catalog_prices=catalog,
        agent_types=agent_types_final,
        agent_params=agent_params,
        profiles=profiles,
        n_steps=n_steps,
        initial_balance=initial_balance,
        buy_missing_products=buy_missing_products,
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
@given(buy_missing=st.booleans(), n_processes=st.integers(2, 4))
@settings(deadline=300_000, max_examples=20)
def test_can_run_with_a_single_agent_type(agent_type, buy_missing, n_processes):
    world = generate_world(
        [agent_type],
        buy_missing_products=buy_missing,
        n_processes=n_processes,
        name=unique_name(
            f"scml2020/single/{agent_type.__name__}"
            f"{'Buy' if buy_missing else 'Fine'}{n_processes}",
            add_time=True,
            rand_digits=4,
        ),
    )
    world.run()
    save_stats(world, world.log_folder)


@given(
    agent_types=st.lists(
        st.sampled_from(active_types),
        min_size=1,
        max_size=len(active_types),
        unique=True,
    ),
    buy_missing=st.booleans(),
    n_processes=st.integers(2, 4)
)
@settings(deadline=300_000, max_examples=20)
def test_can_run_with_a_multiple_agent_types(agent_types, buy_missing, n_processes):
    world = generate_world(
        agent_types, compact=True, buy_missing_products=buy_missing, name=unique_name("scml2020/multi")
        , n_processes=n_processes
    )
    world.run()
    save_stats(world, world.log_folder)
