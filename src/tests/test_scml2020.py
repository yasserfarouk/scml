import numpy as np
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
)
import random
from scml.scml2020.agents.satisfiser import SatisfiserAgent

# agent types to be tested
types = [DoNothingAgent, RandomAgent, BuyCheapSellExpensiveAgent, SatisfiserAgent]
active_types = [_ for _ in types if _ != DoNothingAgent]


@mark.parametrize("agent_type", types)
def test_can_run_with_a_single_agent_type(agent_type):
    n_processes = 5
    n_steps = 10
    n_agents_per_level = 4
    profiles = []
    catalog = 20 * np.arange(1, n_processes + 2)
    for l in range(n_processes):
        supply = np.zeros((n_steps, n_processes + 1), dtype=int)
        sales = np.zeros((n_steps, n_processes + 1), dtype=int)
        supply_prices = np.random.randint(
            catalog[0] // 2, catalog[0] + 1, size=(n_steps, n_processes + 1), dtype=int
        )
        sales_prices = np.random.randint(
            catalog[-1], catalog[-1] * 2 + 1, size=(n_steps, n_processes + 1), dtype=int
        )

        if l == 0:
            supply[:, l] = np.random.randint(0, 20, size=n_steps, dtype=int)
        elif l == n_processes - 1:
            sales[:, l] = np.random.randint(0, 40, size=n_steps, dtype=int)
        for a in range(n_agents_per_level):
            costs = np.zeros((n_steps, n_processes))
            costs[:, l] = np.random.randint(1, 5, size=n_steps, dtype=int)
            profiles.append(
                FactoryProfile(
                    costs=costs,
                    guaranteed_sale_prices=sales_prices,
                    guaranteed_sales=sales,
                    guaranteed_supplies=supply,
                    guaranteed_supply_prices=supply_prices,
                )
            )
    world = SCML2020World(
        process_inputs=np.ones(n_processes, dtype=int),
        process_outputs=np.ones(n_processes, dtype=int),
        catalog_prices=catalog,
        agent_types=agent_type,
        profiles=profiles,
        n_steps=n_steps,
        name=unique_name(agent_type.__name__, rand_digits=2),
    )

    world.run()
    assert True


@given(
    agent_types=st.lists(
        st.sampled_from(active_types),
        min_size=1,
        max_size=len(active_types),
        unique=True,
    )
)
def test_can_run_with_a_multiple_agent_types(agent_types):
    print([_.__name__ for _ in agent_types])
    n_processes = 5
    n_steps = 100
    n_agents_per_level = 4
    profiles = []
    catalog = 20 * np.arange(1, n_processes + 2)
    for l in range(n_processes):
        supply = np.zeros((n_steps, n_processes + 1), dtype=int)
        sales = np.zeros((n_steps, n_processes + 1), dtype=int)
        supply_prices = np.random.randint(
            catalog[0] // 2, catalog[0] + 1, size=(n_steps, n_processes + 1)
        )
        sales_prices = np.random.randint(
            catalog[-1], catalog[-1] * 2 + 1, size=(n_steps, n_processes + 1)
        )
        if l == 0:
            supply[:, l] = np.random.randint(0, 20, size=n_steps)
        elif l == n_processes - 1:
            sales[:, l] = np.random.randint(0, 40, size=n_steps)
        for a in range(n_agents_per_level):
            costs = np.zeros((n_steps, n_processes))
            costs[:, l] = np.random.randint(1, 5, size=n_steps)
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
    print([_.__name__ for _ in agent_types_final])
    world = SCML2020World(
        process_inputs=np.ones(n_processes),
        process_outputs=np.ones(n_processes),
        catalog_prices=catalog,
        agent_types=agent_types_final,
        profiles=profiles,
        n_steps=n_steps,
    )
    world.run()
    assert True


def test_can_run_with_a_single_agent_type():
    agent_type = RandomAgent
    n_processes = 5
    n_steps = 10
    n_agents_per_level = 4
    profiles = []
    catalog = 20 * np.arange(1, n_processes + 2)
    for l in range(n_processes):
        supply = np.zeros((n_steps, n_processes + 1), dtype=int)
        sales = np.zeros((n_steps, n_processes + 1), dtype=int)
        supply_prices = np.random.randint(
            catalog[0] // 2, catalog[0] + 1, size=(n_steps, n_processes + 1), dtype=int
        )
        sales_prices = np.random.randint(
            catalog[-1], catalog[-1] * 2 + 1, size=(n_steps, n_processes + 1), dtype=int
        )

        if l == 0:
            supply[:, l] = np.random.randint(0, 20, size=n_steps, dtype=int)
        elif l == n_processes - 1:
            sales[:, l] = np.random.randint(0, 40, size=n_steps, dtype=int)
        for a in range(n_agents_per_level):
            costs = np.zeros((n_steps, n_processes))
            costs[:, l] = np.random.randint(1, 5, size=n_steps, dtype=int)
            profiles.append(
                FactoryProfile(
                    costs=costs,
                    guaranteed_sale_prices=sales_prices,
                    guaranteed_sales=sales,
                    guaranteed_supplies=supply,
                    guaranteed_supply_prices=supply_prices,
                )
            )
    world = SCML2020World(
        process_inputs=np.ones(n_processes, dtype=int),
        process_outputs=np.ones(n_processes, dtype=int),
        catalog_prices=catalog,
        agent_types=agent_type,
        profiles=profiles,
        n_steps=n_steps,
        name=unique_name(agent_type.__name__, rand_digits=2),
    )

    world.run()
    assert True
