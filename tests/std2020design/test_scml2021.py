import warnings

import pytest


import random

import hypothesis.strategies as st
import numpy as np
from hypothesis import example, given, settings
from negmas import save_stats
from negmas.helpers import unique_name
from numpy.testing import assert_allclose
from pytest import mark, raises

import scml
from scml.oneshot import OneShotSingleAgreementAgent
from scml.scml2020 import (
    BuyCheapSellExpensiveAgent,
    DoNothingAgent,
    IndependentNegotiationsAgent,
    RandomAgent,
    SatisficerAgent,
    SCML2021World,
    is_system_agent,
)
from scml.scml2020.agents.decentralizing import DecentralizingAgent

from ..switches import SCML_ON_GITHUB, SCML_RUN2021_STD

warnings.filterwarnings("ignore")
random.seed(0)

COMPACT = True
NOLOGS = True
# agent types to be tested
types = scml.scml2020.builtin_agent_types(as_str=False)
oneshot_types = [
    _
    for _ in scml.oneshot.builtin_agent_types(as_str=False)
    if not issubclass(_, OneShotSingleAgreementAgent)
]
active_types = [_ for _ in types if _ != DoNothingAgent]


def generate_world(
    agent_types,
    n_processes=3,
    n_steps=10,
    n_agents_per_process=2,
    n_lines=10,
    initial_balance=None,
    buy_missing_products=True,
    **kwargs,
):
    kwargs["no_logs"] = True
    kwargs["compact"] = True
    world = SCML2021World(
        **SCML2021World.generate(
            agent_types,
            n_processes=n_processes,
            n_steps=n_steps,
            n_lines=n_lines,
            n_agents_per_process=n_agents_per_process,
            initial_balance=initial_balance,
            buy_missing_products=buy_missing_products,
            **kwargs,
        )
    )
    for s1, s2 in zip(world.suppliers[:-1], world.suppliers[1:]):
        assert len(set(s1).intersection(set(s2))) == 0
    for s1, s2 in zip(world.consumers[:-1], world.consumers[1:]):
        assert len(set(s1).intersection(set(s2))) == 0
    for p in range(n_processes):
        assert len(world.suppliers[p + 1]) == n_agents_per_process
        assert len(world.consumers[p]) == n_agents_per_process
    for a in world.agents.keys():
        if is_system_agent(a):
            continue
        assert len(world.agent_inputs[a]) == 1
        assert len(world.agent_outputs[a]) == 1
        assert len(world.agent_processes[a]) == 1
        assert len(world.agent_suppliers[a]) == (
            n_agents_per_process if world.agent_inputs[a][0] != 0 else 1
        )
        assert len(world.agent_consumers[a]) == (
            n_agents_per_process if world.agent_outputs[a][0] != n_processes else 1
        )
    return world


# def test_can_run_with_a_single_agent_type_example():
#     agent_type=IndependentNegotiationsAgent; buy_missing=True; n_processes=2
#     world = generate_world(
#         [agent_type],
#         buy_missing_products=buy_missing,
#         n_processes=n_processes,
#         name=unique_name(
#             f"scml2020tests/single/{agent_type.__name__}"
#             f"{'Buy' if buy_missing else 'Fine'}{n_processes}",
#             add_time=True,
#             rand_digits=4,
#         ),
#         compact=COMPACT,
#         no_logs=NOLOGS,
#     )
#     world.run()
#     save_stats(world, world.log_folder)


# @mark.parametrize("agent_type", types)
@given(
    agent_type=st.sampled_from(types),
    buy_missing=st.booleans(),
    n_processes=st.integers(2, 4),
)
@settings(deadline=500_000, max_examples=20)
@example(agent_type=IndependentNegotiationsAgent, buy_missing=True, n_processes=2)
@mark.skipif(
    condition=not SCML_RUN2021_STD,
    reason="Environment set to ignore running 2020 or tournament tests. See switches.py",
)
def test_can_run_with_a_single_agent_type(agent_type, buy_missing, n_processes):
    world = generate_world(
        [agent_type],
        buy_missing_products=buy_missing,
        n_processes=n_processes,
        name=unique_name(
            f"scml2020tests/single/{agent_type.__name__}"
            f"{'Buy' if buy_missing else 'Fine'}{n_processes}",
            add_time=True,
            rand_digits=4,
        ),
        compact=COMPACT,
        no_logs=NOLOGS,
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
    n_processes=st.integers(2, 4),
)
@settings(deadline=300_000, max_examples=20)
@example(
    agent_types=[scml.scml2020.agents.indneg.IndependentNegotiationsAgent],
    buy_missing=False,
    n_processes=2,
)
@mark.skipif(
    condition=not SCML_RUN2021_STD,
    reason="Environment set to ignore running 2020 or tournament tests. See switches.py",
)
def test_can_run_with_multiple_agent_types(agent_types, buy_missing, n_processes):
    world = generate_world(
        agent_types,
        buy_missing_products=buy_missing,
        name=unique_name(
            f"scml2020tests/multi/{'-'.join(_.__name__[:3] for _ in agent_types)}/"
            f"{'Buy' if buy_missing else 'Fine'}_p{n_processes}",
            add_time=True,
            rand_digits=4,
        ),
        n_processes=n_processes,
        initial_balance=10_000,
        compact=COMPACT,
        no_logs=NOLOGS,
    )
    world.run()
    save_stats(world, world.log_folder)


@given(
    buy_missing=st.booleans(),
    n_processes=st.integers(2, 4),
    initial_balance=st.sampled_from([50, 10_000, 10_000_000]),
)
@settings(deadline=500_000)
@mark.skipif(
    condition=not SCML_RUN2021_STD,
    reason="Environment set to ignore running 2020 or tournament tests. See switches.py",
)
def test_nothing_happens_with_do_nothing(buy_missing, n_processes, initial_balance):
    world = generate_world(
        [DoNothingAgent],
        buy_missing_products=buy_missing,
        n_processes=n_processes,
        name=unique_name(
            f"scml2020tests/single/doing_nothing/"
            f"{'Buy' if buy_missing else 'Fine'}_p{n_processes}_b{initial_balance}",
            add_time=True,
            rand_digits=4,
        ),
        initial_balance=initial_balance,
        bankruptcy_limit=initial_balance,
        compact=COMPACT,
        no_logs=NOLOGS,
    )
    world.run()
    assert len(world.contracts_per_step) == 0
    for a, f, p in world.afp:
        if is_system_agent(a.id):
            continue
        if (
            a.awi.my_input_product == 0
            or a.awi.my_input_product == a.awi.n_processes - 1
        ):
            assert f.current_balance <= initial_balance, (
                f"{a.name} (process {a.awi.my_input_product} of {a.awi.n_processes})'s balance "
                f"should go down"
            )
        # else:
        #     assert f.current_balance == initial_balance, (
        #         f"{a.name} (process {a.awi.my_input_product} of {a.awi.n_processes})'s balance "
        #         f"should not change"
        #     )


@pytest.mark.skipif(SCML_ON_GITHUB, reason="Known to timeout on CI")
@given(buy_missing=st.booleans(), n_processes=st.integers(2, 4))
@settings(deadline=300_000, max_examples=20)
def test_something_happens_with_random_agents(buy_missing, n_processes):
    world = generate_world(
        [RandomAgent],
        buy_missing_products=buy_missing,
        n_processes=n_processes,
        name=unique_name(
            f"scml2020tests/single/do_something/"
            f"{'Buy' if buy_missing else 'Fine'}_p{n_processes}",
            add_time=True,
            rand_digits=4,
        ),
        initial_balance=10_000,
        bankruptcy_limit=10_000,
        compact=COMPACT,
        no_logs=NOLOGS,
        n_steps=15,
    )
    world.run()
    assert len(world.signed_contracts) + len(world.cancelled_contracts) != 0


@given(n_processes=st.integers(2, 4))
@settings(deadline=300_000, max_examples=3)
def test_agents_go_bankrupt(n_processes):
    buy_missing = True
    world = generate_world(
        [RandomAgent],
        buy_missing_products=buy_missing,
        n_processes=n_processes,
        name=unique_name(
            f"scml2020tests/single/bankrupt/"
            f"{'Buy' if buy_missing else 'Fine'}_p{n_processes}",
            add_time=True,
            rand_digits=4,
        ),
        initial_balance=0,
        bankruptcy_limit=0,
        n_steps=10,
        compact=COMPACT,
        no_logs=NOLOGS,
    )
    world.run()
    #    assert len(world.signed_contracts) + len(world.cancelled_contracts) == 0
    for a, f, p in world.afp:
        if is_system_agent(a.id):
            continue
        if (
            a.awi.my_input_product == 0
            or a.awi.my_input_product == a.awi.n_processes - 1
        ):
            assert f.current_balance <= 0, (
                f"{a.name} (process {a.awi.my_input_product} of {a.awi.n_processes})'s balance "
                f"should go down"
            )
            assert f.is_bankrupt, (
                f"{a.name} (process {a.awi.my_input_product} of {a.awi.n_processes}) should "
                f"be bankrupt (balance = {f.current_balance}, inventory={f.current_inventory})"
            )
        # else:
        #     assert f.current_balance == 0, (
        #         f"{a.name} (process {a.awi.my_input_product} of {a.awi.n_processes})'s balance "
        #         f"should not change"
        #     )
        #     assert not f.is_bankrupt, (
        #         f"{a.name} (process {a.awi.my_input_product} of {a.awi.n_processes}) should "
        #         f"NOT be bankrupt (balance = {f.current_balance}, "
        #         f"inventory={f.current_inventory})"
        #     )


@mark.parametrize("n_processes", [2, 3, 4, 5, 6])
def test_generate(n_processes):
    world = SCML2021World(
        **SCML2021World.generate(
            agent_types=DoNothingAgent,
            n_steps=50,
            n_processes=n_processes,
            initial_balance=None,
        )
    )
    world.run()


def test_a_tiny_world():
    world = generate_world(
        [DecentralizingAgent],
        n_processes=2,
        n_steps=5,
        n_agents_per_process=2,
        n_lines=5,
        initial_balance=10_000,
        buy_missing_products=True,
    )
    world.run()
    assert True


def test_graph():
    world = generate_world(
        [DecentralizingAgent],
        n_processes=2,
        n_steps=10,
        n_agents_per_process=2,
        n_lines=5,
        initial_balance=10_000,
        buy_missing_products=True,
    )
    world.graph(together=True)
    world.step()
    world.graph(steps=None, together=True)
    world.graph(steps=None, together=False)
    world.run()
    world.graph((0, world.n_steps), together=False)
    world.graph((0, world.n_steps), together=True)


def test_graphs_lead_to_no_unknown_nodes():
    world = SCML2021World(
        **SCML2021World.generate(
            agent_types=[DecentralizingAgent, BuyCheapSellExpensiveAgent], n_steps=10
        ),
        construct_graphs=True,
        no_logs=True,
    )
    world.graph((0, world.n_steps))


@pytest.mark.skip("known to fail. Will not be supported anymore")
@pytest.mark.skipif(SCML_ON_GITHUB, reason="known to fail on CI")
@given(
    atype=st.lists(
        st.sampled_from(oneshot_types + types), unique=True, min_size=2, max_size=6
    )
)
@example(
    atype=[
        scml.scml2020.agents.random.RandomAgent,
    ],
)
@settings(deadline=300_000, max_examples=30)
def test_adapter(atype):
    world = SCML2021World(
        **SCML2021World.generate(agent_types=atype, n_steps=10),
        construct_graphs=False,
        no_logs=True,
        compact=True,
    )
    world.run()


def test_production_cost_increase():
    from scml.scml2020 import SCML2021World
    from scml.scml2020.agents import DecentralizingAgent

    NPROCESSES = 5
    costs = [[] for _ in range(NPROCESSES)]
    for _ in range(100):
        world = SCML2021World(
            **SCML2021World.generate(
                DecentralizingAgent,
                n_agents_per_process=10,
                n_processes=NPROCESSES,
            ),
            compact=True,
            no_logs=True,
        )
        for aid in world.agent_profiles.keys():
            if is_system_agent(aid):
                continue
            profile = world.agent_profiles[aid]
            costs[profile.input_products[0]].append(
                profile.costs[:, profile.input_products[0]].mean()
            )
    mean_costs = [sum(_) / len(_) for _ in costs]
    assert all(
        [
            b > (0.5 * (i + 2) / (i + 1)) * a
            for i, (a, b) in enumerate(zip(mean_costs[:-1], mean_costs[1:]))
        ]
    ), f"non-ascending costs {mean_costs}"


@mark.parametrize("n_processes", [2, 3, 4])
def test_satisficer(n_processes):
    world = generate_world(
        [SatisficerAgent],
        buy_missing_products=True,
        n_processes=n_processes,
        compact=COMPACT,
        no_logs=NOLOGS,
    )
    world.run()
    save_stats(world, world.log_folder)


N_AGENTS_PER_COMPETITORS = 3


class MyColluders(DoNothingAgent):
    # define a class-level variable to share the information
    my_friends = dict()

    def init(self):
        # share any information that will be static for the whole competition
        self.my_friends[self.id] = dict(
            level=self.awi.my_input_product, initial_balance=self.awi.current_balance
        )

    def before_step(self):
        # here you can access my_frieds and it will be fully populated.
        # any thing shared in `init()` or `step()` is available here
        assert self.id in self.my_friends
        if self.awi.current_step > 0:
            for v in self.my_friends.values():
                assert "last_step_balance" in v
                if v["level"] < self.awi.my_current_input:
                    assert "current_step_balance" in v

        # information shared here is consistently available to agents at
        # higher levels
        self.my_friends[self.id].update(dict(current_balance=self.awi.current_balance))

    def step(self):
        assert self.id in self.my_friends
        for v in self.my_friends.values():
            assert "current_step_balance" in v

        # share any information that is available by the end of the step
        self.my_friends[self.id].update(
            dict(
                last_step_balance=self.awi.current_balance,
                last_step_inventory=self.awi.current_inventory,
            )
        )


# def test_colluding_agents_find_each_other():
#     anac2021_collusion(
#         competitors=[MyColluders, RandomAgent],
#         n_agents_per_competitor=N_AGENTS_PER_COMPETITORS,
#         n_configs=3,
#         n_processes=2,
#         n_steps=5,
#     )


@mark.parametrize(
    ["method", "n_agents", "n_processes", "n_steps"],
    [
        ("guaranteed_profit", 1, 2, 5),
        ("profitable", 1, 2, 5),
        ("guaranteed_profit", 1, 2, 8),
        ("profitable", 1, 2, 8),
        ("guaranteed_profit", 2, 2, 8),
        ("profitable", 2, 2, 8),
        ("guaranteed_profit", 5, 2, 8),
        ("profitable", 5, 2, 8),
        ("guaranteed_profit", 1, 2, 16),
        ("profitable", 1, 2, 16),
        ("guaranteed_profit", 2, 2, 16),
        ("profitable", 2, 2, 16),
        ("guaranteed_profit", 5, 2, 16),
        ("profitable", 5, 2, 16),
        ("guaranteed_profit", 1, 3, 8),
        ("profitable", 1, 3, 8),
        ("guaranteed_profit", 2, 3, 8),
        ("profitable", 2, 3, 8),
        ("guaranteed_profit", 5, 3, 8),
        ("profitable", 5, 3, 8),
        ("guaranteed_profit", 1, 3, 16),
        ("profitable", 1, 3, 16),
        ("guaranteed_profit", 2, 3, 16),
        ("profitable", 2, 3, 16),
        ("guaranteed_profit", 5, 3, 16),
        ("profitable", 5, 3, 16),
        # ("guaranteed_profit", 2, 3, 80),
        # ("profitable", 2, 3, 80),
    ],
)
def test_satisficer_n_agent_per_level(method, n_agents, n_processes, n_steps):
    from pathlib import Path

    from negmas.helpers import force_single_thread
    from negmas.situated import save_stats

    from scml.scml2020 import SCML2021World

    force_single_thread(True)
    world = SCML2021World(
        **SCML2021World.generate(
            [SatisficerAgent] * n_processes,
            n_agents_per_process=n_agents,
            n_processes=n_processes,
            n_steps=n_steps,
            exogenous_generation_method=method,
            random_agent_types=False,
            neg_step_time_limit=float("inf"),
            neg_time_limit=float("inf"),
        ),
        compact=True,
        no_logs=True,
        end_negotiation_on_refusal_to_propose=True,
        name=f"Satisficer1{method}-a{n_agents}p{n_processes}s{n_steps}",
    )
    world.run()
    world.save_negotiations = True
    save_stats(
        world,
        log_dir=Path.home() / "negmas" / "logs" / "scml" / "scml2021" / world.name,
    )
    force_single_thread(False)

    assert True


class MyRandomAgent(RandomAgent):
    def has_trade(self, s=None):
        if s is None:
            s = self.awi.current_step
        return (self.awi._world._sold_quantity[:, s] > 0) & (
            np.abs(self.awi._world._real_price[:, s] - self.catalog) >= 1
        )

    def trading_(self, s=None):
        if s is None:
            s = self.awi.current_step
        return self.awi._world._trading_price[:, s]

    def init(self):
        super().init()
        self.trading, self.trading_before = [], []
        self.catalog = self.awi.catalog_prices.copy()
        self.n_products = self.awi.n_products
        assert_allclose(self.catalog, self.trading_())
        assert_allclose(self.catalog, self.awi.trading_prices)

    def before_step(self):
        super().before_step()
        self.trading_before.append(self.awi.trading_prices.tolist())

        assert_allclose(self.trading_before[-1], self.trading_())
        assert_allclose(self.catalog, self.awi.catalog_prices)

    def step(self):
        super().step()
        self.trading.append(self.awi.trading_prices.tolist())

        assert_allclose(self.catalog, self.awi.catalog_prices)
        assert_allclose(self.trading[-1], self.trading_())
        with_trade = self.has_trade()

        if not np.any(with_trade):
            return
        with raises(AssertionError):
            assert_allclose(self.trading[-1], self.catalog)


@mark.parametrize(
    ["n_agents", "n_processes", "n_steps"],
    [
        (1, 2, 16),
        (2, 2, 16),
        (5, 2, 16),
        (1, 3, 16),
        (2, 3, 16),
        (5, 3, 16),
    ],
)
def test_trading_prices_updated(n_agents, n_processes, n_steps):
    from negmas.helpers import force_single_thread

    from scml.scml2020 import SCML2021World

    eps = 1e-3

    world = SCML2021World(
        **SCML2021World.generate(
            [MyRandomAgent] * n_processes,
            n_agents_per_process=n_agents,
            n_processes=n_processes,
            n_steps=n_steps,
        ),
        compact=True,
        no_logs=True,
    )
    catalog_prices = world.catalog_prices
    diffs = np.zeros_like(catalog_prices)

    # we start at catlaog prices
    for aid, agent in world.agents.items():
        assert np.abs(agent.awi.trading_prices - catalog_prices).max() < eps

    force_single_thread(True)
    for _ in range(n_steps):
        world.step()
        trading_prices = None
        for aid, agent in world.agents.items():
            if is_system_agent(aid):
                continue
            trading_prices = agent.awi.trading_prices.copy()
            break
        diffs = np.maximum(diffs, np.abs(trading_prices - catalog_prices))

    assert diffs.max() > eps
    force_single_thread(False)
