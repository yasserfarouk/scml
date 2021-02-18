import copy
from unittest import mock

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import example
from hypothesis import given
from hypothesis.stateful import Bundle
from hypothesis.stateful import RuleBasedStateMachine
from hypothesis.stateful import rule

from scml.scml2020 import NO_COMMAND
from scml.scml2020 import Factory
from scml.scml2020 import FactoryProfile
from scml.scml2020 import FactoryState
from scml.scml2020.components.simulation import FactorySimulator

PROCESSES = 5
LINES = 10
STEPS = 50
INITIAL = 1000


def create_factory():
    return Factory(
        create_profile(),
        INITIAL,
        np.ones(PROCESSES, dtype=int),
        np.ones(PROCESSES, dtype=int),
        agent_id="aid",
        agent_name="aname",
        world=create_world(),
        # breach processing parameters
        buy_missing_products=True,
        # compensation parameters (for victims of bankrupt agents)
        compensate_before_past_debt=False,
        # external contracts parameters
        production_no_borrow=False,
        production_no_bankruptcy=True,
        production_penalty=0.15,
        production_buy_missing=False,
        catalog_prices=np.random.randint(1, 20, size=PROCESSES + 1, dtype=int),
    )


def create_profile():
    return FactoryProfile(np.random.randint(1, 10, (LINES, PROCESSES), dtype=int))


def create_world():
    return mock.Mock(current_step=0, n_steps=STEPS, bankruptcy_limit=-100)


@pytest.fixture
def profile():
    return create_profile()


@pytest.fixture()
def world_mock():
    return create_world()


@pytest.fixture()
def factory():
    return create_factory()


def test_factory_profile():
    p = create_profile()
    assert p.n_lines == LINES
    assert p.n_products == PROCESSES + 1
    assert p.n_processes == PROCESSES


class TestFactory:
    @staticmethod
    def confirm_empty(state):
        return (
            state.balance == INITIAL
            and np.all(state.commands == NO_COMMAND)
            and state.balance_change == 0
            and np.all(state.inventory_changes == 0)
            and np.all(state.inventory == 0)
        )

    @staticmethod
    def confirm_same(s1: FactoryState, s2: FactoryState):
        return (
            s1.balance == s2.balance
            and s1.balance_change == s2.balance_change
            and np.all(s1.commands == s2.commands)
            and np.all(s1.inventory == s2.inventory)
            and np.all(s1.inventory_changes == s2.inventory_changes)
        )

    def test_construction(self, factory):
        assert self.confirm_empty(factory.state)

    @given(
        process=st.integers(0, PROCESSES - 1),
        step=st.integers(-1, STEPS - 1),
        line=st.integers(-1, LINES - 1),
    )
    def test_scheduling(self, process, step, line):
        factory = create_factory()
        assert self.confirm_empty(factory.state)
        initial_state = copy.deepcopy(factory.state)

        factory.schedule_production(process, 1, step, line)
        state = factory.state
        assert not self.confirm_same(initial_state, state)
        assert len(state.commands[np.nonzero(state.commands == process)]) == 1
        if step >= 0:
            commands = state.commands[step, :]
            assert len(commands[np.nonzero(commands == process)]) == 1
        if line >= 0:
            commands = state.commands[:, line]
            assert len(commands[np.nonzero(commands == process)]) == 1

        if step >= 0 and line >= 0:
            assert factory.cancel_production(step, line)
            self.confirm_empty(factory.state)
            assert self.confirm_same(initial_state, factory.state)
        else:
            assert not factory.cancel_production(step, line)
            assert self.confirm_same(state, factory.state)

    def test_step(self):
        process, step, line = 0, 0, 0
        factory = create_factory()
        factory.confirm_production = False
        assert self.confirm_empty(factory.state)
        initial_state = copy.deepcopy(factory.state)

        factory.schedule_production(process, 1, step, line)
        factory.schedule_production(process, 1, step + 1, line)
        state = factory.state
        assert not self.confirm_same(initial_state, state)
        assert len(state.commands[np.nonzero(state.commands == process)]) == 2
        factory._inventory[process] = 1
        factory._inventory[process + 1] = 0
        factory.step()
        assert factory.current_inventory[process] == 0
        assert factory.current_inventory[process + 1] == 1
        assert factory.inventory_changes[process] == -1
        assert factory.inventory_changes[process + 1] == 1
        assert sum(factory.inventory_changes) == 0
        assert (
            factory.state.balance
            == factory.initial_balance - factory.profile.costs[line, process]
        )
        assert factory.state.balance_change == -factory.profile.costs[line, process]
        factory.step()
        assert factory.current_inventory[process] == 0
        assert factory.current_inventory[process + 1] == 1
        assert factory.inventory_changes[process] == 0
        assert factory.inventory_changes[process + 1] == 0
        assert sum(factory.inventory_changes) == 0
        assert (
            factory.state.balance
            == factory.initial_balance - factory.profile.costs[line, process]
        )
        assert factory.state.balance_change == 0


def test_simulator_runs():
    breach_penalty = 0.15
    profile = create_profile()
    factory = create_factory()
    bankruptcy_limit = factory.initial_balance
    simulator = FactorySimulator(
        profile=profile,
        initial_balance=factory.initial_balance,
        bankruptcy_limit=bankruptcy_limit,
        spot_market_global_loss=breach_penalty,
        catalog_prices=np.ones(profile.n_products, dtype=int),
        n_steps=50,
    )
    assert simulator.balance_at(simulator.n_steps - 1) == factory.initial_balance
    assert np.all(simulator.inventory_at(simulator.n_steps - 1) == 0)
    assert not simulator.is_bankrupt()
    assert not simulator.pay(
        factory.initial_balance * 3, t=1, ignore_money_shortage=False
    )
    assert simulator.balance_at(simulator.n_steps - 1) == factory.initial_balance
    simulator.pay(factory.initial_balance * 3, t=1, ignore_money_shortage=True)
    assert simulator.is_bankrupt()
