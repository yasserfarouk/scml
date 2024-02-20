import random
from collections import defaultdict
from pprint import pprint

import matplotlib.pyplot as plt
from scml.oneshot.agent import OneShotAgent
from scml.oneshot.agents.rand import SyncRandomOneShotAgent
from scml.oneshot.world import SCML2022OneShotWorld, SCML2023OneShotWorld
from negmas import ResponseType

from scml.oneshot import RandomOneShotAgent
from scml.scml2020.common import is_system_agent


def try_agent(agent_type, n_processes=2, **kwargs):
    """Runs an agent in a world simulation against a randomly behaving agent"""
    return try_agents([RandomOneShotAgent, agent_type], n_processes, **kwargs)


def try_agents(
    agent_types, n_processes=2, n_trials=1, draw=True, agent_params=None, year=2023
):
    """
    Runs a simulation with the given agent_types, and n_processes n_trial times.
    Optionally also draws a graph showing what happened
    """
    type_scores = defaultdict(float)
    counts = defaultdict(int)
    agent_scores = dict()
    world = None
    for _ in range(n_trials):
        p = (
            n_processes
            if isinstance(n_processes, int)
            else random.randint(*n_processes)
        )
        cls = {2022: SCML2022OneShotWorld, 2023: SCML2023OneShotWorld}[year]
        world = cls(
            **cls.generate(
                agent_types,
                agent_params=agent_params,
                n_steps=10,
                n_processes=p,
                random_agent_types=True,
            ),
            construct_graphs=True,
            sync_calls=True,
            neg_step_time_limit=float("inf"),
            neg_time_limit=float("inf"),
        )
        world.run()

        all_scores = world.scores()
        for aid, agent in world.agents.items():
            if is_system_agent(aid):
                continue
            key = aid if n_trials == 1 else f"{aid}@{world.id[:4]}"
            agent_scores[key] = (
                agent.type_name.split(":")[-1].split(".")[-1],
                all_scores[aid],
                "(bankrupt)" if world.is_bankrupt[aid] else "",
            )
        for aid, agent in world.agents.items():
            if is_system_agent(aid):
                continue
            type_ = agent.type_name.split(":")[-1].split(".")[-1]
            type_scores[type_] += all_scores[aid]
            counts[type_] += 1
    type_scores = {k: v / counts[k] if counts[k] else v for k, v in type_scores.items()}
    if draw and world:
        world.draw(
            what=["contracts-concluded"],
            steps=(0, world.n_steps - 1),
            together=True,
            ncols=1,
            figsize=(20, 20),
        )
        plt.show()

    return world, agent_scores, type_scores


def analyze_contracts(world):
    """
    Analyzes the contracts signed in the given world
    """
    import pandas as pd

    data = pd.DataFrame.from_records(world.saved_contracts)
    return data.groupby(["seller_name", "buyer_name"])[
        ["quantity", "unit_price"]
    ].mean()


def print_agent_scores(agent_scores):
    """
    Prints scores of individiual agent instances
    """
    for aid, (type_, score, bankrupt) in agent_scores.items():
        print(f"Agent {aid} of type {type_} has a final score of {score} {bankrupt}")


def print_type_scores(type_scores):
    """Prints scores of agent types"""
    pprint(sorted(tuple(type_scores.items()), key=lambda x: -x[1]))


class MyOneShotDoNothing(OneShotAgent):
    """My Agent that does nothing"""

    def propose(self, negotiator_id, state):
        return None

    def respond(self, negotiator_id, state, source=None):
        return ResponseType.END_NEGOTIATION


def test_do_nothing():
    world, ascores, tscores = try_agent(MyOneShotDoNothing, draw=False)


def test_sync_random():
    world, ascores, tscores = try_agent(SyncRandomOneShotAgent, draw=False)
