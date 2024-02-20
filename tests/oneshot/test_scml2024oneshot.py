from collections import defaultdict

import pytest
from rich import print

from scml.oneshot import PLACEHOLDER_AGENT_PREFIX
from scml.oneshot.agents.greedy import (
    GreedyOneShotAgent,
    GreedySingleAgreementAgent,
    GreedySyncAgent,
)
from scml.oneshot.agents.rand import (
    RandDistOneShotAgent,
    RandomOneShotAgent,
    SyncRandomOneShotAgent,
)
from scml.oneshot.common import is_system_agent
from scml.oneshot.context import ANACOneShotContext
from scml.oneshot.rl.agent import OneShotRLAgent
from scml.oneshot.world import SCML2024OneShotWorld

from ..switches import DefaultOneShotWorld


@pytest.mark.parametrize(
    "atype,no_bankrupt,some_profits",
    [
        (SyncRandomOneShotAgent, True, True),
        (RandomOneShotAgent, False, False),
        (GreedySingleAgreementAgent, False, False),
        (GreedyOneShotAgent, False, False),
        (GreedySyncAgent, True, False),
    ],
)
def test_run_single_agent(atype, no_bankrupt, some_profits):
    world = SCML2024OneShotWorld(**SCML2024OneShotWorld.generate(atype, n_steps=50))
    types = [
        a._obj.__class__  # type: ignore
        for k, a in world.agents.items()
        if not is_system_agent(k)
    ]
    assert all(
        [issubclass(_, atype) for _ in types]
    ), f"Not all types are {atype.__name__}: {types}"
    world.run()
    d = world.scores()
    scores = list(d.values())
    if no_bankrupt:
        assert min(scores) >= 0, (
            "Some negative scores (i.e. some agents went bankrupt)!!\n"
            "{k:v for k, v in d.items() if v < 0}"
        )
    if some_profits:
        assert max(scores) >= 1, f"No agents got a any profits:\n{d}"


class NonCompetitorRecordingAgent(RandDistOneShotAgent):
    def __init__(self, *args, print_everything=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_proposal_calls = defaultdict(int)
        self.counter_calls = defaultdict(int)
        self.print_everything = print_everything

    def first_proposals(self):
        proposals = super().first_proposals()
        self.first_proposal_calls[self.awi.current_step] += 1
        if self.print_everything:
            print(
                f"{self.awi.current_step} First Proposal for {self.id}:{self.awi.needed_sales}, {self.awi.needed_supplies}"
            )
            print(proposals)
        return proposals

    def counter_all(self, offers, states):
        responses = super().counter_all(offers, states)
        self.counter_calls[self.awi.current_step] += 1
        if self.print_everything:
            print(
                f"{self.awi.current_step} Responses for {self.id}:{self.awi.needed_sales}, {self.awi.needed_supplies}"
            )
            print(f"{offers=}\n{responses=}")
        return responses


class RecordingAgent(NonCompetitorRecordingAgent):
    ...


class AlwaysFallingBack(OneShotRLAgent):
    def has_no_valid_model(self):
        return True

    def __init__(
        self,
        *args,
        fallback_type=RecordingAgent,
        **kwargs,
    ):
        super().__init__(
            *args,
            fallback_type=fallback_type,
            **kwargs,
        )


@pytest.mark.parametrize("allow_zero", (True, False))
def test_recording(allow_zero):
    context = ANACOneShotContext(
        world_type=DefaultOneShotWorld,
        non_competitors=(NonCompetitorRecordingAgent,),
        world_params=dict(allow_zero_quantity=allow_zero),
    )
    world, (agent,) = context.generate((RecordingAgent,))
    assert isinstance(agent, RecordingAgent)
    world.run()
    assert (
        len(agent.counter_calls) + len(agent.first_proposal_calls) >= world.current_step
    )
    # print(world.scores())
    # assert (
    #     False
    # ), f"{world.current_step=}\n{agent.first_proposal_calls=}\n{agent.counter_calls=}"


@pytest.mark.parametrize("allow_zero", (True, False))
def test_fallingback(allow_zero):
    context = ANACOneShotContext(
        world_type=DefaultOneShotWorld,
        non_competitors=(NonCompetitorRecordingAgent,),
        world_params=dict(allow_zero_quantity=allow_zero),
    )
    world, (agent,) = context.generate((AlwaysFallingBack,))
    assert isinstance(agent, AlwaysFallingBack)
    world.run()
    assert agent._fallback_agent is not None
    assert (
        len(agent._fallback_agent.counter_calls)
        + len(agent._fallback_agent.first_proposal_calls)
        >= world.current_step
    )
    # print(world.scores())
    # assert (
    #     False
    # ), f"{world.current_step=}\n{agent._fallback_agent.first_proposal_calls=}\n{agent._fallback_agent.counter_calls=}"


CONFIGS = dict()
DefaultType = RandDistOneShotAgent


def try_agent(agent_type, alone=False, **kwargs):
    """Runs an agent in a world simulation against a randomly behaving agent"""
    if alone:
        return try_agents([agent_type, agent_type], **kwargs)
    return try_agents([DefaultType, agent_type], **kwargs)


def make_configs(n, n_trials, n_steps=10):
    types = [f"{PLACEHOLDER_AGENT_PREFIX}{i}" for i in range(n)]
    return [
        (
            types,
            SCML2024OneShotWorld.generate(
                tuple(types),
                agent_params=None,
                n_steps=10,
                random_agent_types=True,
                construct_graphs=True,
            ),
        )
        for _ in range(n_trials)
    ]


def try_agents(
    agent_types, n_trials=4, n_steps=10, draw=True, agent_params=None, debug=True
):
    """
    Runs a simulation with the given agent_types, and n_processes n_trial times.
    Optionally also draws a graph showing what happened
    """
    n = len(agent_types)
    if n not in CONFIGS:
        CONFIGS[n] = make_configs(n, n_trials, n_steps)
    n_rem = n_trials - len(CONFIGS[n])
    if n_rem > 0:
        CONFIGS[n] += make_configs(n, n_rem, n_steps)
    configs = CONFIGS[n]
    type_scores = defaultdict(float)
    counts = defaultdict(int)
    agent_scores = dict()
    worlds = []
    for old_types, config in configs[:n_trials]:
        world = SCML2024OneShotWorld(
            **SCML2024OneShotWorld.replace_agents(
                config, old_types, agent_types, agent_params
            ),
            debug=debug,
        )
        worlds.append(world)
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

    return worlds, agent_scores, type_scores


def test_trying_agent():
    worlds, ascores, tscores = try_agent(RandomOneShotAgent, alone=True)


def test_combining_stats():
    worlds, ascores, tscores = try_agent(SyncRandomOneShotAgent, alone=True)
    SCML2024OneShotWorld.plot_combined_stats(
        tuple(worlds),
        stats="score",
        n_steps=None,
        pertype=False,
        legend=True,
        makefig=True,
        ylegend=1.0,
    )
