from scml.oneshot.context import ANACOneShotContext
from scml.runner import WorldInfo, WorldRunner
from scml.oneshot.agents import (
    EqualDistOneShotAgent,
    RandDistOneShotAgent,
    GreedySyncAgent,
    RandomOneShotAgent,
)
from typing import Iterable
from scml.oneshot.world import SCMLBaseWorld
from pytest import mark
from scml.std.agents.greedy import GreedyStdAgent
from scml.std.agents.rand import RandomStdAgent
from scml.std.context import ANACStdContext
from scml.std.world import StdWorld


@mark.parametrize("all_agents", [True, False])
def test_simple_run(all_agents):
    runner = WorldRunner(
        ANACOneShotContext(n_steps=10),
        n_configs=3,
        n_repetitions=2,
        save_worlds=True,
        control_all_agents=all_agents,
    )

    assert len(runner(EqualDistOneShotAgent)) == 3 * 2
    assert len(runner(RandDistOneShotAgent)) == 3 * 2
    for config in runner.existing_config_names:
        assert len(runner.agents_of(EqualDistOneShotAgent, config=config)) == len(
            runner.agents_of(RandDistOneShotAgent, config=config)
        )
    score_summary = runner.score_summary()
    assert "score" in score_summary.columns, f"{score_summary}"
    assert len(score_summary) == 2, f"{score_summary}"
    if not all_agents:
        assert len(runner.scores) == 3 * 2 * 2
    assert len(runner.worlds) == 3 * 2
    for v in runner.worlds.values():
        assert len(v) == 2
        for x in v:
            assert isinstance(x, WorldInfo)
            assert isinstance(x.world, SCMLBaseWorld)
    assert len(runner.all_worlds) == 3 * 2 * 2
    assert len(runner.worlds_of(RandDistOneShotAgent)) == 3 * 2
    assert len(runner.worlds_of(EqualDistOneShotAgent)) == 3 * 2
    for i in range(3):
        assert len(runner.worlds_of(RandDistOneShotAgent, f"c{i}")) == 2
        assert len(runner.worlds_of(EqualDistOneShotAgent, f"c{i}")) == 2
    assert len(runner.stats) == 3 * 2 * 2 * 10, f"{runner.stats}"


def test_copying():
    r1 = WorldRunner(
        ANACOneShotContext(n_steps=10), n_configs=3, n_repetitions=2, save_worlds=True
    )
    r2 = WorldRunner.from_runner(r1)
    r3 = WorldRunner.from_configs(r1.generator.world_type, tuple(r1.configs))

    def equal(a, b):
        if not a:
            assert not b
            return True
        if isinstance(a, dict):
            for k, v in a.items():
                assert v == b[k]
            for k, v in b.items():
                assert v == a[k]
            return True
        if isinstance(a, Iterable):
            for x, y in zip(a, b, strict=True):
                assert equal(x, y)
            return True
        assert a == b
        return True

    for i, (c1, c2) in enumerate(((r1.configs, r2.configs), (r1.configs, r3.configs))):
        assert len(c1) == len(
            c2
        ), f"Error in comparing {'r1, r2' if i == 0 else 'r1, r3'}"
        for a, b in zip(c1, c2, strict=True):
            for k in ("agent_types", "agent_params"):
                assert len(a) == len(
                    b
                ), f"Error in comparing {k} for {'r1, r2' if i == 0 else 'r1, r3'}"
                for x, y in zip(a[k], b[k], strict=True):
                    assert equal(x, y), f"{k}: {x=}, {y=}"


@mark.parametrize("all_agents", [True, False])
def test_std_runner(all_agents):
    context = ANACStdContext(n_steps=10, n_processes=3)
    runner = WorldRunner(context, n_configs=3, n_repetitions=2, save_worlds=True)
    assert len(runner(GreedyStdAgent)) == 3 * 2
    assert len(runner(RandomStdAgent)) == 3 * 2
    for config in runner.existing_config_names:
        assert len(runner.agents_of(GreedyStdAgent, config=config)) == len(
            runner.agents_of(RandomStdAgent, config=config)
        )
    score_summary = runner.score_summary()
    assert len(score_summary) == 2, f"{score_summary}"
    if not all_agents:
        assert len(runner.scores) == 3 * 2 * 2
    assert len(runner.worlds) == 3 * 2
    for v in runner.worlds.values():
        assert len(v) == 2
        for x in v:
            assert isinstance(x, WorldInfo)
            assert isinstance(x.world, StdWorld)
    assert len(runner.all_worlds) == 3 * 2 * 2
    assert len(runner.worlds_of(GreedyStdAgent)) == 3 * 2
    assert len(runner.worlds_of(RandomStdAgent)) == 3 * 2
    for i in range(3):
        assert len(runner.worlds_of(GreedyStdAgent, f"c{i}")) == 2
        assert len(runner.worlds_of(RandomStdAgent, f"c{i}")) == 2
    assert len(runner.stats) == 3 * 2 * 2 * 10, f"{runner.stats}"


def test_plot():
    runner = WorldRunner(
        ANACOneShotContext(n_steps=4),
        n_configs=2,
        n_repetitions=1,
        save_worlds=True,
        shorten_names=True,
    )

    runner(EqualDistOneShotAgent)
    runner(RandDistOneShotAgent)
    runner.plot_stats()


def test_agents_per_world():
    runner = WorldRunner(
        ANACOneShotContext(n_steps=4),
        n_configs=2,
        n_repetitions=1,
        save_worlds=True,
        shorten_names=True,
    )

    runner(EqualDistOneShotAgent)
    runner(RandDistOneShotAgent)
    runner.agents_per_world_of(EqualDistOneShotAgent)


@mark.parametrize("progress", (True, False))
def test_runner_without_saving(progress):
    runner = WorldRunner(
        ANACOneShotContext(name="myc", n_steps=20),
        n_configs=3,
        n_repetitions=2,
        save_worlds=False,
    )

    for t in (
        EqualDistOneShotAgent,
        RandDistOneShotAgent,
        GreedySyncAgent,
        RandomOneShotAgent,
    ):
        infos = runner.add(t)
        runner._runall(infos, progress=progress)

    assert all([_.world.current_step == 20 for _ in runner.all_world_infos])
    assert all([_.world.name.startswith("myc") for _ in runner.all_world_infos])


@mark.parametrize("progress", (True, False))
def test_runner_with_call(progress):
    runner = WorldRunner(
        ANACOneShotContext(name="myc", n_steps=20), n_configs=3, n_repetitions=2
    )

    for t in (
        EqualDistOneShotAgent,
        RandDistOneShotAgent,
        GreedySyncAgent,
        RandomOneShotAgent,
    ):
        runner(t, progress=progress)

    assert all([_.world.current_step == 20 for _ in runner.all_world_infos])
    assert all([_.world.name.startswith("myc") for _ in runner.all_world_infos])


@mark.parametrize("progress", (True, False))
def test_runner_in_two_stages_saving(progress):
    runner = WorldRunner(
        ANACOneShotContext(name="myc", n_steps=20), n_configs=3, n_repetitions=2
    )

    for t in (
        EqualDistOneShotAgent,
        RandDistOneShotAgent,
        GreedySyncAgent,
        RandomOneShotAgent,
    ):
        runner.add(t)

    runner.runall(progress=progress)

    assert all([_.world.current_step == 20 for _ in runner.all_world_infos])
    assert all([_.world.name.startswith("myc") for _ in runner.all_world_infos])
