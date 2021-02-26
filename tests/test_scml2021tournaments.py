from pathlib import Path

import pytest

from scml.oneshot.agents import RandomOneShotAgent
from scml.scml2020.utils import anac2021_oneshot


class MyAgent0(RandomOneShotAgent):
    pass


class MyAgent1(RandomOneShotAgent):
    pass


class MyAgent2(RandomOneShotAgent):
    pass


class MyAgent3(RandomOneShotAgent):
    pass


class MyAgent4(RandomOneShotAgent):
    pass


class MyAgent5(RandomOneShotAgent):
    pass


class MyAgent6(RandomOneShotAgent):
    pass


class MyAgent7(RandomOneShotAgent):
    pass


class MyAgent8(RandomOneShotAgent):
    pass


class MyAgent9(RandomOneShotAgent):
    pass


@pytest.mark.parametrize("n", [2, 3])
def test_oneshot(n):
    competitors = [eval(f"MyAgent{_}") for _ in range(n)]
    results = anac2021_oneshot(
        competitors=competitors,
        n_steps=10,
        n_configs=1,
        n_runs_per_world=1,
        parallelism="serial",
        log_folder=str(Path.home() / "negmas" / "logs" / "tests"),
    )
    df = (
        results.scores[["agent_type", "score"]]
        .groupby(["agent_type"])
        .count()
        .reset_index()
    )
    assert len(results.total_scores) == n
    assert (
        len(df["score"].unique()) == 1
    ), f"Agents do not appear the same number of times:\n{df}"
