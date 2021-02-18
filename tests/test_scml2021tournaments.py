from pathlib import Path

import pytest

from scml.oneshot.builtin import RandomOneShotAgent
from scml.scml2020.utils import anac2021_oneshot

BaseAgent = RandomOneShotAgent


class MyAgent0(BaseAgent):
    pass


class MyAgent1(BaseAgent):
    pass


class MyAgent2(BaseAgent):
    pass


class MyAgent3(BaseAgent):
    pass


class MyAgent4(BaseAgent):
    pass


class MyAgent5(BaseAgent):
    pass


class MyAgent6(BaseAgent):
    pass


class MyAgent7(BaseAgent):
    pass


class MyAgent8(BaseAgent):
    pass


class MyAgent9(BaseAgent):
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
