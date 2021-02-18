from pathlib import Path

from scml.scml2019 import DoNothingFactoryManager
from scml.scml2019 import GreedyFactoryManager
from scml.scml2019 import anac2019_collusion
from scml.scml2019 import anac2019_std
from scml.scml2019.utils import anac2019_sabotage

PARALLELISM = "serial"


def test_std():
    results = anac2019_std(
        competitors=[DoNothingFactoryManager, GreedyFactoryManager],
        n_steps=5,
        n_configs=1,
        n_runs_per_world=1,
        parallelism=PARALLELISM,
        max_worlds_per_config=2,
        log_folder=str(Path.home() / "negmas" / "logs" / "tests"),
    )
    assert len(results.total_scores) >= 2
    assert (
        results.total_scores.loc[
            results.total_scores.agent_type
            == "scml.scml2019.factory_managers.builtins.DoNothingFactoryManager",
            "score",
        ].values[0]
        == 0.0
    )


def test_collusion():
    results = anac2019_collusion(
        competitors=[DoNothingFactoryManager, GreedyFactoryManager],
        n_steps=5,
        n_configs=1,
        n_runs_per_world=1,
        max_worlds_per_config=2,
        parallelism=PARALLELISM,
    )
    assert len(results.total_scores) >= 2
    assert (
        results.total_scores.loc[
            results.total_scores.agent_type
            == "scml.scml2019.factory_managers.builtins.DoNothingFactoryManager",
            "score",
        ].values[0]
        == 0.0
    )


class Greedy1(GreedyFactoryManager):
    pass


def test_sabotage():
    results = anac2019_sabotage(
        competitors=[DoNothingFactoryManager, Greedy1],
        parallelism="serial",
        n_steps=5,
        n_configs=1,
        n_runs_per_world=1,
        min_factories_per_level=1,
        n_default_managers=1,
        n_agents_per_competitor=2,
        max_worlds_per_config=2,
    )
    assert len(results.total_scores) >= 2
