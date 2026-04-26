from pathlib import Path

import pytest

from scml.scml2019 import DoNothingFactoryManager, GreedyFactoryManager
from scml.scml2019.utils19 import anac2019_collusion, anac2019_sabotage, anac2019_std

from ..switches import (
    SCML_RUN2019,
    SCML_RUN_COLLUSION_TOURNAMENTS,
    SCML_RUN_SABOTAGE_TOURNAMENTS,
    SCML_RUN_STD_TOURNAMENTS,
)

PARALLELISM = "serial"


@pytest.mark.skipif(
    condition=not SCML_RUN2019 or not SCML_RUN_STD_TOURNAMENTS,
    reason="Environment set to ignore running 2019 or tournament tests. See switches.py",
)
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
    assert len(results.total_scores) >= 2  # type: ignore
    assert (
        results.total_scores.loc[  # type: ignore
            results.total_scores.agent_type  # type: ignore
            == "scml.scml2019.factory_managers.builtins.DoNothingFactoryManager",
            "score",
        ].values[0]
        == 0.0
    )


@pytest.mark.skipif(
    condition=not SCML_RUN2019 or not SCML_RUN_COLLUSION_TOURNAMENTS,
    reason="Environment set to ignore running 2019 or tournament tests. See switches.py",
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
    assert len(results.total_scores) >= 2  # type: ignore
    assert (
        results.total_scores.loc[  # type: ignore
            results.total_scores.agent_type  # type: ignore
            == "scml.scml2019.factory_managers.builtins.DoNothingFactoryManager",
            "score",
        ].values[0]
        == 0.0
    )


class Greedy1(GreedyFactoryManager):
    pass


@pytest.mark.skipif(
    condition=not SCML_RUN2019 or not SCML_RUN_SABOTAGE_TOURNAMENTS,
    reason="Environment set to ignore running 2019 or tournament tests. See switches.py",
)
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
    assert len(results.total_scores) >= 2  # type: ignore
