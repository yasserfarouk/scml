"""Tests that scml participates in negmas's global random seed."""

import pytest

import scml.experiment as exp


def test_run_config_seeded_seeds_the_process(monkeypatch):
    applied = []
    monkeypatch.setattr(exp, "seed_all", applied.append)
    monkeypatch.setattr(exp, "run_config", lambda config, funcs: {"config": config})

    assert exp.run_config_seeded(7, {"a": 1}, []) == {"config": {"a": 1}}
    assert applied == [7]

    applied.clear()
    exp.run_config_seeded(None, {"a": 1}, [])
    assert not applied, "Nothing should be seeded when there is no seed"


def test_task_seeds_are_none_when_unseeded(monkeypatch):
    monkeypatch.setattr(exp, "task_seed", None)
    assert exp._task_seeds(4) == [None] * 4


def test_task_seeds_are_distinct_when_seeded(monkeypatch):
    rand = pytest.importorskip("negmas.helpers.rand", reason="negmas has no global seed")
    monkeypatch.setattr(rand, "_seed", 42)
    monkeypatch.setattr(exp, "task_seed", rand.task_seed)

    seeds = exp._task_seeds(4)
    assert all(s is not None for s in seeds)
    assert len(set(seeds)) == 4, "Every task must get its own stream"
    assert seeds == exp._task_seeds(4), "Seeds must be reproducible"
