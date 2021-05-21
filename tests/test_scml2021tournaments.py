import random
from pathlib import Path

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, example

from scml.oneshot.agents import RandomOneShotAgent
from scml.scml2020.utils import anac2021_oneshot
from scml.scml2020.utils import truncated_mean


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


class TestTruncatedMean:

    @given(s=st.floats(0.0, 100.0), m=st.floats(-50, 50))
    def test_tukey(self, s, m):
        limit = 1.5
        scores = np.hstack(
            (m + s * np.random.randn(90), m + limit * s + 0.1 + s * np.random.rand(10))
        )
        tmean, limits = truncated_mean(scores, top_limit=limit, bottom_limit=float("inf"), base="tukey", return_limits=True)
        assert not np.isnan(tmean), f"limits are {(limits)}"
        assert tmean <= np.mean(scores)

    @given(s=st.floats(0.0, 100.0), m=st.floats(-50, 50))
    def test_zscore(self, s, m):
        limit, eps = 3, max(0.001, s * 1e-2)
        scores = np.hstack(
            (m + s * np.random.randn(90), m + limit * s + 0.1 + s * np.random.rand(10))
        )
        m, s = np.mean(scores), np.std(scores)
        non_outlier = scores[scores <= m + s * limit]
        tmean, limits = truncated_mean(scores, top_limit=limit, bottom_limit=float("inf"), base="zscore", return_limits=True)
        assert abs(tmean - np.mean(non_outlier)) < eps, f"limits are {(limits)}"
        non_outlier = scores[scores <= m + s * (limit+5)]
        tmean, limits = truncated_mean(scores, top_limit=limit+5, bottom_limit=float("inf"), base="zscore", return_limits=True)
        assert abs(tmean - np.mean(non_outlier)) < eps, f"limits are {(limits)}"
        scores = - scores
        m, s = np.mean(scores), np.std(scores)
        non_outlier = scores[scores <= m + s * limit]
        tmean, limits = truncated_mean(scores, top_limit=limit, bottom_limit=float("inf"), base="zscore", return_limits=True)
        assert abs(tmean - np.mean(non_outlier)) < eps, f"limits are {(limits)}"
        non_outlier = scores[scores <= m + s * (limit+5)]
        tmean, limits = truncated_mean(scores, top_limit=limit+5, bottom_limit=float("inf"), base="zscore", return_limits=True)
        assert abs(tmean - np.mean(non_outlier)) < eps, f"limits are {(limits)}"
        # non_outlier = scores[scores >= -m - s * (limit + 5)]
        # tmean, limits = truncated_mean(scores, top_limit=float("inf"), bottom_limit=limit + 5, base="zscore", return_limits=True)
        # assert tmean == np.mean(non_outlier)
        # non_outlier = scores[scores >= -m - s * limit]
        # tmean, limits = truncated_mean(scores, top_limit=float("inf"), bottom_limit=limit, base="zscore", return_limits=True)
        # assert abs(tmean - np.mean(non_outlier)) < eps, f"limits are {(limits)}"

        # scores = np.hstack((scores, -scores))
        # m, s = np.mean(scores), np.std(scores)
        # non_outlier = scores[scores <= m + s * limit]
        # non_outlier = non_outlier[non_outlier >= -m - s * limit]
        # tmean, limits = truncated_mean(scores, top_limit=limit, bottom_limit=limit, base="zscore", return_limits=True)
        # assert abs(tmean - np.mean(non_outlier)) < eps, f"limits are {(limits)}"




    def test_truncates_expected_part_fraction(self):
        # testing fractions method
        scores = np.arange(100)
        tmean = truncated_mean(scores, top_limit=0.1, bottom_limit=0.0, base="fraction")
        assert tmean == np.mean(scores[:-10])
        tmean = truncated_mean(scores, top_limit=0.0, bottom_limit=0.1, base="fraction")
        assert tmean == np.mean(scores[9:])
        tmean = truncated_mean(scores, top_limit=0.1, bottom_limit=0.1, base="fraction")
        assert tmean == np.mean(scores[9:-10])
        tmean = truncated_mean(scores, top_limit=0.5, bottom_limit=0.5, base="fraction")
        assert tmean == 49.0
        tmean = truncated_mean(scores, top_limit=1.0, bottom_limit=1.0, base="fraction")
        assert np.isnan(tmean)

    def test_truncates_expected_part_scores(self):
        # testing scores method
        scores = np.arange(100)
        tmean = truncated_mean(scores, top_limit=0.1, bottom_limit=0.0, base="scores")
        assert tmean == np.mean(scores[:-10])
        tmean = truncated_mean(scores, top_limit=0.0, bottom_limit=0.1, base="scores")
        assert tmean == np.mean(scores[10:])
        tmean = truncated_mean(scores, top_limit=0.1, bottom_limit=0.1, base="scores")
        assert tmean == np.mean(scores[10:-10])
        tmean = truncated_mean(scores, top_limit=0.5, bottom_limit=0.5, base="scores")
        assert np.isnan(tmean)
        tmean = truncated_mean(scores, top_limit=1.0, bottom_limit=1.0, base="scores")
        assert np.isnan(tmean)

        # adding outliers
        scores = np.hstack((scores[:90], 1000 * np.ones(10)))
        tmean = truncated_mean(scores, top_limit=0.1, bottom_limit=0.0, base="scores")
        assert tmean == np.mean(scores[:-10])

        # order does not matter
        scores2 = np.random.permutation(scores)
        tmean = truncated_mean(scores, top_limit=0.1, bottom_limit=0.0, base="scores")
        tmean2 = truncated_mean(scores2, top_limit=0.1, bottom_limit=0.0, base="scores")
        assert tmean == tmean2

    def test_truncates_expected_part_iqr_fraction(self):
        # testing iqr method
        scores = np.hstack(
            (
                np.arange(25),
                50 + np.arange(25),
                100 + np.arange(25),
                150 + np.arange(25),
            )
        )
        tmean, limits = truncated_mean(
            scores,
            top_limit=0.2,
            bottom_limit=0.0,
            base="iqr_fraction",
            return_limits=True,
        )
        assert tmean == np.mean(scores[:-5]), f"limits are {(limits)}"
        tmean, limits = truncated_mean(
            scores,
            top_limit=0.0,
            bottom_limit=0.2,
            base="iqr_fraction",
            return_limits=True,
        )
        assert tmean == np.mean(scores[4:]), f"limits are {(limits)}"
        tmean, limits = truncated_mean(
            scores,
            top_limit=0.2,
            bottom_limit=0.2,
            base="iqr_fraction",
            return_limits=True,
        )
        assert tmean == np.mean(scores[4:-5]), f"limits are {(limits)}"
        tmean, limits = truncated_mean(
            scores,
            top_limit=1.0,
            bottom_limit=1.0,
            base="iqr_fraction",
            return_limits=True,
        )
        assert tmean == np.mean(scores[25:-25]), f"limits are {(limits)}"
