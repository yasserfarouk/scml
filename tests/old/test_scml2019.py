import pytest

from scml.scml2019.utils19 import anac2019_world
from tests.switches import SCML_RUN2019


@pytest.mark.parametrize(
    "n_steps,consumption_horizon", [(5, 5), (10, 5)], ids=["tiny", "short"]
)
@pytest.mark.skipif(
    condition=not SCML_RUN2019,
    reason="Environment set to ignore running 2019 tests. See switches.py",
)
def test_anac2019(n_steps, consumption_horizon):
    world = anac2019_world(n_steps=n_steps, consumption_horizon=consumption_horizon)
    world.run()


if __name__ == "__main__":
    pytest.main(args=[__file__])
