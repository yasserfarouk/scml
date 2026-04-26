import sys
from time import perf_counter

from negmas.helpers import humanize_time
from rich import print
from rich.progress import track

from scml.oneshot import SCML2024OneShotWorld
from scml.oneshot.agent import OneShotAgent
from scml.oneshot.world import SCMLBaseWorld
from scml.std import SCML2024StdWorld
from scml.utils import DefaultAgentsOneShot2024, DefaultAgentsStd2024

INFO = dict(
    std=dict(world_type=SCML2024StdWorld, agent_types=DefaultAgentsStd2024),
    oneshot=dict(world_type=SCML2024OneShotWorld, agent_types=DefaultAgentsOneShot2024),
)


def main(
    world_type: type[SCMLBaseWorld],
    agent_types: tuple[type[OneShotAgent]],
    n: int = 10,
    world_params=dict(fast=True),
):
    nsteps = 0
    tick = perf_counter()
    for _ in track(range(n)):
        world = world_type(**world_type.generate(agent_types), **world_params)
        print(f"Running {world_type.__name__} world with {world.n_steps} steps")
        nsteps += world.n_steps
        world.run()
        assert (
            world.current_step == world.n_steps
        ), f"{world.current_step=} of {world.n_steps=}"
    return (perf_counter() - tick) / nsteps


if __name__ == "__main__":
    type_ = "oneshot" if len(sys.argv) < 2 else sys.argv[1]
    n = 10 if len(sys.argv) < 3 else int(sys.argv[2])
    print(f"Ran {n} simulations taking {humanize_time(main(**INFO[type_], n=n), show_ms=True, show_us=True)} per step")  # type: ignore
