from typing import Any, Type

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register

from scml.common import intin, make_array
from scml.oneshot.agent import OneShotAgent
from scml.oneshot.awi import OneShotAWI
from scml.oneshot.rl.action import ActionManager
from scml.oneshot.rl.observation import ObservationManager
from scml.oneshot.world import SCML2023OneShotWorld
from scml.scml2020.utils import (
    DefaultAgentsOneShot2023,
)  # anac2023_oneshot_world_generator,; anac_assigner_oneshot,; anac_config_generator_oneshot,

__all__ = ["OneShotEnv"]


class OneShotEnv(gym.Env):
    def __init__(
        self,
        action_manager: ActionManager,
        observation_manager: ObservationManager,
        render_mode=None,
        year=2023,
        agent_type: Type[OneShotAgent] | None = None,
        level: int = 0,
        n_processes: tuple[int, int] | int = 2,
        n_suppliers: tuple[int, int] | int = 0,
        n_consumers: tuple[int, int] | int = 4,
        n_competitors: tuple[int, int] | int = (3, 7),
        n_agents_per_level: tuple[int, int] | int = (4, 8),
        n_lines: int = 10,
        non_competitors: list[OneShotAgent] = DefaultAgentsOneShot2023,
        **kwargs,
    ):
        from .agent import OneShotRLAgent

        if agent_type is None:
            agent_type = OneShotRLAgent
        # check that the inputs make sense:
        # - the resulting world will always have the same number of suppliers and consumers for the test agent
        # - we do not have level zero agents with suppliers or last level agents with consumers
        assert level != 0 or n_suppliers == 0
        assert (
            not isinstance(n_processes, int)
            or (level != n_processes - 1 and level != -1)
            or n_consumers == 0
        )
        if isinstance(n_processes, tuple):
            assert not (level > 0 and level < n_processes[-1] - 1) or (
                isinstance(n_suppliers, int)
                and isinstance(n_consumers, int)
                and n_suppliers > 0
                and n_consumers > 0
            )
            assert level == -1 or level < n_processes[-1]
        else:
            assert not (level > 0 and level < n_processes - 1) or (
                isinstance(n_suppliers, int)
                and isinstance(n_consumers, int)
                and n_suppliers > 0
                and n_consumers > 0
            )
            assert level == -1 or level < n_processes
        kwargs["random_agent_types"] = False
        # kwargs["n_processes"] = n_processes
        self._n_processes, self._level = n_processes, level
        self._n_suppliers, self._n_consumers = n_suppliers, n_consumers
        self._n_agents_per_level = n_agents_per_level
        self._n_competitors = n_competitors
        self._non_competitors = non_competitors
        self._config = kwargs
        self._world: SCML2023OneShotWorld = None  # type: ignore
        self._agent_type = agent_type
        self._agent_id: str = ""
        self._agent: OneShotAgent = None  # type: ignore
        self._year = year
        self._n_lines = n_lines
        self._obs_manager = observation_manager
        self._action_manager = action_manager
        assert observation_manager.is_valid(self)
        assert action_manager.is_valid(self)
        self.action_space = action_manager.make_space()
        self.observation_space = observation_manager.make_space()
        self.render_mode = render_mode
        # self.reset()

    def _get_obs(self):
        return self._obs_manager.encode(self._agent.awi.state)
        # return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return dict()
        # return {
        #     "distance": np.linalg.norm(
        #         self._agent_location - self._target_location, ord=1
        #     )
        # }

    def _render_frame(self):
        pass

    def close(self):
        pass

    def render(self):
        pass

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        import random

        random.seed(seed)
        year = self._year

        # configs = anac_config_generator_oneshot(
        #     year,
        #     n_competitors=1,
        #     n_agents_per_competitor=1,
        #     one_offer_per_step=True,
        # )
        # assigned = anac_assigner_oneshot(
        #     configs, 1, competitors=(self._agent_type,), params=None, fair=False  # type: ignore
        # )
        # assigned = assigned[0][0]
        # self._world = anac2023_oneshot_world_generator(year=year, **assigned)
        n_processes = intin(self._n_processes)
        # find my level
        my_level = n_processes - 1 if self._level < 0 else self._level

        # distribute agents over production levels (processes)
        n_agents_per_process = make_array(
            self._n_agents_per_level, n_processes, dtype=int
        )
        # override the number of consumers and number of suppliers to match my choice
        if my_level == 0:
            assert isinstance(self._n_consumers, int)
            n_agents_per_process[1] = self._n_consumers
        elif my_level == n_processes - 1:
            assert isinstance(self._n_suppliers, int)
            n_agents_per_process[n_processes - 2] = self._n_suppliers
        else:
            assert isinstance(self._n_consumers, int)
            assert isinstance(self._n_suppliers, int)
            n_agents_per_process[my_level + 1] = self._n_consumers
            n_agents_per_process[my_level - 1] = self._n_suppliers
        n_competitors = intin(self._n_competitors) + 1
        n_agents_per_process[my_level] = n_competitors

        n_agents = sum(n_agents_per_process)
        agent_types = list(random.choices(self._non_competitors, k=n_agents))
        agent_processes = np.zeros(n_agents, dtype=int)
        nxt, indx = 0, -1
        for level in range(n_processes):
            last = nxt + n_agents_per_process[level]
            agent_processes[nxt:last] = level
            if level == my_level:
                indx = random.randint(nxt, last)
                agent_types[indx] = self._agent_type  # type: ignore
            nxt += n_agents_per_process[level]
        assert indx >= 0
        conf = self._config
        conf.update(
            dict(
                n_lines=self._n_lines,
                agent_types=agent_types,
                agent_processes=agent_processes.tolist(),
                n_processes=n_processes,
            )
        )
        self._world = SCML2023OneShotWorld(
            **SCML2023OneShotWorld.generate(
                **conf,
            ),
            one_offer_per_step=True,
        )
        expected_type = self._agent_type._type_name()
        agents = [
            i
            for i, type_ in enumerate(self._world.agent_types)
            if type_.split(":")[-1] == expected_type
        ]
        assert (
            len(agents) == 1
        ), f"Found the following agent of type {self._agent_type}: {agents}"
        for aid, agent in self._world.agents.items():
            if agent.type_name.split(":")[-1] == expected_type:
                self._agent_id, self._agent = aid, agent  # type: ignore
                break

        awi: OneShotAWI = self._agent.awi  # type: ignore
        if my_level == 0:
            assert (
                len(awi.my_consumers) == self._n_consumers
            ), f"{len(awi.my_consumers)=} != {self._n_consumers=}"
            assert len(awi.my_suppliers) == 1
        elif my_level == n_processes - 1:
            assert (
                len(awi.my_suppliers) == self._n_suppliers
            ), f"{len(awi.my_suppliers)=} != {self._n_suppliers=}"
            assert len(awi.my_consumers) == 1
        else:
            assert (
                len(awi.my_suppliers) == self._n_suppliers
            ), f"{len(awi.my_suppliers)=} != {self._n_suppliers=}"
            assert (
                len(awi.my_consumers) == self._n_consumers
            ), f"{len(awi.my_consumers)=} != {self._n_consumers=}"
        assert (
            awi.n_competitors == n_competitors - 1
        ), f"{awi.n_competitors=} != {n_competitors=}"
        self._world.step_with(dict(), init=True)
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        score_before = self._world.scores()[self._agent_id]
        terminated = not self._world.step_with(
            {self._agent_id: self._action_manager.decode(self._agent.awi, action)}  # type: ignore
        )
        reward = self._world.scores()[self._agent_id] - score_before
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info


register(
    id="scml/OneShot-v0",
    entry_point="scml.oneshot.rl.env:GridWorldEnv",
    max_episode_steps=None,
)
