import random
import warnings
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from attr import define
from negmas import Agent, World

from scml.common import intin, make_array
from scml.oneshot.agent import OneShotAgent
from scml.oneshot.rl.agent import OneShotDummyAgent
from scml.oneshot.world import (
    SCML2021OneShotWorld,
    SCML2022OneShotWorld,
    SCML2023OneShotWorld,
)

from ..agents import (
    GreedyOneShotAgent,
    SingleAgreementAspirationAgent,
    SyncRandomOneShotAgent,
)
from .common import WorldFactory, isin

# from scml.utils import (
#     anac_assigner_oneshot,
#     anac_config_generator_oneshot,
#     anac_oneshot_world_generator,
# )


__all__ = [
    "OneShotWorldFactory",
    # "ANACOneShotFactory",
    "FixedPartnerNumbersOneShotFactory",
    "LimitedPartnerNumbersOneShotFactory",
]


DefaultAgentsOneShot2023 = [
    GreedyOneShotAgent,
    SingleAgreementAspirationAgent,
    SyncRandomOneShotAgent,
]


@define(frozen=True)
class OneShotWorldFactory(WorldFactory, ABC):
    agent_type: type[OneShotAgent] = OneShotDummyAgent

    @abstractmethod
    def make(self) -> SCML2023OneShotWorld:
        """Generates the oneshot world and assigns an agent of type `agent_type` to it"""
        ...

    def __call__(self) -> tuple[SCML2023OneShotWorld, OneShotAgent]:
        """Generates the world and assigns an agent to it"""
        agent_type = self.agent_type
        world = self.make()  # type: ignore
        world: SCML2023OneShotWorld
        expected_type = agent_type._type_name()
        agents = [
            i
            for i, type_ in enumerate(world.agent_types)
            if type_.split(":")[-1] == expected_type
        ]
        assert (
            len(agents) == 1
        ), f"Found the following agent of type {agent_type}: {agents}"
        for a in world.agents.values():
            if a.type_name.split(":")[-1] == expected_type:
                return world, a  # type: ignore
        raise RuntimeError(f"Cannot find a world of type {expected_type}")


# @define(frozen=True)
# class ANACOneShotFactory(OneShotWorldFactory):
#     """Generates a oneshot world compatible with the settings of a given year in the ANAC competition"""
#
#     year: int = 2023
#
#     def make(self) -> SCML2023OneShotWorld:
#         """Generates a world"""
#         agent_type = self.agent_type
#         configs = anac_config_generator_oneshot(
#             self.year,
#             n_competitors=1,
#             n_agents_per_competitor=1,
#             one_offer_per_step=True,
#         )
#         assigned = anac_assigner_oneshot(
#             configs, 1, competitors=(agent_type,), params=None, fair=False  # type: ignore
#         )
#         assigned = assigned[0][0]
#         return anac_oneshot_world_generator(year=self.year, **assigned)
#
#     def is_valid_world(self, world: SCML2023OneShotWorld) -> bool:
#         """Checks that the given world could have been generated from this generator"""
#         warnings.warn(f"N. processes must be 2")
#         return world.n_processes == 2
#
#     def contains_factory(self, generator: WorldFactory) -> bool:
#         """Checks that the any world generated from the given `generator` could have been generated from this generator"""
#         if not isinstance(generator, ANACOneShotFactory):
#             warnings.warn(
#                 f"factories of type: {generator.__class__} are incompatible with this factory of type {self.__class__}"
#             )
#             return False
#         return self == generator


@define(frozen=True)
class FixedPartnerNumbersOneShotFactory(OneShotWorldFactory):
    """Generates a oneshot world fixing the agent level, production capacity and the number of suppliers, consumers, and optionally same-level competitors."""

    year: int = 2023
    level: int = 0
    n_processes: int = 2
    n_consumers: int = 4
    n_suppliers: int = 0
    n_competitors: tuple[int, int] | int = (3, 7)
    n_agents_per_level: tuple[int, int] | int = (4, 8)
    n_lines: tuple[int, int] | int = 10
    non_competitors: list[str | type[OneShotAgent]] = DefaultAgentsOneShot2023

    def __attrs_post_init__(self):
        assert self.level != 0 or self.n_suppliers == 0
        assert (
            not isinstance(self.n_processes, int)
            or (self.level != self.n_processes - 1 and self.level != -1)
            or self.n_consumers == 0
        )
        if isinstance(self.n_processes, tuple):
            assert not (self.level > 0 and self.level < self.n_processes[-1] - 1) or (
                self.n_suppliers > 0 and self.n_consumers > 0
            )
            assert self.level == -1 or self.level < self.n_processes[-1]
        else:
            assert not (self.level > 0 and self.level < self.n_processes - 1) or (
                self.n_suppliers > 0 and self.n_consumers > 0
            )
            assert self.level == -1 or self.level < self.n_processes

    def make(self) -> SCML2023OneShotWorld:
        """Generates a world"""
        agent_type = self.agent_type
        n_processes = intin(self.n_processes)
        n_lines = intin(self.n_lines)
        # find my level
        my_level = n_processes - 1 if self.level < 0 else self.level

        # distribute agents over production levels (processes)
        n_agents_per_process = make_array(
            self.n_agents_per_level, n_processes, dtype=int
        )
        # override the number of consumers and number of suppliers to match my choice
        if my_level == 0:
            n_agents_per_process[1] = self.n_consumers
        elif my_level == n_processes - 1:
            n_agents_per_process[n_processes - 2] = self.n_suppliers
        else:
            n_agents_per_process[my_level + 1] = self.n_consumers
            n_agents_per_process[my_level - 1] = self.n_suppliers
        n_competitors = intin(self.n_competitors) + 1
        n_agents_per_process[my_level] = n_competitors

        n_agents = sum(n_agents_per_process)
        agent_types = list(random.choices(self.non_competitors, k=n_agents))
        agent_processes = np.zeros(n_agents, dtype=int)
        nxt, indx = 0, -1
        for level in range(n_processes):
            last = nxt + n_agents_per_process[level]
            agent_processes[nxt:last] = level
            if level == my_level:
                indx = random.randint(nxt, last)
                agent_types[indx] = agent_type  # type: ignore
            nxt += n_agents_per_process[level]
        assert indx >= 0
        type_ = {
            2023: SCML2023OneShotWorld,
            2022: SCML2022OneShotWorld,
            2021: SCML2021OneShotWorld,
        }[self.year]
        return type_(
            **type_.generate(
                n_lines=n_lines,
                agent_types=agent_types,
                agent_processes=agent_processes.tolist(),
                n_processes=n_processes,
                random_agent_types=False,
            ),
            one_offer_per_step=True,
        )

    def is_valid_world(self, world: SCML2023OneShotWorld) -> bool:
        """Checks that the given world could have been generated from this generator"""
        agent_type = self.agent_type
        expected_type = agent_type._type_name()
        agents = [
            i
            for i, type_ in enumerate(world.agent_types)
            if type_.split(":")[-1] == expected_type
        ]
        assert (
            len(agents) == 1
        ), f"Found the following agent of type {agent_type}: {agents}"
        agent: OneShotAgent = None  # type: ignore
        for a in world.agents.values():
            if a.type_name.split(":")[-1] == expected_type:
                agent = a  # type: ignore
                break
        else:
            warnings.warn(f"cannot find any agent of type {expected_type}")
            return False
        if not isin(world.n_processes, self.n_processes):
            warnings.warn(
                f"Invalid n_processes: {world.n_processes=} != {self.n_processes=}"
            )
            return False
        if not isin(agent.awi.n_lines, self.n_lines):
            warnings.warn(f"Invalid n_lines: {agent.awi.n_lines=} != {self.n_lines=}")
            return False
        # TODO: check non-competitor types
        # find my level
        n_processes = world.n_processes
        my_level = n_processes - 1 if self.level < 0 else self.level
        awi: OneShotAWI = agent.awi  # type: ignore
        n_partners = self.n_consumers + self.n_suppliers
        if not isin(len(awi.my_partners), n_partners):
            warnings.warn(
                f"Invalid n_partners: {len(awi.my_partners)=} != {n_partners=}"
            )
            return False
        if my_level == 0:
            if not isin(len(awi.my_consumers), self.n_consumers):
                warnings.warn(
                    f"Invalid n_consumers: {len(awi.my_consumers)=} != {self.n_consumers=}"
                )
                return False
            if len(awi.my_suppliers) != 1:
                warnings.warn(f"Invalid n_suppliers: {len(awi.my_suppliers)=} != 1")
                return False
        elif my_level == n_processes - 1:
            if not isin(len(awi.my_suppliers), self.n_suppliers):
                warnings.warn(
                    f"Invalid n_suppliers: {len(awi.my_suppliers)=} != {self.n_suppliers=}"
                )
                return False
            if len(awi.my_consumers) != 1:
                warnings.warn(f"Invalid n_conumsers: {len(awi.my_consumers)=} != 1")
                return False
        else:
            if not isin(len(awi.my_suppliers), self.n_suppliers):
                warnings.warn(
                    f"Invalid n_suppliers: {len(awi.my_suppliers)=} != {self.n_suppliers=}"
                )
                return False
            if not isin(len(awi.my_consumers), self.n_consumers):
                warnings.warn(
                    f"Invalid n_consumers: {len(awi.my_consumers)=} != {self.n_consumers=}"
                )
                return False
        n_competitors = awi.n_competitors
        if not isin(n_competitors, self.n_competitors):
            warnings.warn(
                f"Invalid n_competitors: {awi.n_competitors=} != {n_competitors=}"
            )
            return False
        return True

    def contains_factory(self, generator: WorldFactory) -> bool:
        """Checks that the any world generated from the given `generator` could have been generated from this generator"""
        if not isinstance(generator, self.__class__):
            return False
        if not isin(generator.year, self.year):
            return False
        if not isin(generator.n_processes, self.n_processes):
            return False
        if not isin(generator.level, self.level):
            return False
        if not isin(generator.n_consumers, self.n_consumers):
            return False
        if not isin(generator.n_suppliers, self.n_suppliers):
            return False
        if not isin(generator.n_competitors, self.n_competitors):
            return False
        if not isin(generator.n_agents_per_level, self.n_agents_per_level):
            return False
        if not isin(generator.n_lines, self.n_lines):
            return False
        if set(generator.non_competitors).difference(
            list(self.non_competitors) + [self.agent_type]
        ):
            return False
        return True


@define(frozen=True)
class LimitedPartnerNumbersOneShotFactory(OneShotWorldFactory):
    """Generates a oneshot world limiting the range of the agent level, production capacity
    and the number of suppliers, consumers, and optionally same-level competitors."""

    year: int = 2023
    level: int = 0
    n_processes: tuple[int, int] = (2, 2)
    n_consumers: tuple[int, int] = (4, 8)
    n_suppliers: tuple[int, int] = (0, 0)
    n_competitors: tuple[int, int] = (3, 7)
    n_agents_per_level: tuple[int, int] = (4, 8)
    n_lines: tuple[int, int] | int = 10
    non_competitors: list[str | type[OneShotAgent]] = DefaultAgentsOneShot2023

    def __attrs_post_init__(self):
        assert self.level != 0 or self.n_suppliers == (0, 0)
        assert self.level != -1 or self.n_consumers == (0, 0)
        assert not (self.level > 0 and self.level < self.n_processes[-1] - 1) or (
            self.n_suppliers[-1] > 0 and self.n_consumers[-1] > 0
        )
        assert self.level == -1 or self.level < self.n_processes[-1]

    def make(self) -> SCML2023OneShotWorld:
        """Generates a world"""
        agent_type = self.agent_type
        n_processes = intin(self.n_processes)
        n_lines = intin(self.n_lines)
        # find my level
        my_level = n_processes - 1 if self.level < 0 else self.level
        n_suppliers = intin(self.n_suppliers)
        n_consumers = intin(self.n_consumers)
        n_competitors = intin(self.n_competitors)
        n_lines = intin(self.n_lines)
        n_agents_per_level = self.n_agents_per_level

        # distribute agents over production levels (processes)
        n_agents_per_process = make_array(n_agents_per_level, n_processes, dtype=int)
        # override the number of consumers and number of suppliers to match my choice
        if my_level == 0:
            n_agents_per_process[1] = n_consumers
        elif my_level == n_processes - 1:
            n_agents_per_process[n_processes - 2] = n_suppliers
        else:
            n_agents_per_process[my_level + 1] = n_consumers
            n_agents_per_process[my_level - 1] = n_suppliers
        n_competitors = intin(n_competitors) + 1
        n_agents_per_process[my_level] = n_competitors

        n_agents = sum(n_agents_per_process)
        agent_types = list(random.choices(self.non_competitors, k=n_agents))
        agent_processes = np.zeros(n_agents, dtype=int)
        nxt, indx = 0, -1
        for level in range(n_processes):
            last = nxt + n_agents_per_process[level]
            agent_processes[nxt:last] = level
            if level == my_level:
                indx = random.randint(nxt, last)
                agent_types[indx] = agent_type  # type: ignore
            nxt += n_agents_per_process[level]
        assert indx >= 0
        type_ = {
            2023: SCML2023OneShotWorld,
            2022: SCML2022OneShotWorld,
            2021: SCML2021OneShotWorld,
        }[self.year]
        return type_(
            **type_.generate(
                n_lines=n_lines,
                agent_types=agent_types,
                agent_processes=agent_processes.tolist(),
                n_processes=n_processes,
                random_agent_types=False,
            ),
            one_offer_per_step=True,
        )

    def is_valid_world(self, world: SCML2023OneShotWorld) -> bool:
        """Checks that the given world could have been generated from this generator"""
        agent_type = self.agent_type
        expected_type = agent_type._type_name()
        agents = [
            i
            for i, type_ in enumerate(world.agent_types)
            if type_.split(":")[-1] == expected_type
        ]
        assert (
            len(agents) == 1
        ), f"Found the following agent of type {agent_type}: {agents}"
        agent: OneShotAgent = None  # type: ignore
        my_level = -1
        for a in world.agents.values():
            if a.type_name.split(":")[-1] == expected_type:
                agent, my_level = a, a.awi.level  # type: ignore
                break
        else:
            warnings.warn(f"cannot find any agent of type {expected_type}")
            return False
        if not isin(world.n_processes, self.n_processes):
            warnings.warn(
                f"Invalid n_processes: {world.n_processes=} != {self.n_processes=}"
            )
            return False
        if not isin(agent.awi.n_lines, self.n_lines):
            warnings.warn(f"Invalid n_lines: {agent.awi.n_lines=} != {self.n_lines=}")
            return False
        # TODO: check non-competitor types
        # find my level
        n_processes = world.n_processes
        awi: OneShotAWI = agent.awi  # type: ignore
        n_partners = (
            self.n_consumers[0] + self.n_suppliers[0],
            self.n_consumers[-1] + self.n_suppliers[-1],
        )
        if not isin(len(awi.my_partners), n_partners):
            warnings.warn(
                f"Invalid n_partners: {len(awi.my_partners)=} != {n_partners=}"
            )
            return False
        if my_level == 0:
            if not isin(len(awi.my_consumers), self.n_consumers):
                warnings.warn(
                    f"Invalid n_consumers: {len(awi.my_consumers)=} != {self.n_consumers=}"
                )
                return False
            if len(awi.my_suppliers) != 1:
                warnings.warn(f"Invalid n_suppliers: {len(awi.my_suppliers)=} != 1")
                return False
        elif my_level == n_processes - 1:
            if not isin(len(awi.my_suppliers), self.n_suppliers):
                warnings.warn(
                    f"Invalid n_suppliers: {len(awi.my_suppliers)=} != {self.n_suppliers=}"
                )
                return False
            if len(awi.my_consumers) != 1:
                warnings.warn(f"Invalid n_conumsers: {len(awi.my_consumers)=} != 1")
                return False
        else:
            if not isin(len(awi.my_suppliers), self.n_suppliers):
                warnings.warn(
                    f"Invalid n_suppliers: {len(awi.my_suppliers)=} != {self.n_suppliers=}"
                )
                return False
            if not isin(len(awi.my_consumers), self.n_consumers):
                warnings.warn(
                    f"Invalid n_consumers: {len(awi.my_consumers)=} != {self.n_consumers=}"
                )
                return False
        n_competitors = awi.n_competitors
        if not isin(n_competitors, n_competitors):
            warnings.warn(
                f"Invalid n_competitors: {awi.n_competitors=} != {n_competitors=}"
            )
            return False
        return True

    def contains_factory(self, generator: WorldFactory) -> bool:
        """Checks that the any world generated from the given `generator` could have been generated from this generator"""
        if not isinstance(generator, self.__class__):
            return False
        if not isin(generator.year, self.year):
            return False
        if not isin(generator.n_processes, self.n_processes):
            return False
        if not isin(generator.level, self.level):
            return False
        if not isin(generator.n_consumers, self.n_consumers):
            return False
        if not isin(generator.n_suppliers, self.n_suppliers):
            return False
        if not isin(generator.n_competitors, self.n_competitors):
            return False
        if not isin(generator.n_agents_per_level, self.n_agents_per_level):
            return False
        if not isin(generator.n_lines, self.n_lines):
            return False
        if set(generator.non_competitors).difference(
            list(self.non_competitors) + [self.agent_type]
        ):
            return False
        return True
