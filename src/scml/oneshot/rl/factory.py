import random
import warnings
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from attr import define

from scml.common import intin, make_array
from scml.oneshot.agent import OneShotAgent
from scml.oneshot.agents import OneShotDummyAgent
from scml.oneshot.awi import OneShotAWI
from scml.oneshot.world import (
    SCML2020OneShotWorld,
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
#     anac_config_factory_oneshot,
#     anac_oneshot_world_factory,
# )


__all__ = [
    "OneShotWorldFactory",
    # "ANACOneShotFactory",
    "FixedPartnerNumbersOneShotFactory",
    "LimitedPartnerNumbersOneShotFactory",
    "ANACOneShotFactory",
]


DefaultAgentsOneShot2023 = [
    GreedyOneShotAgent,
    SingleAgreementAspirationAgent,
    SyncRandomOneShotAgent,
]


@define(frozen=True)
class OneShotWorldFactory(WorldFactory, ABC):
    """A factory that generates oneshot worlds with a single agent of type `agent_type` with predetermined structure and settings"""

    @abstractmethod
    def make(
        self,
        types: tuple[type[OneShotAgent], ...] = (OneShotDummyAgent,),
        params: tuple[dict[str, Any], ...] | None = None,
    ) -> SCML2020OneShotWorld:
        """Generates the oneshot world and assigns an agent of type `agent_type` to it"""
        ...

    def __call__(
        self,
        types: tuple[type[OneShotAgent], ...] = (OneShotDummyAgent,),
        params: tuple[dict[str, Any], ...] | None = None,
    ) -> tuple[SCML2020OneShotWorld, tuple[OneShotAgent]]:
        """Generates the world and assigns an agent to it"""
        world = self.make(types, params)
        agents = []
        if types:
            expected_types = [type._type_name() for type in types]
            expected_set = set(expected_types)
            agents = [
                i
                for i, type_ in enumerate(world.agent_types)
                if type_.split(":")[-1] in expected_set
            ]
            assert len(agents) == len(
                types
            ), f"Found the following agent of type {types=}: {agents=}"
            agents = []
            for expected_type in expected_types:
                for a in world.agents.values():
                    if a.type_name.split(":")[-1] == expected_type:
                        agents.append(a)
        return world, tuple(agents)


# @define(frozen=True)
# class ANACOneShotFactory(OneShotWorldFactory):
#     """Generates a oneshot world compatible with the settings of a given year in the ANAC competition"""
#
#     year: int = 2023
#
#     def make(self) -> SCML2023OneShotWorld:
#         """Generates a world"""
#         agent_type = self.agent_type
#         configs = anac_config_factory_oneshot(
#             self.year,
#             n_competitors=1,
#             n_agents_per_competitor=1,
#             one_offer_per_step=True,
#         )
#         assigned = anac_assigner_oneshot(
#             configs, 1, competitors=(agent_type,), params=None, fair=False  # type: ignore
#         )
#         assigned = assigned[0][0]
#         return anac_oneshot_world_factory(year=self.year, **assigned)
#
#     def is_valid_world(self, world: SCML2023OneShotWorld) -> bool:
#         """Checks that the given world could have been generated from this factory"""
#         warnings.warn(f"N. processes must be 2")
#         return world.n_processes == 2
#
#     def contains_factory(self, factory: WorldFactory) -> bool:
#         """Checks that the any world generated from the given `factory` could have been generated from this factory"""
#         if not isinstance(factory, ANACOneShotFactory):
#             warnings.warn(
#                 f"factories of type: {factory.__class__} are incompatible with this factory of type {self.__class__}"
#             )
#             return False
#         return self == factory


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

    def make(
        self,
        types: tuple[type[OneShotAgent], ...] = (OneShotDummyAgent,),
        params: tuple[dict[str, Any], ...] | None = None,
    ) -> SCML2020OneShotWorld:
        """Generates a world"""
        if types and params is None:
            params = tuple(dict() for _ in types)
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
        n_agents_per_process[my_level] = max(
            len(types), n_agents_per_process[my_level], n_competitors
        )

        n_agents = sum(n_agents_per_process)
        agent_types = list(random.choices(self.non_competitors, k=n_agents))
        agent_params = None
        if params:
            agent_params: list[dict[str, Any]] | None = [dict() for _ in agent_types]
        agent_processes = np.zeros(n_agents, dtype=int)
        nxt, indx = 0, -1
        for level in range(n_processes):
            last = nxt + n_agents_per_process[level]
            agent_processes[nxt:last] = level
            if level == my_level:
                indices = random.sample(range(nxt, last), k=len(types))
                assert params is not None and agent_params is not None
                for indx, agent_type, p in zip(indices, types, params):
                    agent_types[indx] = agent_type
                    if params:
                        agent_params[indx]["controller_params"] = p
            nxt += n_agents_per_process[level]
        assert indx >= 0
        type_ = {
            2023: SCML2023OneShotWorld,
            2022: SCML2022OneShotWorld,
            2021: SCML2021OneShotWorld,
            2020: SCML2020OneShotWorld,
        }[self.year]
        return type_(
            **type_.generate(
                n_lines=n_lines,
                agent_types=agent_types,
                agent_params=agent_params,
                agent_processes=agent_processes.tolist(),
                n_processes=n_processes,
                random_agent_types=False,
            ),
            one_offer_per_step=True,
        )

    def is_valid_world(
        self,
        world: SCML2020OneShotWorld,
        types: tuple[type[OneShotAgent], ...] = (OneShotDummyAgent,),
    ) -> bool:
        """Checks that the given world could have been generated from this factory"""
        for agent_type in types:
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
                warnings.warn(
                    f"Invalid n_lines: {agent.awi.n_lines=} != {self.n_lines=}"
                )
                return False
        # TODO: check non-competitor types
        return self.is_valid_awi(agent.awi)  # type: ignore

    def is_valid_awi(self, awi: OneShotAWI) -> bool:
        # find my level
        my_level = awi.n_processes - 1 if self.level < 0 else self.level
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
        elif my_level == awi.n_processes - 1:
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

    def contains_factory(self, factory: WorldFactory) -> bool:
        """Checks that the any world generated from the given `factory` could have been generated from this factory"""
        if not isinstance(factory, self.__class__):
            return False
        if not isin(factory.year, self.year):
            return False
        if not isin(factory.n_processes, self.n_processes):
            return False
        if not isin(factory.level, self.level):
            return False
        if not isin(factory.n_consumers, self.n_consumers):
            return False
        if not isin(factory.n_suppliers, self.n_suppliers):
            return False
        if not isin(factory.n_competitors, self.n_competitors):
            return False
        if not isin(factory.n_agents_per_level, self.n_agents_per_level):
            return False
        if not isin(factory.n_lines, self.n_lines):
            return False
        if set(factory.non_competitors).difference(list(self.non_competitors)):
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
    n_lines: tuple[int, int] | int = (10, 10)
    non_competitors: list[str | type[OneShotAgent]] = DefaultAgentsOneShot2023

    def __attrs_post_init__(self):
        assert self.level != 0 or self.n_suppliers == (0, 0)
        assert self.level != -1 or self.n_consumers == (0, 0)
        assert not (self.level > 0 and self.level < self.n_processes[-1] - 1) or (
            self.n_suppliers[-1] > 0 and self.n_consumers[-1] > 0
        )
        assert self.level == -1 or self.level < self.n_processes[-1]

    def make(
        self,
        types: tuple[type[OneShotAgent], ...] = (OneShotDummyAgent,),
        params: tuple[dict[str, Any], ...] | None = None,
    ) -> SCML2020OneShotWorld:
        """Generates a world"""
        if types and params is None:
            params = tuple(dict() for _ in types)
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
        n_agents_per_process[my_level] = max(
            len(types), n_agents_per_process[my_level], n_competitors
        )

        n_agents = sum(n_agents_per_process)
        agent_types = list(random.choices(self.non_competitors, k=n_agents))
        agent_params = None
        if params:
            agent_params: list[dict[str, Any]] | None = [dict() for _ in agent_types]
        agent_processes = np.zeros(n_agents, dtype=int)
        nxt, indx = 0, -1
        for level in range(n_processes):
            last = nxt + n_agents_per_process[level]
            agent_processes[nxt:last] = level
            if level == my_level:
                indices = random.sample(range(nxt, last), k=len(types))
                assert params is not None and agent_params is not None
                for indx, agent_type, p in zip(indices, types, params):
                    agent_types[indx] = agent_type
                    if params:
                        agent_params[indx]["controller_params"] = p
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
                agent_params=agent_params,
                agent_processes=agent_processes.tolist(),
                n_processes=n_processes,
                random_agent_types=False,
            ),
            one_offer_per_step=True,
        )

    def is_valid_world(
        self,
        world: SCML2020OneShotWorld,
        types: tuple[type[OneShotAgent], ...] = (OneShotDummyAgent,),
    ) -> bool:
        """Checks that the given world could have been generated from this factory"""
        for agent_type in types:
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
                warnings.warn(
                    f"Invalid n_lines: {agent.awi.n_lines=} != {self.n_lines=}"
                )
                return False
            # TODO: check non-competitor types
        return self.is_valid_awi(agent.awi)  # type: ignore

    def is_valid_awi(self, awi: OneShotAWI) -> bool:
        # find my level
        n_processes = awi.n_processes
        my_level = n_processes - 1 if self.level < 0 else self.level
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

    def contains_factory(self, factory: WorldFactory) -> bool:
        """Checks that the any world generated from the given `factory` could have been generated from this factory"""
        if not isinstance(factory, self.__class__):
            return False
        if not isin(factory.year, self.year):
            return False
        if not isin(factory.n_processes, self.n_processes):
            return False
        if not isin(factory.level, self.level):
            return False
        if not isin(factory.n_consumers, self.n_consumers):
            return False
        if not isin(factory.n_suppliers, self.n_suppliers):
            return False
        if not isin(factory.n_competitors, self.n_competitors):
            return False
        if not isin(factory.n_agents_per_level, self.n_agents_per_level):
            return False
        if not isin(factory.n_lines, self.n_lines):
            return False
        if set(factory.non_competitors).difference(list(self.non_competitors)):
            return False
        return True


@define(frozen=True)
class ANACOneShotFactory(OneShotWorldFactory):
    """Generates a oneshot world with no constraints except compatibility with a specific ANAC competition year."""

    year: int = 2023

    def make(
        self,
        types: tuple[type[OneShotAgent], ...] = (OneShotDummyAgent,),
        params: tuple[dict[str, Any], ...] | None = None,
    ) -> SCML2020OneShotWorld:
        """Generates a world"""
        type_ = {
            2023: SCML2023OneShotWorld,
            2022: SCML2022OneShotWorld,
            2021: SCML2021OneShotWorld,
        }[self.year]
        d = type_.generate()
        if types:
            if params is None:
                params = tuple(dict() for _ in types)
            n = len(d["agent_types"])
            indices = random.sample(range(n), k=len(types))
            for i, agent_type, p in zip(indices, types, params):
                d["agent_params"][i].update(
                    dict(
                        controller_type=agent_type,
                        controller_params=p,
                    )
                )

        return type_(**d, one_offer_per_step=True)

    def is_valid_world(
        self,
        world: SCML2020OneShotWorld,
        types: tuple[type[OneShotAgent], ...] = (OneShotDummyAgent,),
    ) -> bool:
        """Checks that the given world could have been generated from this factory"""
        for agent_type in types:
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
        return self.is_valid_awi(agent.awi)  # type: ignore

    def is_valid_awi(self, awi: OneShotAWI) -> bool:
        return isinstance(awi, OneShotAWI)

    def contains_factory(self, factory: WorldFactory) -> bool:
        """Checks that the any world generated from the given `factory` could have been generated from this factory"""
        return isinstance(factory, WorldFactory)
