import random
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import Any, Iterable, Union, Sequence

import numpy as np
from attr import define, field
from scml.oneshot.agents.greedy import GreedyOneShotAgent
from scml.oneshot.agents.rand import (
    EqualDistOneShotAgent,
    NiceAgent,
    RandDistOneShotAgent,
)
from negmas.helpers.strings import unique_name

from scml.common import intin, isin, isinclass, isinfloat, isinobject, make_array
from scml.oneshot.agent import OneShotAgent
from scml.oneshot.agents import Placeholder
from scml.oneshot.awi import OneShotAWI
from scml.oneshot.common import is_system_agent
from scml.oneshot.world import (
    SCML2021OneShotWorld,
    SCML2022OneShotWorld,
    SCML2023OneShotWorld,
    SCML2024OneShotWorld,
    SCMLBaseWorld,
    get_n_agents_per_process,
    get_n_lines,
)

__all__ = [
    "Context",
    "GeneralContext",
    "ANACContext",
    "LimitedPartnerNumbersContext",
    "FixedPartnerNumbersContext",
    "ANACOneShotContext",
    "LimitedPartnerNumbersOneShotContext",
    "FixedPartnerNumbersOneShotContext",
    "SupplierContext",
    "ConsumerContext",
    "StrongSupplierContext",
    "StrongConsumerContext",
    "WeakSupplierContext",
    "WeakConsumerContext",
    "BalancedSupplierContext",
    "BalancedConsumerContext",
    "RepeatingContext",
    "ContextParams",
    "MonopolicContext",
    "SingleAgentPerLevelSupplierContext",
    "EutopiaContext",
    "EutopiaConsumerContext",
    "EutopiaSupplierContext",
]


DefaultAgentsOneShot = (
    GreedyOneShotAgent,
    RandDistOneShotAgent,
    EqualDistOneShotAgent,
)


@define
class ContextParams:
    """Basic Parameters you can assume about a context. Returned by `extract_context_params`"""

    perishable: bool
    nlines: int
    nsuppliers: int
    nconsumers: int


class Strength(Enum):
    Weak = -1
    Balanced = 0
    Strong = 1


def sample_with_strength(
    c: int | tuple[int, int], n: int | tuple[int, int], s: Strength | None
):
    if s is None:
        c = intin(c)
        n = intin(n)
        return c, n
    mnc, mxc = c if isinstance(c, Iterable) else (c, c)
    if s == Strength.Balanced:
        c = intin((mnc + 1, mxc - 1))
        mn, mx = c, c + 2
    elif s == Strength.Strong:
        c = intin((mnc + 1, mxc))
        mn = n[0] if isinstance(n, Iterable) else n
        mx = max(mn, c - 1)
    else:
        c = intin((mnc, mxc - 1))
        mx = n[1] if isinstance(n, Iterable) else n
        mn = min(c + 2, mx)
    return c, intin((mn, mx))


def safemax(x: Iterable | int):
    if isinstance(x, Iterable):
        return max(x)
    return x


def safemin(x: Iterable | int):
    if isinstance(x, Iterable):
        return min(x)
    return x


class Context(ABC):
    """A context used for generating worlds satisfying predefined conditions and testing for them"""

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    @abstractmethod
    def generate(
        self,
        types: tuple[type[OneShotAgent], ...] | None = None,
        params: tuple[dict[str, Any], ...] | None = None,
        name: str | None = None,
    ) -> tuple[SCMLBaseWorld, tuple[OneShotAgent]]:
        """
        Generates a world with one or more agents to be controlled externally and returns both

        Args:
            agent_types: The types of a list of agents to be guaranteed to exist in the world
            agent_params: The parameters to pass to the constructors of these agents. None means no parameters for any agents
            name: The name of the worlds to generate. Uses a random name if not given

        Returns:
            The constructed world and a tuple of the agents created corresponding (in order) to the given agent types/params
        """
        ...

    @abstractmethod
    def is_valid_world(
        self,
        world: SCMLBaseWorld,
    ) -> bool:
        """Checks that the given world could have been generated from this context"""
        ...

    @abstractmethod
    def is_valid_awi(self, awi: OneShotAWI) -> bool:
        """Checks that the given AWI is connected to a world that could have been generated from this context"""
        ...

    @abstractmethod
    def contains_context(self, context: "Context") -> bool:
        """Checks that the any world generated from the given `context` could have been generated from this context"""
        ...

    def __contains__(self, other: "Union[SCMLBaseWorld, OneShotAWI, Context]") -> bool:
        if isinstance(other, Context):
            return self.contains_context(other)
        if isinstance(other, OneShotAWI):
            return self.is_valid_awi(other)
        return self.is_valid_world(other)


N_SUPPLIERS = (4, 8)
"""Numbers of suppliers supported"""
N_CONSUMERS = (4, 8)
"""Numbers of consumers supported"""
NTESTS = 20
DEFAULT_PLACEHOLDER_AGENT_TYPES = (Placeholder,)

WARN_ON_FAILURE = True
RAISE_ON_FAILURE = False


def _is(
    condition: bool,
    raise_on_failure=RAISE_ON_FAILURE,
    warn_on_failure=WARN_ON_FAILURE,
    message: str = "",
) -> bool:
    if not condition:
        return False
    if raise_on_failure:
        raise AssertionError(message)
    if warn_on_failure:
        warnings.warn(message)
    return True


def _not(
    condition: bool,
    raise_on_failure=RAISE_ON_FAILURE,
    warn_on_failure=WARN_ON_FAILURE,
    message: str = "",
) -> bool:
    if condition:
        return False
    if raise_on_failure:
        raise AssertionError(message)
    if warn_on_failure:
        warnings.warn(message)
    return True


def _safeget(d: dict[str, dict[str, Any]], x: str, y: str):
    return d.get(x, dict()).get(y, None)


def _world_matches_config(
    world: SCMLBaseWorld,
    config: dict[str, Any],
    expected_types: Iterable[type[OneShotAgent] | str] | None,
    expected_world_type: type[SCMLBaseWorld] | None = None,
    raise_on_failure: bool = False,
    warn_on_failure: bool = False,
):
    if _is(
        world.perishable != _safeget(config, "info", "perishable"),
        raise_on_failure,
        warn_on_failure,
        f'{world.perishable=} != f{_safeget(config, "info", "perishable")=}',
    ):
        return False
    if _not(
        isin(world.n_steps, _safeget(config, "info", "n_steps")),
        raise_on_failure,
        warn_on_failure,
        f'not isin({world.n_steps=}, {_safeget(config, "info", "n_steps")=})',
    ):
        return False
    if _not(
        isin(world.n_processes, _safeget(config, "info", "n_processes")),
        raise_on_failure,
        warn_on_failure,
        "not isin(world.n_processes, self.n_processes)",
    ):
        return False
    if _not(
        isin(world.info["n_lines"], _safeget(config, "info", "n_lines")),
        raise_on_failure,
        warn_on_failure,
        'not isin(world.info["n_lines"], _safeget(config,"info","n_lines"))',
    ):
        return False
    if _not(
        all(
            isin(_, _safeget(config, "info", "n_agents_per_process"))
            for _ in world.info["n_agents_per_process"]
        ),
        raise_on_failure,
        warn_on_failure,
        "not all( isin(_, self.n_agents_per_process) for _ in world.info['n_agents_per_process'])",
    ):
        return False
    if _not(
        isin(
            world.info["process_inputs_generator"],
            _safeget(config, "info", "process_inputs"),
        ),
        raise_on_failure,
        warn_on_failure,
        "not isin(world.info['process_inputs_generator'], self.process_inputs)",
    ):
        return False
    if _not(
        isin(
            world.info["process_outputs_generator"],
            _safeget(config, "info", "process_outputs"),
        ),
        raise_on_failure,
        warn_on_failure,
        "not isin(world.info['process_outputs_generator'], self.process_outputs)",
    ):
        return False
    if _not(
        isin(
            world.info["production_costs"],
            _safeget(config, "info", "production_costs"),
        ),
        raise_on_failure,
        warn_on_failure,
        "not isin(world.info['production_costs'], self.production_costs)",
    ):
        return False
    if _not(
        isinfloat(world.info["profit_means"], _safeget(config, "info", "profit_means")),
        raise_on_failure,
        warn_on_failure,
        "not isinfloat(world.info['profit_means'], self.profit_means)",
    ):
        return False
    if _not(
        isinfloat(
            world.info["profit_stddevs"],
            _safeget(config, "info", "profit_stddevs"),
        ),
        raise_on_failure,
        warn_on_failure,
        "not isinfloat(world.info['profit_stddevs'], self.profit_stddevs)",
    ):
        return False
    if _not(
        isinfloat(
            world.info["max_productivity"],
            _safeget(config, "info", "max_productivity"),
        ),
        raise_on_failure,
        warn_on_failure,
        "not isinfloat(world.info['max_productivity'], self.max_productivity)",
    ):
        return False
    if _is(
        _safeget(config, "info", "initial_balance") is not None
        and not isin(
            world.info["initial_balance"],
            _safeget(config, "info", "initial_balance"),
        ),
        raise_on_failure,
        warn_on_failure,
        "self.initial_balance is not None and not isin(world.info['initial_balance'], self.initial_balance)",
    ):
        return False
    if _not(
        isinfloat(
            world.info["exogenous_supply_predictability"],
            _safeget(config, "info", "exogenous_supply_predictability"),
        ),
        raise_on_failure,
        warn_on_failure,
        "not isinfloat( world.info['exogenous_supply_predictability'], self.exogenous_supply_predictability,)",
    ):
        return False
    if _not(
        isinfloat(
            world.info["exogenous_sales_predictability"],
            _safeget(config, "info", "exogenous_sales_predictability"),
        ),
        raise_on_failure,
        warn_on_failure,
        "not isinfloat( world.info['exogenous_sales_predictability'], self.exogenous_sales_predictability,)",
    ):
        return False
    if _not(
        isinfloat(
            world.info["exogenous_control"],
            _safeget(config, "info", "exogenous_control"),
        ),
        raise_on_failure,
        warn_on_failure,
        "not isinfloat(world.info['exogenous_control'], self.exogenous_control)",
    ):
        return False
    if _not(
        isinfloat(
            world.info["cash_availability"],
            _safeget(config, "info", "cash_availability"),
        ),
        raise_on_failure,
        warn_on_failure,
        "not isinfloat(world.info['cash_availability'], self.cash_availability)",
    ):
        return False
    if _not(
        isinfloat(
            world.info["shortfall_penalty"],
            _safeget(config, "info", "shortfall_penalty"),
        ),
        raise_on_failure,
        warn_on_failure,
        "not isinfloat(world.info['shortfall_penalty'], self.shortfall_penalty)",
    ):
        return False
    if _not(
        isinfloat(
            world.info["shortfall_penalty_dev"],
            _safeget(config, "info", "shortfall_penalty_dev"),
        ),
        raise_on_failure,
        warn_on_failure,
        "not isinfloat( world.info['shortfall_penalty_dev'], self.shortfall_penalty_dev)",
    ):
        return False
    if _not(
        isinfloat(
            world.info["disposal_cost"], _safeget(config, "info", "disposal_cost")
        ),
        raise_on_failure,
        warn_on_failure,
        "not isinfloat(world.info['disposal_cost'], self.disposal_cost)",
    ):
        return False
    if _not(
        isinfloat(
            world.info["disposal_cost_dev"],
            _safeget(config, "info", "disposal_cost_dev"),
        ),
        raise_on_failure,
        warn_on_failure,
        "not isinfloat(world.info['disposal_cost_dev'], self.disposal_cost_dev)",
    ):
        return False
    if _not(
        isinfloat(world.info["storage_cost"], _safeget(config, "info", "storage_cost")),
        raise_on_failure,
        warn_on_failure,
        "not isinfloat(world.info['storage_cost'], self.storage_cost)",
    ):
        return False
    if _not(
        isinfloat(
            world.info["storage_cost_dev"],
            _safeget(config, "info", "storage_cost_dev"),
        ),
        raise_on_failure,
        warn_on_failure,
        "not isinfloat(world.info['storage_cost_dev'], self.storage_cost_dev)",
    ):
        return False
    if _not(
        isinfloat(
            world.info["exogenous_price_dev"],
            _safeget(config, "info", "exogenous_price_dev"),
        ),
        raise_on_failure,
        warn_on_failure,
        "not isinfloat(world.info['exogenous_price_dev'], self.exogenous_price_dev)",
    ):
        return False
    if _not(
        isinfloat(
            world.info["price_multiplier"],
            _safeget(config, "info", "price_multiplier"),
        ),
        raise_on_failure,
        warn_on_failure,
        "not isinfloat(world.info['price_multiplier'], self.price_multiplier)",
    ):
        return False
    if _is(
        world.info["cost_increases_with_level"]
        != _safeget(config, "info", "cost_increases_with_level"),
        raise_on_failure,
        warn_on_failure,
        "world.info['cost_increases_with_level'] != self.cost_increases_with_level",
    ):
        return False
    if _is(
        world.info["equal_exogenous_supply"]
        != _safeget(config, "info", "equal_exogenous_supply"),
        raise_on_failure,
        warn_on_failure,
        "world.info['equal_exogenous_supply'] != self.equal_exogenous_supply",
    ):
        return False
    if _is(
        world.info["equal_exogenous_sales"]
        != _safeget(config, "info", "equal_exogenous_sales"),
        raise_on_failure,
        warn_on_failure,
        "world.info['equal_exogenous_sales'] != self.equal_exogenous_sales",
    ):
        return False
    if _is(
        world.info["cap_exogenous_quantities"]
        != _safeget(config, "info", "cap_exogenous_quantities"),
        raise_on_failure,
        warn_on_failure,
        "world.info['cap_exogenous_quantities'] != self.cap_exogenous_quantities",
    ):
        return False
    if _is(
        world.info["force_signing"] != _safeget(config, "info", "force_signing"),
        raise_on_failure,
        warn_on_failure,
        "world.info['force_signing'] != self.force_signing",
    ):
        return False
    if _is(
        world.info["random_agent_types"]
        != _safeget(config, "info", "random_agent_types"),
        raise_on_failure,
        warn_on_failure,
        "world.info['random_agent_types'] != self.random_agent_types",
    ):
        return False
    if _is(
        world.info["penalties_scale"] != _safeget(config, "info", "penalties_scale"),
        raise_on_failure,
        warn_on_failure,
        "world.info['penalties_scale'] != self.penalties_scale",
    ):
        return False
    if _is(
        world.info["exogenous_generation_method"]
        != _safeget(config, "info", "exogenous_generation_method"),
        raise_on_failure,
        warn_on_failure,
        "world.info['exogenous_generation_method'] != self.method",
    ):
        return False
    if expected_world_type and _not(
        isinstance(world, expected_world_type),
        raise_on_failure,
        warn_on_failure,
        "not isinstance(world, self.world_type)",
    ):
        return False

    if expected_types:
        world_agent_types = [
            type(_._obj)  # type: ignore
            for aid, _ in world.agents.items()
            if not is_system_agent(aid)
        ]
        if _not(
            isinclass(world_agent_types, list(expected_types)),
            raise_on_failure,
            warn_on_failure,
            f"not isinclass({world_agent_types=}, {list(expected_types)=})",
        ):
            return False
    return True


def _config_matches_base(
    config: dict[str, Any],
    base: dict[str, Any],
    raise_on_failure: bool,
    warn_on_failure: bool,
):
    if _is(
        _safeget(config, "info", "perishable") != _safeget(base, "info", "perishable"),
        raise_on_failure,
        warn_on_failure,
        f'{_safeget(config, "info", "perishable")=} != {_safeget(base, "info", "perishable")=}',
    ):
        return False
    if _not(
        isin(
            _safeget(config, "info", "n_steps"),
            _safeget(base, "info", "n_steps"),
        ),
        raise_on_failure,
        warn_on_failure,
        "not isin(config.n_steps, self.n_steps)",
    ):
        return False
    if _not(
        isin(
            _safeget(config, "info", "n_processes"),
            _safeget(base, "info", "n_processes"),
        ),
        raise_on_failure,
        warn_on_failure,
        "not isin(config.n_processes, self.n_processes)",
    ):
        return False
    if _not(
        isin(
            _safeget(config, "info", "n_lines"),
            _safeget(base, "info", "n_lines"),
        ),
        raise_on_failure,
        warn_on_failure,
        'not isin(_safeget(config,"info", dict())["n_lines"], config.get("info","n_lines"))',
    ):
        return False
    if _not(
        all(
            isin(_, _safeget(base, "info", "n_agents_per_process"))
            for _ in _safeget(config, "info", "n_agents_per_process")
        ),
        raise_on_failure,
        warn_on_failure,
        f"not all( isin(_, self.n_agents_per_process) for _ in config.get('info', dict())['n_agents_per_process'])\n"
        f'{_safeget(base, "info", "n_agents_per_process")=}\n {_safeget(config, "info", "n_agents_per_process")=}',
    ):
        return False
    if _not(
        isin(
            _safeget(config, "info", "process_inputs_generator"),
            _safeget(base, "info", "process_inputs"),
        ),
        raise_on_failure,
        warn_on_failure,
        "not isin(config.get('info', dict())['process_inputs_generator'], self.process_inputs)",
    ):
        return False
    if _not(
        isin(
            _safeget(config, "info", "process_outputs_generator"),
            _safeget(base, "info", "process_outputs"),
        ),
        raise_on_failure,
        warn_on_failure,
        "not isin(config.get('info', dict())['process_outputs_generator'], self.process_outputs)",
    ):
        return False
    if _not(
        isin(
            _safeget(config, "info", "production_costs"),
            _safeget(base, "info", "production_costs"),
        ),
        raise_on_failure,
        warn_on_failure,
        "not isin(config.get('info', dict())['production_costs'], self.production_costs)",
    ):
        return False
    if _not(
        isinfloat(
            _safeget(config, "info", "profit_means"),
            _safeget(base, "info", "profit_means"),
        ),
        raise_on_failure,
        warn_on_failure,
        "not isinfloat(config.get('info', dict())['profit_means'], self.profit_means)",
    ):
        return False
    if _not(
        isinfloat(
            _safeget(config, "info", "profit_stddevs"),
            _safeget(base, "info", "profit_stddevs"),
        ),
        raise_on_failure,
        warn_on_failure,
        "not isinfloat(config.get('info', dict())['profit_stddevs'], self.profit_stddevs)",
    ):
        return False
    if _not(
        isinfloat(
            _safeget(config, "info", "max_productivity"),
            _safeget(base, "info", "max_productivity"),
        ),
        raise_on_failure,
        warn_on_failure,
        "not isinfloat(config.get('info', dict())['max_productivity'], self.max_productivity)",
    ):
        return False
    if _is(
        _safeget(base, "info", "initial_balance") is None
        and not isin(
            _safeget(config, "info", "initial_balance"),
            _safeget(base, "info", "initial_balance"),
        ),
        raise_on_failure,
        warn_on_failure,
        "self.initial_balance is not None and not isin(config.get('info', dict())['initial_balance'], self.initial_balance)",
    ):
        return False
    if _not(
        isinfloat(
            _safeget(config, "info", "exogenous_supply_predictability"),
            _safeget(base, "info", "exogenous_supply_predictability"),
        ),
        raise_on_failure,
        warn_on_failure,
        "not isinfloat( config.get('info', dict())['exogenous_supply_predictability'], self.exogenous_supply_predictability,)",
    ):
        return False
    if _not(
        isinfloat(
            _safeget(config, "info", "exogenous_sales_predictability"),
            _safeget(base, "info", "exogenous_sales_predictability"),
        ),
        raise_on_failure,
        warn_on_failure,
        "not isinfloat( config.get('info', dict())['exogenous_sales_predictability'], self.exogenous_sales_predictability,)",
    ):
        return False
    if _not(
        isinfloat(
            _safeget(config, "info", "exogenous_control"),
            _safeget(base, "info", "exogenous_control"),
        ),
        raise_on_failure,
        warn_on_failure,
        "not isinfloat(config.get('info', dict())['exogenous_control'], self.exogenous_control)",
    ):
        return False
    if _not(
        isinfloat(
            _safeget(config, "info", "cash_availability"),
            _safeget(base, "info", "cash_availability"),
        ),
        raise_on_failure,
        warn_on_failure,
        "not isinfloat(config.get('info', dict())['cash_availability'], self.cash_availability)",
    ):
        return False
    if _not(
        isinfloat(
            _safeget(config, "info", "shortfall_penalty"),
            _safeget(base, "info", "shortfall_penalty"),
        ),
        raise_on_failure,
        warn_on_failure,
        "not isinfloat(config.get('info', dict())['shortfall_penalty'], self.shortfall_penalty)",
    ):
        return False
    if _not(
        isinfloat(
            _safeget(config, "info", "shortfall_penalty_dev"),
            _safeget(base, "info", "shortfall_penalty_dev"),
        ),
        raise_on_failure,
        warn_on_failure,
        "not isinfloat( config.get('info', dict())['shortfall_penalty_dev'], self.shortfall_penalty_dev)",
    ):
        return False
    if _not(
        isinfloat(
            _safeget(config, "info", "disposal_cost"),
            _safeget(base, "info", "disposal_cost"),
        ),
        raise_on_failure,
        warn_on_failure,
        "not isinfloat(config.get('info', dict())['disposal_cost'], self.disposal_cost)",
    ):
        return False
    if _not(
        isinfloat(
            _safeget(config, "info", "disposal_cost_dev"),
            _safeget(base, "info", "disposal_cost_dev"),
        ),
        raise_on_failure,
        warn_on_failure,
        "not isinfloat(config.get('info', dict())['disposal_cost_dev'], self.disposal_cost_dev)",
    ):
        return False
    if _not(
        isinfloat(
            _safeget(config, "info", "storage_cost"),
            _safeget(base, "info", "storage_cost"),
        ),
        raise_on_failure,
        warn_on_failure,
        "not isinfloat(config.get('info', dict())['storage_cost'], self.storage_cost)",
    ):
        return False
    if _not(
        isinfloat(
            _safeget(config, "info", "storage_cost_dev"),
            _safeget(base, "info", "storage_cost_dev"),
        ),
        raise_on_failure,
        warn_on_failure,
        "not isinfloat(config.get('info', dict())['storage_cost_dev'], self.storage_cost_dev)",
    ):
        return False
    if _not(
        isinfloat(
            _safeget(config, "info", "exogenous_price_dev"),
            _safeget(base, "info", "exogenous_price_dev"),
        ),
        raise_on_failure,
        warn_on_failure,
        "not isinfloat(config.get('info', dict())['exogenous_price_dev'], self.exogenous_price_dev)",
    ):
        return False
    if _not(
        isinfloat(
            _safeget(config, "info", "price_multiplier"),
            _safeget(base, "info", "price_multiplier"),
        ),
        raise_on_failure,
        warn_on_failure,
        "not isinfloat(config.get('info', dict())['price_multiplier'], self.price_multiplier)",
    ):
        return False
    if _is(
        _safeget(config, "info", "cost_increases_with_level")
        != _safeget(base, "info", "cost_increases_with_level"),
        raise_on_failure,
        warn_on_failure,
        "config.get('info', dict())['cost_increases_with_level'] != self.cost_increases_with_level",
    ):
        return False
    if _is(
        _safeget(config, "info", "equal_exogenous_supply")
        != _safeget(base, "info", "equal_exogenous_supply"),
        raise_on_failure,
        warn_on_failure,
        "config.get('info', dict())['equal_exogenous_supply'] != self.equal_exogenous_supply",
    ):
        return False
    if _is(
        _safeget(config, "info", "equal_exogenous_sales")
        != _safeget(base, "info", "equal_exogenous_sales"),
        raise_on_failure,
        warn_on_failure,
        "config.get('info', dict())['equal_exogenous_sales'] != self.equal_exogenous_sales",
    ):
        return False
    if _is(
        _safeget(config, "info", "cap_exogenous_quantities")
        != _safeget(base, "info", "cap_exogenous_quantities"),
        raise_on_failure,
        warn_on_failure,
        "config.get('info', dict())['cap_exogenous_quantities'] != self.cap_exogenous_quantities",
    ):
        return False
    if _is(
        _safeget(config, "info", "force_signing")
        != _safeget(base, "info", "force_signing"),
        raise_on_failure,
        warn_on_failure,
        "config.get('info', dict())['force_signing'] != self.force_signing",
    ):
        return False
    if _is(
        _safeget(config, "info", "random_agent_types")
        != _safeget(base, "info", "random_agent_types"),
        raise_on_failure,
        warn_on_failure,
        "config.get('info', dict())['random_agent_types'] != self.random_agent_types",
    ):
        return False
    if _is(
        _safeget(config, "info", "penalties_scale")
        != _safeget(base, "info", "penalties_scale"),
        raise_on_failure,
        warn_on_failure,
        "config.get('info', dict())['penalties_scale'] != self.penalties_scale",
    ):
        return False
    if _is(
        _safeget(config, "info", "exogenous_generation_method")
        != _safeget(base, "info", "exogenous_generation_method"),
        raise_on_failure,
        warn_on_failure,
        f' {_safeget(config, "info", "exogenous_generation_method")=} != {_safeget(base, "info", "exogenous_generation_method")=}, ',
    ):
        return False
    return True


@define
class BaseContext(Context, ABC):
    """A context that generates oneshot worlds with agents of a given `types` with predetermined structure and settings"""

    name: str | None = None
    world_type: type[SCMLBaseWorld] = SCML2024OneShotWorld
    world_params: dict[str, Any] = field(factory=dict)
    non_competitors: tuple[str | type[OneShotAgent], ...] = DefaultAgentsOneShot
    placeholder_types: tuple[type[OneShotAgent], ...] = DEFAULT_PLACEHOLDER_AGENT_TYPES
    placeholder_params: tuple[dict, ...] | None = None
    placeholder_levels: tuple[int, ...] | None = None

    @abstractmethod
    def make_config(self) -> dict[str, Any]:
        """Generates a config for a world"""

    @abstractmethod
    def is_valid_world(  # type: ignore
        self,
        world: SCMLBaseWorld,
        raise_on_failure: bool = False,
        warn_on_failure: bool = False,
        types: tuple[str | type[OneShotAgent], ...] | None = None,
    ) -> bool:
        """Checks that the given world could have been generated from this context"""

    def extract_context_params(
        self, min_values: bool, level: int | None = None
    ) -> ContextParams:
        _ = min_values, level
        raise NotImplementedError(
            f"{self.__class__.__name__} did not implement `extrtact_context_params`"
        )

    def make(
        self,
        types: tuple[type[OneShotAgent], ...] | None = None,
        params: tuple[dict[str, Any], ...] | None = None,
        name: str | None = None,
    ) -> SCMLBaseWorld:
        """Generates the oneshot world and assigns an agent of type `agent_type` to it"""
        if types is None:
            types = self.placeholder_types
            params = self.placeholder_params
        return self.make_world(types, params, name=name)

    def make_world(
        self,
        types: tuple[type[OneShotAgent], ...] | None = None,
        params: tuple[dict[str, Any], ...] | None = None,
        config: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> SCMLBaseWorld:
        """Generates a world"""
        if types is None:
            types = self.placeholder_types
            params = self.placeholder_params
        test_world = (config is not None,)
        if config is None:
            config = self.make_config()
        config = self.world_type.replace_agents(
            config, self.placeholder_types, types, params
        )
        if name is None:
            name = unique_name(self.name, sep=".")
        if name is not None:
            config["name"] = name

        world = self.world_type(
            **(self.world_params | config),
            one_offer_per_step=True,
        )
        if test_world:
            assert self.is_valid_world(world, types=types)
        return world

    def generate(  # type: ignore
        self,
        types: tuple[type[OneShotAgent], ...] | None = None,
        params: tuple[dict[str, Any], ...] | None = None,
        name: str | None = None,
    ) -> tuple[SCMLBaseWorld, tuple[OneShotAgent, ...]]:
        """Generates the world and assigns an agent to it"""
        if not name:
            name = self.name
        if types is None:
            types = self.placeholder_types
            params = self.placeholder_params
        if isinstance(types, OneShotAgent):
            types = (types,)  # type: ignore
        if isinstance(params, dict):
            params = (params,)
        world = self.make(types, params, name)
        ids = []
        if types:
            ids = [id for id, a in world.agents.items() if isinobject(a._obj, types)]  # type: ignore
            assert len(ids) == len(
                types
            ), f"Found the following agent of type {types=}: {ids=}"
        agents = tuple(world.agents[id]._obj for id in ids)  # type: ignore
        return world, agents  # type: ignore

    def is_valid_awi(
        self,
        awi: OneShotAWI,
        raise_on_failure: bool = False,
        warn_on_failure: bool = False,
        types: tuple[str | type[OneShotAgent], ...] | None = None,
    ) -> bool:  # type: ignore
        # todo: what should I do with tupes input to is_invalid_world
        return self.is_valid_world(
            awi._world,
            raise_on_failure=raise_on_failure,
            warn_on_failure=warn_on_failure,
            types=types,
        )

    def contains_context(
        self,
        context: Context,
        raise_on_failure: bool = False,
        warn_on_failure: bool = False,
        n_tests: int = NTESTS,
    ) -> bool:
        for _ in range(n_tests):
            world, _ = context.generate()
            if not self.is_valid_world(
                world,
                raise_on_failure=raise_on_failure,
                warn_on_failure=warn_on_failure,
            ):
                return False
        return True


@define
class GeneralContext(BaseContext):
    """A context that generates oneshot worlds with agents of a given `types` with predetermined structure and settings"""

    # std vs oneshot
    perishable: bool = True
    # negotiation parameters
    price_multiplier: np.ndarray | tuple[float, float] | float = (1.5, 2.0)
    force_signing = True
    # production graph parameters
    n_steps: tuple[int, int] | int = (20, 200)
    n_processes: tuple[int, int] | int = 2
    n_lines: tuple[int, int] | int = 10
    n_agents_per_process: np.ndarray | list[int] | tuple[int, int] | int = (
        min(N_CONSUMERS[0], N_SUPPLIERS[0]),
        max(N_CONSUMERS[-1], N_SUPPLIERS[-1]),
    )
    # profile parameters
    production_costs: np.ndarray | tuple[int, int] | int = (1, 4)
    cash_availability: tuple[float, float] | float = (1.5, 2.5)
    shortfall_penalty: tuple[float, float] | float = (0.2, 1.0)
    shortfall_penalty_dev: tuple[float, float] | float = (0.0, 0.1)
    disposal_cost: tuple[float, float] | float = (0.0, 0.2)
    disposal_cost_dev: tuple[float, float] | float = (0.0, 0.02)
    storage_cost: tuple[float, float] | float = (0.0, 0.02)
    storage_cost_dev: tuple[float, float] | float = 0
    cost_increases_with_level = True
    penalties_scale: str | list[str] = "trading"
    process_inputs: tuple[int, int] | int = 1
    process_outputs: np.ndarray | tuple[int, int] | int = 1
    # exogenous contract generation parameters
    exogenous_generation_method = "profitable"
    profit_means: np.ndarray | tuple[float, float] | float = (0.1, 0.2)
    profit_stddevs: np.ndarray | tuple[float, float] | float = 0.05
    max_productivity: np.ndarray | tuple[float, float] | float = (0.8, 1.0)
    initial_balance: np.ndarray | tuple[int, int] | int | None = None
    exogenous_supply_predictability: tuple[float, float] | float = (0.6, 0.9)
    exogenous_sales_predictability: tuple[float, float] | float = (0.6, 0.9)
    exogenous_control: tuple[float, float] | float = -1
    exogenous_price_dev: tuple[float, float] | float = (0.1, 0.2)
    equal_exogenous_supply = False
    equal_exogenous_sales = False
    cap_exogenous_quantities: bool = True

    def __attrs_post_init__(self):
        from scml.std.world import StdWorld

        if self.perishable:
            assert not issubclass(self.world_type, StdWorld)
        else:
            assert issubclass(self.world_type, StdWorld)

    def extract_context_params(
        self, min_values: bool, level: int | None = None
    ) -> ContextParams:
        nlines = safemin(self.n_lines) if min_values else safemax(self.n_lines)
        app = self.n_agents_per_process
        if not isinstance(app, Iterable):
            nsuppliers = nconsumers = app
        if isinstance(app, tuple) and len(app) == 2:
            nsuppliers = nconsumers = app[0] if min_values else app[-1]
        else:
            assert isinstance(app, Sequence)
            n_processes = len(app)

            if level is None:
                mn_consumers, mn_suppliers = min(app[1:]), min(app[:-1])
                mx_consumers, mx_suppliers = max(app[1:]), max(app[:-1])
            else:
                mx_consumers = mn_consumers = (
                    app[level + 1] if 0 <= level < n_processes - 1 else 0
                )
                mx_suppliers = mn_suppliers = app[level - 1] if level > 0 else 0
            nsuppliers = mn_suppliers if min_values else mx_suppliers
            nconsumers = mn_consumers if min_values else mx_consumers
        if level is None:
            return ContextParams(self.perishable, nlines, nsuppliers, nconsumers)
        if level == 0:
            nsuppliers = 0
        elif (
            level == -1
            or not isinstance(self.n_processes, Iterable)
            and level == self.n_processes - 1
        ):
            nconsumers = 0
        return ContextParams(self.perishable, nlines, nsuppliers, nconsumers)

    def make_predefined_config(
        self,
        agent_types: list[type[OneShotAgent]],
        agent_processes: list[int],
        agent_params: list[dict[str, Any]],
        n_agents_per_process: list[int],
    ) -> dict[str, Any]:
        """Generates a config for a world"""
        if agent_params is None:
            agent_params = [dict() for _ in agent_types]

        return self.world_params | self.world_type.generate(
            agent_types=agent_types,  # type: ignore
            agent_params=agent_params,
            agent_processes=agent_processes,
            perishable=self.perishable,
            n_steps=self.n_steps,
            n_processes=len(n_agents_per_process),
            n_lines=self.n_lines,
            n_agents_per_process=np.asarray(n_agents_per_process),
            process_inputs=self.process_inputs,
            process_outputs=self.process_outputs,
            production_costs=self.production_costs,
            profit_means=self.profit_means,
            profit_stddevs=self.profit_stddevs,
            max_productivity=self.max_productivity,
            initial_balance=self.initial_balance,
            exogenous_supply_predictability=self.exogenous_supply_predictability,
            exogenous_sales_predictability=self.exogenous_sales_predictability,
            exogenous_control=self.exogenous_control,
            cash_availability=self.cash_availability,
            shortfall_penalty=self.shortfall_penalty,
            shortfall_penalty_dev=self.shortfall_penalty_dev,
            disposal_cost=self.disposal_cost,
            disposal_cost_dev=self.disposal_cost_dev,
            storage_cost=self.storage_cost,
            storage_cost_dev=self.storage_cost_dev,
            exogenous_price_dev=self.exogenous_price_dev,
            price_multiplier=self.price_multiplier,
            cost_increases_with_level=self.cost_increases_with_level,
            equal_exogenous_supply=self.equal_exogenous_supply,
            equal_exogenous_sales=self.equal_exogenous_sales,
            cap_exogenous_quantities=self.cap_exogenous_quantities,
            force_signing=self.force_signing,
            random_agent_types=False,
            penalties_scale=self.penalties_scale,
            exogenous_generation_method=self.exogenous_generation_method,
        )

    def contains_context(
        self,
        context: Context,
        raise_on_failure: bool = False,
        warn_on_failure: bool = False,
        n_tests: int = NTESTS,
    ) -> bool:
        if isinstance(context, GeneralContext):
            return self.contains_general_context(context)
        return super().contains_context(
            context, raise_on_failure, warn_on_failure, n_tests
        )

    def _assign_types(self, n_processes, types, params, levels, n_agents_per_process):
        n_agents = sum(n_agents_per_process)
        perlevel = defaultdict(list)
        for i, t, p in zip(levels, types, params):
            perlevel[i].append((t, p))

        agent_types = list(random.choices(self.non_competitors, k=n_agents))
        agent_params: list[dict[str, Any]] = list(dict() for _ in agent_types)
        agent_processes = np.zeros(n_agents, dtype=int)
        nxt, indx = 0, -1
        rngs = []
        for level in range(n_processes):
            last = nxt + n_agents_per_process[level]
            agent_processes[nxt:last] = level
            rngs.append((nxt, last))
            nxt += n_agents_per_process[level]
        for i, tp in perlevel.items():
            first, last = rngs[i]
            assert (
                last - first + 1 >= len(tp)
            ), f"Cannot put agents of type {tp=} in level {i} which has only {last - first + 1} agents"

            random.shuffle(tp)
            selected = list(range(first, last))
            random.shuffle(selected)
            selected = selected[: len(tp)]
            for indx, (my_type, my_params) in zip(selected, tp):
                agent_types[indx] = my_type
                if params:
                    agent_params[indx]["controller_params"] = my_params
        return agent_types, agent_processes, agent_params

    def _distribute_agents(self, n_types):
        n_processes = intin(self.n_processes)

        # distribute agents over production levels (processes)
        n_agents_per_process = make_array(
            self.n_agents_per_process, n_processes, dtype=int, min_total=n_types
        )
        return n_processes, n_agents_per_process

    def make_config(self) -> dict[str, Any]:
        """Generates a config for a world"""
        types = self.placeholder_types
        params = [dict() for _ in types]
        levels = self.placeholder_levels
        n_processes, n_agents_per_process = self._distribute_agents(len(types))
        assert len(n_agents_per_process) == n_processes

        n_agents = sum(n_agents_per_process)
        assert n_agents >= len(types)

        # find my levels
        if not levels:
            levels = tuple(random.randint(0, n_processes - 1) for _ in types)

        return self.make_predefined_config(
            *self._assign_types(
                n_processes, types, params, levels, n_agents_per_process
            ),
            n_agents_per_process,  # type: ignore
        )

    def is_valid_world(  # type: ignore
        self,
        world: SCMLBaseWorld,
        raise_on_failure: bool = False,
        warn_on_failure: bool = False,
        types: tuple[str | type[OneShotAgent], ...] | None = None,
    ) -> bool:
        """Checks that the given world could have been generated from this context"""
        if _is(
            world.perishable != self.perishable,
            raise_on_failure,
            warn_on_failure,
            f"{world.perishable=} != {self.perishable=}",
        ):
            return False
        if _not(
            isin(world.n_steps, self.n_steps),
            raise_on_failure,
            warn_on_failure,
            f"not isin({world.n_steps=}, {self.n_steps=})",
        ):
            return False
        if _not(
            isin(world.n_processes, self.n_processes),
            raise_on_failure,
            warn_on_failure,
            "not isin(world.n_processes, self.n_processes)",
        ):
            return False
        if _not(
            isin(world.info["n_lines"], self.n_lines),
            raise_on_failure,
            warn_on_failure,
            'not isin(world.info["n_lines"], self.n_lines)',
        ):
            return False
        if _not(
            all(
                isin(_, self.n_agents_per_process)
                for _ in world.info["n_agents_per_process"]
            ),
            raise_on_failure,
            warn_on_failure,
            f"not all( isin(_, self.n_agents_per_process) for _ in world.info['n_agents_per_process'])\n"
            f"{self.n_agents_per_process=}\n{world.info['n_agents_per_process']=}",
        ):
            return False
        if _not(
            isin(world.info["process_inputs_generator"], self.process_inputs),
            raise_on_failure,
            warn_on_failure,
            "not isin(world.info['process_inputs_generator'], self.process_inputs)",
        ):
            return False
        if _not(
            isin(world.info["process_outputs_generator"], self.process_outputs),
            raise_on_failure,
            warn_on_failure,
            "not isin(world.info['process_outputs_generator'], self.process_outputs)",
        ):
            return False
        if _not(
            isin(world.info["production_costs"], self.production_costs),
            raise_on_failure,
            warn_on_failure,
            "not isin(world.info['production_costs'], self.production_costs)",
        ):
            return False
        if _not(
            isinfloat(world.info["profit_means"], self.profit_means),
            raise_on_failure,
            warn_on_failure,
            "not isinfloat(world.info['profit_means'], self.profit_means)",
        ):
            return False
        if _not(
            isinfloat(world.info["profit_stddevs"], self.profit_stddevs),
            raise_on_failure,
            warn_on_failure,
            "not isinfloat(world.info['profit_stddevs'], self.profit_stddevs)",
        ):
            return False
        if _not(
            isinfloat(world.info["max_productivity"], self.max_productivity),
            raise_on_failure,
            warn_on_failure,
            "not isinfloat(world.info['max_productivity'], self.max_productivity)",
        ):
            return False
        if _is(
            self.initial_balance is not None
            and not isin(world.info["initial_balance"], self.initial_balance),
            raise_on_failure,
            warn_on_failure,
            "self.initial_balance is not None and not isin(world.info['initial_balance'], self.initial_balance)",
        ):
            return False
        if _not(
            isinfloat(
                world.info["exogenous_supply_predictability"],
                self.exogenous_supply_predictability,
            ),
            raise_on_failure,
            warn_on_failure,
            "not isinfloat( world.info['exogenous_supply_predictability'], self.exogenous_supply_predictability,)",
        ):
            return False
        if _not(
            isinfloat(
                world.info["exogenous_sales_predictability"],
                self.exogenous_sales_predictability,
            ),
            raise_on_failure,
            warn_on_failure,
            "not isinfloat( world.info['exogenous_sales_predictability'], self.exogenous_sales_predictability,)",
        ):
            return False
        if _not(
            isinfloat(world.info["exogenous_control"], self.exogenous_control),
            raise_on_failure,
            warn_on_failure,
            "not isinfloat(world.info['exogenous_control'], self.exogenous_control)",
        ):
            return False
        if _not(
            isinfloat(world.info["cash_availability"], self.cash_availability),
            raise_on_failure,
            warn_on_failure,
            "not isinfloat(world.info['cash_availability'], self.cash_availability)",
        ):
            return False
        if _not(
            isinfloat(world.info["shortfall_penalty"], self.shortfall_penalty),
            raise_on_failure,
            warn_on_failure,
            "not isinfloat(world.info['shortfall_penalty'], self.shortfall_penalty)",
        ):
            return False
        if _not(
            isinfloat(world.info["shortfall_penalty_dev"], self.shortfall_penalty_dev),
            raise_on_failure,
            warn_on_failure,
            "not isinfloat( world.info['shortfall_penalty_dev'], self.shortfall_penalty_dev)",
        ):
            return False
        if _not(
            isinfloat(world.info["disposal_cost"], self.disposal_cost),
            raise_on_failure,
            warn_on_failure,
            "not isinfloat(world.info['disposal_cost'], self.disposal_cost)",
        ):
            return False
        if _not(
            isinfloat(world.info["disposal_cost_dev"], self.disposal_cost_dev),
            raise_on_failure,
            warn_on_failure,
            "not isinfloat(world.info['disposal_cost_dev'], self.disposal_cost_dev)",
        ):
            return False
        if _not(
            isinfloat(world.info["storage_cost"], self.storage_cost),
            raise_on_failure,
            warn_on_failure,
            "not isinfloat(world.info['storage_cost'], self.storage_cost)",
        ):
            return False
        if _not(
            isinfloat(world.info["storage_cost_dev"], self.storage_cost_dev),
            raise_on_failure,
            warn_on_failure,
            "not isinfloat(world.info['storage_cost_dev'], self.storage_cost_dev)",
        ):
            return False
        if _not(
            isinfloat(world.info["exogenous_price_dev"], self.exogenous_price_dev),
            raise_on_failure,
            warn_on_failure,
            "not isinfloat(world.info['exogenous_price_dev'], self.exogenous_price_dev)",
        ):
            return False
        if _not(
            isinfloat(world.info["price_multiplier"], self.price_multiplier),
            raise_on_failure,
            warn_on_failure,
            "not isinfloat(world.info['price_multiplier'], self.price_multiplier)",
        ):
            return False
        if _is(
            world.info["cost_increases_with_level"] != self.cost_increases_with_level,
            raise_on_failure,
            warn_on_failure,
            "world.info['cost_increases_with_level'] != self.cost_increases_with_level",
        ):
            return False
        if _is(
            world.info["equal_exogenous_supply"] != self.equal_exogenous_supply,
            raise_on_failure,
            warn_on_failure,
            "world.info['equal_exogenous_supply'] != self.equal_exogenous_supply",
        ):
            return False
        if _is(
            world.info["equal_exogenous_sales"] != self.equal_exogenous_sales,
            raise_on_failure,
            warn_on_failure,
            "world.info['equal_exogenous_sales'] != self.equal_exogenous_sales",
        ):
            return False
        if _is(
            world.info["cap_exogenous_quantities"] != self.cap_exogenous_quantities,
            raise_on_failure,
            warn_on_failure,
            "world.info['cap_exogenous_quantities'] != self.cap_exogenous_quantities",
        ):
            return False
        if _is(
            world.info["force_signing"] != self.force_signing,
            raise_on_failure,
            warn_on_failure,
            "world.info['force_signing'] != self.force_signing",
        ):
            return False
        if _is(
            world.info["random_agent_types"] is not False,
            raise_on_failure,
            warn_on_failure,
            "world.info['random_agent_types'] != False",
        ):
            return False
        if _is(
            world.info["penalties_scale"] != self.penalties_scale,
            raise_on_failure,
            warn_on_failure,
            "world.info['penalties_scale'] != self.penalties_scale",
        ):
            return False
        if _is(
            world.info["exogenous_generation_method"]
            != self.exogenous_generation_method,
            raise_on_failure,
            warn_on_failure,
            "world.info['exogenous_generation_method'] != self.method",
        ):
            return False
        if _not(
            isinstance(world, self.world_type),
            raise_on_failure,
            warn_on_failure,
            "not isinstance(world, self.world_type)",
        ):
            return False
        if types:
            world_agent_types = [
                type(_._obj)  # type: ignore
                for aid, _ in world.agents.items()
                if not is_system_agent(aid)
            ]
            if _not(
                isinclass(world_agent_types, list(self.non_competitors) + list(types)),
                raise_on_failure,
                warn_on_failure,
                f"not isinclass({world_agent_types=}, {list(self.non_competitors)=}), {types=}",
            ):
                return False
        return True

    def contains_general_context(self, context: "GeneralContext") -> bool:
        """Checks that the any world generated from the given `context` could have been generated from this context"""
        if context.perishable != self.perishable:
            return False
        if not isin(context.n_steps, self.n_steps):
            return False
        if not isin(context.n_processes, self.n_processes):
            return False
        if not isin(context.n_lines, self.n_lines):
            return False
        if not isin(context.n_agents_per_process, self.n_agents_per_process):
            return False
        if not isin(context.process_inputs, self.process_inputs):
            return False
        if not isin(context.process_outputs, self.process_outputs):
            return False
        if not isin(context.production_costs, self.production_costs):
            return False
        if not isinfloat(context.profit_means, self.profit_means):
            return False
        if not isinfloat(context.profit_stddevs, self.profit_stddevs):
            return False
        if not isinfloat(context.max_productivity, self.max_productivity):
            return False
        if (
            self.initial_balance is not None
            and not isin(context.initial_balance, self.initial_balance)  # type: ignore
        ) or (
            self.initial_balance is not None
            and not isin(context.initial_balance, self.initial_balance)  # type: ignore
        ):
            return False
        if not isinfloat(
            context.exogenous_supply_predictability,
            self.exogenous_supply_predictability,
        ):
            return False
        if not isinfloat(
            context.exogenous_sales_predictability,
            self.exogenous_sales_predictability,
        ):
            return False
        if not isinfloat(context.exogenous_control, self.exogenous_control):
            return False
        if not isinfloat(context.cash_availability, self.cash_availability):
            return False
        if not isinfloat(context.shortfall_penalty, self.shortfall_penalty):
            return False
        if not isinfloat(context.shortfall_penalty_dev, self.shortfall_penalty_dev):
            return False
        if not isinfloat(context.disposal_cost, self.disposal_cost):
            return False
        if not isinfloat(context.disposal_cost_dev, self.disposal_cost_dev):
            return False
        if not isinfloat(context.storage_cost, self.storage_cost):
            return False
        if not isinfloat(context.storage_cost_dev, self.storage_cost_dev):
            return False
        if not isinfloat(context.exogenous_price_dev, self.exogenous_price_dev):
            return False
        if not isinfloat(context.price_multiplier, self.price_multiplier):
            return False
        if context.cost_increases_with_level != self.cost_increases_with_level:
            return False
        if context.equal_exogenous_supply != self.equal_exogenous_supply:
            return False
        if context.equal_exogenous_sales != self.equal_exogenous_sales:
            return False
        if context.cap_exogenous_quantities != self.cap_exogenous_quantities:
            return False
        if context.force_signing != self.force_signing:
            return False
        # if context.random_agent_types != self.random_agent_types:
        #     return False
        if context.penalties_scale != self.penalties_scale:
            return False
        if context.exogenous_generation_method != self.exogenous_generation_method:
            return False
        if isinstance(context.world_type, self.world_type):
            return False
        if not isinclass(list(context.non_competitors), list(self.non_competitors)):
            return False
        return True


@define
class LimitedPartnerNumbersContext(GeneralContext):
    """Generates a world limiting the range of the agent level, production capacity
    and the number of suppliers, consumers, and optionally same-level competitors."""

    level: int = 0
    n_consumers: tuple[int, int] = N_CONSUMERS
    n_suppliers: tuple[int, int] = (0, 0)
    n_competitors: tuple[int, int] = (N_CONSUMERS[0] - 1, N_CONSUMERS[1] - 1)
    buying_strength: Strength | None = None
    selling_strength: Strength | None = None

    def __attrs_post_init__(self):
        max_n_proceses = (
            max(self.n_processes)
            if isinstance(self.n_processes, Iterable)
            else self.n_processes
        )
        assert isin(
            tuple(_ + 1 for _ in self.n_competitors),  # type: ignore
            self.n_agents_per_process,  # type: ignore
        ), f"{self.n_competitors=}, {self.n_agents_per_process=}"
        assert (
            not (self.level > 0 and self.level < max_n_proceses - 1)
            or (self.n_suppliers[-1] > 1 and self.n_consumers[-1] > 1)
        ), f"{self.n_suppliers=}, {self.n_consumers=}, {self.level=}, {self.n_processes=}"
        if self.level == 0:
            assert isin(
                self.n_consumers, self.n_agents_per_process
            ), f"{self.n_consumers=}, {self.n_agents_per_process=}, {self.level=}"
            assert max(self.n_suppliers) < 1, f"{self.n_suppliers=}, {self.level=}"
            assert (
                min(self.n_consumers) > 0
            ), f"{self.n_consumers=}, {self.n_agents_per_process=}, {self.level=}"
        elif (
            self.level == -1
            or isinstance(self.n_processes, int)
            and self.level == max_n_proceses - 1
        ):
            assert self.level < max_n_proceses, f"{max_n_proceses=}, {self.level=}"
            assert isin(
                self.n_suppliers, self.n_agents_per_process
            ), f"{self.n_suppliers=}, {self.n_agents_per_process=}, {self.level=}"
            assert (
                max(self.n_consumers) < 1
            ), f"{self.n_consumers=}, {self.n_agents_per_process=}, {self.level=}"
            assert min(self.n_suppliers) > 0, f"{self.n_suppliers=}, {self.level=}"
        else:
            assert isin(
                self.n_consumers, self.n_agents_per_process
            ), f"{self.n_consumers=}, {self.n_agents_per_process=}, {self.level=}"
            assert isin(
                self.n_suppliers, self.n_agents_per_process
            ), f"{self.n_suppliers=}, {self.n_agents_per_process=}, {self.level=}"

    def extract_context_params(
        self, min_values: bool, level: int | None = None
    ) -> ContextParams:
        assert level is None or level == self.level
        nlines = safemin(self.n_lines) if min_values else safemax(self.n_lines)
        nsuppliers = self.n_suppliers[0 if min_values else -1]
        nconsumers = self.n_consumers[0 if min_values else -1]
        return ContextParams(self.perishable, nlines, int(nsuppliers), int(nconsumers))

    def make_config(self) -> dict[str, Any]:
        """Generates a config"""
        types = self.placeholder_types
        params = (
            [dict() for _ in types]
            if self.placeholder_params is None
            else self.placeholder_params
        )
        levels = self.placeholder_levels
        assert levels is None or all(_ == self.level for _ in levels), (
            "LimitedPartnerNumbersContext does not allow you to decide the levels of "
            "the agents when creating the config as it uses its internal level "
            "and assigns all placeholder agents to it: {self.level=}, {levels=}"
        )
        levels = tuple(self.level for _ in types)
        if params is None:
            params = tuple(dict() for _ in types)
        n_processes, n_agents_per_process = self._distribute_agents(len(types))
        # find my level
        my_level = n_processes - 1 if self.level < 0 else self.level
        n_competitors = self.n_competitors
        n_suppliers = n_agents_per_process[my_level - 1] if my_level > 0 else 0
        n_consumers = (
            n_agents_per_process[my_level + 1] if my_level < n_processes - 1 else 0
        )
        if self.buying_strength is not None:
            n_competitors, n_suppliers = sample_with_strength(
                n_competitors, self.n_suppliers, self.buying_strength
            )
        if self.selling_strength is not None:
            n_competitors, n_consumers = sample_with_strength(
                n_competitors, self.n_consumers, self.selling_strength
            )
        n_competitors = intin(n_competitors)

        # override the number of consumers and number of suppliers to match my choice
        if my_level == 0:
            n_agents_per_process[1] = n_consumers
        elif my_level == n_processes - 1:
            n_agents_per_process[my_level - 1] = n_suppliers
        else:
            n_agents_per_process[my_level + 1] = n_consumers
            n_agents_per_process[my_level - 1] = n_suppliers
        n_agents_per_process[my_level] = max(len(types), n_competitors + 1)
        for k, n in enumerate(n_agents_per_process):
            assert isin(n, self.n_agents_per_process), (
                f"Level {k} has {n} agents which is not in {self.n_agents_per_process}"
                f": {self.n_suppliers=}, {self.n_competitors=}, {self.n_consumers=}"
                f": {self.selling_strength=}, {self.buying_strength=}\n{n_agents_per_process}"
                f"\n{self.n_agents_per_process}"
                f"\n {n_suppliers=}, {n_competitors=}, {n_consumers=}"
            )

        return self.make_predefined_config(
            *self._assign_types(
                n_processes, types, params, levels, n_agents_per_process
            ),
            n_agents_per_process,  # type: ignore
        )

    def find_test_agents(
        self,
        world: SCMLBaseWorld,
        types: tuple[type[OneShotAgent], ...] | None = None,
    ) -> list[str]:
        if types is None:
            types = self.placeholder_types
        return [aid for aid, agent in world.agents.items() if isinobject(agent, types)]

    def is_valid_world(  # type: ignore
        self,
        world: SCMLBaseWorld,
        types: tuple[type[OneShotAgent], ...] | None = None,
        raise_on_failure: bool = False,
        warn_on_failure: bool = False,
    ) -> bool:
        """Checks that the given world could have been generated from this context"""
        if types is None:
            types = self.placeholder_types
        agent_ids = self.find_test_agents(world, types)
        n_processes = world.n_processes
        expected_level = self.level
        for aid in agent_ids:
            my_level = world.agent_profiles[aid].input_product
            if _is(
                my_level == expected_level,
                raise_on_failure,
                warn_on_failure,
                f"Agent {aid} of type {world.agents[aid]._obj.__class__.__name__} "
                f"is on level {my_level} but expected to be on level {expected_level}",
            ):
                return False
            is_first_level = my_level == 0
            is_last_level = my_level == n_processes - 1
            my_suppliers = [
                _ for _ in world.agent_suppliers[aid] if not is_system_agent(_)
            ]
            my_consumers = [
                _ for _ in world.agent_consumers[aid] if not is_system_agent(_)
            ]
            my_competitors = (
                world.suppliers[my_level + 1]
                if not is_last_level
                else world.consumers[my_level - 1]
            )
            assert (
                aid in my_competitors
            ), f"{aid} not found in its competitors!! {my_competitors=}"
            my_competitors = [_ for _ in my_competitors if _ != aid]
            n_consumers, n_suppliers = len(my_consumers), len(my_suppliers)
            n_competitors = len(my_competitors)
            if not isin(n_competitors, self.n_competitors):
                warnings.warn(
                    f"Invalid n_competitors: {n_competitors=} != {self.n_competitors=}"
                )
                return False
            if self.buying_strength is not None:
                if self.buying_strength == Strength.Strong:
                    if _not(
                        n_suppliers >= n_competitors + 1,
                        raise_on_failure=raise_on_failure,
                        warn_on_failure=warn_on_failure,
                        message=f"Strength {self.buying_strength} but {n_suppliers=} and {n_competitors=}",
                    ):
                        return False
                elif self.buying_strength == Strength.Weak:
                    if _not(
                        n_suppliers <= n_competitors - 1,
                        raise_on_failure=raise_on_failure,
                        warn_on_failure=warn_on_failure,
                        message=f"Strength {self.buying_strength} but {n_suppliers=} and {n_competitors=}",
                    ):
                        return False
                elif self.buying_strength == Strength.Balanced:
                    if _not(
                        n_competitors - 1 <= n_suppliers <= n_competitors + 1,
                        raise_on_failure=raise_on_failure,
                        warn_on_failure=warn_on_failure,
                        message=f"Strength {self.buying_strength} but {n_suppliers=} and {n_competitors=}",
                    ):
                        return False

            if self.selling_strength is not None:
                if self.selling_strength == Strength.Strong:
                    if _not(
                        n_consumers >= n_competitors + 1,
                        raise_on_failure=raise_on_failure,
                        warn_on_failure=warn_on_failure,
                        message=f"Strength {self.selling_strength} but {n_consumers=} and {n_competitors=}",
                    ):
                        return False
                elif self.selling_strength == Strength.Weak:
                    if _not(
                        n_consumers <= n_competitors - 1,
                        raise_on_failure=raise_on_failure,
                        warn_on_failure=warn_on_failure,
                        message=f"Strength {self.selling_strength} but {n_consumers=} and {n_competitors=}",
                    ):
                        return False
                elif self.selling_strength == Strength.Balanced:
                    if _not(
                        n_competitors - 1 <= n_consumers <= n_competitors + 1,
                        raise_on_failure=raise_on_failure,
                        warn_on_failure=warn_on_failure,
                        message=f"Strength {self.selling_strength} but {n_consumers=} and {n_competitors=}",
                    ):
                        return False

            if is_first_level:
                if _not(
                    isin(n_consumers, self.n_consumers),
                    raise_on_failure=raise_on_failure,
                    warn_on_failure=warn_on_failure,
                    message=f"Invalid n_consumers: {n_consumers=} != {self.n_consumers=}",
                ):
                    return False
                if _is(
                    n_suppliers != 0,
                    raise_on_failure=raise_on_failure,
                    warn_on_failure=warn_on_failure,
                    message=(
                        f"Invalid n_suppliers for {aid} (at level {my_level} "
                        f"[of {world.n_processes} processes]): {len(my_suppliers)=} != 1\nAll Suppliers: {world.suppliers}"
                    ),
                ):
                    return False
            elif is_last_level:
                if _not(
                    isin(n_suppliers, self.n_suppliers),
                    raise_on_failure=raise_on_failure,
                    warn_on_failure=warn_on_failure,
                    message=f"Invalid n_suppliers: {n_suppliers=} != {self.n_suppliers=}",
                ):
                    return False
                if _is(
                    n_consumers != 0,
                    raise_on_failure=raise_on_failure,
                    warn_on_failure=warn_on_failure,
                    message=f"Invalid n_conumsers: {len(my_consumers)=} != 1",
                ):
                    return False
            else:
                if _not(
                    isin(n_suppliers, self.n_suppliers),
                    raise_on_failure=raise_on_failure,
                    warn_on_failure=warn_on_failure,
                    message=f"Invalid n_suppliers: {n_suppliers=} not in {self.n_suppliers=}",
                ):
                    return False
                if _not(
                    isin(n_consumers, self.n_consumers),
                    raise_on_failure,
                    warn_on_failure,
                    message=f"Invalid n_consumers: {n_consumers=} not in {self.n_consumers=}",
                ):
                    return False
        return super().is_valid_world(world, raise_on_failure=raise_on_failure)

    def contains_limited_partner_context(
        self,
        context: "LimitedPartnerNumbersContext",
        raise_on_failure: bool = False,
        warn_on_failure: bool = False,
    ) -> bool:
        if _not(
            isin(context.n_processes, self.n_processes),
            raise_on_failure,
            warn_on_failure,
            "not isin(context.n_processes, self.n_processes)",
        ):
            return False
        if _not(
            isin(context.level, self.level),
            raise_on_failure,
            warn_on_failure,
            "not isin(context.level, self.level)",
        ):
            return False
        if _not(
            isin(context.n_consumers, self.n_consumers),
            raise_on_failure,
            warn_on_failure,
            "not isin(context.n_consumers, self.n_consumers)",
        ):
            return False
        if _not(
            isin(context.n_suppliers, self.n_suppliers),
            raise_on_failure,
            warn_on_failure,
            "not isin(context.n_suppliers, self.n_suppliers)",
        ):
            return False
        if _not(
            isin(context.n_competitors, self.n_competitors),
            raise_on_failure,
            warn_on_failure,
            "not isin(context.n_competitors, self.n_competitors)",
        ):
            return False
        return super().contains_context(context, raise_on_failure)

    def contains_context(
        self,
        context: Context,
        raise_on_failure: bool = False,
        warn_on_failure: bool = False,
        n_tests: int = NTESTS,
    ) -> bool:
        """Checks that the any world generated from the given `context` could have been generated from this context"""
        if isinstance(context, self.__class__):
            return self.contains_limited_partner_context(
                context,
                raise_on_failure=raise_on_failure,
                warn_on_failure=warn_on_failure,
            )
        return super().contains_context(
            context, raise_on_failure, warn_on_failure, n_tests
        )


@define
class FixedPartnerNumbersContext(LimitedPartnerNumbersContext):
    """Generates a world limiting the range of the agent level, production capacity
    and the number of suppliers, consumers, and optionally same-level competitors."""

    level: int = 0
    n_consumers: int = 4  # type: ignore
    n_suppliers: int = 0  # type: ignore
    n_competitors: int = 3  # type: ignore

    def __attrs_post_init__(self):
        object.__setattr__(self, "n_consumers", (self.n_consumers, self.n_consumers))
        object.__setattr__(self, "n_suppliers", (self.n_suppliers, self.n_suppliers))
        object.__setattr__(
            self, "n_competitors", (self.n_competitors, self.n_competitors)
        )
        super().__attrs_post_init__()
        object.__setattr__(self, "n_consumers", self.n_consumers[0])  # type: ignore
        object.__setattr__(self, "n_suppliers", self.n_suppliers[0])  # type: ignore
        object.__setattr__(self, "n_competitors", self.n_competitors[0])  # type: ignore

    def extract_context_params(
        self, min_values: bool, level: int | None = None
    ) -> ContextParams:
        assert level is None or level == self.level
        nlines = safemin(self.n_lines) if min_values else safemax(self.n_lines)
        nsuppliers = self.n_suppliers
        nconsumers = self.n_consumers
        return ContextParams(self.perishable, nlines, int(nsuppliers), int(nconsumers))


@define
class MonopolicContext(LimitedPartnerNumbersContext):
    """An agent that has no competitors in the same level as themselves"""

    # n_competitors: tuple[int, int] = field(default=(0, 0), converter=lambda _: (0, 0))
    n_competitors: tuple[int, int] = (0, 0)
    n_agents_per_process: np.ndarray | list[int] | tuple[int, int] | int = field(
        default=(1, max(N_CONSUMERS[-1], N_SUPPLIERS[-1])),
        converter=lambda _: (1, max(N_CONSUMERS[-1], N_SUPPLIERS[-1])),
    )

    def __attrs_post_init__(self):
        npp = self.n_agents_per_process
        if isinstance(npp, int):
            assert (
                npp == 1
            ), f"You passed {self.n_agents_per_process=} to a MonopolicContext but this MUST be one in this case"
        elif isinstance(npp, tuple):
            npp = (min(1, npp[0]), npp[1])
        else:
            npp = list(set(list(npp) + [1]))
        object.__setattr__(self, "n_agents_per_process", npp)
        return super().__attrs_post_init__()


@define
class SingleAgentPerLevelConsumerContext(MonopolicContext):
    """A world in which every level has exactly one factory and the agent is a consumer"""

    level: int = -1
    n_consumers: tuple[int, int] = (0, 0)
    n_suppliers: tuple[int, int] = (1, 1)
    n_agents_per_process: np.ndarray | list[int] | tuple[int, int] | int = field(
        default=1, converter=lambda _: 1
    )


@define
class SingleAgentPerLevelSupplierContext(MonopolicContext):
    """A world in which every level has exactly one factory and the agent is a supplier"""

    level: int = 0
    n_consumers: tuple[int, int] = (1, 1)
    n_suppliers: tuple[int, int] = (0, 0)
    n_agents_per_process: np.ndarray | list[int] | tuple[int, int] | int = field(
        default=1, converter=lambda _: 1
    )


@define
class EutopiaContext(MonopolicContext):
    """An unrealistic context in which the agent is the only one in its level and all other agents are nice."""

    non_competitors: tuple[str | type[OneShotAgent], ...] = field(
        default=(NiceAgent,), converter=lambda _: (NiceAgent,)
    )


@define
class EutopiaSupplierContext(EutopiaContext):
    """An unrealistic context in which the agent is the only supplier and all consumers are nice."""

    level: int = field(default=0, converter=lambda _: 0)
    n_consumers: tuple[int, int] = N_CONSUMERS
    n_suppliers: tuple[int, int] = (0, 0)


@define
class EutopiaConsumerContext(EutopiaContext):
    """An unrealistic context in which the agent is the only consumer and all suppliers are nice."""

    level: int = field(default=-1, converter=lambda _: -1)
    n_consumers: tuple[int, int] = (0, 0)
    n_suppliers: tuple[int, int] = N_SUPPLIERS


@define
class FixedPartnerNumbersOneShotContext(FixedPartnerNumbersContext):
    ...


@define
class ANACContext(GeneralContext):
    """Generates a oneshot world with no constraints except compatibility with a specific ANAC competition year."""

    year: int = 2024

    def __attrs_post_init__(self):
        object.__setattr__(
            self,
            "world_type",
            {
                2024: SCML2024OneShotWorld,
                2023: SCML2023OneShotWorld,
                2022: SCML2022OneShotWorld,
                2021: SCML2021OneShotWorld,
                2020: SCMLBaseWorld,
            }[self.year],
        )


@define
class LimitedPartnerNumbersOneShotContext(LimitedPartnerNumbersContext):
    """Generates a oneshot world limiting the range of the agent level, production capacity
    and the number of suppliers, consumers, and optionally same-level competitors."""

    year: int = 2024

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        object.__setattr__(
            self,
            "world_type",
            {
                2024: SCML2024OneShotWorld,
                2023: SCML2023OneShotWorld,
                2022: SCML2022OneShotWorld,
                2021: SCML2021OneShotWorld,
                2020: SCMLBaseWorld,
            }[self.year],
        )


@define
class ANACOneShotContext(ANACContext):
    """Generates a oneshot world with no constraints except compatibility with a specific ANAC competition year."""

    def __attrs_post_init__(self):
        object.__setattr__(
            self,
            "world_type",
            {
                2024: SCML2024OneShotWorld,
                2023: SCML2023OneShotWorld,
                2022: SCML2022OneShotWorld,
                2021: SCML2021OneShotWorld,
                2020: SCMLBaseWorld,
            }[self.year],
        )


@define
class SupplierContext(LimitedPartnerNumbersOneShotContext):
    """A world context that can generate any world compatible with the observation manager"""

    def __init__(self, *args, **kwargs):
        n_agents_per_process = (
            min(N_SUPPLIERS[0], N_CONSUMERS[0]),  # type: ignore
            max(N_SUPPLIERS[1], N_CONSUMERS[1]),  # type: ignore
        )
        kwargs |= dict(
            n_suppliers=(0, 0),  # suppliers have no suppliers
            n_consumers=N_CONSUMERS,
            n_competitors=(N_SUPPLIERS[0] - 1, N_SUPPLIERS[1] - 1),
            n_agents_per_process=n_agents_per_process,
            level=0,  # suppliers are always in the first level
        )
        super().__init__(*args, **kwargs)


@define
class StrongSupplierContext(SupplierContext):
    """A supplier with almost many consumers relative to competitors"""

    def __init__(self, *args, **kwargs):
        kwargs |= dict(selling_strength=Strength.Strong)
        super().__init__(*args, **kwargs)


@define
class BalancedSupplierContext(SupplierContext):
    """A supplier with almost same number of consumers as competitors"""

    def __init__(self, *args, **kwargs):
        kwargs |= dict(selling_strength=Strength.Balanced)
        super().__init__(*args, **kwargs)


@define
class WeakSupplierContext(SupplierContext):
    """A supplier with few consumers relative to competitors"""

    def __init__(self, *args, **kwargs):
        kwargs |= dict(selling_strength=Strength.Weak)
        super().__init__(*args, **kwargs)


@define
class ConsumerContext(LimitedPartnerNumbersOneShotContext):
    """A world context that can generate any world compatible with the observation manager"""

    def __init__(self, *args, **kwargs):
        n_agents_per_process = (
            min(N_SUPPLIERS[0], N_CONSUMERS[0]),  # type: ignore
            max(N_SUPPLIERS[1], N_CONSUMERS[1]),  # type: ignore
        )
        kwargs |= dict(
            n_suppliers=N_SUPPLIERS,
            n_consumers=(0, 0),  # consumers have no consumers
            n_competitors=(N_CONSUMERS[0] - 1, N_CONSUMERS[1] - 1),
            n_agents_per_process=n_agents_per_process,
            level=-1,  # consumers are always in the last level
        )
        super().__init__(*args, **kwargs)


@define
class StrongConsumerContext(ConsumerContext):
    """A consumer with almost many suppliers relative to competitors"""

    def __init__(self, *args, **kwargs):
        kwargs |= dict(buying_strength=Strength.Strong)
        super().__init__(*args, **kwargs)


@define
class BalancedConsumerContext(ConsumerContext):
    """A consumer with almost same number of suppliers as competitors"""

    def __init__(self, *args, **kwargs):
        kwargs |= dict(buying_strength=Strength.Balanced)
        super().__init__(*args, **kwargs)


@define
class WeakConsumerContext(ConsumerContext):
    """A consumer with few suppliers relative to competitors"""

    def __init__(self, *args, **kwargs):
        kwargs |= dict(buying_strength=Strength.Weak)
        super().__init__(*args, **kwargs)


@define
class OneShotContext(GeneralContext):
    """A basic context fixing stationary world config parameters"""


@define
class RepeatingContext(BaseContext):
    """Encapsulates one or more configs and switches between them when asked to generate or make something."""

    configs: tuple[dict[str, Any], ...] = field(
        factory=lambda: (GeneralContext().make_config(),)
    )
    randomize: bool = True
    rename: bool = True
    _next: int = field(init=False, default=0)

    def __attrs_post_init__(self):
        if not self.configs:
            raise ValueError("RepeatingContext with no configs")

    def extract_context_params(
        self, min_values: bool, level: int | None = None
    ) -> ContextParams:
        nlines, nsuppliers, nconsumers = 0, 0, 0
        nlines = min(get_n_lines(_)[0] for _ in self.configs)
        perishables = set(_.get("perishable", None) for _ in self.configs)
        assert (
            len(perishables) == 1
        ), f"Found {perishables} perishables. We cannot combine OneShot and Std worlds here"
        perishable = list(perishables)[0]
        assert len(self.placeholder_types) == 1
        mn_suppliers, mx_suppliers = float("inf"), float("-inf")
        mn_consumers, mx_consumers = float("inf"), float("-inf")
        mn_competitors = float("inf")
        for config in self.configs:
            app = get_n_agents_per_process(config)
            n_processes = len(app)
            existing_types = [
                _.get("controller_type", None) for _ in config["agent_params"]
            ]
            try:
                agent_indx = existing_types.index(self.placeholder_types[0])
            except ValueError:
                raise ValueError(
                    f"Cannot find {self.placeholder_types[0]=} in {existing_types=}"
                )

            my_level = config["profiles"][agent_indx].level
            assert level is None or my_level == level

            mn_suppliers = min(mn_suppliers, app[my_level - 1] if my_level else 0)
            mx_suppliers = max(mx_suppliers, app[my_level - 1] if my_level else 0)
            mn_consumers = min(
                mn_consumers,
                app[my_level + 1] if my_level < n_processes - 1 else 0,
            )
            mx_consumers = max(
                mx_consumers,
                app[my_level + 1] if my_level < n_processes - 1 else 0,
            )
            mn_competitors = min(mn_competitors, app[my_level])
        nsuppliers = mn_suppliers if min_values else mx_suppliers
        nconsumers = mn_consumers if min_values else mx_consumers
        return ContextParams(perishable, nlines, int(nsuppliers), int(nconsumers))

    def make_config(
        self,
        types: tuple[type[OneShotAgent], ...] = DEFAULT_PLACEHOLDER_AGENT_TYPES,
        params: tuple[dict[str, Any], ...] | None = None,
    ) -> dict[str, Any]:
        if not self.configs:
            raise ValueError("No configs to generate from")
        if self.randomize:
            self._next = random.randint(0, len(self.configs) - 1)
        config = self.configs[self._next]
        self._next = (self._next + 1) % len(self.configs)
        config = self.world_type.replace_agents(
            config, self.placeholder_types, types, params
        )
        if self.rename:
            config["name"] = unique_name(
                f"c{self._next}", add_time=False, rand_digits=6, sep=""
            )
        return config

    @classmethod
    def from_context(
        cls: type,
        context: BaseContext,
        n: int = 1,
        types: tuple[type[OneShotAgent]] = DEFAULT_PLACEHOLDER_AGENT_TYPES,
        rename: bool = False,
        randomize: bool = False,
    ):
        return cls(
            configs=tuple(context.make_config() for _ in range(n)),
            placeholder_types=types,
            rename=rename,
            randomize=randomize,
        )

    def contains_repeating_context(
        self,
        context: "RepeatingContext",
        raise_on_failure: bool = False,
        warn_on_failure: bool = False,
    ):
        for config in context.configs:
            if any(
                _config_matches_base(config, base, raise_on_failure, warn_on_failure)
                for base in self.configs
            ):
                break
        else:
            return False
        return True

    def is_valid_world(
        self,
        world: SCMLBaseWorld,
        raise_on_failure=RAISE_ON_FAILURE,
        warn_on_failure=WARN_ON_FAILURE,
        types: tuple[str | type[OneShotAgent], ...] | None = None,
    ) -> bool:
        """Checks that the given world could have been generated from this context"""
        for config in self.configs:
            if _world_matches_config(
                world,
                config,
                expected_types=list(self.non_competitors) + list(types)
                if types
                else None,
                expected_world_type=self.world_type,
                raise_on_failure=raise_on_failure,
                warn_on_failure=warn_on_failure,
            ):
                return True
        return False

    def contains_context(
        self,
        context: Context,
        raise_on_failure: bool = False,
        warn_on_failure: bool = False,
        n_tests: int = NTESTS,
    ) -> bool:
        if isinstance(context, RepeatingContext):
            return self.contains_repeating_context(
                context, raise_on_failure, warn_on_failure
            )
        return super().contains_context(
            context, raise_on_failure, warn_on_failure, n_tests
        )
