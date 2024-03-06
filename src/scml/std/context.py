from attr import define
import numpy as np
from scml.oneshot.agent import OneShotAgent
from scml.std.agents.greedy import GreedyStdAgent
from scml.std.agents.rand import RandomStdAgent, SyncRandomStdAgent
from scml.std.world import SCML2024StdWorld

from scml.oneshot.world import (
    SCML2021OneShotWorld,
    SCML2022OneShotWorld,
    SCML2023OneShotWorld,
    SCML2024OneShotWorld,
    SCMLBaseWorld,
)
from scml.oneshot.context import (
    BaseContext,
    ConsumerContext,
    FixedPartnerNumbersContext,
    GeneralContext,
    LimitedPartnerNumbersOneShotContext,
    RepeatingContext,
    SupplierContext,
    Strength,
    N_SUPPLIERS,
    N_CONSUMERS,
)
from scml.std.world import STD_DEFAULT_PARAMS

__all__ = [
    "BaseStdContext",
    "GeneralStdContext",
    "FixedPartnerNumbersStdContext",
    "LimitedPartnerNumbersStdContext",
    "ANACStdContext",
    "SupplierStdContext",
    "StrongSupplierStdContext",
    "BalancedSupplierStdContext",
    "WeakSupplierStdContext",
    "MiddleManStdContext",
    "StrongMiddleManStdContext",
    "BalancedMiddleManStdContext",
    "WeakMiddleManStdContext",
    "ConsumerStdContext",
    "StrongConsumerStdContext",
    "BalancedConsumerStdContext",
    "WeakConsumerStdContext",
    "StdContext",
    "RepeatingStdContext",
]


WARN_ON_FAILURE = True
RAISE_ON_FAILURE = False

DefaultAgentsStd = (
    GreedyStdAgent,
    RandomStdAgent,
    SyncRandomStdAgent,
)


@define
class BaseStdContext(BaseContext):
    """A context that generates std worlds with agents of a given `types` with predetermined structure and settings"""

    world_type: type[SCMLBaseWorld] = SCML2024StdWorld
    non_competitors: tuple[str | type[OneShotAgent], ...] = DefaultAgentsStd


@define
class GeneralStdContext(GeneralContext):
    """A context that generates std worlds with agents of a given `types` with predetermined structure and settings"""

    world_type: type[SCMLBaseWorld] = SCML2024StdWorld
    non_competitors: tuple[str | type[OneShotAgent], ...] = DefaultAgentsStd
    perishable: bool = False
    horizon: int = STD_DEFAULT_PARAMS["horizon"]  # type: ignore
    n_processes: tuple[int, int] | int = STD_DEFAULT_PARAMS["n_processes"]  # type: ignore
    disposal_cost: tuple[float, float] | float = STD_DEFAULT_PARAMS["disposal_cost"]
    disposal_cost_dev: tuple[float, float] | float = STD_DEFAULT_PARAMS[
        "disposal_cost_dev"
    ]
    storage_cost: tuple[float, float] | float = STD_DEFAULT_PARAMS["storage_cost"]
    storage_cost_dev: tuple[float, float] | float = STD_DEFAULT_PARAMS[
        "storage_cost_dev"
    ]
    price_range_fraction: float | tuple[float, float] = STD_DEFAULT_PARAMS[
        "price_range_fraction"
    ]
    price_multiplier: np.ndarray | tuple[float, float] | float = STD_DEFAULT_PARAMS[
        "price_multiplier"
    ]
    wide_price_range: bool = STD_DEFAULT_PARAMS["wide_price_range"]  # type: ignore
    one_time_per_negotiation: bool = STD_DEFAULT_PARAMS["one_time_per_negotiation"]  # type: ignore
    quantity_multiplier: int = STD_DEFAULT_PARAMS["quantity_multiplier"]  # type: ignore
    max_productivity: tuple[float, float] | float = STD_DEFAULT_PARAMS[  # type: ignore
        "max_productivity"
    ]
    max_supply: tuple[float, float] | float = STD_DEFAULT_PARAMS["max_supply"]
    exogenous_supply_predictability: tuple[float, float] | float = STD_DEFAULT_PARAMS[
        "exogenous_supply_predictability"
    ]
    exogenous_sales_predictability: tuple[float, float] | float = STD_DEFAULT_PARAMS[
        "exogenous_sales_predictability"
    ]
    cap_exogenous_quantities: bool = STD_DEFAULT_PARAMS["cap_exogenous_quantities"]  # type: ignore


@define
class FixedPartnerNumbersStdContext(FixedPartnerNumbersContext):
    """Generates a world limiting the range of the agent level, production capacity
    and the number of suppliers, consumers, and optionally same-level competitors."""

    world_type: type[SCMLBaseWorld] = SCML2024StdWorld
    non_competitors: tuple[str | type[OneShotAgent], ...] = DefaultAgentsStd


@define
class LimitedPartnerNumbersStdContext(LimitedPartnerNumbersOneShotContext):
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
class ANACStdContext(GeneralStdContext):
    """Generates a oneshot world with no constraints except compatibility with a specific ANAC competition year."""

    year: int = 2024

    def __attrs_post_init__(self):
        object.__setattr__(
            self,
            "world_type",
            {
                2024: SCML2024StdWorld,
            }[self.year],
        )


@define
class MiddleManStdContext(LimitedPartnerNumbersOneShotContext):
    """A world context that can generate any world compatible with the observation manager"""

    def __init__(self, *args, **kwargs):
        n_agents_per_process = (
            min(N_SUPPLIERS[0], N_CONSUMERS[0]),  # type: ignore
            max(N_SUPPLIERS[1], N_CONSUMERS[1]),  # type: ignore
        )
        kwargs |= dict(
            n_suppliers=N_SUPPLIERS,  # suppliers have no suppliers
            n_consumers=N_CONSUMERS,
            n_competitors=(N_SUPPLIERS[0] - 1, N_SUPPLIERS[1] - 1),
            n_agents_per_process=n_agents_per_process,
            level=0,  # suppliers are always in the first level
        )
        super().__init__(*args, **kwargs)


@define
class StrongMiddleManStdContext(MiddleManStdContext):
    """A supplier with almost many consumers relative to competitors"""

    def __init__(self, *args, **kwargs):
        kwargs |= dict(selling_strength=Strength.Strong)
        super().__init__(*args, **kwargs)


@define
class BalancedMiddleManStdContext(MiddleManStdContext):
    """A supplier with almost same number of consumers as competitors"""

    def __init__(self, *args, **kwargs):
        kwargs |= dict(selling_strength=Strength.Balanced)
        super().__init__(*args, **kwargs)


@define
class WeakMiddleManStdContext(MiddleManStdContext):
    """A supplier with few consumers relative to competitors"""

    def __init__(self, *args, **kwargs):
        kwargs |= dict(selling_strength=Strength.Weak)
        super().__init__(*args, **kwargs)


@define
class SupplierStdContext(SupplierContext):
    """A world context that can generate any world compatible with the observation manager"""

    world_type: type[SCMLBaseWorld] = SCML2024StdWorld
    non_competitors: tuple[str | type[OneShotAgent], ...] = DefaultAgentsStd


@define
class StrongSupplierStdContext(SupplierStdContext):
    """A supplier with almost many consumers relative to competitors"""

    def __init__(self, *args, **kwargs):
        kwargs |= dict(selling_strength=Strength.Strong)
        super().__init__(*args, **kwargs)


@define
class BalancedSupplierStdContext(SupplierStdContext):
    """A supplier with almost same number of consumers as competitors"""

    def __init__(self, *args, **kwargs):
        kwargs |= dict(selling_strength=Strength.Balanced)
        super().__init__(*args, **kwargs)


@define
class WeakSupplierStdContext(SupplierStdContext):
    """A supplier with few consumers relative to competitors"""

    def __init__(self, *args, **kwargs):
        kwargs |= dict(selling_strength=Strength.Weak)
        super().__init__(*args, **kwargs)


@define
class ConsumerStdContext(ConsumerContext):
    """A world context that can generate any world compatible with the observation manager"""

    world_type: type[SCMLBaseWorld] = SCML2024StdWorld
    non_competitors: tuple[str | type[OneShotAgent], ...] = DefaultAgentsStd


@define
class StrongConsumerStdContext(ConsumerStdContext):
    """A consumer with almost many suppliers relative to competitors"""

    def __init__(self, *args, **kwargs):
        kwargs |= dict(buying_strength=Strength.Strong)
        super().__init__(*args, **kwargs)


@define
class BalancedConsumerStdContext(ConsumerStdContext):
    """A consumer with almost same number of suppliers as competitors"""

    def __init__(self, *args, **kwargs):
        kwargs |= dict(buying_strength=Strength.Balanced)
        super().__init__(*args, **kwargs)


@define
class WeakConsumerStdContext(ConsumerStdContext):
    """A consumer with few suppliers relative to competitors"""

    def __init__(self, *args, **kwargs):
        kwargs |= dict(buying_strength=Strength.Weak)
        super().__init__(*args, **kwargs)


@define
class StdContext(GeneralStdContext):
    """A basic context fixing stationary world config parameters"""


@define
class RepeatingStdContext(RepeatingContext):
    """Encapsulates one or more configs and switches between them when asked to generate or make something."""

    world_type: type[SCMLBaseWorld] = SCML2024StdWorld
    non_competitors: tuple[str | type[OneShotAgent], ...] = DefaultAgentsStd
