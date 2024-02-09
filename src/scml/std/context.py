from typing import Any

from attr import define, field

from scml.oneshot.context import (
    ANACContext,
    Context,
    FixedPartnerNumbersContext,
    GeneralContext,
    LimitedPartnerNumbersContext,
)
from scml.std.agent import StdAgent
from scml.std.agents.nothing import StdDummyAgent
from scml.std.world import SCML2024StdWorld

__all__ = [
    "Context",
    "FixedPartnerNumbersContext",
    "LimitedPartnerNumbersContext",
    "ANACContext",
    "FixedPartnerNumbersStdContext",
    "LimitedPartnerNumbersStdContext",
    "ANACStdContext",
    "GeneralContext",
]


@define(frozen=True)
class FixedPartnerNumbersStdContext(FixedPartnerNumbersContext):
    def make(
        self,
        types: tuple[type[StdAgent], ...] = (StdDummyAgent,),
        params: tuple[dict[str, Any], ...] | None = None,
    ) -> SCML2024StdWorld:
        return super().make_world(SCML2024StdWorld, types, params)  # type: ignore


@define(frozen=True)
class LimitedPartnerNumbersStdContext(LimitedPartnerNumbersContext):
    """Generates a standard world limiting the range of the agent level, production capacity
    and the number of suppliers, consumers, and optionally same-level competitors."""

    submanager_context: type[FixedPartnerNumbersContext] = field(
        init=False, default=FixedPartnerNumbersStdContext
    )

    def make(
        self,
        types: tuple[type[StdAgent], ...] = (StdDummyAgent,),
        params: tuple[dict[str, Any], ...] | None = None,
    ) -> SCML2024StdWorld:
        return super().make_world(SCML2024StdWorld, types, params)  # type: ignore


@define(frozen=True)
class ANACStdContext(ANACContext):
    """Generates a standard world with no constraints except compatibility with a specific ANAC competition year."""

    def make(
        self,
        types: tuple[type[StdAgent], ...] = (StdDummyAgent,),
        params: tuple[dict[str, Any], ...] | None = None,
    ) -> SCML2024StdWorld:
        return super().make_world(SCML2024StdWorld, types, params)  # type: ignore
