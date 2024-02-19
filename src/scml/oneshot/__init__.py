from .agent import *
from .agents import *
from .awi import *
from .common import *
from .context import *
from .policy import *
from .rl import *
from .sysagents import *
from .ufun import *
from .world import *


def builtin_agent_types(as_str=False):
    """
    Returns all built-in agents.

    Args:
        as_str: If true, the full type name will be returned otherwise the
                type object itself.
    """
    from negmas.helpers import get_class

    types = [
        f"scml.oneshot.agents.{_}"
        for _ in agents.__all__
        if _ not in ("StdPlaceholder", "Placeholder", "OneShotDummyAgent")
    ]
    if as_str:
        return types
    return [get_class(_) for _ in types]


__all__ = (
    common.__all__
    + world.__all__
    + ufun.__all__
    + agent.__all__
    + policy.__all__
    + agents.__all__
    + awi.__all__
    + context.__all__
    + rl.__all__
    + ["builtin_agent_types"]
)
