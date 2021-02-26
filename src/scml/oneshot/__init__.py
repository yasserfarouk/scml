from .common import *
from .awi import *
from .sysagents import *
from .ufun import *
from .world import *
from .agent import *
from .agents import *


def builtin_agent_types(as_str=False):
    """
    Returns all built-in agents.

    Args:
        as_str: If true, the full type name will be returned otherwise the
                type object itself.
    """
    from negmas.helpers import get_class

    types = [f"scml.oneshot.agents.{_}" for _ in agents.__all__]
    if as_str:
        return types
    return [get_class(_) for _ in types]


__all__ = (
    common.__all__
    + world.__all__
    + ufun.__all__
    + agent.__all__
    + agents.__all__
    + awi.__all__
    + ["builtin_agent_types"]
)
