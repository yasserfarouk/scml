"""Implements the SCML 2020 world design.

The detailed description of this world simulation can be found here_.

.. _here: http://www.yasserm.com/scml/scml2020.pdf

"""

from .common import *
from .factory import *
from .awi import *
from .world import *
from .agent import *
from .components import *
from .agents import *
from . import utils
from .utils import *


def builtin_agent_types(as_str=False):
    """
    Returns all built-in agents.

    Args:
        as_str: If true, the full type name will be returned otherwise the
                type object itself.
    """
    from negmas.helpers import get_class

    types = [
        f"scml.scml2020.agents.{_}" for _ in agents.__all__ if not _.startswith("Java")
    ]
    if as_str:
        return types
    return [get_class(_) for _ in types]


__all__ = (
    common.__all__
    + agents.__all__
    + world.__all__
    + components.__all__
    + factory.__all__
    + awi.__all__
    + agent.__all__
    + ["utils"]
)
