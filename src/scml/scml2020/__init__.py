"""Implements the SCML 2020 world design"""

from .world import *
from .agents import *

__all__ = agents.__all__ + world.__all__
