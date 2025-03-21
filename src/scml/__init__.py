__author__ = """Yasser Mohammad"""
__email__ = "yasserfarouk@gmail.com"
__version__ = "0.7.6"

from .scml2019 import *
from .scml2020 import *
from .oneshot import *

__all__ = scml2020.__all__ + oneshot.__all__ + ["utils", "runner"]
