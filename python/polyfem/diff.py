"""Differentiable simulations and objectives for PyTorch."""

from typing import TYPE_CHECKING as _TYPE_CHECKING

from polyfempy import differentiable_api as _api
from polyfempy.differentiable_api import __all__ as __all__

if _TYPE_CHECKING:
    from polyfempy.differentiable_api import *

def __getattr__(name):
    if name in __all__:
        return getattr(_api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()).union(__all__))
