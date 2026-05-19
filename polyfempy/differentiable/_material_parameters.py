"""Compatibility shim for ``polyfempy.differentiable.material.parameters``."""

from .material.parameters import *  # noqa: F401,F403
from .material.parameters import (  # noqa: F401
    _expand_material_parameter_to_slots,
    _solver_n_element_assembly_slots,
)
