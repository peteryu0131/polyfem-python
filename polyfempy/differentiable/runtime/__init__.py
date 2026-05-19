"""Runtime pieces for differentiable PolyFEM solves."""

from .result import DifferentiableMaterialResult, DifferentiableResult
from .settings import (
    DifferentiableSolveContract,
    build_solver_from_settings,
    prepare_differentiable_solve_contract,
    prepare_settings_only_differentiable_contract,
)
from .solve import (
    prepare_differentiable_simulation,
    solve_differentiable,
    solve_differentiable_material,
    solve_differentiable_material_from_youngs,
)

__all__ = [
    "DifferentiableMaterialResult",
    "DifferentiableResult",
    "DifferentiableSolveContract",
    "build_solver_from_settings",
    "prepare_differentiable_simulation",
    "prepare_differentiable_solve_contract",
    "prepare_settings_only_differentiable_contract",
    "solve_differentiable",
    "solve_differentiable_material",
    "solve_differentiable_material_from_youngs",
]
