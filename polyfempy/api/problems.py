"""Problem helper classes used by :mod:`polyfempy.api`.

These classes provide the small set of predefined problems that
``SimulationConfig.to_settings()`` still supports. They intentionally keep the
backend-facing ``name()`` and ``params()`` shape used by the original bindings.
"""

from __future__ import annotations

import types
from typing import Any, Dict, Optional, Type


class Problem:
    """Generic scalar/tensor problem payload for backend settings."""

    def __init__(self, rhs=None, exact=None, is_time_dependent: bool = False):
        self.rhs = rhs
        self.is_time_dependent = is_time_dependent
        self.exact = exact
        self.dirichlet_boundary = []
        self.neumann_boundary = []
        self.pressure_boundary = []
        self.initial_solution = None
        self.initial_velocity = None
        self.initial_acceleration = None

    def set_dirichlet_value(self, id, value, is_dirichlet_dim=None, linear_ramp_to=None):
        self.add_dirichlet_value(id, value, is_dirichlet_dim, linear_ramp_to)

    def set_neumann_value(self, id, value, linear_ramp_to=None):
        self.add_neumann_value(id, value, linear_ramp_to)

    def set_pressure_value(self, id, value, linear_ramp_to=None):
        self.add_pressure_value(id, value, linear_ramp_to)

    def add_dirichlet_value(self, id, value, is_dirichlet_dim=None, linear_ramp_to=None):
        entry = {"id": id, "value": value}
        if is_dirichlet_dim is not None:
            assert len(is_dirichlet_dim) in (2, 3)
            assert len(value) == len(is_dirichlet_dim)
            entry["dimension"] = is_dirichlet_dim
        if linear_ramp_to is not None:
            entry["linear_ramp"] = {"to": linear_ramp_to}
        if not isinstance(value, (types.LambdaType, types.FunctionType)):
            self.dirichlet_boundary.append(entry)

    def add_neumann_value(self, id, value, linear_ramp_to=None):
        entry = {"id": id, "value": value}
        if linear_ramp_to is not None:
            entry["linear_ramp"] = {"to": linear_ramp_to}
        if not isinstance(value, (types.LambdaType, types.FunctionType)):
            self.neumann_boundary.append(entry)

    def add_pressure_value(self, id, value, linear_ramp_to=None):
        entry = {"id": id, "value": value}
        if linear_ramp_to is not None:
            entry["linear_ramp"] = {"to": linear_ramp_to}
        if not isinstance(value, (types.LambdaType, types.FunctionType)):
            self.pressure_boundary.append(entry)

    def set_initial_solution(self, value):
        self.initial_solution = value

    def set_initial_velocity(self, value):
        self.initial_velocity = value

    def set_initial_acceleration(self, value):
        self.initial_acceleration = value

    def set_velocity(self, id, value, is_dim_fixed=None, linear_ramp_to=None):
        self.add_dirichlet_value(id, value, is_dim_fixed, linear_ramp_to)

    def set_displacement(self, id, value, is_dim_fixed=None, linear_ramp_to=None):
        self.add_dirichlet_value(id, value, is_dim_fixed, linear_ramp_to)

    def set_force(self, id, value, linear_ramp_to=None):
        self.add_neumann_value(id, value, linear_ramp_to)

    def set_x_symmetric(self, id):
        self.add_dirichlet_value(id, [0, 0], [True, False])

    def set_y_symmetric(self, id):
        self.add_dirichlet_value(id, [0, 0], [False, True])

    def set_xy_symmetric(self, id):
        self.add_dirichlet_value(id, [0, 0, 0], [True, True, False])

    def set_xz_symmetric(self, id):
        self.add_dirichlet_value(id, [0, 0, 0], [True, False, True])

    def set_yz_symmetric(self, id):
        self.add_dirichlet_value(id, [0, 0, 0], [False, True, True])

    def params(self):
        result = dict(self.__dict__)
        if self.initial_solution is None:
            result.pop("initial_solution", None)
        if self.initial_velocity is None:
            result.pop("initial_velocity", None)
        if self.initial_acceleration is None:
            result.pop("initial_acceleration", None)
        return result


class Franke:
    """Franke scalar problem with an exact solution."""

    def name(self):
        return "Franke"

    def params(self):
        return {}


class GenericScalar:
    """Generic scalar problem."""

    def __init__(self):
        self.rhs = 0
        self.exact = None
        self.dirichlet_boundary = []
        self.neumann_boundary = []

    def add_dirichlet_value(self, id, value):
        self.dirichlet_boundary.append({"id": id, "value": value})

    def add_neumann_value(self, id, value):
        self.neumann_boundary.append({"id": id, "value": value})

    def name(self):
        return "GenericScalar"

    def params(self):
        return self.__dict__


class Gravity:
    """Time-dependent gravity problem."""

    def __init__(self, force=0.1):
        self.force = force

    def name(self):
        return "Gravity"

    def params(self):
        return self.__dict__


class Torsion:
    """3D torsion problem.

    The backend-facing problem name is ``TorsionElastic``. The misspelled
    ``axis_coordiante`` attribute is preserved for compatibility with older
    JSON/problem payloads.
    """

    def __init__(
        self,
        axis_coordiante=None,
        axis_coordinate=2,
        n_turns=0.5,
        fixed_boundary=5,
        turning_boundary=6,
    ):
        self.axis_coordiante = axis_coordiante if axis_coordiante is not None else axis_coordinate
        self.n_turns = n_turns
        self.fixed_boundary = fixed_boundary
        self.turning_boundary = turning_boundary

    def name(self):
        return "TorsionElastic"

    def params(self):
        return self.__dict__


TorsionElastic = Torsion


class GenericTensor:
    """Generic tensor problem."""

    def __init__(self):
        self.rhs = [0, 0, 0]
        self.exact = None
        self.dirichlet_boundary = []
        self.neumann_boundary = []

    def set_velocity(self, id, value, is_dim_fixed=None):
        self.add_dirichlet_value(id, value, is_dim_fixed)

    def set_displacement(self, id, value, is_dim_fixed=None):
        self.add_dirichlet_value(id, value, is_dim_fixed)

    def set_force(self, id, value):
        self.add_neumann_value(id, value)

    def add_dirichlet_value(self, id, value, is_dirichlet_dim=None):
        assert len(value) in (2, 3)
        entry = {"id": id, "value": value}
        if is_dirichlet_dim is not None:
            assert len(is_dirichlet_dim) in (2, 3)
            entry["dimension"] = is_dirichlet_dim
        self.dirichlet_boundary.append(entry)

    def add_neumann_value(self, id, value):
        self.neumann_boundary.append({"id": id, "value": value})

    def name(self):
        return "GenericTensor"

    def params(self):
        return self.__dict__


class Flow:
    """Inflow/outflow Stokes problem."""

    def __init__(
        self,
        inflow=1,
        outflow=3,
        inflow_amout=None,
        outflow_amout=None,
        direction=0,
        obstacle=None,
        inflow_amount=None,
        outflow_amount=None,
    ):
        if obstacle is None:
            obstacle = [7]
        if inflow_amout is None:
            inflow_amout = 0.25 if inflow_amount is None else inflow_amount
        if outflow_amout is None:
            outflow_amout = 0.25 if outflow_amount is None else outflow_amount
        self.inflow = inflow
        self.outflow = outflow
        self.inflow_amout = inflow_amout
        self.outflow_amout = outflow_amout
        self.direction = direction
        self.obstacle = obstacle

    def name(self):
        return "Flow"

    def params(self):
        return self.__dict__


class DrivenCavity:
    """Classical driven-cavity flow problem."""

    def name(self):
        return "DrivenCavity"

    def params(self):
        return {}


class FlowWithObstacle:
    """Flow with obstacle problem."""

    def __init__(self, U=1.5, time_dependent=True):
        self.U = U
        self.time_dependent = time_dependent

    def name(self):
        return "FlowWithObstacle"

    def params(self):
        return self.__dict__


_PROBLEM_CLASSES: Dict[str, Type[Any]] = {
    "Franke": Franke,
    "GenericScalar": GenericScalar,
    "Gravity": Gravity,
    "Torsion": Torsion,
    "TorsionElastic": Torsion,
    "GenericTensor": GenericTensor,
    "Flow": Flow,
    "DrivenCavity": DrivenCavity,
    "FlowWithObstacle": FlowWithObstacle,
}


def get_problem_class(name: str) -> Optional[Type[Any]]:
    """Return the predefined problem class for ``name`` if one is supported."""

    return _PROBLEM_CLASSES.get(name)


def available_problem_names():
    """Return supported predefined problem names."""

    return tuple(_PROBLEM_CLASSES)


__all__ = [
    "Problem",
    "Franke",
    "GenericScalar",
    "Gravity",
    "Torsion",
    "TorsionElastic",
    "GenericTensor",
    "Flow",
    "DrivenCavity",
    "FlowWithObstacle",
    "available_problem_names",
    "get_problem_class",
]
