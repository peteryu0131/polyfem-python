"""Reusable helpers for scalar-Young's-modulus material experiments.

These helpers keep experiment scripts focused on their config and optimization
loop while sharing the common material-objective plumbing:

- probing PolyFEM objective RHS terms
- summarizing the gradient chain back to ``E`` / ``log_E``
- finite-difference fallback and validation for scalar ``E`` examples
"""

# pyright: reportMissingImports=false

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from .objective_bridge import create_polyfem_objective, material_design_vector
from .summary import gradient_norm
from .solve_diff import (
    _differentiable_config_and_settings,
    build_solver_from_settings,
    solve_differentiable_material_from_youngs,
    solver_body_slot_mask,
    youngs_value_to_internal,
)
from ..api.report import summarize_history_bundle
from ..api.runtime import format_history_summary


def _as_materials_list(settings: dict[str, Any]) -> list[dict[str, Any]]:
    materials = settings.get("materials", [])
    if isinstance(materials, dict):
        materials = [materials]
    if not isinstance(materials, list):
        return []
    return [dict(item) for item in materials if isinstance(item, dict)]


def _material_id(material: dict[str, Any]) -> int | None:
    raw = material.get("id")
    if isinstance(raw, list):
        if len(raw) != 1:
            return None
        raw = raw[0]
    if raw in (None, ""):
        return None
    return int(raw)


def _material_for_body(settings: dict[str, Any], *, body_id: int) -> dict[str, Any]:
    for material in _as_materials_list(settings):
        if _material_id(material) == int(body_id):
            return material
    raise ValueError(f"could not find material with id/body_id={body_id!r} in cfg.materials")


def _other_material_for_body(
    settings: dict[str, Any],
    *,
    body_id: int,
    other_body_id: Optional[int] = None,
) -> dict[str, Any] | None:
    if other_body_id is not None:
        return _material_for_body(settings, body_id=int(other_body_id))

    others = [
        material
        for material in _as_materials_list(settings)
        if _material_id(material) != int(body_id)
    ]
    if not others:
        return None
    if len(others) == 1:
        return others[0]
    ids = [_material_id(material) for material in others]
    raise ValueError(
        "material optimization can only infer one non-design material; "
        f"found other material ids {ids}. Pass other_body_id or explicit other_E_value/other_nu."
    )


def _value_and_unit(raw: Any, *, default_unit: str = "Pa") -> tuple[float, str]:
    if isinstance(raw, dict):
        if "value" in raw:
            unit = str(raw.get("unit", default_unit))
            return float(raw["value"]), unit
        if "amount" in raw:
            unit = str(raw.get("unit", default_unit))
            return float(raw["amount"]), unit
    return float(raw), default_unit


def _youngs_from_material(
    material: dict[str, Any],
    *,
    default_unit: str = "Pa",
) -> tuple[float, str]:
    for key in ("E", "e", "young", "youngs", "youngs_modulus", "young_modulus"):
        if key in material:
            return _value_and_unit(material[key], default_unit=default_unit)
    raise ValueError(f"material id={material.get('id')} does not define Young's modulus E")


def _nu_from_material(material: dict[str, Any]) -> float:
    for key in ("nu", "poisson", "poisson_ratio"):
        if key in material:
            return float(material[key])
    raise ValueError(f"material id={material.get('id')} does not define Poisson ratio nu")


def _objective_probe_param_type(
    name: str,
    *,
    elastic_objective_names: tuple[str, ...],
) -> str:
    return "elastic" if str(name) in elastic_objective_names else "shape"


def _objective_probe_vector(result: Any, *, param_type: str) -> np.ndarray:
    if str(param_type) == "elastic":
        return material_design_vector(result.lam, result.mu)
    return np.asarray(result.solver.mesh().vertices(), dtype=np.float64).reshape(-1)


def objective_solution_rhs_diagnostics(
    *,
    result: Any,
    objective_names: tuple[str, ...],
    volume_selection: int,
    power: int,
    state_cols: list[int],
    elastic_objective_names: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Probe PolyFEM objective value and raw ``dJ/du`` before backward."""
    out: dict[str, Any] = {}
    for name in objective_names:
        param_type = _objective_probe_param_type(
            str(name),
            elastic_objective_names=elastic_objective_names,
        )
        x = _objective_probe_vector(result, param_type=param_type)
        per_state = []
        for state_col in state_cols:
            record: dict[str, Any] = {"state_col": int(state_col)}
            try:
                obj = create_polyfem_objective(
                    solver=result.solver,
                    objective_name=str(name),
                    param_type=param_type,
                    volume_selection=int(volume_selection),
                    state=int(state_col),
                    power=int(power),
                )
                raw = np.asarray(
                    obj.derivative(result.solver, x, wrt="solution"),
                    dtype=np.float64,
                )
                record.update(
                    {
                        "value": float(obj.value(x)),
                        "rhs_shape": [int(v) for v in raw.shape],
                        "rhs_norm": float(np.linalg.norm(raw.reshape(-1))),
                        "rhs_max_abs": float(np.max(np.abs(raw))) if raw.size else 0.0,
                    }
                )
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            per_state.append(record)
        out[str(name)] = per_state
    return out


def print_objective_rhs_diagnostics(diag: dict[str, Any]) -> None:
    """Print a compact summary of objective RHS probes."""
    print("[objective dJ/du probe]")
    for name, rows in diag.items():
        for row in rows:
            prefix = f"{name}[state={row['state_col']}]"
            if "error" in row:
                print(f"{prefix}: ERROR {row['error']}")
            else:
                print(
                    f"{prefix}: value={row['value']:.6e} "
                    f"rhs_norm={row['rhs_norm']:.6e} "
                    f"rhs_max_abs={row['rhs_max_abs']:.6e} "
                    f"rhs_shape={tuple(row['rhs_shape'])}"
                )


def _tensor_norm_or_none(tensor: torch.Tensor | None) -> float | None:
    if tensor is None:
        return None
    return float(torch.linalg.norm(tensor.detach()).cpu().item())


def _masked_tensor_norm_or_none(
    tensor: torch.Tensor | None,
    mask: torch.Tensor,
    *,
    invert: bool = False,
) -> float | None:
    if tensor is None:
        return None
    m = mask.to(device=tensor.device, dtype=torch.bool)
    if invert:
        m = ~m
    if int(m.sum().detach().cpu().item()) == 0:
        return 0.0
    return float(torch.linalg.norm(tensor.detach()[m]).cpu().item())


def _scalar_grad_or_none(tensor: torch.Tensor | None) -> float | None:
    if tensor is None or tensor.grad is None:
        return None
    return float(tensor.grad.detach().cpu().reshape(-1)[0].item())


def format_optional(value: float | None) -> str:
    """Render optional floats compactly for terminal diagnostics."""
    if value is None:
        return "None"
    return f"{value:.6e}"


def usable_scalar_gradient(value: float | None, *, zero_tol: float = 1e-20) -> bool:
    """Return True when a scalar gradient is finite and above a tiny threshold."""
    return value is not None and np.isfinite(value) and abs(float(value)) > float(zero_tol)


def _dE_dlogE(log_e: torch.Tensor) -> float:
    """Derivative of ``softplus(log_e) + floor`` wrt ``log_e``."""
    return float(torch.sigmoid(log_e.detach()).cpu().item())


def apply_finite_difference_gradient_fallback(
    *,
    log_e: torch.nn.Parameter,
    E_lattice: torch.Tensor,
    fd: dict[str, Any],
) -> dict[str, float]:
    """Replace the scalar optimizer gradient with finite-difference ``dL/dlog_E``."""
    grad_E = float(fd["finite_difference_grad_E"])
    dE_dlog = _dE_dlogE(log_e)
    grad_log_E = grad_E * dE_dlog
    log_e.grad = torch.tensor(
        grad_log_E,
        dtype=log_e.dtype,
        device=log_e.device,
    )
    E_lattice.grad = torch.tensor(
        grad_E,
        dtype=E_lattice.dtype,
        device=E_lattice.device,
    )
    return {
        "finite_difference_grad_E": grad_E,
        "dE_dlogE": dE_dlog,
        "finite_difference_grad_log_E": grad_log_E,
    }


def collect_material_chain_diagnostics(
    *,
    result: Any,
    E_lattice: torch.Tensor,
    log_e: torch.Tensor,
    lattice_mask: torch.Tensor,
) -> dict[str, Any]:
    """Summarize every gradient hop from loss back to scalar ``E``."""
    lam_grad = result.lam.grad
    mu_grad = result.mu.grad
    u_grad = result.u.grad
    return {
        "solution_shape": [int(x) for x in result.u.shape],
        "u_grad_norm_dL_du": _tensor_norm_or_none(u_grad),
        "lam_grad_norm_all": _tensor_norm_or_none(lam_grad),
        "lam_grad_norm_lattice": _masked_tensor_norm_or_none(lam_grad, lattice_mask),
        "lam_grad_norm_non_lattice": _masked_tensor_norm_or_none(
            lam_grad,
            lattice_mask,
            invert=True,
        ),
        "mu_grad_norm_all": _tensor_norm_or_none(mu_grad),
        "mu_grad_norm_lattice": _masked_tensor_norm_or_none(mu_grad, lattice_mask),
        "mu_grad_norm_non_lattice": _masked_tensor_norm_or_none(
            mu_grad,
            lattice_mask,
            invert=True,
        ),
        "grad_E_lattice": _scalar_grad_or_none(E_lattice),
        "grad_log_E": _scalar_grad_or_none(log_e),
    }


def print_material_chain_diagnostics(diag: dict[str, Any]) -> None:
    """Print the material-gradient chain in the order it should light up."""
    print("[gradient chain]")
    print(f"solution_shape: {tuple(diag['solution_shape'])}")
    print(f"u_grad_norm_dL_du: {format_optional(diag['u_grad_norm_dL_du'])}")
    print(
        "lambda_grad_norm: "
        f"all={format_optional(diag['lam_grad_norm_all'])} "
        f"lattice={format_optional(diag['lam_grad_norm_lattice'])} "
        f"non_lattice={format_optional(diag['lam_grad_norm_non_lattice'])}"
    )
    print(
        "mu_grad_norm: "
        f"all={format_optional(diag['mu_grad_norm_all'])} "
        f"lattice={format_optional(diag['mu_grad_norm_lattice'])} "
        f"non_lattice={format_optional(diag['mu_grad_norm_non_lattice'])}"
    )
    print(f"grad_E_lattice: {format_optional(diag['grad_E_lattice'])}")
    print(f"grad_log_E: {format_optional(diag['grad_log_E'])}")


@dataclass
class ScalarMaterialOptimizationStep:
    """Compact record for one completed scalar-``E`` optimization step."""

    iteration: int
    E_value: float
    E_unit: str
    loss: torch.Tensor
    objective_info: Any
    gradient: Optional[torch.Tensor]
    history_bundle: Optional[dict[str, Any]] = None


@dataclass
class ScalarMaterialOptimizationProblem:
    """Reusable payload for optimizing one positive Young's modulus scalar."""

    cfg: dict[str, Any]
    solver: Any
    slot_mask: torch.Tensor
    E_unit: str
    nu: float
    other_E_internal: float
    other_nu: float
    report_cfg: Any = None
    solve_log_level: int = 3
    e_floor: float = 1.0
    log_E: Optional[torch.nn.Parameter] = None
    current_E: Optional[torch.Tensor] = None

    def solve(self) -> Any:
        """Run one differentiable solve at the current scalar ``E``."""
        if self.log_E is None:
            raise ValueError("log_E is not initialized on this optimization problem")

        self.current_E = F.softplus(self.log_E) + float(self.e_floor)
        self.current_E.retain_grad()
        solver_E = youngs_value_to_internal(
            self.current_E,
            pressure_unit=self.E_unit,
            solver_settings=self.cfg,
        )
        result = solve_differentiable_material_from_youngs(
            solver=self.solver,
            E=solver_E,
            nu=float(self.nu),
            slot_mask=self.slot_mask,
            other_E=float(self.other_E_internal),
            other_nu=float(self.other_nu),
            solve_log_level=int(self.solve_log_level),
        )
        result.meta.setdefault("_solve_settings", self.cfg)
        return result

    def make_optimizer(self, *, name: str = "adam", lr: float = 1e-2) -> torch.optim.Optimizer:
        """Create a PyTorch optimizer for the scalar design variable."""
        if self.log_E is None:
            raise ValueError("log_E is not initialized on this optimization problem")

        opt_name = str(name).strip().lower()
        if opt_name == "adam":
            return torch.optim.Adam([self.log_E], lr=float(lr))
        if opt_name == "sgd":
            return torch.optim.SGD([self.log_E], lr=float(lr))
        raise ValueError(f"unsupported optimizer {name!r}")

    def release_solver(self) -> None:
        """Drop the reusable C++ solver reference held by this problem."""
        self.solver = None

    def optimize(
        self,
        *,
        steps: int,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[Any], Any],
        collect_history: bool = False,
    ) -> Iterator[ScalarMaterialOptimizationStep]:
        """Yield completed optimization steps for a user-supplied loss."""
        for iteration in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            result = self.solve()
            try:
                loss_out = loss_fn(result)
                if isinstance(loss_out, tuple):
                    loss, objective_info = loss_out
                else:
                    loss, objective_info = loss_out, None

                loss.backward()
                gradient = None
                if self.current_E is not None and self.current_E.grad is not None:
                    gradient = self.current_E.grad.detach().clone()
                E_value = (
                    float(self.current_E.detach().item())
                    if self.current_E is not None
                    else float("nan")
                )
                history_bundle = (
                    summarize_history_bundle(
                        result,
                        cfg=self.report_cfg if self.report_cfg is not None else self.cfg,
                    )
                    if collect_history
                    else None
                )
                optimizer.step()

                yield ScalarMaterialOptimizationStep(
                    iteration=iteration,
                    E_value=E_value,
                    E_unit=self.E_unit,
                    loss=loss.detach(),
                    objective_info=objective_info,
                    gradient=gradient,
                    history_bundle=history_bundle,
                )
            finally:
                result.release_solver()


def prepare_material_optimization_problem(
    *,
    cfg: Any,
    body_id: int,
    initial_E_value: Optional[float] = None,
    E_unit: Optional[str] = None,
    nu: Optional[float] = None,
    other_body_id: Optional[int] = None,
    other_E_value: Optional[float] = None,
    other_E_unit: Optional[str] = None,
    other_nu: Optional[float] = None,
    solve_log_level: int = 3,
    e_floor: float = 1.0,
    root_path: Optional[str] = None,
) -> ScalarMaterialOptimizationProblem:
    """Prepare a config-backed reusable solver for material ``E`` optimization.

    E/nu values are inferred from ``cfg.materials`` by ``body_id`` when omitted.
    Explicit arguments remain available as overrides for experiments that want
    to start from values different from the config. By default, repeated
    optimization solves use warning-level solver output and keep ``E`` above
    ``e_floor=1.0`` in the selected ``E_unit``.
    """
    _, _, settings, _ = _differentiable_config_and_settings(
        cfg,
        root_path=root_path,
    )
    material = _material_for_body(settings, body_id=int(body_id))
    cfg_E_value, cfg_E_unit = _youngs_from_material(material)
    cfg_nu = _nu_from_material(material)

    initial_E_value = cfg_E_value if initial_E_value is None else float(initial_E_value)
    E_unit = cfg_E_unit if E_unit is None else str(E_unit)
    nu = cfg_nu if nu is None else float(nu)

    other_material = _other_material_for_body(
        settings,
        body_id=int(body_id),
        other_body_id=other_body_id,
    )
    if other_E_value is None or other_nu is None:
        if other_material is None:
            other_cfg_E_value, other_cfg_E_unit, other_cfg_nu = (
                initial_E_value,
                E_unit,
                nu,
            )
        else:
            other_cfg_E_value, other_cfg_E_unit = _youngs_from_material(other_material)
            other_cfg_nu = _nu_from_material(other_material)
        other_E_value = (
            other_cfg_E_value
            if other_E_value is None
            else float(other_E_value)
        )
        other_E_unit = (
            other_cfg_E_unit
            if other_E_unit is None
            else str(other_E_unit)
        )
        other_nu = other_cfg_nu if other_nu is None else float(other_nu)
    elif other_E_unit is None:
        other_E_unit = E_unit

    solver = build_solver_from_settings(settings)
    slot_mask = solver_body_slot_mask(solver, body_id=int(body_id))
    other_E_internal = float(
        youngs_value_to_internal(
            float(other_E_value),
            pressure_unit=other_E_unit,
            solver_settings=settings,
        )
    )
    init_log_E = np.log(np.expm1(max(float(initial_E_value) - float(e_floor), 1e-12)))
    log_E = torch.nn.Parameter(torch.tensor(init_log_E, dtype=torch.get_default_dtype()))
    return ScalarMaterialOptimizationProblem(
        cfg=settings,
        solver=solver,
        slot_mask=slot_mask,
        E_unit=E_unit,
        nu=float(nu),
        other_E_internal=other_E_internal,
        other_nu=float(other_nu),
        report_cfg=cfg,
        solve_log_level=int(solve_log_level),
        e_floor=float(e_floor),
        log_E=log_E,
    )


def prepare_material_differentiable_simulation(
    problem: ScalarMaterialOptimizationProblem,
) -> Any:
    """Run one differentiable material simulation from a prepared problem.

    This mirrors ``prepare_differentiable_simulation(...)`` at the experiment
    level, but uses the material problem's current scalar E design variable
    instead of treating mesh vertices as the differentiable input.
    """
    return problem.solve()


def make_material_optimizer(
    problem: ScalarMaterialOptimizationProblem,
    *,
    name: str = "adam",
    lr: float = 1e-2,
) -> torch.optim.Optimizer:
    """Create a PyTorch optimizer for a scalar material optimization problem."""
    return problem.make_optimizer(name=name, lr=lr)


def format_scalar_material_optimization_step(step: ScalarMaterialOptimizationStep) -> str:
    """Format one completed scalar material optimization step."""
    loss_value = float(step.loss.detach().cpu().item())
    objective_info = step.objective_info if isinstance(step.objective_info, dict) else {}
    unit_suffix = f"_{step.E_unit}" if step.E_unit else ""
    lines = [
        f"iter {step.iteration}: E{unit_suffix}={step.E_value:.6e}",
        f"loss: {loss_value:.6e}",
        f"grad_norm: {gradient_norm(step.gradient):.6e}",
    ]
    if objective_info.get("resolved_objective_name") is not None:
        lines.append(f"objective: {objective_info['resolved_objective_name']}")
    if objective_info.get("time_aggregation") is not None:
        lines.append(f"time_aggregation: {objective_info['time_aggregation']}")
    if objective_info.get("selected_state_col") is not None:
        lines.append(f"selected_state_col: {objective_info['selected_state_col']}")
    return "\n".join(lines)


def format_scalar_material_optimization_history_summary(
    steps: list[ScalarMaterialOptimizationStep],
) -> str:
    """Format per-iteration PolyFEM history summaries for scalar material optimization."""
    sections: list[str] = ["# Scalar Material Optimization History Summary"]
    if not steps:
        sections.append("iterations: 0")
        return "\n".join(sections) + "\n"

    for step in steps:
        sections.extend(
            [
                "",
                f"## iter {step.iteration}",
                format_scalar_material_optimization_step(step),
                "",
            ]
        )
        if step.history_bundle is None:
            sections.append("history_available: false")
            sections.append("note: history collection was not enabled")
        else:
            sections.append(format_history_summary(step.history_bundle).rstrip())
    return "\n".join(sections) + "\n"


def run_scalar_material_optimization(
    problem: ScalarMaterialOptimizationProblem,
    *,
    steps: int,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[Any], Any],
    print_steps: bool = True,
    preferred_objective_name: Optional[str] = None,
    summary_path: Optional[Union[str, Path]] = None,
    history_summary_path: Optional[Union[str, Path]] = None,
    release_solver: bool = True,
) -> list[ScalarMaterialOptimizationStep]:
    """Run a complete scalar material optimization loop and return completed steps."""
    completed_steps: list[ScalarMaterialOptimizationStep] = []
    report_lines: list[str] = []

    def write_reports() -> None:
        if summary_path is not None:
            path = Path(summary_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            text = "\n\n".join(report_lines)
            path.write_text(text + ("\n" if text else ""), encoding="utf-8")

        if history_summary_path is not None:
            path = Path(history_summary_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                format_scalar_material_optimization_history_summary(completed_steps),
                encoding="utf-8",
            )

    try:
        for step in problem.optimize(
            steps=steps,
            optimizer=optimizer,
            loss_fn=loss_fn,
            collect_history=history_summary_path is not None,
        ):
            objective_info = step.objective_info if isinstance(step.objective_info, dict) else {}
            resolved_objective_name = str(objective_info.get("resolved_objective_name", "unknown"))
            step_text = format_scalar_material_optimization_step(step)
            report_lines.append(step_text)
            if print_steps:
                print(step_text)
                if (
                    step.iteration == 0
                    and preferred_objective_name is not None
                    and resolved_objective_name != str(preferred_objective_name)
                ):
                    print(
                        f"objective fallback: preferred {preferred_objective_name!r} is not available "
                        f"in the loaded extension, so using legacy {resolved_objective_name!r}"
                    )
            completed_steps.append(step)
            write_reports()
    finally:
        if completed_steps:
            write_reports()
        if release_solver:
            problem.release_solver()

    return completed_steps


def evaluate_material_loss_value_for_E(
    *,
    solver_settings: dict[str, Any],
    body_id: int,
    E_value: float,
    E_unit: str,
    nu: float,
    other_E_value: float,
    other_E_unit: str,
    other_nu: float,
    solve_log_level: int,
    loss_builder: Callable[[Any], Any],
) -> tuple[float, Any]:
    """Evaluate a material loss at one scalar Young's modulus value."""
    solver = build_solver_from_settings(solver_settings)
    result = None
    try:
        slot_mask = solver_body_slot_mask(solver, body_id=int(body_id))
        E_tensor = torch.tensor(float(E_value), dtype=torch.get_default_dtype())
        solver_E = youngs_value_to_internal(
            E_tensor,
            pressure_unit=E_unit,
            solver_settings=solver_settings,
        )
        solver_other_E = float(
            youngs_value_to_internal(
                float(other_E_value),
                pressure_unit=other_E_unit,
                solver_settings=solver_settings,
            )
        )
        result = solve_differentiable_material_from_youngs(
            solver=solver,
            E=solver_E,
            nu=float(nu),
            slot_mask=slot_mask,
            other_E=solver_other_E,
            other_nu=float(other_nu),
            solve_log_level=int(solve_log_level),
        )
        built = loss_builder(result)
        if isinstance(built, tuple) and len(built) == 2:
            loss, info = built
        else:
            loss, info = built, None
        return float(loss.detach().cpu().item()), info
    finally:
        if result is not None:
            result.release_solver()
        solver = None


def finite_difference_material_E_gradient(
    *,
    solver_settings: dict[str, Any],
    body_id: int,
    center_E_value: float,
    epsilon_E_value: float,
    e_floor_value: float,
    E_unit: str,
    nu: float,
    other_E_value: float,
    other_E_unit: str,
    other_nu: float,
    solve_log_level: int,
    loss_builder: Callable[[Any], Any],
) -> dict[str, Any]:
    """Centered finite-difference check for ``d loss / d E``."""
    center = float(center_E_value)
    eps = abs(float(epsilon_E_value))
    lower = float(e_floor_value) + max(1e-9, 1e-9 * abs(center))
    E_plus = center + eps
    E_minus = max(center - eps, lower)
    loss_plus, info_plus = evaluate_material_loss_value_for_E(
        solver_settings=solver_settings,
        body_id=body_id,
        E_value=E_plus,
        E_unit=E_unit,
        nu=nu,
        other_E_value=other_E_value,
        other_E_unit=other_E_unit,
        other_nu=other_nu,
        solve_log_level=solve_log_level,
        loss_builder=loss_builder,
    )
    loss_minus, info_minus = evaluate_material_loss_value_for_E(
        solver_settings=solver_settings,
        body_id=body_id,
        E_value=E_minus,
        E_unit=E_unit,
        nu=nu,
        other_E_value=other_E_value,
        other_E_unit=other_E_unit,
        other_nu=other_nu,
        solve_log_level=solve_log_level,
        loss_builder=loss_builder,
    )
    return {
        "epsilon_E": eps,
        "E_plus": E_plus,
        "E_minus": E_minus,
        "loss_plus": loss_plus,
        "loss_minus": loss_minus,
        "finite_difference_grad_E": (loss_plus - loss_minus) / (E_plus - E_minus),
        "info_plus": info_plus,
        "info_minus": info_minus,
    }


__all__ = [
    "ScalarMaterialOptimizationProblem",
    "ScalarMaterialOptimizationStep",
    "apply_finite_difference_gradient_fallback",
    "collect_material_chain_diagnostics",
    "evaluate_material_loss_value_for_E",
    "finite_difference_material_E_gradient",
    "format_optional",
    "format_scalar_material_optimization_history_summary",
    "format_scalar_material_optimization_step",
    "make_material_optimizer",
    "objective_solution_rhs_diagnostics",
    "print_material_chain_diagnostics",
    "print_objective_rhs_diagnostics",
    "prepare_material_differentiable_simulation",
    "prepare_material_optimization_problem",
    "run_scalar_material_optimization",
    "usable_scalar_gradient",
]
