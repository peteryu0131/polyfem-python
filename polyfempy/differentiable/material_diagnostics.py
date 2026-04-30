"""Advanced diagnostics for scalar Young's-modulus material optimization."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from .objective_bridge import create_polyfem_objective, material_design_vector
from .solve_diff import (
    build_solver_from_settings,
    solve_differentiable_material_from_youngs,
    solver_body_slot_mask,
    youngs_value_to_internal,
)


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
        param_type = "elastic" if str(name) in elastic_objective_names else "shape"
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


def format_optional(value: float | None) -> str:
    """Render optional floats compactly for terminal diagnostics."""
    if value is None:
        return "None"
    return f"{value:.6e}"


def usable_scalar_gradient(value: float | None, *, zero_tol: float = 1e-20) -> bool:
    """Return True when a scalar gradient is finite and above a tiny threshold."""
    return value is not None and np.isfinite(value) and abs(float(value)) > float(zero_tol)


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
    log_e: Optional[torch.Tensor],
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
    loss_builder: Any,
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
    loss_builder: Any,
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


def _objective_probe_vector(result: Any, *, param_type: str) -> np.ndarray:
    if str(param_type) == "elastic":
        return material_design_vector(result.lam, result.mu)
    return np.asarray(result.solver.mesh().vertices(), dtype=np.float64).reshape(-1)


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


def _dE_dlogE(log_e: torch.Tensor) -> float:
    """Derivative of ``softplus(log_e) + floor`` wrt ``log_e``."""
    return float(torch.sigmoid(log_e.detach()).cpu().item())


__all__ = [
    "apply_finite_difference_gradient_fallback",
    "collect_material_chain_diagnostics",
    "evaluate_material_loss_value_for_E",
    "finite_difference_material_E_gradient",
    "format_optional",
    "objective_solution_rhs_diagnostics",
    "print_material_chain_diagnostics",
    "print_objective_rhs_diagnostics",
    "usable_scalar_gradient",
]
