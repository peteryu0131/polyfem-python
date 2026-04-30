"""Legacy PyTorch-outer / PolyFEM-inner bridge helpers.

These helpers were useful while validating the first differentiable API:
- patch differentiable runtime settings
- run one differentiable bridge step
- run a tiny optimizer-style probe loop

New user code should prefer the clearer problem-based API:

``prepare_differentiable_simulation(...)``
``prepare_optimization_problem(...)``
``make_optimizer(...)``
``run_optimization(...)``

The functions in this file are kept for compatibility with older experiments
and for low-level probes. They are also re-exported from
``polyfempy.differentiable.advanced``.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polyfempy as pf
import torch

from .objective_bridge import get_direct_von_mises_monitor, make_polyfem_autograd_loss
from .result_diff import DifferentiableResult
from .solve_diff import solve_differentiable
from .torch_integration import PolyFEMFunction


@dataclass
class PolyFEMTorchBridgeStep:
    """One PyTorch-style PolyFEM bridge step."""

    result: Any
    loss: torch.Tensor
    state_col: int
    forward_elapsed: float
    backward_elapsed: float | None
    grad_norm: float


@dataclass
class PolyFEMTorchBridgeOptimizerStep:
    """One optimizer iteration in the PyTorch outer loop."""

    iteration: int
    loss: float
    grad_norm: float
    step_norm: float


@dataclass
class PolyFEMTorchBridgeOptimizerProbe:
    """Summary of a thin optimizer-style probe."""

    optimizer_name: str
    lr: float
    state_col: int
    forward_elapsed_total: float
    backward_elapsed_total: float
    direct_von_mises_monitor: dict[str, Any]
    gradient_summary: dict[str, Any] | None
    steps: list[PolyFEMTorchBridgeOptimizerStep]


TORCH_BRIDGE_METHOD_NAME = "PolyFEM Torch Bridge (legacy)"
TORCH_BRIDGE_METHOD_PATTERN = "PyTorch outer optimize + PolyFEM autograd inner objective"


def console_log_level_from_settings(settings: dict[str, Any]) -> int:
    """Map ``output.log.level`` to the integer solve log level."""
    level = 2
    out = settings.get("output")
    if isinstance(out, dict):
        log = out.get("log")
        if isinstance(log, dict):
            raw = log.get("level", "info")
            name = raw.strip().lower() if isinstance(raw, str) else str(raw).strip().lower()
            level_map = {
                "trace": 0,
                "debug": 1,
                "info": 2,
                "warn": 3,
                "warning": 3,
                "error": 4,
                "critical": 5,
                "off": 6,
            }
            level = level_map.get(name, level)
    return level


def solver_set_log_level_off(solver: Any) -> None:
    """Best-effort helper to silence setup logs."""
    if hasattr(solver, "set_log_level"):
        try:
            solver.set_log_level(6)
        except Exception:
            pass


def make_torch_optimizer(*, name: str, param: torch.nn.Parameter, lr: float) -> torch.optim.Optimizer:
    """Create a simple PyTorch optimizer for a single design parameter tensor."""
    opt_name = str(name).strip().lower()
    if opt_name == "adam":
        return torch.optim.Adam([param], lr=float(lr))
    if opt_name == "sgd":
        return torch.optim.SGD([param], lr=float(lr))
    raise ValueError(f"unsupported optimizer {name!r}")


def apply_differentiable_runtime_patches(*, cfg: Any, optimizer_probe: bool) -> dict[str, Any]:
    """Patch config for differentiable runs.

    1. differentiable contact currently needs a constant barrier stiffness
    2. optimizer-probe mode should avoid writing a large VTU/PVD sequence
    """
    patches: dict[str, Any] = {
        "solver_contact_barrier_stiffness": None,
        "optimizer_probe_save_paraview_disabled": False,
    }

    solver = getattr(cfg, "solver", None)
    solver_contact = getattr(solver, "contact", None)
    if solver_contact is not None and hasattr(solver_contact, "barrier_stiffness"):
        solver_contact.barrier_stiffness = 1e3
        patches["solver_contact_barrier_stiffness"] = 1e3

    if optimizer_probe:
        output = getattr(cfg, "output", None)
        if output is not None and hasattr(output, "save_paraview"):
            output.save_paraview = False
            patches["optimizer_probe_save_paraview_disabled"] = True

    return patches


def gradient_summary(grad: torch.Tensor | None) -> dict[str, Any] | None:
    """Compute a compact numeric summary for a gradient tensor."""
    if grad is None:
        return None

    a = np.asarray(grad.detach().cpu().numpy())
    out: dict[str, Any] = {
        "shape": [int(x) for x in a.shape],
        "dtype": str(a.dtype),
        "size": int(a.size),
    }
    if a.size == 0:
        return out

    flat = a.astype(np.float64, copy=False).reshape(-1)
    out.update(
        {
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
            "mean": float(np.mean(flat)),
            "p95": float(np.percentile(flat, 95.0)),
        }
    )
    return out


def array_summary(arr: Any) -> dict[str, Any]:
    """Compute a compact numeric summary for an arbitrary array-like object."""
    a = np.asarray(arr)
    out: dict[str, Any] = {
        "shape": [int(x) for x in a.shape],
        "dtype": str(a.dtype),
        "size": int(a.size),
    }
    if a.size == 0:
        return out

    flat = a.astype(np.float64, copy=False).reshape(-1)
    out.update(
        {
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
            "mean": float(np.mean(flat)),
            "p95": float(np.percentile(flat, 95.0)),
        }
    )
    return out


def summarize_gradient_norm(result: Any) -> float:
    """Return the L2 norm of vertex gradients, or NaN when missing."""
    grad = result.vertices.grad
    if grad is None:
        return float("nan")
    return float(torch.linalg.norm(grad.detach()).cpu().item())


def run_polyfem_differentiable_forward(
    *,
    cfg: Any,
    root_path: str | None = None,
    polyfem_console_log: bool = False,
) -> tuple[Any, float]:
    """Run differentiable forward solve and return ``(result, elapsed_seconds)``."""
    t0 = time.perf_counter()
    result = solve_differentiable(
        cfg=cfg,
        root_path=root_path,
        derivative_type="shape",
        quiet_polyfem_setup=not bool(polyfem_console_log),
    )
    elapsed = time.perf_counter() - t0
    return result, elapsed


def evaluate_polyfem_loss(
    *,
    result: Any,
    objective_name: str,
    objective_power: int,
    objective_volume_selection: int,
    objective_state: str,
) -> tuple[torch.Tensor, int]:
    """Build the autograd-connected optimization loss from the differentiable result."""
    return make_polyfem_autograd_loss(
        result=result,
        objective_name=str(objective_name),
        objective_power=int(objective_power),
        objective_volume_selection=int(objective_volume_selection),
        objective_state=str(objective_state),
    )


def run_backward_if_requested(*, loss: torch.Tensor, run_backward: bool) -> float | None:
    """Run backward if enabled and return elapsed seconds, otherwise ``None``."""
    if not bool(run_backward):
        return None
    t0 = time.perf_counter()
    loss.backward()
    return time.perf_counter() - t0


def run_polyfem_bridge_step(
    *,
    cfg: Any,
    root_path: str | None = None,
    objective_name: str,
    objective_power: int,
    objective_volume_selection: int,
    objective_state: str,
    polyfem_console_log: bool = False,
    run_backward: bool = True,
) -> PolyFEMTorchBridgeStep:
    """Run one complete bridge step: forward -> loss -> backward -> grad summary."""
    result, forward_elapsed = run_polyfem_differentiable_forward(
        cfg=cfg,
        root_path=root_path,
        polyfem_console_log=polyfem_console_log,
    )
    loss, state_col = evaluate_polyfem_loss(
        result=result,
        objective_name=objective_name,
        objective_power=objective_power,
        objective_volume_selection=objective_volume_selection,
        objective_state=objective_state,
    )
    backward_elapsed = run_backward_if_requested(loss=loss, run_backward=run_backward)
    grad_norm = summarize_gradient_norm(result)
    return PolyFEMTorchBridgeStep(
        result=result,
        loss=loss,
        state_col=state_col,
        forward_elapsed=forward_elapsed,
        backward_elapsed=backward_elapsed,
        grad_norm=grad_norm,
    )


def _prepare_optimizer_probe_solver(*, cfg: Any, polyfem_console_log: bool) -> tuple[Any, torch.nn.Parameter, int]:
    settings = cfg.to_dict()
    solver = pf.Solver()
    if not bool(polyfem_console_log):
        solver_set_log_level_off(solver)
    solver.set_settings(json.dumps(settings), strict_validation=False)
    solver.load_mesh_from_settings()
    solver.build_basis()
    vertices_np = np.asarray(solver.mesh().vertices(), dtype=np.float64)
    vertices = torch.nn.Parameter(
        torch.as_tensor(vertices_np, dtype=torch.get_default_dtype())
    )
    solve_log_level = console_log_level_from_settings(settings)
    return solver, vertices, solve_log_level


def run_polyfem_bridge_optimizer_probe(
    *,
    cfg: Any,
    objective_name: str,
    objective_power: int,
    objective_volume_selection: int,
    objective_state: str,
    optimizer_name: str = "adam",
    optimizer_lr: float = 1e-5,
    optimizer_probe_iters: int = 3,
    polyfem_console_log: bool = False,
) -> PolyFEMTorchBridgeOptimizerProbe:
    """Run a tiny optimizer-style outer loop around the PolyFEM bridge."""
    solver, design_vertices, solve_log_level = _prepare_optimizer_probe_solver(
        cfg=cfg,
        polyfem_console_log=polyfem_console_log,
    )
    optimizer = make_torch_optimizer(
        name=str(optimizer_name),
        param=design_vertices,
        lr=float(optimizer_lr),
    )

    steps: list[PolyFEMTorchBridgeOptimizerStep] = []
    forward_elapsed_total = 0.0
    backward_elapsed_total = 0.0
    last_result: DifferentiableResult | None = None
    last_state_col = 0
    prev_vertices = design_vertices.detach().clone()

    for iteration in range(int(optimizer_probe_iters)):
        optimizer.zero_grad(set_to_none=True)

        t_forward = time.perf_counter()
        solution = PolyFEMFunction.apply(solver, design_vertices, "shape", int(solve_log_level))
        forward_elapsed_total += time.perf_counter() - t_forward

        result = DifferentiableResult(
            u=solution,
            solver=solver,
            derivative_type="shape",
            differentiable_params=["geometry"],
            vertices=design_vertices,
        )
        loss, state_col = evaluate_polyfem_loss(
            result=result,
            objective_name=objective_name,
            objective_power=objective_power,
            objective_volume_selection=objective_volume_selection,
            objective_state=objective_state,
        )

        t_backward = time.perf_counter()
        loss.backward()
        backward_elapsed_total += time.perf_counter() - t_backward

        grad = design_vertices.grad
        grad_norm = float("nan") if grad is None else float(torch.linalg.norm(grad.detach()).cpu().item())

        optimizer.step()
        step_norm = float(torch.linalg.norm(design_vertices.detach() - prev_vertices).cpu().item())
        prev_vertices = design_vertices.detach().clone()

        steps.append(
            PolyFEMTorchBridgeOptimizerStep(
                iteration=iteration,
                loss=float(loss.detach().cpu().item()),
                grad_norm=grad_norm,
                step_norm=step_norm,
            )
        )
        last_result = result
        last_state_col = state_col

    if last_result is None:
        raise RuntimeError("optimizer probe ran with zero iterations")

    probe = PolyFEMTorchBridgeOptimizerProbe(
        optimizer_name=str(optimizer_name),
        lr=float(optimizer_lr),
        state_col=int(last_state_col),
        forward_elapsed_total=float(forward_elapsed_total),
        backward_elapsed_total=float(backward_elapsed_total),
        direct_von_mises_monitor=get_direct_von_mises_monitor(last_result),
        gradient_summary=gradient_summary(design_vertices.grad),
        steps=steps,
    )
    last_result.solver = None
    return probe


def write_polyfem_bridge_step_report(
    *,
    workspace: Path,
    cfg: Any,
    cfg_path: Path,
    step: PolyFEMTorchBridgeStep,
    objective_name: str,
    objective_power: int,
    objective_volume_selection: int,
    objective_state_spec: str,
    runtime_patches: dict[str, Any],
    expected_time_points: int | None = None,
    report_stem: str = "api_diff_probe",
    gradient_filename: str = "shape_gradient.npy",
    probe_name: str = "polyfem_torch_bridge_step",
    probe_goal: str | None = None,
) -> Path:
    """Write a generic single-step bridge report and companion text summary."""
    vm_monitor = get_direct_von_mises_monitor(step.result)
    grad = step.result.vertices.grad
    grad_summary = gradient_summary(grad)

    grad_npy_path: str | None = None
    if grad is not None:
        grad_path = (workspace / gradient_filename).resolve()
        np.save(grad_path, grad.detach().cpu().numpy())
        grad_npy_path = str(grad_path)

    goal = probe_goal or (
        "Run one differentiable bridge step: PolyFEM forward, PolyFEM objective loss, "
        "then PyTorch backward."
    )
    report = {
        "probe": {
            "name": probe_name,
            "goal": goal,
        },
        "method": {
            "name": TORCH_BRIDGE_METHOD_NAME,
            "pattern": TORCH_BRIDGE_METHOD_PATTERN,
            "loss_builder": "make_polyfem_autograd_loss",
            "monitor_channel": "get_direct_von_mises_monitor(result)",
            "optimization_channel": "make_polyfem_autograd_loss",
        },
        "run": {
            "workspace": str(workspace.resolve()),
            "config": str(cfg_path.resolve()),
            "forward_elapsed_seconds": float(step.forward_elapsed),
            "backward_elapsed_seconds": None
            if step.backward_elapsed is None
            else float(step.backward_elapsed),
            "time": {
                "t0": float(cfg.time.t0),
                "dt": float(cfg.time.dt),
                "tend": float(cfg.time.tend),
                "expected_time_points": expected_time_points,
            },
            "objective": {
                "name": str(objective_name),
                "power": int(objective_power),
                "volume_selection": int(objective_volume_selection),
                "state_spec": str(objective_state_spec),
                "state_col": int(step.state_col),
            },
        },
        "runtime_patches": runtime_patches,
        "result": {
            "u": array_summary(step.result.u.detach().cpu().numpy()),
            "direct_von_mises_available": bool(vm_monitor["available"]),
            "direct_von_mises": vm_monitor["summary"],
            "direct_von_mises_role": vm_monitor["role"],
        },
        "loss": {
            "value": float(step.loss.detach().cpu().item()),
            "backward_ran": step.backward_elapsed is not None,
            "source": "make_polyfem_autograd_loss",
        },
        "gradient": {
            "available": bool(grad is not None),
            "summary": grad_summary,
            "npy_path": grad_npy_path,
        },
    }

    report_path = (workspace / f"{report_stem}_report.json").resolve()
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        probe_name,
        f"workspace: {workspace.resolve()}",
        f"method: {TORCH_BRIDGE_METHOD_NAME}",
        f"pattern: {TORCH_BRIDGE_METHOD_PATTERN}",
        f"forward_elapsed_seconds: {step.forward_elapsed:.2f}",
        (
            "backward_elapsed_seconds: not_run"
            if step.backward_elapsed is None
            else f"backward_elapsed_seconds: {step.backward_elapsed:.2f}"
        ),
        f"time: t0={float(cfg.time.t0):g} dt={float(cfg.time.dt):g} tend={float(cfg.time.tend):g}",
        f"barrier_stiffness: {runtime_patches.get('solver_contact_barrier_stiffness')}",
        (
            "objective: "
            f"{str(objective_name)} power={int(objective_power)} "
            f"volume_selection={int(objective_volume_selection)} "
            f"state={int(step.state_col)}"
        ),
        f"loss: {float(step.loss.detach().cpu().item()):.6e}",
        f"direct_von_mises_available: {bool(vm_monitor['available'])}",
    ]
    if vm_monitor["summary"] is not None:
        summary = vm_monitor["summary"]
        lines.append(f"direct_von_mises_max: {float(summary['max']):.6e}")
        lines.append(f"direct_von_mises_p95: {float(summary['p95']):.6e}")
    if grad_summary is not None:
        lines.append(f"grad_shape: {tuple(int(x) for x in grad.shape)}")
        lines.append(f"grad_norm: {float(step.grad_norm):.6e}")
        lines.append(f"grad_abs_max: {float(np.max(np.abs(grad.detach().cpu().numpy()))):.6e}")
    else:
        lines.append("grad_norm: not_available")

    summary_path = (workspace / f"{report_stem}_summary.txt").resolve()
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def write_polyfem_bridge_optimizer_probe_report(
    *,
    workspace: Path,
    cfg: Any,
    cfg_path: Path,
    probe: PolyFEMTorchBridgeOptimizerProbe,
    objective_name: str,
    objective_power: int,
    objective_volume_selection: int,
    objective_state_spec: str,
    runtime_patches: dict[str, Any],
    report_stem: str = "api_diff_optimizer_probe",
    probe_name: str = "polyfem_torch_bridge_optimizer_probe",
    probe_goal: str | None = None,
) -> Path:
    """Write a generic optimizer-probe report and companion text summary."""
    goal = probe_goal or (
        "Show a thin PyTorch optimizer-style outer loop around the PolyFEM autograd bridge."
    )
    report = {
        "probe": {
            "name": probe_name,
            "goal": goal,
        },
        "method": {
            "name": TORCH_BRIDGE_METHOD_NAME,
            "pattern": TORCH_BRIDGE_METHOD_PATTERN,
            "mode": "optimizer_probe",
        },
        "optimizer": {
            "name": probe.optimizer_name,
            "lr": probe.lr,
            "iterations": len(probe.steps),
        },
        "run": {
            "workspace": str(workspace.resolve()),
            "config": str(cfg_path.resolve()),
            "time": {
                "t0": float(cfg.time.t0),
                "dt": float(cfg.time.dt),
                "tend": float(cfg.time.tend),
            },
            "objective": {
                "name": str(objective_name),
                "power": int(objective_power),
                "volume_selection": int(objective_volume_selection),
                "state_spec": str(objective_state_spec),
                "state_col": int(probe.state_col),
            },
        },
        "runtime_patches": runtime_patches,
        "timing": {
            "forward_elapsed_total_seconds": float(probe.forward_elapsed_total),
            "backward_elapsed_total_seconds": float(probe.backward_elapsed_total),
        },
        "direct_von_mises_monitor": probe.direct_von_mises_monitor,
        "gradient": {
            "summary": probe.gradient_summary,
        },
        "iterations": [
            {
                "iteration": int(step.iteration),
                "loss": float(step.loss),
                "grad_norm": float(step.grad_norm),
                "step_norm": float(step.step_norm),
            }
            for step in probe.steps
        ],
    }

    report_path = (workspace / f"{report_stem}_report.json").resolve()
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        probe_name,
        f"workspace: {workspace.resolve()}",
        f"method: {TORCH_BRIDGE_METHOD_NAME}",
        f"pattern: {TORCH_BRIDGE_METHOD_PATTERN}",
        f"optimizer: {probe.optimizer_name} lr={probe.lr:g} iters={len(probe.steps)}",
        f"objective: {str(objective_name)} power={int(objective_power)} state={int(probe.state_col)}",
        f"barrier_stiffness: {runtime_patches.get('solver_contact_barrier_stiffness')}",
        (
            "paraview_sequence_disabled: "
            f"{bool(runtime_patches.get('optimizer_probe_save_paraview_disabled', False))}"
        ),
        "note: this is a proof-of-concept PyTorch outer loop, not a full production optimizer.",
        "",
        "[iterations]",
        "iter  loss        grad_norm    step_norm",
    ]
    for step in probe.steps:
        lines.append(
            f"{int(step.iteration):<4}  {float(step.loss):<10.6e}  "
            f"{float(step.grad_norm):<10.6e}  {float(step.step_norm):<10.6e}"
        )

    if probe.direct_von_mises_monitor.get("summary") is not None:
        vm = probe.direct_von_mises_monitor["summary"]
        lines.extend(
            [
                "",
                "[direct_von_mises_monitor]",
                f"available: {bool(probe.direct_von_mises_monitor['available'])}",
                f"max: {float(vm['max']):.6e}",
                f"p95: {float(vm['p95']):.6e}",
            ]
        )

    summary_path = (workspace / f"{report_stem}_summary.txt").resolve()
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def print_polyfem_bridge_step_summary(
    *,
    cfg_path: Path,
    report_path: Path,
    step: PolyFEMTorchBridgeStep,
    objective_name: str,
    runtime_patches: dict[str, Any],
) -> None:
    """Print a compact terminal summary for a single bridge step."""
    print(f"differentiable forward took {step.forward_elapsed:.2f}s")
    if step.backward_elapsed is not None:
        print(f"backward() took {step.backward_elapsed:.2f}s")
    print(f"method: {TORCH_BRIDGE_METHOD_NAME}")
    print(f"pattern: {TORCH_BRIDGE_METHOD_PATTERN}")
    print(f"barrier_stiffness: {runtime_patches.get('solver_contact_barrier_stiffness')}")
    print(f"loss ({objective_name}): {float(step.loss.detach().cpu().item()):.6e}")
    print(f"grad_norm: {float(step.grad_norm):.6e}")
    print(f"config: {cfg_path}")
    print(f"api diff report: {report_path}")


def print_polyfem_bridge_optimizer_probe_summary(
    *,
    cfg_path: Path,
    report_path: Path,
    probe: PolyFEMTorchBridgeOptimizerProbe,
    runtime_patches: dict[str, Any],
) -> None:
    """Print a compact terminal summary for an optimizer probe."""
    print(f"method: {TORCH_BRIDGE_METHOD_NAME}")
    print(f"pattern: {TORCH_BRIDGE_METHOD_PATTERN}")
    print(f"barrier_stiffness: {runtime_patches.get('solver_contact_barrier_stiffness')}")
    print(
        f"optimizer probe: {probe.optimizer_name} lr={probe.lr:g} "
        f"iters={len(probe.steps)}"
    )
    print(
        "paraview_sequence_disabled: "
        f"{bool(runtime_patches.get('optimizer_probe_save_paraview_disabled', False))}"
    )
    print(
        f"forward_total: {probe.forward_elapsed_total:.2f}s  "
        f"backward_total: {probe.backward_elapsed_total:.2f}s"
    )
    if probe.steps:
        last = probe.steps[-1]
        print(f"last_loss: {last.loss:.6e}")
        print(f"last_grad_norm: {last.grad_norm:.6e}")
        print(f"last_step_norm: {last.step_norm:.6e}")
    print("note: this is a tiny torch.optim-style outer loop for explanation/probing.")
    print(f"config: {cfg_path}")
    print(f"optimizer probe report: {report_path}")


__all__ = [
    "PolyFEMTorchBridgeStep",
    "PolyFEMTorchBridgeOptimizerStep",
    "PolyFEMTorchBridgeOptimizerProbe",
    "TORCH_BRIDGE_METHOD_NAME",
    "TORCH_BRIDGE_METHOD_PATTERN",
    "console_log_level_from_settings",
    "solver_set_log_level_off",
    "make_torch_optimizer",
    "apply_differentiable_runtime_patches",
    "array_summary",
    "gradient_summary",
    "summarize_gradient_norm",
    "run_polyfem_differentiable_forward",
    "evaluate_polyfem_loss",
    "run_backward_if_requested",
    "run_polyfem_bridge_step",
    "run_polyfem_bridge_optimizer_probe",
    "write_polyfem_bridge_step_report",
    "write_polyfem_bridge_optimizer_probe_report",
    "print_polyfem_bridge_step_summary",
    "print_polyfem_bridge_optimizer_probe_summary",
]
