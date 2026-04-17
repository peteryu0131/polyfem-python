"""PyTorch integration for PolyFEM differentiable simulations.

Route A only:
- Always `import polyfempy as pf` (stable C++ extension module name).
- nanobind vs pybind11 is a build-time backend choice and must not change Python imports.

Important:
- After ``solve()``, displacement for autograd is taken **first** from ``get_solutions()`` so
  ``solve_adjoint(grad_output)`` matches the cached layout (especially transient). The return
  value dict/tuple is used only as fallback.
"""

# pyright: reportMissingImports=false
# torch is an optional dependency, so type checker may not find it

import numpy as np
from typing import Any

try:
    import torch  # pyright: ignore[reportMissingImports]
    from torch.autograd import Function  # pyright: ignore[reportMissingImports]
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    Function = object  # Placeholder for type hints

from .cpp_ext import get_cpp_polyfempy


class PolyFEMFunction(Function):
    """PyTorch Function wrapper for PolyFEM with automatic gradient computation.

    Handles cache management, adjoint solving, and derivative computation.
    Use solve_differentiable() instead of this directly.
    """
    
    @staticmethod
    def forward(
        ctx,
        solver,
        vertices: "torch.Tensor",
        derivative_type: str = "shape",
        solve_log_level: int = 2,
    ) -> "torch.Tensor":
        """Forward pass: run simulation and cache results.

        ``solve_log_level`` matches ``api.solve`` / JSON ``output.log.level`` (0=trace … 6=off).
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for differentiable simulations")
        
        vertices_np = vertices.detach().cpu().numpy()
        solver.mesh().set_vertices(vertices_np)
        
        # After set_vertices() changes the geometry, we MUST rebuild basis and reassemble.
        # This is critical: the old basis and matrices were built for the old geometry.
        # solve() will also call build_basis() internally, but we need to do it here
        # explicitly to ensure the geometry change is properly handled.
        if hasattr(solver, "build_basis"):
            solver.build_basis()
        if hasattr(solver, "assemble"):
            solver.assemble()
        
        # Enable derivative caching (required for adjoint)
        pf = get_cpp_polyfempy()
        solver.set_cache_level(pf.CacheLevel.Derivatives)

        try:
            ret = solver.solve(log_level=solve_log_level)
        except TypeError:
            ret = solver.solve()

        # Prefer get_solutions() so adjoint_rhs matches diff_cached layout (transient / multi-step).
        solutions_np = None
        if hasattr(solver, "get_solutions"):
            try:
                solutions_np = np.asarray(solver.get_solutions())
                if solutions_np.size == 0:
                    solutions_np = None
            except Exception:
                solutions_np = None
        if solutions_np is None and isinstance(ret, dict) and ret.get("_result_bundle") and "u" in ret:
            solutions_np = np.asarray(ret["u"])
        elif solutions_np is None and isinstance(ret, (tuple, list)) and len(ret) > 0:
            solutions_np = np.asarray(ret[0])
        elif solutions_np is None:
            # Last-resort compatibility: try cache object if exposed
            if hasattr(solver, "get_solution_cache"):
                cache = solver.get_solution_cache()
                if hasattr(cache, "solution"):
                    solutions_np = np.asarray(cache.solution(0))
                elif hasattr(cache, "displacement"):
                    solutions_np = np.asarray(cache.displacement(0))

        if solutions_np is None:
            raise RuntimeError(
                "Failed to retrieve solution after solve(): no return tuple and no known getters."
            )
        
        sol_tensor = torch.tensor(solutions_np, dtype=vertices.dtype, device=vertices.device)
        
        ctx.solver = solver
        ctx.derivative_type = derivative_type
        ctx.vertices_shape = vertices.shape
        ctx.vertices_dtype = vertices.dtype
        ctx.vertices_device = vertices.device
        
        return sol_tensor
    
    @staticmethod
    def backward(ctx, grad_output: "torch.Tensor") -> tuple:
        """Backward pass: solve adjoint and compute derivatives."""
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for differentiable simulations")
        
        pf = get_cpp_polyfempy()

        grad_output_np = grad_output.detach().cpu().numpy()
        ctx.solver.solve_adjoint(grad_output_np)
        
        if ctx.derivative_type == "shape":
            grad_np = pf.shape_derivative(ctx.solver)
        elif ctx.derivative_type == "periodic_shape":
            # For periodic problems, use periodic_shape_derivative if available
            # Otherwise fall back to regular shape_derivative
            # Note: This requires periodic_shape_derivative to be exposed in polyfempy
            if hasattr(pf, "periodic_shape_derivative"):
                grad_np = pf.periodic_shape_derivative(ctx.solver)
            else:
                # Fallback: use regular shape_derivative for now
                # TODO: Add periodic_shape_derivative to polyfempy bindings
                import warnings
                warnings.warn(
                    "periodic_shape_derivative not available in polyfempy. "
                    "Using regular shape_derivative instead. "
                    "For proper periodic support, add periodic_shape_derivative to adjoint.cpp bindings.",
                    RuntimeWarning
                )
                grad_np = pf.shape_derivative(ctx.solver)
        elif ctx.derivative_type == "material":
            grad_np = pf.elastic_material_derivative(ctx.solver)
        elif ctx.derivative_type == "initial_velocity":
            grad_dict = pf.initial_velocity_derivative(ctx.solver)
            grad_np = np.array(list(grad_dict.values())).flatten()
        else:
            raise ValueError(f"Unknown derivative type: {ctx.derivative_type}")
        
        grad_tensor = torch.tensor(
            grad_np, 
            dtype=ctx.vertices_dtype,
            device=ctx.vertices_device
        )
        
        if grad_tensor.shape != ctx.vertices_shape:
            if grad_tensor.numel() == np.prod(ctx.vertices_shape):
                grad_tensor = grad_tensor.reshape(ctx.vertices_shape)

        # forward(ctx, solver, vertices, derivative_type, solve_log_level) → 5 slots; no grad for non-tensors.
        return None, grad_tensor, None, None


def _elastic_material_derivative_to_grad_lam_mu(
    raw: np.ndarray,
    n_el: int,
    lam: "torch.Tensor",
    mu: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Map C++ ``elastic_material_derivative`` array to per-element ``dL/dλ``, ``dL/dμ`` tensors."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim == 2 and raw.shape[0] == 2 and raw.shape[1] > n_el:
        raw = raw[:, :n_el]
    if raw.ndim == 2 and raw.shape[1] == 2 and raw.shape[0] > n_el:
        raw = raw[:n_el, :]
    if raw.ndim == 2:
        if raw.shape == (2, n_el):
            g_lam = torch.as_tensor(raw[0], dtype=lam.dtype, device=lam.device)
            g_mu = torch.as_tensor(raw[1], dtype=mu.dtype, device=mu.device)
        elif raw.shape == (n_el, 2):
            g_lam = torch.as_tensor(raw[:, 0], dtype=lam.dtype, device=lam.device)
            g_mu = torch.as_tensor(raw[:, 1], dtype=mu.dtype, device=mu.device)
        else:
            raise RuntimeError(
                f"elastic_material_derivative matrix unexpected shape {raw.shape}, n_el={n_el}"
            )
    elif raw.ndim == 1 and raw.size == 2 * n_el:
        g_lam = torch.as_tensor(raw[:n_el], dtype=lam.dtype, device=lam.device)
        g_mu = torch.as_tensor(raw[n_el:], dtype=mu.dtype, device=mu.device)
    else:
        raise RuntimeError(
            f"elastic_material_derivative unexpected shape {raw.shape}, n_el={n_el}"
        )
    return g_lam, g_mu


class PolyFEMPerElementMaterialFunction(Function):
    """Per-element Lamé (λ, μ) as differentiable inputs: ``set_per_element_material`` → solve → adjoint.

    Use ``solve_differentiable_material()`` instead of calling this class directly.
    Pair with a scalar design variable by building ``lam, mu`` from ``E`` in PyTorch (chain rule).
    """

    @staticmethod
    def forward(
        ctx,
        solver: Any,
        lam: "torch.Tensor",
        mu: "torch.Tensor",
        solve_log_level: int,
        forward_solve_cache: str,
    ) -> "torch.Tensor":
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for differentiable simulations")

        pf = get_cpp_polyfempy()
        solver.set_per_element_material(
            lam.detach().cpu().numpy(),
            mu.detach().cpu().numpy(),
        )

        cache = str(forward_solve_cache).strip().lower()
        if cache in ("none", "off", "0"):
            pass
        elif cache in ("solution",):
            solver.set_cache_level(pf.CacheLevel.Solution)
        else:
            solver.set_cache_level(pf.CacheLevel.Derivatives)

        try:
            ret = solver.solve(log_level=int(solve_log_level))
        except TypeError:
            ret = solver.solve()

        solutions_np = None
        if hasattr(solver, "get_solutions"):
            try:
                solutions_np = np.asarray(solver.get_solutions())
                if solutions_np.size == 0:
                    solutions_np = None
            except Exception:
                solutions_np = None
        if solutions_np is None and isinstance(ret, dict) and ret.get("_result_bundle") and "u" in ret:
            solutions_np = np.asarray(ret["u"])
        elif solutions_np is None and isinstance(ret, (tuple, list)) and len(ret) > 0:
            solutions_np = np.asarray(ret[0])
        elif solutions_np is None:
            if hasattr(solver, "get_solution_cache"):
                cache_obj = solver.get_solution_cache()
                if hasattr(cache_obj, "solution"):
                    solutions_np = np.asarray(cache_obj.solution(0))
                elif hasattr(cache_obj, "displacement"):
                    solutions_np = np.asarray(cache_obj.displacement(0))

        if solutions_np is None:
            raise RuntimeError(
                "solve_differentiable_material: failed to retrieve solution after solve()."
            )

        sol_tensor = torch.tensor(
            solutions_np, dtype=lam.dtype, device=lam.device
        )

        ctx.solver = solver
        ctx.save_for_backward(lam, mu)
        return sol_tensor

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: "torch.Tensor") -> tuple:
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for differentiable simulations")

        pf = get_cpp_polyfempy()
        grad_output_np = grad_output.detach().cpu().numpy()
        ctx.solver.solve_adjoint(grad_output_np)

        lam, mu = ctx.saved_tensors
        n_el = int(lam.numel())
        raw = np.asarray(pf.elastic_material_derivative(ctx.solver), dtype=np.float64)
        g_lam, g_mu = _elastic_material_derivative_to_grad_lam_mu(raw, n_el, lam, mu)

        # forward(ctx, solver, lam, mu, solve_log_level, forward_solve_cache) — grads for lam, mu only
        return None, g_lam, g_mu, None, None

