"""DifferentiableResult: result container for differentiable simulations."""

# pyright: reportMissingImports=false
# torch is an optional dependency, so type checker may not find it

from __future__ import annotations

from typing import Optional, Dict, Any, TYPE_CHECKING, Tuple
import numpy as np

if TYPE_CHECKING:
    import torch

try:
    import torch  # pyright: ignore[reportMissingImports]
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def scalar_stats_1d(arr: np.ndarray) -> dict[str, float | int]:
    """mean / max / p95 / n for a 1d sample array."""
    a = np.asarray(arr, dtype=np.float64).ravel()
    if a.size == 0:
        return {"mean": float("nan"), "max": float("nan"), "p95": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(a)),
        "max": float(np.max(a)),
        "p95": float(np.percentile(a, 95)),
        "n": int(a.size),
    }


def sampled_mises_flat_from_solver(solver: Any) -> Optional[Tuple[np.ndarray, str]]:
    """Try PolyFEM Solver methods that expose sampled von Mises (post-solve, not autograd).

    Tries in order: ``get_sampled_mises_avg_frames``, ``get_sampled_mises_frames``,
    ``get_sampled_mises_avg``, ``get_sampled_mises``. Returns ``(flattened array, method_name)``
    or ``None`` if nothing works.

    Requires C++ bindings that forward to ``polyfem::State`` when your PolyFEM build provides
    these members; otherwise methods may be missing or raise at runtime.
    """
    names = (
        "get_sampled_mises_avg_frames",
        "get_sampled_mises_frames",
        "get_sampled_mises_avg",
        "get_sampled_mises",
    )
    for name in names:
        if not hasattr(solver, name):
            continue
        fn = getattr(solver, name)
        if not callable(fn):
            continue
        try:
            out = fn()
            arr = np.asarray(out)
            if arr.size == 0:
                continue
            return arr.ravel(), name
        except Exception:
            continue
    return None


def try_numpy_von_mises_from_stress(solver: Any) -> Optional[np.ndarray]:
    """Optional path: if bindings expose ``get_stress`` (or similar), compute von Mises in numpy.

    Not wired by default — depends on your ``Solver`` API. Override or extend locally if needed.
    """
    return None


class DifferentiableResult:
    """Result container with PyTorch tensors supporting automatic gradients."""
    
    def __init__(
        self,
        u: "torch.Tensor",
        solver: Any,
        derivative_type: str = "shape",
        differentiable_params: Optional[list] = None,
        vertices: Optional["torch.Tensor"] = None,
        strain: Optional["torch.Tensor"] = None,
        stress: Optional["torch.Tensor"] = None,
        meta: Optional[Dict[str, Any]] = None
    ):
        """Initialize DifferentiableResult."""
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DifferentiableResult")
        
        self.u = u
        self.solver = solver
        self.derivative_type = derivative_type
        self.differentiable_params = differentiable_params or []
        self.vertices = vertices  # 可微顶点，backward 后可用 result.vertices.grad
        self.strain = strain
        self.stress = stress
        self.meta = meta or {}
        self.meta["backend"] = "nanobind"
        self.meta["differentiable"] = True
        self.meta["derivative_type"] = derivative_type

    def release_solver(self) -> None:
        """Release the C++ Solver reference and detach the solution from the autograd graph.

        Call this after backward() when you no longer need gradients. This allows the
        C++ Solver to be freed and avoids nanobind's 'leaked instances' message at exit.
        """
        self.solver = None
        self.u = self.u.detach()
    
    def to_numpy(self) -> Dict[str, np.ndarray]:
        """Convert PyTorch tensors to numpy arrays."""
        result = {
            "u": self.u.detach().cpu().numpy()
        }
        if self.strain is not None:
            result["strain"] = self.strain.detach().cpu().numpy()
        if self.stress is not None:
            result["stress"] = self.stress.detach().cpu().numpy()
        return result
    
    def __repr__(self) -> str:
        return (
            f"DifferentiableResult(u.shape={self.u.shape}, "
            f"derivative_type={self.derivative_type}, "
            f"differentiable_params={self.differentiable_params})"
        )

