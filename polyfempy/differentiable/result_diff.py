"""DifferentiableResult: result container for differentiable simulations."""

# pyright: reportMissingImports=false
# torch is an optional dependency, so type checker may not find it

from typing import Optional, Dict, Any, TYPE_CHECKING, Literal
import numpy as np

if TYPE_CHECKING:
    import torch

try:
    import torch  # pyright: ignore[reportMissingImports]
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


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
        self._history = None

    @property
    def history(self):
        """Best-effort per-step history, mirroring ``Result.history`` enough for reporting."""
        if self._history is not None:
            return self._history

        from ..api.result import HistoryView

        if self.solver is None:
            self._history = HistoryView()
            return self._history

        full_json = self.meta.get("_solve_settings")
        if not isinstance(full_json, dict):
            full_json = None

        try:
            from ..api._solve_pipeline import _collect_solver_history

            self._history = _collect_solver_history(self.solver, full_json)
        except Exception:
            self._history = HistoryView()

        if getattr(self._history, "available", False):
            self.meta.setdefault(
                "history_source",
                getattr(self._history, "source", "solver.solution_frames"),
            )
        return self._history

    @property
    def body_ids(self):
        """Per-sample body ids from in-memory history, when available."""
        history = self.history
        body_ids = getattr(history, "body_ids", None)
        if body_ids is None:
            return None
        arr = np.asarray(body_ids)
        return arr if arr.size > 0 else None

    @property
    def shape_gradient(self):
        """Return the shape gradient populated by ``loss.backward()``."""
        if self.vertices is None:
            return None
        return self.vertices.grad

    def _try_get_stress_numpy(self) -> Optional[np.ndarray]:
        """Best-effort fetch stress from underlying C++ solver (if exposed).

        This avoids VTU I/O, but is NOT connected to PyTorch autograd.
        Availability depends on the compiled polyfempy extension exposing stress getters.
        """
        if self.solver is None:
            return None
        for name in ("get_stress", "get_cauchy_stress", "stress"):
            if hasattr(self.solver, name):
                try:
                    out = getattr(self.solver, name)()
                    if out is None:
                        continue
                    arr = np.asarray(out)
                    if arr.size == 0:
                        continue
                    return arr
                except Exception:
                    continue
        return None

    @staticmethod
    def _von_mises_from_stress_voigt(stress: np.ndarray) -> np.ndarray:
        """Compute von Mises from stress in Voigt form.

        Supported shapes:
        - (n, 6): [sxx, syy, szz, sxy, syz, szx]
        - (n, 3): [sxx, syy, sxy] (assume szz=syz=szx=0)
        """
        s = np.asarray(stress, dtype=np.float64)
        if s.ndim != 2 or s.shape[1] not in (3, 6):
            raise ValueError(f"Expected stress shape (n,3) or (n,6), got {s.shape}")

        if s.shape[1] == 3:
            sxx = s[:, 0]
            syy = s[:, 1]
            szz = np.zeros_like(sxx)
            sxy = s[:, 2]
            syz = np.zeros_like(sxx)
            szx = np.zeros_like(sxx)
        else:
            sxx = s[:, 0]
            syy = s[:, 1]
            szz = s[:, 2]
            sxy = s[:, 3]
            syz = s[:, 4]
            szx = s[:, 5]

        vm2 = 0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3.0 * (
            sxy**2 + syz**2 + szx**2
        )
        vm2 = np.maximum(vm2, 0.0)
        return np.sqrt(vm2)

    def get_von_mises_numpy(self) -> Optional[np.ndarray]:
        """Return von Mises stress array from solver (if possible), without VTU I/O.

        Returns None if stress is not available via bindings.
        """
        stress = self._try_get_stress_numpy()
        if stress is None:
            return None
        return self._von_mises_from_stress_voigt(stress)

    def get_percentile_from_von_mises(
        self, q: float = 95.0, *, method: Literal["linear"] = "linear"
    ) -> Optional[float]:
        """Compute percentile (e.g. p95) of von Mises (numpy), if available."""
        vm = self.get_von_mises_numpy()
        if vm is None:
            return None
        if vm.size == 0:
            return float("nan")
        return float(np.percentile(vm, q, method=method))

    def release_solver(self) -> None:
        """Release the C++ Solver reference and detach the solution from the autograd graph.

        Call this after backward() when you no longer need gradients. This allows the
        C++ Solver to be freed and avoids nanobind's 'leaked instances' message at exit.
        """
        self.solver = None
        self.u = self.u.detach()
        self._history = None
    
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


class DifferentiableMaterialResult:
    """Result of ``solve_differentiable_material``: displacement + per-element Lamé tensors."""

    def __init__(
        self,
        u: "torch.Tensor",
        solver: Any,
        lam: "torch.Tensor",
        mu: "torch.Tensor",
        meta: Optional[Dict[str, Any]] = None,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DifferentiableMaterialResult")
        self.u = u
        self.solver = solver
        self.lam = lam
        self.mu = mu
        self.meta = meta or {}
        self.meta["backend"] = "nanobind"
        self.meta["differentiable"] = True
        self.meta["derivative_type"] = "per_element_material"
        self._history = None

    @property
    def history(self):
        """Best-effort per-step history, mirroring ``DifferentiableResult`` enough for reports."""
        if self._history is not None:
            return self._history

        from ..api.result import HistoryView

        if self.solver is None:
            self._history = HistoryView()
            return self._history

        full_json = self.meta.get("_solve_settings")
        if not isinstance(full_json, dict):
            full_json = None

        try:
            from ..api._solve_pipeline import _collect_solver_history

            self._history = _collect_solver_history(self.solver, full_json)
        except Exception:
            self._history = HistoryView()

        if getattr(self._history, "available", False):
            self.meta.setdefault(
                "history_source",
                getattr(self._history, "source", "solver.solution_frames"),
            )
        return self._history

    @property
    def body_ids(self):
        """Per-sample body ids from in-memory history, when available."""
        history = self.history
        body_ids = getattr(history, "body_ids", None)
        if body_ids is None:
            return None
        arr = np.asarray(body_ids)
        return arr if arr.size > 0 else None

    def release_solver(self) -> None:
        """Release the C++ Solver reference and detach the solution tensor."""
        self.solver = None
        self.u = self.u.detach()
        self._history = None

    def __repr__(self) -> str:
        return (
            f"DifferentiableMaterialResult(u.shape={self.u.shape}, "
            f"lam.shape={self.lam.shape}, mu.shape={self.mu.shape})"
        )
