"""DifferentiableResult: result container for differentiable simulations."""

# pyright: reportMissingImports=false
# torch is an optional dependency, so type checker may not find it

from typing import Optional, Dict, Any, TYPE_CHECKING
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

