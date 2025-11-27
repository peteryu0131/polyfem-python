"""DifferentiableResult: result container for differentiable simulations.

This module provides a result container that holds PyTorch tensors and
supports automatic gradient computation.
"""

from typing import Optional, Dict, Any
import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # Placeholder


class DifferentiableResult:
    """Result container for differentiable simulations.
    
    This class holds PyTorch tensors that support automatic gradient computation.
    It's similar to Result but designed for differentiable operations.
    
    Attributes:
        u: Solution tensor (torch.Tensor), shape (n_dof, n_time_steps)
        solver: Underlying Solver object (for advanced use)
        derivative_type: Type of derivative computed
        differentiable_params: List of parameters that are differentiable
        strain: Strain tensor (optional, torch.Tensor or None)
        stress: Stress tensor (optional, torch.Tensor or None)
        meta: Metadata dictionary
    """
    
    def __init__(
        self,
        u: "torch.Tensor",
        solver: Any,  # pf.Solver object
        derivative_type: str = "shape",
        differentiable_params: Optional[list] = None,
        strain: Optional["torch.Tensor"] = None,
        stress: Optional["torch.Tensor"] = None,
        meta: Optional[Dict[str, Any]] = None
    ):
        """Initialize DifferentiableResult.
        
        Args:
            u: Solution tensor
            solver: Solver object (for backward pass)
            derivative_type: Type of derivative
            differentiable_params: List of differentiable parameters
            strain: Strain tensor (optional)
            stress: Stress tensor (optional)
            meta: Metadata dictionary
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DifferentiableResult")
        
        self.u = u
        self.solver = solver
        self.derivative_type = derivative_type
        self.differentiable_params = differentiable_params or []
        self.strain = strain
        self.stress = stress
        self.meta = meta or {}
        
        # Add backend info
        self.meta["backend"] = "nanobind"
        self.meta["differentiable"] = True
        self.meta["derivative_type"] = derivative_type
    
    def to_numpy(self) -> Dict[str, np.ndarray]:
        """Convert PyTorch tensors to numpy arrays.
        
        Returns:
            Dictionary with numpy arrays
        """
        result = {
            "u": self.u.detach().cpu().numpy()
        }
        if self.strain is not None:
            result["strain"] = self.strain.detach().cpu().numpy()
        if self.stress is not None:
            result["stress"] = self.stress.detach().cpu().numpy()
        return result
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"DifferentiableResult(u.shape={self.u.shape}, "
            f"derivative_type={self.derivative_type}, "
            f"differentiable_params={self.differentiable_params})"
        )

