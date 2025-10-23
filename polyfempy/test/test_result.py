# test/test_result.py

import importlib
import numpy as np
import pytest

from polyfempy.api.result import Result

_HAS_MESHIO = importlib.util.find_spec("meshio") is not None
_HAS_TORCH  = importlib.util.find_spec("torch") is not None
torch = importlib.import_module("torch") if _HAS_TORCH else None


def _mini_mesh():
    """Create a tiny 2D square mesh split into two triangles.

    Returns:
        tuple[np.ndarray, np.ndarray]: 
            vertices V with shape (4, 2) and int32 triangle cells C with shape (2, 3).
    """
    V = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=float)   # (4,2)
    C = np.array([[0,1,2],[0,2,3]], dtype=np.int32)        # (2,3) triangles
    return V, C


def test_numpy_roundtrip_and_summary(tmp_path):
    """Verify NumPy roundtrip (as_numpy/to_backend are no-ops) and summary shapes.

    Args:
        tmp_path: PyTest fixture for a temporary directory (not used for I/O here).

    Asserts:
        - Field 'u' remains a NumPy array with shape (4, 2).
        - Summary dict contains expected backend and shapes.
    """
    V, C = _mini_mesh()
    r = Result(backend="numpy", vertices=V, cells=C, fields={"u": np.ones((4,2))})

    # as_numpy / to_backend should be no-ops for NumPy backend
    r.as_numpy().to_backend()
    u = r.field("u")
    assert isinstance(u, np.ndarray)
    assert u.shape == (4, 2)

    # short summary
    s = r.summary()
    assert s["backend"] == "numpy"
    assert s["vertices"] == (4, 2)
    assert s["cells"] == (2, 3)
    assert s["fields"]["u"] == (4, 2)


def test_magnitude_adds_norm_field():
    """Check that magnitude('u') creates a scalar field 'u_norm' of length N.

    Asserts:
        - 'u_norm' exists and has shape (4,).
        - For vectors [1, 1], the norm equals sqrt(2).
    """
    V, C = _mini_mesh()
    r = Result("numpy", V, C, fields={"u": np.ones((4,2))})
    r.magnitude("u")  # produce u_norm
    u_norm = r.field("u_norm")
    assert u_norm is not None
    assert u_norm.shape == (4,)
    # each vector is [1,1] → norm should be sqrt(2)
    assert np.allclose(u_norm, np.sqrt(2.0))


def test_field_set_and_remove():
    """Verify basic field management: set_field / remove_field / field_names."""
    V, C = _mini_mesh()
    r = Result("numpy", V, C)
    r.set_field("p", np.arange(4))
    assert "p" in r.field_names()
    r.remove_field("p")
    assert "p" not in r.field_names()


def test_to_vtk_falls_back_to_npz_when_no_meshio(tmp_path):
    """Ensure exporting with '.npz' suffix always writes NPZ (fallback path).

    Args:
        tmp_path: PyTest fixture providing a temporary directory.

    Asserts:
        - Output file exists.
        - NPZ contains 'vertices', 'cells', and field 'u'.
    """
    V, C = _mini_mesh()
    r = Result("numpy", V, C, fields={"u": np.zeros((4,2))})
    out = tmp_path / "out_data.npz"
    r.to_vtk(str(out))
    assert out.exists()

    # readable content
    data = np.load(out)
    assert "vertices" in data and "cells" in data and "u" in data


@pytest.mark.skipif(not _HAS_MESHIO, reason="meshio not installed")
def test_to_vtk_with_meshio(tmp_path):
    """If meshio is installed, writing a '.vtk' file should succeed.

    Args:
        tmp_path: PyTest fixture providing a temporary directory.

    Asserts:
        - The '.vtk' file exists after export.
    """
    V, C = _mini_mesh()
    r = Result("numpy", V, C, fields={"u": np.zeros((4,2))})
    out = tmp_path / "mesh.vtk"
    r.to_vtk(str(out))
    assert out.exists()


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_to_backend_torch_cpu_roundtrip_zero_copy_like():
    """When backend is 'torch', to_backend() should return torch.Tensors for fields.

    Asserts:
        - Field 'u' is a torch.Tensor of shape (4, 2).
    """
    V, C = _mini_mesh()
    u_np = np.zeros((4,2), dtype=np.float32)
    r = Result("torch", V, C, fields={"u": u_np})

    r.to_backend()
    u_t = r.field("u")
    assert isinstance(u_t, torch.Tensor)
    assert tuple(u_t.shape) == (4, 2)
