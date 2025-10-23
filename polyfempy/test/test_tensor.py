# test/test_tensor.py
# Tests for polyfempy.api.tensor backend-preserving behavior.

import importlib
import numpy as np
import pytest

from polyfempy.api import tensor as T

# ---------------------------------------------------------------------
# Basics: NumPy-only (runs in any environment)
# ---------------------------------------------------------------------

def test_numpy_roundtrip():
    """Check NumPy arrays roundtrip correctly.

    Asserts:
        - as_numpy returns a NumPy array and 'numpy' as backend.
        - to_backend returns the same object (no copy/convert).
    """
    a = np.arange(6).reshape(2, 3)
    arr, backend = T.as_numpy(a)
    assert backend == "numpy"
    assert isinstance(arr, np.ndarray)

    out = T.to_backend(arr, backend)
    assert isinstance(out, np.ndarray)
    assert out is arr


def test_numpy_contiguous_enforced():
    """as_numpy should enforce C-contiguity for NumPy arrays.

    Steps:
        - Create a Fortran-contiguous (non-C) array.
        - Call as_numpy.
    
    Asserts:
        - backend is 'numpy'.
        - returned array is C-contiguous.
    """
    a = np.asfortranarray(np.arange(6).reshape(2, 3))
    assert a.flags.f_contiguous and not a.flags.c_contiguous

    arr, backend = T.as_numpy(a)
    assert backend == "numpy"
    assert arr.flags.c_contiguous


def test_to_backend_none_passthrough():
    """to_backend(None, ...) should return None (passthrough)."""
    assert T.to_backend(None, "numpy") is None


# ---------------------------------------------------------------------
# Optional: PyTorch (skipped if not installed)
# ---------------------------------------------------------------------

_HAS_TORCH = importlib.util.find_spec("torch") is not None
torch = importlib.import_module("torch") if _HAS_TORCH else None

@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_torch_cpu_zero_copy_roundtrip():
    """Check zero-copy roundtrip for torch CPU tensors.

    Asserts:
        - as_numpy returns a NumPy array backed by the same memory.
        - modifying the torch tensor updates the NumPy view.
        - converting back preserves sharing.
    """
    t = torch.zeros(2, 3)  # CPU & contiguous
    arr, backend = T.as_numpy(t)
    assert backend == "torch"
    assert isinstance(arr, np.ndarray)

    t[0, 0] = 7
    assert arr[0, 0] == 7

    out_t = T.to_backend(arr, backend)
    assert isinstance(out_t, torch.Tensor)
    out_t[1, 1] = 9
    assert arr[1, 1] == 9


@pytest.mark.skipif(not _HAS_TORCH or not torch.cuda.is_available(), reason="requires torch+CUDA")
def test_torch_gpu_fallback_to_cpu_numpy():
    """GPU tensors should be moved to CPU inside as_numpy.

    Asserts:
        - backend is 'torch'.
        - returned object is a NumPy array.
        - shape matches.
    """
    t = torch.ones(2, 3, device="cuda")
    arr, backend = T.as_numpy(t)
    assert backend == "torch"
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 3)


# ---------------------------------------------------------------------
# Optional: JAX (skipped if not installed)
# ---------------------------------------------------------------------

_HAS_JAX = importlib.util.find_spec("jax") is not None
jnp = importlib.import_module("jax.numpy") if _HAS_JAX else None

@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed")
def test_jax_roundtrip():
    """Check JAX array roundtrip.

    Asserts:
        - as_numpy returns a NumPy array and backend 'jax'.
        - to_backend returns a JAX array (type from 'jax.*').
        - shape is preserved.
    """
    x = jnp.ones((2, 3))
    arr, backend = T.as_numpy(x)
    assert backend == "jax"
    assert isinstance(arr, np.ndarray)

    y = T.to_backend(arr, backend)
    assert "jax" in type(y).__module__
    assert y.shape == (2, 3)
