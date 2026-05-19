from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from polyfempy.api.result import Result


def _small_result() -> Result:
    return Result(
        backend="numpy",
        vertices=np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        cells=np.array([[0, 1, 2]], dtype=np.int32),
        point_data={"u": np.zeros((3, 2), dtype=np.float64)},
        cell_data={},
    )


def test_to_vtk_raises_by_default_when_mesh_export_fails(tmp_path, monkeypatch):
    result = _small_result()
    out_path = tmp_path / "result.vtu"

    def fail_write(*_args, **_kwargs):
        raise RuntimeError("meshio unavailable")

    monkeypatch.setattr(result, "write", fail_write)

    with pytest.raises(RuntimeError, match="meshio unavailable"):
        result.to_vtk(str(out_path))

    assert not Path(str(out_path) + ".npz").exists()


def test_to_vtk_can_explicitly_fallback_to_npz(tmp_path, monkeypatch):
    result = _small_result()
    out_path = tmp_path / "result.vtu"

    def fail_write(*_args, **_kwargs):
        raise RuntimeError("meshio unavailable")

    monkeypatch.setattr(result, "write", fail_write)

    fallback_path = result.to_vtk(str(out_path), fallback_npz=True)

    assert fallback_path == str(out_path) + ".npz"
    assert Path(fallback_path).exists()
    with np.load(fallback_path) as data:
        np.testing.assert_array_equal(data["vertices"], result.vertices)
        assert "point_u" in data
