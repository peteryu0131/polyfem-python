from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import polyfempy.differentiable.data.training as training_data


class _DummyDifferentiableResult:
    def __init__(self):
        self.shape_gradient = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float64,
        )
        self.vertices = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0]],
            dtype=torch.float64,
        )


def test_save_training_sample_metadata_is_strict_json_when_history_missing(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        training_data,
        "summarize_history_bundle",
        lambda *_args, **_kwargs: {
            "available": False,
            "history_source": "missing",
            "steps_by_body": [],
        },
    )

    paths = training_data.save_training_sample(
        result=_DummyDifferentiableResult(),
        loss=torch.tensor(5.0, dtype=torch.float64),
        workspace=tmp_path,
        body_id=1,
        body_name="lattice",
    )

    metadata = json.loads(paths["metadata_json"].read_text(encoding="utf-8"))

    # Reviewer-facing metadata must be valid strict JSON. NumPy artifacts may
    # still carry NaN labels for missing numeric targets.
    json.dumps(metadata, allow_nan=False)
    assert metadata["lattice_vm_max"] is None
    assert metadata["lattice_vm_p95"] is None
    assert metadata["lattice_vm_mean"] is None

    scalars = np.load(paths["scalars_npy"])
    assert np.isnan(scalars[2:]).all()
