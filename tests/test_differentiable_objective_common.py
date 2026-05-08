from __future__ import annotations

import pytest

from polyfempy.differentiable._objective_common import (
    _resolve_smooth_max_sharpness,
    _resolve_time_aggregation,
    _resolve_volume_selection,
    objective_state_columns,
    resolve_objective_state_column,
)


def test_objective_state_columns_defaults_to_last_state():
    assert objective_state_columns(n_cols=4) == ([3], "state")
    assert resolve_objective_state_column("first", 4) == 0
    assert resolve_objective_state_column("last", 4) == 3


def test_objective_state_columns_smooth_max_aliases_use_all_states():
    assert objective_state_columns(n_cols=3, time_aggregation="smooth_max") == (
        [0, 1, 2],
        "smooth_max",
    )
    assert objective_state_columns(n_cols=3, time_aggregation="softmax") == (
        [0, 1, 2],
        "smooth_max",
    )


def test_time_aggregation_aliases_reject_conflicts():
    assert _resolve_time_aggregation(time="max", time_aggregation="max") == "max"
    with pytest.raises(ValueError, match="Use only one"):
        _resolve_time_aggregation(time="max", time_aggregation="mean")


def test_body_alias_rejects_conflicting_volume_selection():
    assert _resolve_volume_selection(body=2) == 2
    with pytest.raises(ValueError, match="Use either body or volume_selection"):
        _resolve_volume_selection(body=2, volume_selection=3)


def test_smooth_max_sharpness_alias_rejects_conflicts():
    assert _resolve_smooth_max_sharpness(smooth_max_beta=4.0) == 4.0
    with pytest.raises(ValueError, match="Use either smooth_max_sharpness"):
        _resolve_smooth_max_sharpness(smooth_max_sharpness=2.0, smooth_max_beta=4.0)
