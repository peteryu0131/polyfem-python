from __future__ import annotations

from pathlib import Path

import pytest


def test_state_accepts_backend_shaped_dict_without_sharing_payload():
    from polyfempy import differentiable_api as D

    payload = {"geometry": [{"mesh": "beam.msh"}], "materials": [{"type": "NeoHookean"}]}

    state = D.state(payload)
    emitted = state.as_dict()

    assert isinstance(state, D.State)
    assert state.source_kind == "dict"
    assert emitted == payload
    assert emitted is not payload

    emitted["geometry"][0]["mesh"] = "changed.msh"

    assert state.as_dict()["geometry"][0]["mesh"] == "beam.msh"
    assert payload["geometry"][0]["mesh"] == "beam.msh"


def test_state_accepts_generated_config_like_object_with_as_dict():
    from polyfempy import differentiable_api as D

    class GeneratedConfig:
        def as_dict(self):
            return {
                "geometry": [{"mesh": "beam.msh"}],
                "root_path": "case-root",
            }

    cfg = GeneratedConfig()

    state = D.state(cfg)

    assert state.source_kind == "generated"
    assert state.config is cfg
    assert state.as_dict() == {
        "geometry": [{"mesh": "beam.msh"}],
        "root_path": "case-root",
    }


def test_state_rejects_generated_config_when_as_dict_does_not_return_dict():
    from polyfempy import differentiable_api as D

    class BadGeneratedConfig:
        def as_dict(self):
            return ["not", "a", "dict"]

    with pytest.raises(TypeError, match="as_dict\\(\\) must return dict"):
        D.state(BadGeneratedConfig())


@pytest.mark.parametrize("bad_config", [None, 3, ["geometry"], object()])
def test_state_rejects_unsupported_config_types(bad_config):
    from polyfempy import differentiable_api as D

    expected_error = ValueError if bad_config is None else TypeError

    with pytest.raises(expected_error):
        D.state(bad_config)


@pytest.mark.parametrize("path_config", ["case.json", Path("case.json")])
def test_state_json_path_is_not_part_of_d2_a(path_config):
    from polyfempy import differentiable_api as D

    with pytest.raises(TypeError, match="JSON path support is not implemented"):
        D.state(path_config)

