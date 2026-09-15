from __future__ import annotations

import importlib
import sys

import pytest


def test_model_rejects_empty_model_list():
    from polyfempy import differentiable_api as D

    with pytest.raises(ValueError, match="at least one forward model"):
        D.model([])


def test_model_accepts_backend_shaped_dict_without_sharing_payload():
    from polyfempy import differentiable_api as D

    payload = {"geometry": [{"mesh": "beam.msh"}], "materials": [{"type": "NeoHookean"}]}

    diff_model = D.model([payload])
    emitted = diff_model.as_dicts()

    assert isinstance(diff_model, D.DifferentiableModel)
    assert diff_model.source_kinds == ("dict",)
    assert emitted == [payload]
    assert emitted[0] is not payload

    emitted[0]["geometry"][0]["mesh"] = "changed.msh"

    assert diff_model.as_dicts()[0]["geometry"][0]["mesh"] == "beam.msh"
    assert payload["geometry"][0]["mesh"] == "beam.msh"


def test_model_accepts_generated_config_like_object_with_as_dict():
    from polyfempy import differentiable_api as D

    class GeneratedConfig:
        def as_dict(self):
            return {
                "geometry": [{"mesh": "beam.msh"}],
                "root_path": "case-root",
            }

    cfg = GeneratedConfig()

    diff_model = D.model([cfg])

    assert diff_model.inputs == (cfg,)
    assert diff_model.configs == (cfg,)
    assert diff_model.source_kinds == ("generated",)
    assert diff_model.as_dicts() == [
        {
            "geometry": [{"mesh": "beam.msh"}],
            "root_path": "case-root",
        }
    ]


def test_model_accepts_forward_model_builder_with_config_method():
    from polyfempy import differentiable_api as D

    class GeneratedConfig:
        def as_dict(self):
            return {
                "geometry": [{"mesh": "beam.msh"}],
                "materials": [{"type": "NeoHookean"}],
            }

    class ForwardModel:
        def __init__(self):
            self.config_call_count = 0
            self.generated_config = GeneratedConfig()

        def config(self):
            self.config_call_count += 1
            return self.generated_config

    forward_model = ForwardModel()

    diff_model = D.model([forward_model])

    assert forward_model.config_call_count == 1
    assert diff_model.inputs == (forward_model,)
    assert diff_model.configs == (forward_model.generated_config,)
    assert diff_model.source_kinds == ("model",)
    assert diff_model.as_dicts() == [
        {
            "geometry": [{"mesh": "beam.msh"}],
            "materials": [{"type": "NeoHookean"}],
        }
    ]


def test_model_rejects_config_method_that_does_not_produce_dict_payload():
    from polyfempy import differentiable_api as D

    class BadForwardModel:
        def config(self):
            return ["not", "a", "config"]

    with pytest.raises(TypeError, match="config\\(\\) must return"):
        D.model([BadForwardModel()])


def test_model_rejects_unsupported_input_type():
    from polyfempy import differentiable_api as D

    with pytest.raises(TypeError, match="forward model entries must be"):
        D.model([object()])


def test_model_module_import_does_not_load_torch_or_old_reference_package():
    torch_was_loaded = "torch" in sys.modules
    sys.modules.pop("polyfempy.differentiable", None)

    importlib.import_module("polyfempy.differentiable_api.model")

    assert "polyfempy.differentiable" not in sys.modules
    if not torch_was_loaded:
        assert "torch" not in sys.modules

