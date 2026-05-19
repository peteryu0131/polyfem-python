from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from polyfempy.api.config import SimulationConfig, Solver, SolverContactOptions
import polyfempy.differentiable._solve_settings as solve_settings
from polyfempy.api._solve_contract import NoMeshSourceError
from polyfempy.differentiable._solve_settings import (
    _apply_internal_differentiable_runtime_patches,
    _console_log_level_from_settings,
    _differentiable_config_and_settings,
    _geometry_uses_only_absolute_mesh_paths,
    _is_settings_only_no_mesh_error,
    prepare_settings_only_differentiable_contract,
    prepare_differentiable_solve_contract,
)


def test_console_log_level_from_settings_defaults_and_named_levels():
    assert _console_log_level_from_settings({}) == 2
    assert _console_log_level_from_settings({"output": {"log": {"level": "warn"}}}) == 3
    assert _console_log_level_from_settings({"output": {"log": {"level": "off"}}}) == 6


def test_geometry_absolute_mesh_path_detection():
    assert _geometry_uses_only_absolute_mesh_paths(
        {"geometry": [{"mesh": "/tmp/body.msh"}]}
    )
    assert not _geometry_uses_only_absolute_mesh_paths(
        {"geometry": [{"mesh": "relative/body.msh"}]}
    )
    assert not _geometry_uses_only_absolute_mesh_paths(
        {"geometry": [{"type": "ground", "height": 0.0}]}
    )


def test_runtime_patch_sets_constant_differentiable_barrier_stiffness():
    settings = {"solver": {"contact": {"barrier_stiffness": "adaptive"}}}

    patches = _apply_internal_differentiable_runtime_patches(config=None, settings=settings)

    assert settings["solver"]["contact"]["barrier_stiffness"] == 1e3
    assert patches == [
        {
            "path": "solver.contact.barrier_stiffness",
            "old": "adaptive",
            "new": 1e3,
            "reason": "differentiable_contact_requires_constant_barrier_stiffness",
        }
    ]


def test_differentiable_runtime_patch_does_not_mutate_user_config():
    cfg = SimulationConfig.linear_elasticity(20.0, 0.3)
    cfg.solver = Solver(contact=SolverContactOptions(barrier_stiffness="adaptive"))

    _, _, settings, _, diagnostics = _differentiable_config_and_settings(
        cfg,
        return_diagnostics=True,
    )

    assert settings["solver"]["contact"]["barrier_stiffness"] == 1e3
    assert cfg.solver.contact.barrier_stiffness == "adaptive"
    assert diagnostics["runtime_patches"][0]["path"] == "solver.contact.barrier_stiffness"


def test_settings_only_fallback_is_explicit_in_diagnostics():
    cfg = SimulationConfig.linear_elasticity(20.0, 0.3)

    contract = prepare_settings_only_differentiable_contract(
        cfg=cfg,
        reason="Either provide vertices/cells arrays, or use JSON config with geometry",
    )

    assert contract.settings["pde"] == "LinearElasticity"
    assert contract.mesh_source.mode == "settings_only"
    assert contract.diagnostics["mesh_source"] == "settings_only"
    assert contract.diagnostics["contract_path"] == "settings_only_compatibility"
    assert "Either provide vertices/cells" in contract.diagnostics["fallback_reason"]


def test_differentiable_config_wrapper_uses_settings_only_contract(monkeypatch):
    cfg = SimulationConfig.linear_elasticity(20.0, 0.3)
    calls = []
    real_helper = solve_settings.prepare_settings_only_differentiable_contract

    def spy_helper(**kwargs):
        calls.append(kwargs)
        return real_helper(**kwargs)

    monkeypatch.setattr(solve_settings, "prepare_settings_only_differentiable_contract", spy_helper)

    _, _, settings, _, diagnostics = _differentiable_config_and_settings(
        cfg,
        return_diagnostics=True,
    )

    assert calls
    assert calls[0]["cfg"] is cfg
    assert settings["pde"] == "LinearElasticity"
    assert diagnostics["mesh_source"] == "settings_only"


def test_settings_only_classifier_requires_contract_no_mesh_exception():
    assert _is_settings_only_no_mesh_error(NoMeshSourceError("missing mesh"))
    assert not _is_settings_only_no_mesh_error(
        ValueError("Either provide vertices/cells arrays, or use JSON config with geometry")
    )


def test_differentiable_settings_use_canonical_json_cleanup():
    cfg = {
        "pde": "LinearElasticity",
        "materials": [{"type": "LinearElasticity", "E": 20.0, "nu": 0.3}],
        "geometry": [{"mesh": "/tmp/body.msh"}],
        "output": {
            "directory": "/tmp/out",
            "result": {"fields": ["u"]},
            "fallback": {"sampled_vtu": "auto"},
            "save_vtu": True,
        },
    }

    _, _, settings, _, diagnostics = _differentiable_config_and_settings(
        cfg,
        return_diagnostics=True,
    )

    assert "result" not in settings["output"]
    assert "fallback" not in settings["output"]
    assert "save_vtu" not in settings["output"]
    assert diagnostics["mesh_source"] == "json"


def test_differentiable_config_wrapper_delegates_to_solve_contract(monkeypatch):
    cfg = SimulationConfig.linear_elasticity(20.0, 0.3)
    calls = []

    def fake_prepare_contract(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            config=cfg,
            settings={"from": "contract"},
            root_path="/tmp/root.json",
            diagnostics={"mesh_source": "json", "runtime_patches": []},
            mesh_source=SimpleNamespace(mode="json"),
        )

    monkeypatch.setattr(
        solve_settings,
        "prepare_differentiable_solve_contract",
        fake_prepare_contract,
    )

    config, config_obj, settings, root_path, diagnostics = _differentiable_config_and_settings(
        cfg,
        root_path="/tmp/root.json",
        apply_runtime_patches=False,
        return_diagnostics=True,
    )

    assert calls == [
        {
            "cfg": cfg,
            "root_path": "/tmp/root.json",
            "apply_runtime_patches": False,
        }
    ]
    assert config is cfg
    assert config_obj is cfg
    assert settings == {"from": "contract"}
    assert root_path == "/tmp/root.json"
    assert diagnostics["mesh_source"] == "json"


def test_prepare_differentiable_contract_uses_direct_array_mesh_source():
    cfg = SimulationConfig.linear_elasticity(20.0, 0.3)
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    cells = np.array([[0, 1, 2]], dtype=np.int64)

    contract = prepare_differentiable_solve_contract(
        V=vertices,
        C=cells,
        cfg=cfg,
    )

    assert contract.mesh_source.mode == "array"
    assert contract.mesh_source.cells.dtype == np.int32
    assert contract.settings["geometry"][0]["type"] == "ground"
    assert contract.diagnostics["mesh_source"] == "array"


def test_prepare_differentiable_contract_rejects_partial_array_with_json_geometry():
    cfg = {
        "pde": "LinearElasticity",
        "materials": [{"type": "LinearElasticity", "E": 20.0, "nu": 0.3}],
        "geometry": [{"mesh": "/tmp/body.msh"}],
    }
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    try:
        prepare_differentiable_solve_contract(V=vertices, C=None, cfg=cfg)
    except ValueError as exc:
        assert "array mode requires both vertices and cells" in str(exc)
    else:
        raise AssertionError("partial differentiable array input should raise")


def test_prepare_differentiable_contract_uses_guided_array_payload():
    cfg = SimulationConfig.linear_elasticity(20.0, 0.3)
    cfg.extras["_mesh_array_mode"] = {
        "vertices": np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        "cells": np.array([[0, 1, 2]], dtype=np.int64),
        "body_ids": np.array([7], dtype=np.int32),
    }

    contract = prepare_differentiable_solve_contract(V=None, C=None, cfg=cfg)

    assert contract.mesh_source.mode == "guided_array"
    np.testing.assert_array_equal(contract.mesh_source.body_ids, np.array([7], dtype=np.int32))
    assert contract.diagnostics["mesh_source"] == "guided_array"
