from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_forward_solver_binding_targets_varform_state_only():
    source = (ROOT / "src" / "state" / "state.cpp").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert "#include <polyfem/State.hpp>" in source
    assert "solve_problem" not in source
    assert ".build_basis(" not in source
    assert ".assemble_rhs(" not in source
    assert ".assemble_mass_mat(" not in source
    assert "s.assembler" not in source
    assert "s.bases" not in source
    assert "set_per_element_material" not in source
    assert 'bundle["sol"] = sol' in source
    assert 'bundle["u"]' not in source
    assert 'bundle["p"]' not in source
    assert 'bundle["vertices"]' not in source
    assert 'bundle["cells"]' not in source


def test_default_cpp_module_registers_new_differentiable_session_only():
    src_cmake = (ROOT / "src" / "CMakeLists.txt").read_text()
    binding_cpp = (ROOT / "src" / "binding.cpp").read_text()

    assert "add_subdirectory(differentiable_api)" in src_cmake
    assert "add_subdirectory(differentiable)" not in src_cmake
    assert "add_subdirectory(solver)" not in src_cmake
    assert "differentiable_api/binding.hpp" in binding_cpp
    assert "differentiable/binding.hpp" not in binding_cpp
    assert "solver/binding.hpp" not in binding_cpp
    assert "define_differentiable_session" in binding_cpp
    assert "define_differentiable_cache" not in binding_cpp
    assert "define_adjoint" not in binding_cpp
    assert "define_objective" not in binding_cpp
    assert "define_opt_utils" not in binding_cpp
    assert "define_nonlinear_problem" not in binding_cpp
    assert not (ROOT / "src" / "solver" / "binding.hpp").exists()
    assert not (ROOT / "src" / "solver" / "nl_problem.cpp").exists()
    assert not (ROOT / "src" / "solver" / "CMakeLists.txt").exists()


def test_differentiable_session_uses_current_shape_backend_setup():
    source = (ROOT / "src" / "differentiable_api" / "session.cpp").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert "#include <polyfem/optimization/BuildFromJson.hpp>" in source
    assert "#include <polyfem/optimization/DiffCache.hpp>" in source
    assert "#include <polyfem/optimization/var2sims/ShapeVariableToSimulation.hpp>" in source
    assert "#include <polyfem/varforms/diff/DifferentiableVarForm.hpp>" in source
    assert "from_json::build_differentiable_varform" in source
    assert "std::shared_ptr<polyfem::DiffCache>" in source
    assert "std::shared_ptr<polyfem::varform::DifferentiableVarForm>" in source
    assert "std::shared_ptr<polyfem::solver::ShapeVariableToSimulation>" in source
    assert "settings_repr_" not in source
    assert "vertices_repr_" not in source


def test_differentiable_session_stores_direct_shape_vertices():
    source = (ROOT / "src" / "differentiable_api" / "session.cpp").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert "nb::cast<Eigen::MatrixXd>(vertices)" in source
    assert "flatten_vertices_node_major" in source
    assert "current_shape_x_" in source
    assert "input_vertex_count_" in source
    assert "input_dimension_" in source
    assert "varform_->get_mesh().n_vertices()" in source
    assert "varform_->get_mesh().dimension()" in source
    assert "(void)vertices;" not in source


def test_differentiable_session_solve_uses_current_forward_cache_path():
    source = (ROOT / "src" / "differentiable_api" / "session.cpp").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert "Eigen::MatrixXd solve()" in source
    assert "varform::ForwardStepCallback post_step" in source
    assert "diff_cache->cache_transient(step, *varform, solution, nullptr)" in source
    assert "varform_->solve(last_solution_, initial_conditions, post_step, true)" in source
    assert "last_solution_" in source
    assert "has_solution_" in source
    assert "DifferentiableSession.solve is registered" not in source
