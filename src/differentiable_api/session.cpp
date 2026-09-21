#include "binding.hpp"

#include <polyfem/optimization/BuildFromJson.hpp>
#include <polyfem/optimization/DiffCache.hpp>
#include <polyfem/optimization/parametrization/Parametrization.hpp>
#include <polyfem/optimization/var2sims/ShapeVariableToSimulation.hpp>
#include <polyfem/varforms/diff/DifferentiableVarForm.hpp>

#include <Eigen/Core>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

polyfem::json settings_from_python(const py::object &settings)
{
  if (py::isinstance<py::str>(settings))
  {
    const std::string json_string = nb::cast<std::string>(settings);
    return polyfem::json::parse(json_string);
  }

  py::module_ json_module = py::module_::import_("json");
  const std::string json_string =
      nb::cast<std::string>(json_module.attr("dumps")(settings));
  return polyfem::json::parse(json_string);
}

Eigen::VectorXd flatten_vertices_node_major(const Eigen::MatrixXd &vertices)
{
  Eigen::VectorXd x(vertices.rows() * vertices.cols());
  int index = 0;
  for (int vertex = 0; vertex < vertices.rows(); ++vertex)
  {
    for (int dim = 0; dim < vertices.cols(); ++dim)
    {
      x(index++) = vertices(vertex, dim);
    }
  }
  return x;
}

class DifferentiableSession
{
public:
  DifferentiableSession() = default;

  void set_settings(const py::object &settings)
  {
    settings_ = settings_from_python(settings);
    varform_ = polyfem::from_json::build_differentiable_varform(
        settings_,
        max_threads_);
    diff_cache_ = std::make_shared<polyfem::DiffCache>();
    shape_var2sim_ = build_direct_shape_variable_to_simulation();

    has_settings_ = true;
  }

  void set_shape_vertices(const py::object &vertices, const py::object &selection)
  {
    (void)selection;
    if (!has_settings_)
    {
      throw std::runtime_error(
          "DifferentiableSession requires set_settings(...) before "
          "set_shape_vertices(...).");
    }

    const Eigen::MatrixXd vertices_matrix = nb::cast<Eigen::MatrixXd>(vertices);
    input_vertex_count_ = static_cast<int>(vertices_matrix.rows());
    input_dimension_ = static_cast<int>(vertices_matrix.cols());

    const int expected_vertex_count = varform_->get_mesh().n_vertices();
    const int expected_dimension = varform_->get_mesh().dimension();
    if (input_vertex_count_ != expected_vertex_count
        || input_dimension_ != expected_dimension)
    {
      throw std::runtime_error(
          "Shape vertices must have shape ("
          + std::to_string(expected_vertex_count)
          + ", "
          + std::to_string(expected_dimension)
          + "); got ("
          + std::to_string(input_vertex_count_)
          + ", "
          + std::to_string(input_dimension_)
          + ").");
    }

    current_shape_x_ = flatten_vertices_node_major(vertices_matrix);
    shape_var2sim_->update(current_shape_x_);
    has_shape_vertices_ = true;
  }

  Eigen::MatrixXd solve()
  {
    ensure_ready_for_shape_solve();
    shape_var2sim_->update(current_shape_x_);

    const auto *initial_conditions =
        diff_cache_->initial_condition_override
            ? &*diff_cache_->initial_condition_override
            : nullptr;
    const polyfem::varform::ForwardStepCallback post_step =
        [varform = varform_, diff_cache = diff_cache_](
            const int step,
            const Eigen::MatrixXd &solution) {
          diff_cache->cache_transient(step, *varform, solution, nullptr);
        };

    varform_->solve(last_solution_, initial_conditions, post_step, true);
    has_solution_ = true;
    return last_solution_;
  }

  py::object backward_shape(const py::object &grad_u)
  {
    (void)grad_u;
    ensure_ready_for_shape_solve();
    throw std::runtime_error(
        "DifferentiableSession.backward_shape is registered, but the real "
        "shape adjoint backend is not implemented yet.");
  }

private:
  void ensure_ready_for_shape_solve() const
  {
    if (!has_settings_)
    {
      throw std::runtime_error(
          "DifferentiableSession requires set_settings(...) before solve().");
    }
    if (!has_shape_vertices_)
    {
      throw std::runtime_error(
          "DifferentiableSession requires set_shape_vertices(...) before solve().");
    }
  }

  bool has_settings_ = false;
  bool has_shape_vertices_ = false;
  bool has_solution_ = false;
  size_t max_threads_ = 1;
  int input_vertex_count_ = 0;
  int input_dimension_ = 0;
  polyfem::json settings_;
  Eigen::VectorXd current_shape_x_;
  Eigen::MatrixXd last_solution_;
  std::shared_ptr<polyfem::varform::DifferentiableVarForm> varform_;
  std::shared_ptr<polyfem::DiffCache> diff_cache_;
  std::shared_ptr<polyfem::solver::ShapeVariableToSimulation> shape_var2sim_;

  std::shared_ptr<polyfem::solver::ShapeVariableToSimulation>
  build_direct_shape_variable_to_simulation() const
  {
    std::vector<std::shared_ptr<polyfem::varform::DifferentiableVarForm>> varforms{
        varform_};
    std::vector<std::shared_ptr<polyfem::DiffCache>> diff_caches{
        diff_cache_};
    polyfem::solver::CompositeParametrization parametrization;
    Eigen::VectorXi active_dimensions;
    Eigen::VectorXi active_geometry_nodes;

    return std::make_shared<polyfem::solver::ShapeVariableToSimulation>(
        std::move(varforms),
        std::move(diff_caches),
        std::move(parametrization),
        std::move(active_dimensions),
        std::move(active_geometry_nodes));
  }
};

} // namespace

void define_differentiable_session(py::module_ &m)
{
  py::class_<DifferentiableSession>(m, "DifferentiableSession")
      .def(py::init<>())
      .def(
          "set_settings",
          &DifferentiableSession::set_settings,
          "Store differentiable solve settings.",
          py::arg("settings"))
      .def(
          "set_shape_vertices",
          &DifferentiableSession::set_shape_vertices,
          "Store direct shape vertices for a future differentiable solve.",
          py::arg("vertices"),
          py::kw_only(),
          py::arg("selection") = py::none())
      .def(
          "solve",
          &DifferentiableSession::solve,
          "Run the differentiable forward solve.")
      .def(
          "backward_shape",
          &DifferentiableSession::backward_shape,
          "Run the shape adjoint backward pass.",
          py::arg("grad_u"));
}
