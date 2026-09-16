#include "binding.hpp"

#include <stdexcept>
#include <string>

namespace
{

class DifferentiableSession
{
public:
  DifferentiableSession() = default;

  void set_settings(const py::object &settings)
  {
    settings_repr_ = nb::cast<std::string>(py::str(settings));
    has_settings_ = true;
  }

  void set_shape_vertices(const py::object &vertices, const py::object &selection)
  {
    vertices_repr_ = nb::cast<std::string>(py::str(vertices));
    selection_repr_ = nb::cast<std::string>(py::str(selection));
    has_shape_vertices_ = true;
  }

  py::object solve()
  {
    ensure_ready_for_shape_solve();
    throw std::runtime_error(
        "DifferentiableSession.solve is registered, but the real "
        "differentiable backend is not implemented yet.");
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
  std::string settings_repr_;
  std::string vertices_repr_;
  std::string selection_repr_;
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
