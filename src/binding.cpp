#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>

#include <stdexcept>

#include "binding_wrapper.hpp"

#include "differentiable/binding.hpp"
#include "mesh/binding.hpp"
#include "state/binding.hpp"
#include "solver/binding.hpp"

PY_MODULE(polyfempy, m)
{
  define_pde_types(m);

  define_solver(m);
  define_solve(m);

  define_mesh(m);

  define_nonlinear_problem(m);

  define_differentiable_cache(m);
  define_adjoint(m);
  define_objective(m);
  define_opt_utils(m);

  m.def("version", []() { return "polyfempy nanobind backend"; }, "Get version information");
}
