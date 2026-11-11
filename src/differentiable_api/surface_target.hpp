#pragma once

#include <polyfem/optimization/forms/TargetForms.hpp>
#include <polyfem/io/Evaluator.hpp>
#include <polyfem/utils/BoundarySampler.hpp>
#include <polyfem/utils/IntegrableFunctional.hpp>

namespace polyfempy::differentiable_api
{
// Retain TargetForm's reference interpolation and value. Assemble its displacement
// derivative once per basis per face, including elements with several exterior faces.
class SurfaceTargetForm final : public polyfem::solver::TargetForm
{
public:
  using TargetForm::TargetForm;

  Eigen::VectorXd compute_adjoint_rhs_step(int step, const Eigen::VectorXd &,
      const polyfem::varform::DifferentiableVarForm &form, const polyfem::DiffCache &cache) const override
  {
    const auto &space = form.primary_space();
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(space.ndof());
    if (&form != varform_.get()) return rhs;
    const auto &mesh = form.get_mesh();
    const auto &functional = get_integral_functional();
    polyfem::IntegrableFunctional::ParameterType params;
    params.step = step;
    params.t = 0.0;
    if (form.get_problem().is_time_dependent())
    {
      const auto &time = form.get_args().at("time");
      params.t = time.at("t0").get<double>() + step * time.at("dt").get<double>();
    }
    for (const auto &boundary : form.boundary_state().total_local_boundary)
    {
      const int e = boundary.element_id();
      params.elem = e;
      params.body_id = mesh.get_body_id(e);
      for (int face = 0; face < boundary.size(); ++face)
      {
        params.boundary_id = mesh.get_boundary_id(boundary.global_primitive_id(face));
        if (!ids_.empty() && !ids_.count(params.boundary_id)) continue;
        Eigen::MatrixXd uv, points, normals, u, grad_u, derivative;
        Eigen::VectorXd weights;
        polyfem::utils::BoundarySampler::boundary_quadrature(boundary, form.n_boundary_samples(),
            mesh, face, false, uv, points, normals, weights);
        polyfem::assembler::ElementAssemblyValues values;
        values.compute(e, mesh.is_volume(), points, space.basis_list()[e], space.geometry_basis_list()[e]);
        polyfem::io::Evaluator::interpolate_at_local_vals(e, mesh.dimension(), space.value_dim,
            values, cache.u(step), u, grad_u);
        // The position mismatch depends on displacement and geometry only.
        functional.dj_du({}, points, values.val, u, grad_u, normals, values, params, derivative);
        derivative.array().colwise() *= weights.array();
        for (const auto &basis : values.basis_values)
          for (const auto &global : basis.global)
            rhs.segment(global.index * space.value_dim, space.value_dim) +=
                global.val * derivative.transpose() * basis.val;
      }
    }
    return weight() * rhs;
  }
};
}
