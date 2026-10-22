#include "von_mises.hpp"

#include <polyfem/assembler/ElementAssemblyValues.hpp>
#include <polyfem/io/Evaluator.hpp>
#include <polyfem/utils/IntegrableFunctional.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace polyfempy::differentiable_api
{
namespace
{
double inner(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b)
{
  return (a.array() * b.array()).sum();
}

// Analytic derivative of sigma_vm^2, with symmetric Cauchy stress. In 2D use
// the backend's xx^2 - xx*yy + yy^2 + 3*xy^2 convention (no inferred zz).
Eigen::MatrixXd squared_mises_gradient(const Eigen::MatrixXd &sigma)
{
  const Eigen::MatrixXd symmetric = (sigma + sigma.transpose()) / 2.0;
  Eigen::MatrixXd gradient = 3.0 * symmetric;
  gradient.diagonal().array() -= symmetric.trace();
  return gradient;
}

Eigen::MatrixXd deformation(const Eigen::MatrixXd &grad_u)
{
  Eigen::MatrixXd F = Eigen::MatrixXd::Identity(grad_u.rows(), grad_u.cols()) + grad_u;
  if (!F.allFinite() || !std::isfinite(F.determinant()) || F.determinant() <= 0)
    throw std::runtime_error("von_mises_squared requires a positive deformation determinant");
  return F;
}

Eigen::MatrixXd cauchy(const Eigen::MatrixXd &stress, const Eigen::MatrixXd &F, bool nonlinear)
{
  if (!nonlinear)
    return stress;
  return (stress * F.transpose()) / F.determinant();
}
}

VonMisesSquaredForm::VonMisesSquaredForm(
    const polyfem::solver::VariableToSimulationGroup &variables,
    std::shared_ptr<const polyfem::varform::DifferentiableVarForm> varform,
    std::shared_ptr<const polyfem::DiffCache> cache,
    const polyfem::json &args)
    : SpatialIntegralForm(variables, std::move(varform), std::move(cache), args)
{
  const auto formulation = varform_->primary_assembler().name();
  if (formulation != "LinearElasticity" && formulation != "NeoHookean")
    throw std::runtime_error("von_mises_squared supports LinearElasticity and NeoHookean only");
  set_integral_type(polyfem::solver::SpatialIntegralType::Volume);
  const auto selected = args.value("volume_selection", std::vector<int>{});
  ids_ = std::set<int>(selected.begin(), selected.end());
  for (int id : ids_)
  {
    bool found = false;
    for (int e = 0; e < varform_->get_mesh().n_elements(); ++e)
      found = found || varform_->get_mesh().get_body_id(e) == id;
    if (!found)
      throw std::runtime_error("von_mises_squared selection contains an unknown body ID");
  }
}

polyfem::IntegrableFunctional VonMisesSquaredForm::get_integral_functional() const
{
  polyfem::IntegrableFunctional functional;
  const bool nonlinear = varform_->primary_assembler().name() == "NeoHookean";
  // Reuse the backend stress and its constitutive derivative; only the
  // von Mises derivative and PK1-to-Cauchy chain rule are implemented here.
  auto evaluate = [varform = varform_, nonlinear](bool derivative,
      const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts,
      const Eigen::MatrixXd &grad_u, const polyfem::IntegrableFunctional::ParameterType &params,
      Eigen::MatrixXd &value) {
    value.setZero(grad_u.rows(), derivative ? grad_u.cols() : 1);
    for (int q = 0; q < grad_u.rows(); ++q)
    {
      Eigen::MatrixXd G;
      polyfem::utils::vector2matrix(grad_u.row(q), G);
      const Eigen::MatrixXd F = nonlinear ? deformation(G) : Eigen::MatrixXd::Identity(G.rows(), G.cols()).eval();
      // OptAssemblerData stores references: keep the converted rows alive.
      const Eigen::MatrixXd local_point = local_pts.row(q);
      const Eigen::MatrixXd global_point = pts.row(q);
      const double dt = varform->get_problem().is_time_dependent()
          ? varform->get_args()["time"]["dt"].get<double>() : 0.0;
      const polyfem::assembler::OptAssemblerData data(params.t, dt, params.elem,
                                                     local_point, global_point, G);
      Eigen::MatrixXd stress, unused;
      varform->primary_assembler().compute_stress_grad_multiply_mat(
          data, Eigen::MatrixXd::Zero(G.rows(), G.cols()), stress, unused);
      const Eigen::MatrixXd sigma = cauchy(stress, F, nonlinear);
      const Eigen::MatrixXd A = squared_mises_gradient(sigma);
      if (!derivative)
        value(q, 0) = 0.5 * inner(A, sigma);
      else
      {
        Eigen::MatrixXd gradient;
        const Eigen::MatrixXd stress_pullback = nonlinear ? (A * F / F.determinant()).eval() : A;
        varform->primary_assembler().compute_stress_grad_multiply_mat(
            data, stress_pullback, stress, gradient);
        if (nonlinear)
          gradient += A.transpose() * stress / F.determinant() - inner(A, sigma) * F.inverse().transpose();
        value.row(q) = polyfem::utils::flatten(gradient).transpose();
      }
    }
  };
  functional.set_j([evaluate](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts,
      const Eigen::MatrixXd &, const Eigen::MatrixXd &grad_u, const Eigen::VectorXd &,
      const Eigen::VectorXd &, const Eigen::MatrixXd &, const polyfem::assembler::ElementAssemblyValues &,
      const polyfem::IntegrableFunctional::ParameterType &params, Eigen::MatrixXd &value) {
    evaluate(false, local_pts, pts, grad_u, params, value);
  });
  functional.set_dj_dgradu([evaluate](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts,
      const Eigen::MatrixXd &, const Eigen::MatrixXd &grad_u, const Eigen::VectorXd &,
      const Eigen::VectorXd &, const Eigen::MatrixXd &, const polyfem::assembler::ElementAssemblyValues &,
      const polyfem::IntegrableFunctional::ParameterType &params, Eigen::MatrixXd &value) {
    evaluate(true, local_pts, pts, grad_u, params, value);
  });
  return functional;
}

void VonMisesSquaredForm::compute_partial_gradient_step(
    int step, const Eigen::VectorXd &x, Eigen::VectorXd &gradient) const
{
  // Only material variables are accepted by the session for this objective.
  gradient = weight() * variable_to_simulations_.apply_parametrization_jacobian(
      polyfem::solver::ParameterType::LameParameter, *varform_, x, [this, step]() {
    const auto &space = varform_->primary_space();
    const auto &bases = space.basis_list();
    const auto &gbases = space.geometry_basis_list();
    const int count = varform_->get_mesh().n_elements();
    const int dim = varform_->get_mesh().dimension();
    const bool nonlinear = varform_->primary_assembler().name() == "NeoHookean";
    const bool transient = varform_->get_problem().is_time_dependent();
    const double dt = transient ? varform_->get_args()["time"]["dt"].get<double>() : 0.0;
    const double t = transient ? varform_->get_args()["time"]["t0"].get<double>() + step * dt : 0.0;
    Eigen::VectorXd term = Eigen::VectorXd::Zero(2 * count);
    for (int e = 0; e < count; ++e)
    {
      if (!ids_.empty() && !ids_.count(varform_->get_mesh().get_body_id(e)))
        continue;
      polyfem::assembler::ElementAssemblyValues vals;
      varform_->assembly_cache().compute(e, varform_->get_mesh().is_volume(), bases[e], gbases[e], vals);
      Eigen::MatrixXd u, grad_u;
      polyfem::io::Evaluator::interpolate_at_local_vals(e, dim, dim, vals, diff_cache_->u(step), u, grad_u);
      for (int q = 0; q < vals.quadrature.weights.size(); ++q)
      {
        Eigen::MatrixXd G;
        polyfem::utils::vector2matrix(grad_u.row(q), G);
        const Eigen::MatrixXd F = nonlinear ? deformation(G) : Eigen::MatrixXd::Identity(dim, dim).eval();
        // These matrices must outlive both constitutive calls below.
        const Eigen::MatrixXd local_point = vals.quadrature.points.row(q);
        const Eigen::MatrixXd global_point = vals.val.row(q);
        const polyfem::assembler::OptAssemblerData data(t, dt, e, local_point, global_point, G);
        Eigen::MatrixXd stress, unused, dmu, dlambda;
        varform_->primary_assembler().compute_stress_grad_multiply_mat(
            data, Eigen::MatrixXd::Zero(dim, dim), stress, unused);
        varform_->primary_assembler().compute_dstress_dmu_dlambda(data, dmu, dlambda);
        const Eigen::MatrixXd A = squared_mises_gradient(cauchy(stress, F, nonlinear));
        const double da = vals.det(q) * vals.quadrature.weights(q);
        // Backend material ordering: all lambda values followed by all mu values.
        term(e) += da * inner(A, cauchy(dlambda, F, nonlinear));
        term(e + count) += da * inner(A, cauchy(dmu, F, nonlinear));
      }
    }
    return term;
  });
}
VonMisesPeakForm::VonMisesPeakForm(
    const polyfem::solver::VariableToSimulationGroup &variables,
    std::shared_ptr<const polyfem::varform::DifferentiableVarForm> varform,
    std::shared_ptr<const polyfem::DiffCache> cache, const polyfem::json &args)
    : AdjointForm(variables), varform_(std::move(varform)), cache_(std::move(cache))
{
  const auto formulation = varform_->primary_assembler().name();
  if (formulation != "LinearElasticity" && formulation != "NeoHookean")
    throw std::runtime_error("von_mises_peak supports LinearElasticity and NeoHookean only");
  // The upstream linear transient cache dereferences nl_problem, which the
  // linear solver does not initialize. Keep this unsupported path out of solves.
  if (formulation == "LinearElasticity" && varform_->get_problem().is_time_dependent())
    throw std::runtime_error("von_mises_peak transient material gradients currently require NeoHookean");
  const auto selected = args.value("volume_selection", std::vector<int>{});
  ids_ = std::set<int>(selected.begin(), selected.end());
  for (int id : ids_)
  {
    bool found = false;
    for (int e = 0; e < varform_->get_mesh().n_elements(); ++e)
      found = found || varform_->get_mesh().get_body_id(e) == id;
    if (!found)
      throw std::runtime_error("von_mises_peak selection contains an unknown body ID");
  }
}

void VonMisesPeakForm::solution_changed(const Eigen::VectorXd &x)
{
  AdjointForm::solution_changed(x);
  valid_ = false;
}

void VonMisesPeakForm::find_peak() const
{
  if (valid_) return;
  const auto &space = varform_->primary_space();
  const int dim = varform_->get_mesh().dimension();
  const bool nonlinear = varform_->primary_assembler().name() == "NeoHookean";
  const bool transient = varform_->get_problem().is_time_dependent();
  const double dt = transient ? varform_->get_args()["time"]["dt"].get<double>() : 0.;
  const double t0 = transient ? varform_->get_args()["time"]["t0"].get<double>() : 0.;
  peak_ = -1;
  element_ = -1;
  for (int step = 0; step < cache_->size(); ++step)
  {
    const Eigen::VectorXd solution = cache_->u(step);
    for (int e = 0; e < varform_->get_mesh().n_elements(); ++e)
    {
      if (!ids_.empty() && !ids_.count(varform_->get_mesh().get_body_id(e))) continue;
      polyfem::assembler::ElementAssemblyValues vals;
      varform_->assembly_cache().compute(e, varform_->get_mesh().is_volume(),
          space.basis_list()[e], space.geometry_basis_list()[e], vals);
      Eigen::MatrixXd u, gradu;
      polyfem::io::Evaluator::interpolate_at_local_vals(e, dim, dim, vals, solution, u, gradu);
      for (int q = 0; q < vals.quadrature.weights.size(); ++q)
      {
        Eigen::MatrixXd G;
        polyfem::utils::vector2matrix(gradu.row(q), G);
        const Eigen::MatrixXd F = nonlinear ? deformation(G) : Eigen::MatrixXd::Identity(dim, dim).eval();
        const Eigen::MatrixXd local = vals.quadrature.points.row(q), global = vals.val.row(q);
        const polyfem::assembler::OptAssemblerData data(t0 + step * dt, dt, e, local, global, G);
        Eigen::MatrixXd stress, unused;
        varform_->primary_assembler().compute_stress_grad_multiply_mat(data, Eigen::MatrixXd::Zero(dim, dim), stress, unused);
        const Eigen::MatrixXd sigma = cauchy(stress, F, nonlinear);
        const double squared = 0.5 * inner(squared_mises_gradient(sigma), sigma);
        if (!std::isfinite(squared)) throw std::runtime_error("Non-finite von Mises stress");
        const double value = std::sqrt(std::max(0., squared));
        if (value > peak_)
        {
          peak_ = value; step_ = step; element_ = e; point_ = q;
          position_ = vals.val.row(q);
        }
      }
    }
  }
  if (element_ < 0) throw std::runtime_error("von_mises_peak has no cached stress samples");
  valid_ = true;
}

double VonMisesPeakForm::value_unweighted(const Eigen::VectorXd &) const
{
  find_peak();
  return peak_;
}

void VonMisesPeakForm::derivatives(Eigen::MatrixXd &du, Eigen::VectorXd &material, bool need_rhs,
                                  Eigen::VectorXd *shape) const
{
  find_peak();
  const auto &space = varform_->primary_space();
  const int dim = varform_->get_mesh().dimension(), count = varform_->get_mesh().n_elements();
  if (need_rhs) du.setZero(space.ndof(), cache_->size());
  material.setZero(2 * count);
  if (shape) shape->setZero(varform_->get_mesh().n_vertices() * dim);
  // The norm is nonsmooth at zero; choose the zero subgradient.
  if (peak_ == 0) return;
  const bool nonlinear = varform_->primary_assembler().name() == "NeoHookean";
  const bool transient = varform_->get_problem().is_time_dependent();
  const double dt = transient ? varform_->get_args()["time"]["dt"].get<double>() : 0.;
  const double t0 = transient ? varform_->get_args()["time"]["t0"].get<double>() : 0.;
  polyfem::assembler::ElementAssemblyValues vals;
  varform_->assembly_cache().compute(element_, varform_->get_mesh().is_volume(),
      space.basis_list()[element_], space.geometry_basis_list()[element_], vals);
  Eigen::MatrixXd u, gradu, G;
  polyfem::io::Evaluator::interpolate_at_local_vals(element_, dim, dim, vals, cache_->u(step_), u, gradu);
  polyfem::utils::vector2matrix(gradu.row(point_), G);
  const Eigen::MatrixXd F = nonlinear ? deformation(G) : Eigen::MatrixXd::Identity(dim, dim).eval();
  const Eigen::MatrixXd local = vals.quadrature.points.row(point_), global = vals.val.row(point_);
  const polyfem::assembler::OptAssemblerData data(t0 + step_ * dt, dt, element_, local, global, G);
  Eigen::MatrixXd stress, unused, gradient, dmu, dlambda;
  varform_->primary_assembler().compute_stress_grad_multiply_mat(data, Eigen::MatrixXd::Zero(dim, dim), stress, unused);
  const Eigen::MatrixXd sigma = cauchy(stress, F, nonlinear);
  const Eigen::MatrixXd A = squared_mises_gradient(sigma) / (2 * peak_);
  const Eigen::MatrixXd pullback = nonlinear ? (A * F / F.determinant()).eval() : A;
  varform_->primary_assembler().compute_stress_grad_multiply_mat(data, pullback, stress, gradient);
  if (nonlinear)
    gradient += A.transpose() * stress / F.determinant() - inner(A, sigma) * F.inverse().transpose();
  if (shape)
  {
    // Pointwise objective: only the inverse geometry Jacobian contributes.
    // Unlike a spatial integral, there is no volume-measure derivative.
    // d(grad u)/dX_{a,d} = -grad u.col(d) * grad N_a.
    polyfem::assembler::ElementAssemblyValues geometry_values;
    geometry_values.compute(element_, varform_->get_mesh().is_volume(), vals.quadrature.points,
        space.geometry_basis_list()[element_], space.geometry_basis_list()[element_]);
    Eigen::VectorXd node_gradient = Eigen::VectorXd::Zero(space.geometry->n_bases * dim);
    for (const auto &basis : geometry_values.basis_values)
      for (const auto &global_basis : basis.global)
        for (int d = 0; d < dim; ++d)
          node_gradient(global_basis.index * dim + d) -= weight() * global_basis.val
              * inner(gradient, G.col(d) * basis.grad_t_m.row(point_));
    *shape = polyfem::utils::flatten(polyfem::utils::unflatten(node_gradient, dim)(
        varform_->primitive_to_node(), Eigen::all));
  }
  const Eigen::VectorXd flat = polyfem::utils::flatten(gradient);
  if (need_rhs)
    for (const auto &basis : vals.basis_values)
      for (const auto &global_basis : basis.global)
        for (int d = 0; d < dim; ++d)
          du(global_basis.index * dim + d, step_) += weight() * global_basis.val
              * flat.segment(d * dim, dim).dot(basis.grad_t_m.row(point_));
  varform_->primary_assembler().compute_dstress_dmu_dlambda(data, dmu, dlambda);
  material(element_) = weight() * inner(A, cauchy(dlambda, F, nonlinear));
  material(element_ + count) = weight() * inner(A, cauchy(dmu, F, nonlinear));
}

Eigen::MatrixXd VonMisesPeakForm::compute_adjoint_rhs(const Eigen::VectorXd &,
    const polyfem::varform::DifferentiableVarForm &, const polyfem::DiffCache &) const
{
  Eigen::MatrixXd du;
  Eigen::VectorXd material;
  derivatives(du, material);
  return du;
}

void VonMisesPeakForm::compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradient) const
{
  gradient = variable_to_simulations_.apply_parametrization_jacobian(
      polyfem::solver::ParameterType::LameParameter, *varform_, x, [this]() {
        Eigen::MatrixXd du;
        Eigen::VectorXd material;
        derivatives(du, material, false);
        return material;
      });
  gradient += variable_to_simulations_.apply_parametrization_jacobian(
      polyfem::solver::ParameterType::Shape, *varform_, x, [this]() {
        Eigen::MatrixXd du;
        Eigen::VectorXd material, shape;
        derivatives(du, material, false, &shape);
        return shape;
      });
}

polyfem::json VonMisesPeakForm::info() const
{
  find_peak();
  const bool transient = varform_->get_problem().is_time_dependent();
  const double t = transient ? varform_->get_args()["time"]["t0"].get<double>()
      + step_ * varform_->get_args()["time"]["dt"].get<double>() : 0.;
  std::vector<double> position(position_.data(), position_.data() + position_.size());
  return {{"objective", name()}, {"value", weight() * peak_}, {"step", step_},
      {"time", t}, {"element", element_}, {"quadrature_point", point_},
      {"reference_position", position}, {"sampling", "all cached steps, element quadrature points"},
      {"tie_policy", "first step/element/quadrature point; zero subgradient at zero"}};
}

}
