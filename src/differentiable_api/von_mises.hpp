#pragma once

#include <polyfem/optimization/forms/SpatialIntegralForms.hpp>

namespace polyfempy::differentiable_api
{
// Reference-volume integral of squared Cauchy von Mises stress. The session
// limits this implementation to material parameters, static or final-time.
class VonMisesSquaredForm : public polyfem::solver::SpatialIntegralForm
{
public:
  VonMisesSquaredForm(
      const polyfem::solver::VariableToSimulationGroup &variables,
      std::shared_ptr<const polyfem::varform::DifferentiableVarForm> varform,
      std::shared_ptr<const polyfem::DiffCache> cache,
      const polyfem::json &args);

  std::string name() const override { return "von_mises_squared"; }
  void compute_partial_gradient_step(int step, const Eigen::VectorXd &x,
                                     Eigen::VectorXd &gradient) const override;

protected:
  polyfem::IntegrableFunctional get_integral_functional() const override;
};
// Exact maximum over all cached time steps and element quadrature points.
// Ties select the first (step, element, quadrature) in serial traversal order.
class VonMisesPeakForm : public polyfem::solver::AdjointForm
{
public:
  VonMisesPeakForm(const polyfem::solver::VariableToSimulationGroup &variables,
      std::shared_ptr<const polyfem::varform::DifferentiableVarForm> varform,
      std::shared_ptr<const polyfem::DiffCache> cache, const polyfem::json &args);
  std::string name() const override { return "von_mises_peak"; }
  double value_unweighted(const Eigen::VectorXd &x) const override;
  Eigen::MatrixXd compute_adjoint_rhs(const Eigen::VectorXd &x,
      const polyfem::varform::DifferentiableVarForm &varform,
      const polyfem::DiffCache &cache) const override;
  void compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradient) const override;
  void solution_changed(const Eigen::VectorXd &x) override;
  polyfem::json info() const;
private:
  void find_peak() const;
  void derivatives(Eigen::MatrixXd &du, Eigen::VectorXd &material, bool need_rhs = true,
                   Eigen::VectorXd *shape = nullptr) const;
  std::shared_ptr<const polyfem::varform::DifferentiableVarForm> varform_;
  std::shared_ptr<const polyfem::DiffCache> cache_;
  std::set<int> ids_;
  mutable bool valid_ = false;
  mutable double peak_ = 0;
  mutable int step_ = 0, element_ = -1, point_ = 0;
  mutable Eigen::RowVectorXd position_;
};

}
