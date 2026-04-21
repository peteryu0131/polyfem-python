#include "local_objectives.hpp"

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/AssemblerData.hpp>
#include <polyfem/solver/Optimizations.hpp>
#include <polyfem/solver/forms/adjoint_forms/SpatialIntegralForms.hpp>
#include <polyfem/solver/forms/parametrization/Parametrizations.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>
#include <polyfem/utils/IntegrableFunctional.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace polyfem;
using namespace polyfem::solver;

namespace
{
class SelectedStepForm : public AdjointForm
{
public:
    SelectedStepForm(const std::shared_ptr<StaticForm> &base, const int step)
        : AdjointForm(base->get_variable_to_simulations()), base_(base), step_(step)
    {
        set_weight(base_->weight());
    }

    std::string name() const override { return base_->name(); }

    double value_unweighted(const Eigen::VectorXd &x) const override
    {
        return base_->value_unweighted_step(step_, x);
    }

    void solution_changed(const Eigen::VectorXd &new_x) override
    {
        AdjointForm::solution_changed(new_x);
        base_->solution_changed_step(step_, new_x);
    }

    void compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override
    {
        base_->compute_partial_gradient_step(step_, x, gradv);
    }

    Eigen::MatrixXd compute_adjoint_rhs(const Eigen::VectorXd &x, const State &state) const override
    {
        if (step_ < 0 || step_ >= state.diff_cached.size())
            throw std::runtime_error("objective state index is out of range for diff_cached");

        Eigen::MatrixXd term = Eigen::MatrixXd::Zero(state.ndof(), state.diff_cached.size());
        if (base_->depends_on_step_prev() && step_ > 0)
            term.col(step_ - 1) = base_->compute_adjoint_rhs_step_prev(step_, x, state);
        term.col(step_) = base_->compute_adjoint_rhs_step(step_, x, state);
        return term;
    }

private:
    std::shared_ptr<StaticForm> base_;
    int step_;
};

double von_mises_power_value(const Eigen::MatrixXd &sigma, const int power)
{
    const double vm = polyfem::von_mises_stress_for_stress_tensor(sigma);
    return std::pow(std::max(vm, 0.0), static_cast<double>(power));
}

Eigen::MatrixXd numerical_von_mises_power_gradient(const Eigen::MatrixXd &sigma, const int power)
{
    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(sigma.rows(), sigma.cols());
    for (int i = 0; i < sigma.rows(); ++i)
    {
        for (int j = 0; j < sigma.cols(); ++j)
        {
            const double x = sigma(i, j);
            const double h = 1e-6 * std::max(1.0, std::abs(x));

            Eigen::MatrixXd sigma_plus = sigma;
            Eigen::MatrixXd sigma_minus = sigma;
            sigma_plus(i, j) += h;
            sigma_minus(i, j) -= h;

            const double f_plus = von_mises_power_value(sigma_plus, power);
            const double f_minus = von_mises_power_value(sigma_minus, power);
            grad(i, j) = (f_plus - f_minus) / (2.0 * h);
        }
    }
    return grad;
}

class VonMisesForm : public SpatialIntegralForm
{
public:
    VonMisesForm(const VariableToSimulationGroup &variable_to_simulations, const State &state, const polyfem::json &args)
        : SpatialIntegralForm(variable_to_simulations, state, args)
    {
        set_integral_type(SpatialIntegralType::Volume);

        if (args.contains("volume_selection"))
        {
            const auto tmp_ids = args["volume_selection"].get<std::vector<int>>();
            ids_ = std::set(tmp_ids.begin(), tmp_ids.end());
        }

        if (args.contains("power"))
            power_ = std::max(1, args["power"].get<int>());
    }

    std::string name() const override { return "von_mises"; }

    void compute_partial_gradient_step(const int time_step, const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override
    {
        SpatialIntegralForm::compute_partial_gradient_step(time_step, x, gradv);
        gradv += weight() * variable_to_simulations_.apply_parametrization_jacobian(ParameterType::LameParameter, &state_, x, [this]() {
            throw std::runtime_error("von_mises does not support direct material derivatives in this binding");
            return Eigen::VectorXd::Zero(0).eval();
        });
    }

protected:
    IntegrableFunctional get_integral_functional() const override
    {
        IntegrableFunctional j;

        const std::string formulation = state_.formulation();
        const int power = power_;

        j.set_j([formulation, power, &state = std::as_const(state_)](const Eigen::MatrixXd &local_pts,
                                                                     const Eigen::MatrixXd &pts,
                                                                     const Eigen::MatrixXd &u,
                                                                     const Eigen::MatrixXd &grad_u,
                                                                     const Eigen::VectorXd &lambda,
                                                                     const Eigen::VectorXd &mu,
                                                                     const Eigen::MatrixXd &reference_normals,
                                                                     const assembler::ElementAssemblyValues &vals,
                                                                     const IntegrableFunctional::ParameterType &params,
                                                                     Eigen::MatrixXd &val) {
            val.setZero(grad_u.rows(), 1);
            const double dt = state.problem->is_time_dependent() ? state.args["time"]["dt"].get<double>() : 0.0;

            Eigen::MatrixXd grad_u_q;
            for (int q = 0; q < grad_u.rows(); ++q)
            {
                utils::vector2matrix(grad_u.row(q), grad_u_q);

                Eigen::MatrixXd stress;
                Eigen::MatrixXd grad_unused;
                state.assembler->compute_stress_grad_multiply_mat(
                    assembler::OptAssemblerData(params.t, dt, params.elem, local_pts.row(q), pts.row(q), grad_u_q),
                    Eigen::MatrixXd::Zero(grad_u_q.rows(), grad_u_q.cols()),
                    stress,
                    grad_unused);

                Eigen::MatrixXd sigma = stress;
                if (formulation != "LinearElasticity")
                {
                    const Eigen::MatrixXd F = Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols()) + grad_u_q;
                    const double J = F.determinant();
                    sigma = (stress * F.transpose()) / J;
                }

                val(q) = von_mises_power_value(sigma, power);
            }
        });

        j.set_dj_dgradu([formulation, power, &state = std::as_const(state_)](const Eigen::MatrixXd &local_pts,
                                                                              const Eigen::MatrixXd &pts,
                                                                              const Eigen::MatrixXd &u,
                                                                              const Eigen::MatrixXd &grad_u,
                                                                              const Eigen::VectorXd &lambda,
                                                                              const Eigen::VectorXd &mu,
                                                                              const Eigen::MatrixXd &reference_normals,
                                                                              const assembler::ElementAssemblyValues &vals,
                                                                              const IntegrableFunctional::ParameterType &params,
                                                                              Eigen::MatrixXd &val) {
            val.setZero(grad_u.rows(), grad_u.cols());
            const double dt = state.problem->is_time_dependent() ? state.args["time"]["dt"].get<double>() : 0.0;

            Eigen::MatrixXd grad_u_q;
            for (int q = 0; q < grad_u.rows(); ++q)
            {
                utils::vector2matrix(grad_u.row(q), grad_u_q);

                Eigen::MatrixXd stress;
                Eigen::MatrixXd grad_unused;
                const assembler::OptAssemblerData data(params.t, dt, params.elem, local_pts.row(q), pts.row(q), grad_u_q);
                state.assembler->compute_stress_grad_multiply_mat(
                    data,
                    Eigen::MatrixXd::Zero(grad_u_q.rows(), grad_u_q.cols()),
                    stress,
                    grad_unused);

                Eigen::MatrixXd dloss_dgradu;
                if (formulation == "LinearElasticity")
                {
                    const Eigen::MatrixXd sigma = stress;
                    const Eigen::MatrixXd dloss_dsigma = numerical_von_mises_power_gradient(sigma, power);
                    state.assembler->compute_stress_grad_multiply_mat(data, dloss_dsigma, stress, dloss_dgradu);
                }
                else
                {
                    const Eigen::MatrixXd F = Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols()) + grad_u_q;
                    const double J = F.determinant();
                    const Eigen::MatrixXd sigma = (stress * F.transpose()) / J;
                    const Eigen::MatrixXd dloss_dsigma = numerical_von_mises_power_gradient(sigma, power);

                    Eigen::MatrixXd dloss_dpk1;
                    const Eigen::MatrixXd mat = (dloss_dsigma * F) / J;
                    state.assembler->compute_stress_grad_multiply_mat(data, mat, stress, dloss_dpk1);

                    const Eigen::MatrixXd FmT = F.inverse().transpose();
                    const double sigma_dot = (dloss_dsigma.array() * sigma.array()).sum();
                    dloss_dgradu = dloss_dpk1 + (dloss_dsigma.transpose() * stress) / J - sigma_dot * FmT;
                }

                val.row(q) = utils::flatten(dloss_dgradu);
            }
        });

        return j;
    }

private:
    int power_ = 2;
};
} // namespace

namespace polyfempy::differentiable
{
std::shared_ptr<AdjointForm> create_local_objective(
    const std::string &obj_type,
    const std::string &param_type,
    const std::shared_ptr<State> &state,
    const polyfem::json &args)
{
    std::shared_ptr<AdjointForm> obj;

    if (obj_type == "von_mises")
    {
        VariableToSimulationGroup var2sim;
        var2sim.push_back(VariableToSimulation::create(param_type, {state}, CompositeParametrization()));
        obj = std::make_shared<VonMisesForm>(var2sim, *state, args);
    }
    else
    {
        obj = AdjointOptUtils::create_simple_form(obj_type, param_type, state, args);
    }

    if (args.contains("weight"))
        obj->set_weight(args["weight"].get<double>());

    if (args.contains("state"))
    {
        const auto static_obj = std::dynamic_pointer_cast<StaticForm>(obj);
        if (static_obj)
        {
            const int step = args["state"].get<int>();
            if (step < 0 || step >= state->diff_cached.size())
                throw std::runtime_error("objective state index is out of range for diff_cached");
            obj = std::make_shared<SelectedStepForm>(static_obj, step);
        }
    }

    return obj;
}
} // namespace polyfempy::differentiable
