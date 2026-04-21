#pragma once

#include <memory>
#include <string>

#include <polyfem/State.hpp>
#include <polyfem/solver/forms/adjoint_forms/AdjointForm.hpp>
#include <polyfem/utils/JSONUtils.hpp>

namespace polyfempy::differentiable
{
std::shared_ptr<polyfem::solver::AdjointForm> create_local_objective(
    const std::string &obj_type,
    const std::string &param_type,
    const std::shared_ptr<polyfem::State> &state,
    const json &args);
}
