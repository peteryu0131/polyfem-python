// Diagnostic driver for the original optimization path, not DifferentiableSession.
#include <polyfem/optimization/OptState.hpp>
#include <polyfem/optimization/AdjointNLProblem.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include "shape_setup.hpp"

int main(int argc, char **argv)
{
    if (argc != 3 && argc != 4)
    {
        std::cerr << "Usage: polyfem_reference_probe staged-opt.json result.json [--no-strict-validation]\n";
        return 2;
    }
    try
    {
        polyfem::json config;
        std::ifstream input(argv[1]);
        input >> config;
        config["root_path"] = std::filesystem::absolute(argv[1]).string();
        config["compute_objective"] = true;
        polyfem::OptState state;
        const bool strict = argc == 3;
        const bool setup = !strict && std::string(argv[3]) == "--shape-setup";
        if (!strict && !setup && std::string(argv[3]) != "--no-strict-validation")
            return 2;
        // Same setup sequence as OptState::run's non-remeshing path. Export
        // its first initial_guess, rather than inverse-mapping twice.
        state.init(config, strict);
        state.create_varforms(state.args["solver"]["max_threads"].get<int>());
        if (setup)
        {
            std::ofstream output(argv[2]);
            output << shape_setup(*state.varforms.at(0), config.at("variable_to_simulation")).dump();
            return output.good() ? 0 : 1;
        }
        state.init_variables();
        state.create_problem();
        Eigen::VectorXd x, gradient;
        state.initial_guess(x);
        const double value = state.eval(x);
        state.nl_problem->gradient(x, gradient);
        polyfem::json result;
        result["x"] = std::vector<double>(x.data(), x.data() + x.size());
        result["gradient"] = std::vector<double>(gradient.data(), gradient.data() + gradient.size());
        result["value"] = value;
        result["resolved_opt"] = state.args;
        result["state_args"] = state.state_args;
        Eigen::VectorXd mapped;
        const bool material = state.args["variable_to_simulation"][0]["type"] == "elastic";
        // Shared material parameters expand to one Lamé pair per element.
        mapped.setZero(material ? 2 * state.varforms.front()->get_mesh().n_elements() : x.size());
        state.variable_to_simulations.data[0]->update_state_variables(x, mapped);
        result["mapped_parameters"] = std::vector<double>(mapped.data(), mapped.data() + mapped.size());
        // Check the same trial point on both paths without pretending that
        // upstream's optimizer and Adam have identical update rules.
        const double step = 1e-5;
        Eigen::VectorXd trial = x - step * gradient / gradient.norm();
        result["trial_x"] = std::vector<double>(trial.data(), trial.data() + trial.size());
        result["trial_value"] = state.eval(trial);
        result["trial_step_norm"] = step;
        std::ofstream output(argv[2]);
        output << result.dump(2) << '\n';
        return output.good() ? 0 : 1;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
