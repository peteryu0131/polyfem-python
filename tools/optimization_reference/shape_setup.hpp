// Export fixed geometry data for the PyTorch shape examples.
#include <polyfem/optimization/NodeSelectionUtils.hpp>
#include <polyfem/optimization/parametrization/SplineParametrizations.hpp>

static polyfem::json shape_setup(const polyfem::varform::DifferentiableVarForm &form,
                                const polyfem::json &mappings)
{
    Eigen::MatrixXd vertices;
    form.get_vertices(vertices);
    polyfem::json result = polyfem::json::array();
    for (const auto &mapping : mappings)
    {
        const auto kind = mapping.at("composite_map_type").get<std::string>();
        Eigen::VectorXi nodes;
        if (kind == "boundary")
            nodes = polyfem::select_boundary_nodes(form, mapping.at("surface_selection"));
        else if (kind == "interior")
            nodes = polyfem::select_interior_nodes(form, mapping.at("volume_selection"));
        else
            throw std::runtime_error("Unsupported shape node selection: " + kind);
        polyfem::json entry;
        entry["nodes"] = std::vector<int>(nodes.data(), nodes.data() + nodes.size());
        const auto &maps = mapping.at("composition");
        entry["part"] = maps.at(0).at("parameter_index");
        if (maps.size() == 2)
        {
            const auto &spec = maps.at(1);
            if (spec.at("type") != "bounded-biharmonic-weights" || nodes.size() != spec.at("num_vertices").get<int>())
                throw std::runtime_error("Unexpected BBW selection or map");
            const int count = spec.at("num_control_vertices");
            polyfem::solver::BoundedBiharmonicWeights2Dto3D bbw(count, nodes.size(), form, spec.at("allow_rotations"));
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> surface = vertices(nodes, Eigen::all);
            bbw.inverse_eval(Eigen::Map<const Eigen::VectorXd>(surface.data(), surface.size()));
            const Eigen::MatrixXd weights = bbw.get_bbw_weights();
            for (int i = 0; i < weights.rows(); ++i)
            {
                const Eigen::VectorXd row = weights.row(i);
                entry["weights"].push_back(std::vector<double>(row.data(), row.data() + row.size()));
            }
            // Handle vertices satisfy the unit Dirichlet condition for their column.
            for (int j = 0; j < count; ++j)
            {
                Eigen::Index i;
                if (weights.col(j).maxCoeff(&i) < 1.0 - 1e-7)
                    throw std::runtime_error("BBW handle missing its unit weight");
                entry["handles"].push_back(nodes(i));
            }
        }
        result.push_back(entry);
    }
    return result;
}
