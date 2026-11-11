#pragma once

#include <polyfem/optimization/AdjointTools.hpp>
#include <polyfem/optimization/VarFormDiff.hpp>
#include <polyfem/optimization/var2sims/DirichletBoundaryVariableToSimulation.hpp>

#include <algorithm>
#include <map>
#include <set>
#include <unordered_map>

namespace polyfempy::differentiable_api
{
// One prescribed value moves each constrained FE degree of freedom once,
// even when several primitives in the same boundary selection share a node.
class BoundaryDisplacementControl final : public polyfem::solver::DirichletBoundaryVariableToSimulation
{
public:
  BoundaryDisplacementControl(const std::shared_ptr<polyfem::varform::DifferentiableVarForm> &form,
      const std::shared_ptr<polyfem::DiffCache> &cache, const Eigen::VectorXi &ids, const Eigen::VectorXi &times)
      : DirichletBoundaryVariableToSimulation({form}, {cache}, {}, ids, times),
        form_(form), cache_(cache), times_(times)
  {
    const int dim = form->get_mesh().dimension();
    std::vector<int> selected(ids.data(), ids.data() + ids.size());
    if (selected.empty())
    {
      for (const auto &entry : form->boundary_conditions_ids("dirichlet_boundary"))
        selected.push_back(entry.first);
      std::sort(selected.begin(), selected.end());
    }
    if (times_.size() == 0)
    {
      const int steps = form->get_args()["time"]["time_steps"].get<int>();
      times_ = Eigen::VectorXi::LinSpaced(steps, 0, steps - 1);
    }
    const auto &nodes = form->boundary_state().boundary_nodes;
    std::unordered_map<int, int> offset;
    for (int i = 0; i < nodes.size(); ++i) offset.emplace(nodes[i], i);
    std::map<int, int> group;
    for (int i = 0; i < selected.size(); ++i) group.emplace(selected[i], i);
    offsets_.resize(selected.size() * dim);
    for (const auto &boundary : form->boundary_state().local_boundary)
    {
      const auto &basis = form->primary_space().basis_list()[boundary.element_id()];
      for (int i = 0; i < boundary.size(); ++i)
      {
        const int face = boundary.global_primitive_id(i);
        const auto it = group.find(form->get_mesh().get_boundary_id(face));
        if (it == group.end()) continue;
        for (int node : basis.local_nodes_for_primitive(face, form->get_mesh()))
          for (const auto &global : basis.bases[node].global())
            for (int d = 0; d < dim; ++d)
            {
              const auto pos = offset.find(global.index * dim + d);
              if (pos != offset.end()) offsets_[it->second * dim + d].insert(pos->second);
            }
      }
    }
  }

  Eigen::VectorXd compute_adjoint_term(const Eigen::VectorXd &x) const override
  {
    Eigen::VectorXd nodal;
    polyfem::solver::AdjointTools::dJ_dirichlet_transient_adjoint_term(*form_,
        polyfem::get_adjoint_mat(*form_, *cache_, 1), polyfem::get_adjoint_mat(*form_, *cache_, 0), nodal);
    const int stride = form_->boundary_state().boundary_nodes.size();
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(x.size());
    for (int t = 0; t < times_.size(); ++t)
      for (int j = 0; j < offsets_.size(); ++j)
        for (int offset : offsets_[j])
          gradient(t * offsets_.size() + j) += nodal(times_(t) * stride + offset);
    return gradient;
  }

private:
  std::shared_ptr<polyfem::varform::DifferentiableVarForm> form_;
  std::shared_ptr<polyfem::DiffCache> cache_;
  Eigen::VectorXi times_;
  std::vector<std::set<int>> offsets_;
};
}
