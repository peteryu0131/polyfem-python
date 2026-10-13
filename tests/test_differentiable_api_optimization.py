import pytest

from polyfempy import differentiable_api as diff

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("optimizer", [torch.optim.Adam, torch.optim.SGD])
def test_closure_loop_matches_explicit_loop_with_projection_and_recording(optimizer):
    compact = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    explicit = compact.detach().clone().requires_grad_(True)

    def loss(theta):
        velocity = torch.stack((theta.cos(), theta.sin()))
        return (velocity - torch.tensor([0.1, 1.0])).square().sum()

    result = (diff.optimization(lambda: loss(compact), parameters=[compact])
              .optimizer(optimizer, lr=0.4).steps(5)
              .optimize(bounds=[(0.1, 0.8)], record=lambda step, value: {"theta": float(compact.detach())}))
    opt = optimizer([explicit], lr=0.4)
    for row in result.history:
        opt.zero_grad()
        value = loss(explicit)
        value.backward()
        assert row["theta"] == pytest.approx(float(explicit.detach()))
        assert row["loss"] == pytest.approx(float(value.detach()))
        assert row["gradient_norm"] == pytest.approx(float(explicit.grad.abs()))
        opt.step()
        with torch.no_grad():
            explicit.clamp_(0.1, 0.8)
    torch.testing.assert_close(compact, explicit)
    assert result.loss.detach() == value.detach()
    assert float(compact.detach()) == pytest.approx(0.8)


@pytest.mark.parametrize("kind", ["material", "initial_condition"])
def test_direct_primitive_optimizer_uses_same_leaf_parameter(kind, monkeypatch):
    # Numerical loop test independent of backend setup; backend examples run separately.
    parameter = torch.ones((3, 2), dtype=torch.float64, requires_grad=True)
    cls = diff.MaterialOptimization if kind == "material" else diff.InitialConditionOptimization
    monkeypatch.setattr(cls, "__call__", lambda self: parameter.square().sum())
    wrapper = getattr(diff, f"{kind}_opt")(diff.model({}), parameter, objective=diff.Objective.COMPLIANCE)
    result = wrapper.optimizer(torch.optim.SGD, lr=0.1).steps(2).optimize()
    torch.testing.assert_close(parameter, torch.full_like(parameter, 0.64))
    assert result.parameters[0] is parameter


def test_optimizer_rejects_nonleaf_parameters_mismatches_and_invalid_steps():
    p = torch.tensor(1.0, requires_grad=True)
    with pytest.raises(ValueError, match="leaf"):
        diff.optimization(lambda: p.square(), parameters=[p * 2])
    runner = diff.optimization(lambda: p.square(), parameters=[p])
    with pytest.raises(ValueError, match="configure"):
        runner.optimize()
    other = torch.tensor(1.0, requires_grad=True)
    with pytest.raises(ValueError, match="exactly"):
        runner.optimizer(torch.optim.SGD([other], lr=0.1))
    with pytest.raises(TypeError, match="no-argument"):
        runner.optimizer(torch.optim.LBFGS, lr=0.1)
    for steps in (True, 1.5):
        with pytest.raises(TypeError, match="integer"):
            runner.steps(steps)
    with pytest.raises(ValueError, match="positive"):
        runner.steps(0)
    runner.optimizer(torch.optim.SGD([p], lr=0.1)).steps(2)
    with pytest.raises(ValueError, match="within bounds"):
        runner.optimize(bounds=[(0, 0.5)])


def test_multiple_parameters_and_disconnected_gradient():
    p = torch.tensor(1.0, requires_grad=True)
    q = torch.tensor(2.0, requires_grad=True)
    runner = diff.optimization(lambda: p.square() + q.square(), parameters=[p, q])
    result = runner.optimizer(torch.optim.SGD, lr=0.1).steps(2).optimize()
    assert float(p.detach()) == pytest.approx(0.64)
    assert float(q.detach()) == pytest.approx(1.28)
    assert len(result.history) == 2
    with pytest.raises(ValueError, match="every supplied parameter"):
        (diff.optimization(lambda: p.square(), parameters=[p, q])
         .optimizer(torch.optim.SGD, lr=0.1).steps(1).optimize())
