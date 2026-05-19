from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)


def test_make_torch_optimizer_supports_sgd_and_adam():
    from polyfempy.differentiable.optimization.optimizers import make_torch_optimizer

    parameter = torch.nn.Parameter(torch.tensor(1.0))

    sgd = make_torch_optimizer([parameter], name="sgd", lr=0.1)
    adam = make_torch_optimizer([parameter], name="adam", lr=0.01)

    assert isinstance(sgd, torch.optim.SGD)
    assert isinstance(adam, torch.optim.Adam)


def test_make_torch_optimizer_rejects_empty_parameter_list():
    from polyfempy.differentiable.optimization.optimizers import make_torch_optimizer

    with pytest.raises(ValueError, match="no design parameters"):
        make_torch_optimizer([], name="sgd", lr=0.1, empty_error="no design parameters")


def test_make_torch_optimizer_rejects_unknown_optimizer():
    from polyfempy.differentiable.optimization.optimizers import make_torch_optimizer

    parameter = torch.nn.Parameter(torch.tensor(1.0))

    with pytest.raises(ValueError, match="unsupported optimizer"):
        make_torch_optimizer([parameter], name="rmsprop", lr=0.1)
