from types import SimpleNamespace

import numpy as np
import pytest
import torch

from polyfempy import differentiable_api as diff
from test_differentiable_api_controls import _Session


def test_periodic_parameter_initialization_and_independent_graphs():
    class Session(_Session):
        def prepare_control(self, spec):
            assert spec['type'] == 'periodic-shape'
            return 6

        def control_initial_parameters(self):
            return np.arange(6, dtype=float)

    obj = diff.periodic_shape({}, objective=diff.objectives.create('stress', dimensions=[1, 1]),
                              backend=SimpleNamespace(DifferentiableSession=Session))
    x = obj.initial_parameters.requires_grad_()
    assert x.shape == (6,) and x.dtype == torch.float64
    obj.initial_parameters.fill_(99)
    assert obj.initial_parameters.tolist() == list(range(6))
    a, b = obj(parameters=x), obj(parameters=2*x)
    (a+b).square().backward()
    expected = 20*x.detach()*(5*x.detach().square().sum())
    torch.testing.assert_close(x.grad, expected)
    with pytest.raises(ValueError, match='shape'):
        obj(parameters=x.reshape(3, 2))
