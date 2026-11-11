"""Physical control inputs backed by existing PolyFEM parameter mappings."""

import json

from .shape import _load_default_backend


class ControlObject:
    """One control category and one state; boundary values use internal units."""

    def __init__(self, config, *, kind, objective, selection=None,
                 boundary_ids=None, time_slices=None, node_ids=None, target=None, backend=None):
        from .object import _model, _objective
        self.model = _model(config)
        self.objective = _objective(objective, selection)
        self._target = None if target is None else _model(target).as_dicts()[0]
        spatial = self.objective.get('static_objective', self.objective)
        if self._target is not None:
            if (kind not in ('damping', 'friction') or self.objective.get('type') != 'transient_integral'
                    or spatial.get('type') != 'target'
                    or self.objective.get('state', 0) != 0
                    or spatial.get('state', 0) != 0 or spatial.get('target_state') != 1):
                raise ValueError('target requires damping or friction with a transient_integral target objective, state=0, target_state=1')
        elif spatial.get('type') == 'target':
            raise ValueError('A target objective requires target=config')
        self.backend = backend if backend is not None else _load_default_backend()
        self.kind = kind
        self.spec = {'type': kind}
        if kind == 'dirichlet-nodes':
            self.node_ids = self._indices(node_ids, 'node_ids', nonempty=True)
            self.spec['active_geometry_nodes'] = self.node_ids
        if kind in ('dirichlet-boundary', 'pressure'):
            self.boundary_ids = self._indices(boundary_ids, 'boundary_ids', nonempty=True)
            self.time_slices = self._indices([] if time_slices is None else time_slices, 'time_slices')
            self.spec.update(active_boundary_ids=self.boundary_ids,
                             active_time_slices=self.time_slices)
        session = self._session()
        count = session.prepare_control(self.spec)
        settings = json.loads(session.settings())
        if kind == 'friction':
            if not settings.get('contact', {}).get('enabled', False):
                raise ValueError('Friction optimization requires enabled contact')
            stiffness = settings.get('solver', {}).get('contact', {}).get('barrier_stiffness')
            if not isinstance(stiffness, (int, float)) or isinstance(stiffness, bool) or stiffness <= 0:
                raise ValueError('Friction gradients currently require a positive fixed barrier_stiffness')
        if kind == 'periodic-shape':
            self.input_shape = (count,)
            self._initial_parameters = tuple(float(v) for v in session.control_initial_parameters())
        elif kind == 'damping':
            self.input_shape = (2,)
        elif kind == 'friction':
            self.input_shape = ()
        elif kind == 'dirichlet-nodes':
            dimension = session.model_metadata()['dimension']
            self.input_shape = (len(self.node_ids), dimension)
            if count != len(self.node_ids) * dimension:
                raise RuntimeError('Unexpected nodal Dirichlet parameter count')
        else:
            dimension = session.model_metadata()['dimension']
            nb = len(self.boundary_ids)
            nt = count // (nb * dimension if kind == 'dirichlet-boundary' else nb)
            transient = settings.get('time') is not None
            if not transient and self.time_slices:
                raise ValueError('time_slices apply only to transient simulations')
            if transient:
                self._validate_boundary_series(settings, dimension)
            self.input_shape = ((nt, nb, dimension) if kind == 'dirichlet-boundary'
                                else (nt, nb) if transient else (nb,))
        del session

    def _validate_boundary_series(self, settings, dimension):
        import math
        time = settings['time']
        steps = time['time_steps']
        key = 'dirichlet_boundary' if self.kind == 'dirichlet-boundary' else 'pressure_boundary'
        boundaries = settings['boundary_conditions'][key]
        if isinstance(boundaries, dict):
            boundaries = [boundaries]
        for boundary_id in self.boundary_ids:
            boundary = next(b for b in boundaries if b['id'] == boundary_id)
            series = boundary['value'] if self.kind == 'dirichlet-boundary' else [boundary['value']]
            if (len(series) != (dimension if self.kind == 'dirichlet-boundary' else 1)
                    or any(not isinstance(row, list) or len(row) < steps + 1
                           or any(type(v) not in (int, float) or not math.isfinite(v) for v in row)
                           for row in series)):
                raise ValueError('Transient boundary control requires numeric time series including step 0')
            reference = boundary.get('time_reference', [])
            expected = [time.get('t0', 0.) + i*time['dt'] for i in range(len(series[0]))]
            if (not isinstance(reference, list) or len(reference) != len(expected)
                    or any(len(row) != len(expected) for row in series)
                    or any(not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
                           for a, b in zip(reference, expected))):
                raise ValueError('Boundary time_reference must match every simulation time step')

    @staticmethod
    def _indices(values, name, nonempty=False):
        if (not isinstance(values, (tuple, list)) or (nonempty and not values)
                or any(type(v) is not int or v < 0 for v in values)
                or len(set(values)) != len(values)):
            raise ValueError(f'{name} must contain distinct nonnegative integers')
        return list(values)

    def _session(self):
        session = self.backend.DifferentiableSession()
        if not hasattr(session, 'prepare_control'):
            raise RuntimeError('Control inputs require rebuilding the polyfempy backend')
        session.set_settings(self.model.as_dicts()[0])
        return session

    def __call__(self, *, values, V=None):
        import torch
        from .torch_ops import _ControlOpt
        if (not isinstance(values, torch.Tensor) or not values.is_floating_point()
                or tuple(values.shape) != self.input_shape
                or not bool(torch.isfinite(values).all())):
            raise ValueError(f'values must be a finite floating tensor of shape {self.input_shape}')
        if self.kind in ('damping', 'friction') and bool((values < 0).any()):
            raise ValueError('Damping and friction coefficients must be nonnegative')
        if V is not None:
            if self.kind != "dirichlet-boundary" or self._target is not None:
                raise ValueError("V requires a boundary displacement control without a target state")
            if not isinstance(V, torch.Tensor) or V.dtype != values.dtype or V.device != values.device or not torch.isfinite(V).all():
                raise ValueError("V must be a finite tensor with the same dtype and device as values")
            return _ControlOpt.apply(values, self, V)
        return _ControlOpt.apply(values, self)


class DampingObject(ControlObject):
    def __call__(self, *, psi, phi):
        import torch
        if not all(isinstance(v, torch.Tensor) and v.ndim == 0 for v in (psi, phi)):
            raise ValueError('psi and phi must be scalar tensors')
        if psi.dtype != phi.dtype or psi.device != phi.device:
            raise ValueError('psi and phi must have the same dtype and device')
        return super().__call__(values=torch.stack((psi, phi)))


class FrictionObject(ControlObject):
    def __call__(self, *, coefficient):
        return super().__call__(values=coefficient)


class PeriodicShapeObject(ControlObject):
    """Static homogenization with the backend's reduced periodic parameter vector."""

    @property
    def initial_parameters(self):
        import torch
        return torch.tensor(self._initial_parameters, dtype=torch.float64)

    def __call__(self, *, parameters):
        return super().__call__(values=parameters)


def periodic_shape(config, *, objective, selection=None, backend=None):
    """Create a periodic homogenization objective; input is reduced parameters, not V."""
    return PeriodicShapeObject(config, kind='periodic-shape', objective=objective,
                               selection=selection, backend=backend)
