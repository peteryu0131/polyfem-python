import numpy as np


class Result:
    """Minimal VarForm forward-solve result.

    ``sol`` is the raw solution returned by ``polyfem::State::solve``. It is
    not assumed to be aligned with mesh vertices or sampled visualization data.
    """

    def __init__(self, sol, *, meta=None):
        self.sol = np.ascontiguousarray(np.asarray(sol))
        self.meta = {} if meta is None else dict(meta)
