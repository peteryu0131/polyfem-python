"""
Minimal 2D Linear Elasticity (unit square → two triangles).

Run:
     python polyfempy\examples\run_elasticity.py
"""

import numpy as np
from polyfempy.api import solve, SimulationConfig


def main():
    # mesh: unit square split into 2 triangles
    V = np.array([[0., 0.],
                  [1., 0.],
                  [1., 1.],
                  [0., 1.]], dtype=float)
    C = np.array([[0, 1, 2],
                  [0, 2, 3]], dtype=np.int32)

    # config: linear elasticity, order-1, simple BCs
    cfg = SimulationConfig.linear_elasticity(E=1e6, nu=0.3, order=1)
    cfg.boundary_conditions = {
        "dirichlet_boundary": [{"id": 4, "value": [0.0, 0.0]}],  # left side fixed (assumed id=4)
        "rhs": [1.0, 0.0],  # small horizontal body force
    }
    cfg.extras = {"verbosity": 1}

    # solve
    res = solve(V, C, cfg)

    # quick peek
    u = res.fields.get("u")
    print(f"[ok] fields={list(res.fields)}; u.shape={None if u is None else u.shape}")

    # export (VTK if meshio present, else NPZ fallback)
    res.to_vtk("elasticity_output.vtu")
    print(f"[ok] saved → elasticity_output.vtu")


if __name__ == "__main__":
    main()
