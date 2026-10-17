# Lattice impact: forward baseline (stage A)

From the repository root:

```powershell
conda run --no-capture-output -n polyfem python differentiable_example/paper_example/impact_forward.py
```

Requires the compiled backend with Pardiso. The separate experiment tests also
use pytest, NumPy, and meshio. No training or differentiation is performed. The script loads the two meshes beside it; contact_impact.json is reference only.

`impact_forward.py` builds both bodies using the generated API and calls
`polyfempy.solve(cfg=config)`. It uses the original cm/g/s units, material values,
zero initial block velocity, gravity, and contact distance. Both meshes contain
physical boundary label 3: the script overrides the block's labels with 4 so
only the lattice bottom is fixed. Reusing the original labels pinned the block
in the first trial, despite a successful solver return.

The example uses BDF1, `dt=0.01 s`, `tend=0.5 s`: 50 updates and 51 output
frames including the initial state. Change `DT` and `END_TIME` in the script;
the generated API writes the actual settings to `runs/impact_forward/config.json`.
The Newton solver uses `grad_norm_tol=1e-6` and RobustArmijo.
Time-step and mesh convergence are still needed for accurate peak stresses.

## Outputs

Under `runs/impact_forward/`:

- `config.json`: configuration produced by the new API, with resolved mesh paths.
- `impact.pvd`: open in ParaView; the Volume block contains reference coordinates,
  displacement, body IDs, von Mises and stress tensors. Apply **Warp By Vector**
  using `displacement`, scale 1, to see deformed geometry. Contact blocks include
  contact-force vectors.
- `impact_stats.json` and `polyfem.log`: backend statistics and solver diagnostics.

Stress values in VTU use g/(cm*s^2), converted to MPa by dividing by `1e7` in the
test verification report. Reported maxima are visualization-sampled von Mises values,
not a differentiable objective or a certified global peak.

## Validation

The meshes were replaced on October 17. Results from the previous mesh are not
validation of these inputs. The real-backend regression uses a separate temporary
output directory and a short `dt=0.005 s`, `tend=0.03 s` sequence to check free fall
before contact and stress after contact. All experiment-specific checks live in
`tests/test_impact_forward.py`, including finite fields, fixed lattice bottom,
contact forces, and positive deformed triangle areas. Tests write `verification.json`
inside their temporary output directory.

The main script only solves and exports results; completion is not physical
validation. It no longer creates or updates `verification.json` in `runs/`.
Any report left there by an older version belongs to that older run.

Run the focused configuration and real-backend regression checks:

```powershell
conda run -n polyfem python -m pytest differentiable_example/paper_example/tests/test_impact_forward.py -q
```

Material gradients, von Mises objective derivatives, smooth-max aggregation,
geometry parameterization, and ML remain outside this stage.
