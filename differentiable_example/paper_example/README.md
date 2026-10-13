# Lattice impact: forward baseline (stage A)

From the repository root:

```powershell
conda run --no-capture-output -n polyfem python differentiable_example/paper_example/impact_forward.py
```

Requires the compiled backend with Pardiso, NumPy, and meshio for output checks
(all available in the local `polyfem` environment). No training or differentiation
is performed. The supplied meshes and original JSON are left unchanged.

`impact_forward.py` builds both bodies using the generated API and calls
`polyfempy.solve(cfg=config)`. It uses the original cm/g/s units, material values,
zero initial block velocity, gravity, and contact distance. Both meshes contain
physical boundary label 3: the script overrides the block's labels with 4 so
only the lattice bottom is fixed. Reusing the original labels pinned the block
in the first trial, despite a successful solver return.

The baseline uses BDF1, `dt=0.005 s`, `tend=0.03 s` (six updates), rather than the
original JSON's single 0.01 s step. The current Newton API uses
`grad_norm_tol=1e-6` and RobustArmijo; the obsolete JSON solver fields are not
copied. This is a short impact smoke test, not a converged peak-stress study or
a complete rebound simulation.

## Outputs

Under `runs/impact_forward/`:

- `config.json`: configuration produced by the new API, with resolved mesh paths.
- `impact.pvd`: open in ParaView; the Volume block contains reference coordinates,
  displacement, body IDs, von Mises and stress tensors. Apply **Warp By Vector**
  using `displacement`, scale 1, to see deformed geometry. Contact blocks include
  contact-force vectors.
- `impact_stats.json` and `polyfem.log`: backend statistics and solver diagnostics.
- `verification.json`: written only after the complete sequence passes checks:
  finite stress/displacement/contact force, fixed lattice bottom, falling block,
  upward contact force on the block, and positive deformed triangle areas.

Stress values in VTU use g/(cm*s^2), converted to MPa by dividing by `1e7` in the
verification report. Reported maxima are visualization-sampled von Mises values,
not a differentiable objective or a certified global peak.

## Verified local run

With the original 62,768 triangles, the first sampled nonzero block contact force
was at 0.015 s. At 0.03 s:

| Check | Result |
| --- | --- |
| Mean block vertical displacement | -0.384385 cm |
| Lattice bottom displacement | 0 |
| Sampled maximum lattice von Mises stress | 0.483057 MPa |
| Minimum deformed/reference triangle area ratio | 0.984565 |
| Net upward contact force on block | 482480 g*cm/s^2 |

The first two steps match BDF1 free fall; after contact, the block falls less
than the no-contact prediction. Time-step/mesh convergence and a longer event
window are still needed before interpreting stress peaks scientifically.

Run the focused configuration and real-backend regression checks:

```powershell
conda run -n polyfem python -m pytest tests/test_paper_impact_forward.py -q
```

Material gradients, von Mises objective derivatives, smooth-max aggregation,
geometry parameterization, and ML remain outside this stage.
