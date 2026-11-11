# Differentiable PolyFEM examples

Start with the [material beam example](samples/material_and_contact/material_beam.py). It assigns one stiffness variable to each of the beam's 198 elements and prints the compliance before and after one Adam update.

Run it from the repository root:

```powershell
conda run -n polyfem python -m differentiable_example.samples.material_and_contact.material_beam
```

| What would you like to do? | Where to start |
| --- | --- |
| Define a PyTorch loss for a simulated quantity | [Fit a friction coefficient](samples/material_and_contact/friction_matching.py) or [combine two load cases](samples/material_and_contact/material_load_cases.py) |
| Find an example by design variable | [49 samples](samples/), grouped by material, contact, initial conditions, boundary controls, and geometry |
| Run a Python version of a diff-data optimization problem | [18 optimization examples](optimizations/) |
| Study a lattice under impact | [Run the forward simulation](paper_example/impact_forward.py), [change the material](paper_example/material_opt.py), or [change the leg thickness and angle](paper_example/shape_opt.py) |
| Check gradients or compare with the original PolyFEM input | [Validation scripts](../tools/validation/) |
