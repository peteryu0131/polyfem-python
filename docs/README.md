# Documentation Index

Start here when reviewing the repository:

## Reviewer Quick Path

- `REVIEWER_QUICKSTART.md`: first page for reviewer/advisor smoke checks,
  expected outputs, dependency skips, and reading order.
- `TEACHER_REVIEW_GUIDE.md`: recommended reading order for advisor/reviewer
  inspection.
- `TOMS_REVIEW_CHECKLIST.md`: reviewer-style checklist for API clarity,
  software contract, examples, tests, and reproducibility.
- `ARTIFACT_REPRODUCIBILITY.md`: shortest validation path for the software
  artifact.
- `TEST_MATRIX.md`: cleanup-slice to test-subset matrix for future changes.

## Full Index

- `REVIEWER_QUICKSTART.md`: direct command path for checking public imports,
  config/result contracts, backend smoke, examples, and paper-demo boundaries.
- `TEACHER_REVIEW_GUIDE.md`: advisor-facing map of the clean API demos, repo
  layout, and what is or is not hidden behind each helper.
- `API_FUNCTION_MAP.md`: function-by-function call map for the public helpers,
  including what each helper calls internally and which pieces are shared.
- `API_STABILITY.md`: Phase 3 stability contract for stable public API,
  advanced/compatibility names, and internal-only modules.
- `GUIDED_API.md`: guided configuration contract for `polyfempy.api.guided`.
- `CONFIG_CONTRACT.md`: `SimulationConfig` contract, including full/minimal
  JSON semantics and `solve(cfg=...)` input forms.
- `RESULT_CONTRACT.md`: `Result` field namespaces, strict field lookup,
  history, sampled-data, and mesh I/O semantics.
- `DIFFERENTIABLE_CONTRACT.md`: Phase 4 differentiable / optimization API
  contract, including `OptimizationRunResult`.
- `EXAMPLES_MATRIX.md`: examples-to-capability matrix for TOMS-style software
  evaluation.
- `TOMS_REVIEW_CHECKLIST.md`: reviewer-style checklist for API clarity,
  software contract, examples, tests, and reproducibility.
- `ARTIFACT_REPRODUCIBILITY.md`: shortest artifact validation path, including
  no-backend contract checks, backend smoke, and example smoke commands.
- `TEST_MATRIX.md`: cleanup-slice to test-subset matrix for safe future API
  changes.
- `API_CLEANUP_PHASE2_PLAN.md`: detailed Chinese plan for deciding the public
  API surface and safely sequencing the next cleanup phase.
- `API_PUBLIC_SURFACE_DECISION.md`: concrete Phase 2 decision on recommended
  public imports, compatibility exports, and internal-only modules.
- `API_INTERNAL_IMPORT_AUDIT.md`: current import audit showing which examples,
  tests, package modules, and paper demos depend on each API layer.
- `API_CLEANUP_STATUS.md`: engineering status note for the API cleanup work,
  including current stable API paths and remaining cleanup risks.
- `../polyfempy/README.md`: package-level map of the recommended imports and
  implementation layers.

For executable paper-facing demos, see:

- `experiment/paper_experiment/README.md`
- `experiment/paper_experiment/CLEAN_API_WALKTHROUGH.md`
- `experiment/paper_experiment/08_h_theta_shape_optimization.py`
