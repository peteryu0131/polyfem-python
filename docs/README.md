# Documentation Index

Start here when reviewing the repository:

- `TEACHER_REVIEW_GUIDE.md`: advisor-facing map of the clean API demos, repo
  layout, and what is or is not hidden behind each helper.
- `API_FUNCTION_MAP.md`: function-by-function call map for the public helpers,
  including what each helper calls internally and which pieces are shared.
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
