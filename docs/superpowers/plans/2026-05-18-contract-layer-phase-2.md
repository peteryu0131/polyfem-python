# Contract Layer Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the internal solve contract the shared source of truth for config normalization, mesh source selection, and backend settings used by forward and differentiable solve paths.

**Architecture:** Keep public APIs unchanged. Strengthen `polyfempy/api/_solve_contract.py` with a real `CanonicalSolveInput` preparation helper, make `polyfempy/api/_solve_pipeline.py` consume that prepared contract instead of rebuilding settings, and move differentiable config/settings normalization toward the same contract. Existing compatibility wrappers remain importable while delegating to the contract module.

**Tech Stack:** Python dataclasses, NumPy, existing `SimulationConfig`, PyTorch-facing differentiable helpers, pytest.

---

### Task 1: Make CanonicalSolveInput Real

**Files:**
- Modify: `polyfempy/api/_solve_contract.py`
- Modify: `tests/test_pipeline_normalize.py`

- [ ] **Step 1: Write failing tests**

Add tests that call `prepare_canonical_solve_input(...)` for JSON, direct array, and guided array modes. The tests must assert `canonical.mesh_source.mode`, `canonical.backend_settings`, and `canonical.metadata["mesh_source"]`.

- [ ] **Step 2: Run tests to verify RED**

Run: `/home/peteryu/polyfem_env/bin/python -m pytest tests/test_pipeline_normalize.py -q`

Expected: FAIL because `prepare_canonical_solve_input` is not implemented.

- [ ] **Step 3: Implement helper**

Add `prepare_canonical_solve_input(vertices, cells, cfg, dtype=None)` that calls `normalize_config`, `build_full_json`, `choose_mesh_source`, and `build_canonical_solver_settings`, then returns `CanonicalSolveInput`.

- [ ] **Step 4: Run tests to verify GREEN**

Run: `/home/peteryu/polyfem_env/bin/python -m pytest tests/test_pipeline_normalize.py -q`

Expected: PASS.

### Task 2: Make Forward Pipeline Consume CanonicalSolveInput

**Files:**
- Modify: `polyfempy/api/_solve_pipeline.py`
- Modify: `tests/test_pipeline_normalize.py`
- Modify: `tests/test_pipeline_extract_outputs.py` if `NormalizedInputs` construction changes

- [ ] **Step 1: Write failing test**

Add a test around `configure_solver(...)` using a fake array-mode solver and a guided-array `NormalizedInputs`. Assert the JSON passed to `set_settings(...)` does not contain `__array_body__`.

- [ ] **Step 2: Run test to verify RED**

Run: `/home/peteryu/polyfem_env/bin/python -m pytest tests/test_pipeline_normalize.py -q`

Expected: FAIL before pipeline consumes canonical backend settings directly.

- [ ] **Step 3: Implement pipeline changes**

Update `run_pipeline(...)` to prepare `CanonicalSolveInput` once. Convert `canonical.mesh_source` into `NormalizedInputs`, pass `canonical.backend_settings` into `configure_solver`, and avoid rebuilding backend settings inside array/json configure helpers.

- [ ] **Step 4: Run tests to verify GREEN**

Run: `/home/peteryu/polyfem_env/bin/python -m pytest tests/test_pipeline_normalize.py tests/test_pipeline_extract_outputs.py -q`

Expected: PASS.

### Task 3: Move Differentiable Settings Toward Shared Contract

**Files:**
- Modify: `polyfempy/differentiable/_solve_settings.py`
- Modify: `polyfempy/differentiable/solve_diff.py`
- Modify: `tests/test_differentiable_solve_settings.py`

- [ ] **Step 1: Write failing tests**

Add tests that `_differentiable_config_and_settings(...)` preserves full JSON settings, records runtime patches, and keeps user `SimulationConfig` unchanged. Add a partial array mode test for `solve_differentiable(...)` if it can be tested without backend construction.

- [ ] **Step 2: Run tests to verify RED**

Run: `/home/peteryu/polyfem_env/bin/python -m pytest tests/test_differentiable_solve_settings.py -q`

Expected: FAIL for missing diagnostics/contract behavior.

- [ ] **Step 3: Implement minimal changes**

Use shared `normalize_config`, `build_full_json`, and `build_canonical_solver_settings` inside differentiable settings normalization for config+mesh mode. Keep array-mode solver execution unchanged except for shared partial-input validation and metadata.

- [ ] **Step 4: Run tests to verify GREEN**

Run: `/home/peteryu/polyfem_env/bin/python -m pytest tests/test_differentiable_solve_settings.py -q`

Expected: PASS.

### Task 4: Verification

**Files:**
- No new production files unless previous tasks require them

- [ ] **Step 1: Run focused contract tests**

Run: `/home/peteryu/polyfem_env/bin/python -m pytest tests/test_pipeline_normalize.py tests/test_pipeline_sampled_fallback.py tests/test_differentiable_solve_settings.py tests/test_pipeline_runtime_options.py tests/test_pipeline_clean_json.py tests/test_pipeline_extract_outputs.py tests/test_import_public_api.py -q`

Expected: PASS.

- [ ] **Step 2: Run full tests**

Run: `/home/peteryu/polyfem_env/bin/python -m pytest tests -q`

Expected: PASS.

- [ ] **Step 3: Run py_compile**

Run: `/home/peteryu/polyfem_env/bin/python -m py_compile polyfempy/api/_solve_contract.py polyfempy/api/_solve_pipeline.py polyfempy/differentiable/_solve_settings.py polyfempy/differentiable/solve_diff.py`

Expected: PASS with no output.
