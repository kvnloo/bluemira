# Lane 11 local evidence recovery (Waves 1–22)

<!-- oss-factory:v1
{"id":"20261015-lane11-recovery-001","kind":"result","project":"bluemira","from":"chatgpt","to":["factory","energy","civ"],"status":"working","branch":"qa/lane11-waves1-22-recovery-20261015","civ":{"city_id":"oss:bluemira"}}
-->

**QA classification: LIMITED / LOCAL_ONLY. This is a fork-only recovery branch.**
This is **not** a production patch, completed CAD→H5M→OpenMC validation, or origin promotion.

The original ChatGPT Lane 11 conversation retains all 22 ZIP packages. Exact ZIP SHA-256 digests and sizes are in `WAVE_ARCHIVES.sha256`; the original ZIP bytes are **not** included merely by listing them. Do not treat a manifest row as successful source publication. Local Wave 22 was independently rerun on October 15, 2026 (Python 3.13.5): `61 passed in 17.55s` using `pytest -q` on seven included test modules; native DAGMC/MOAB and OpenMC were not run.

## Workstream provenance
1. Waves 1–3: analytic toroidal-chord/STEP geometry and tessellation fixtures.
2. Waves 4–12: standalone Bluemira FastCTD cache correctness, material provenance, caller invalidation and concurrency patches; only mocked backends and source excerpts.
3. Waves 13–18: H5M preflight checks for units, float dtype, underflow, streaming, watertight log parsing.
4. Waves 19–22: raw material group identity, membership coverage, H5M volume/material crosswalk and accompanying regression tests.

**Risk:** Several source excerpts/candidate patches were successively superseded; do not concatenate them or apply directly to upstream Bluemira. Validate each on the correct parent and current production source. Validate OpenMC/DAGMC input semantics against actual geometry. Report physical energy, heat flux and safety metrics as unverified/null.

## Factory gates
Bluemira is registered under `oss:bluemira`. OpenMC registration is pending [oss-factory #23](https://github.com/kvnloo/oss-factory/pull/23); DAGMC project registration incomplete. Global six-worker lease occupancy and Linear origin approval have not been independently confirmed. This branch grants neither a worker lease nor upstream publication authority. No default branch is being merged.

## Original author credit
Existing Bluemira/UKAEA, Svalinn DAGMC/MOAB, OpenMC, and fusion-energy H5M-inspector contributors own their respective source and standards. DarioV86 proposed Bluemira 3D Monte Carlo field-line transport in [#4474](https://github.com/Fusion-Power-Plant-Framework/bluemira/issues/4474). Prior downstream fork tests and patches retain their authorship. These local candidate tests and recovery notes were AI-assisted.

## Next gate
Run pinned, authentic CAD/STEP → Bluemira FastCTD → DAGMC H5M validation (native `check_watertight`), verify Group/CATEGORY/NAME assignments and material-volume mapping, then open the H5M using actual OpenMC with explicit unit conversion. Only then reconsider source patches and Energy/Civilization conclusions.
