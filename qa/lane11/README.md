# Lane 11 local evidence recovery (Waves 1–22)

<!-- oss-factory:v1
{"id":"20261015-lane11-recovery-001","kind":"result","project":"bluemira","from":"chatgpt","to":["factory","energy","civ"],"status":"working","branch":"qa/lane11-waves1-22-recovery-20261015","civ":{"city_id":"oss:bluemira"}}
-->

**QA classification: LIMITED / LOCAL_ONLY. This is a fork-only recovery branch.**
This is **not** a production patch, completed CAD→H5M→OpenMC validation, or origin promotion.

All **22/22 original ZIP archives** are now included as bytes in [the lossless compressed tar bundle](archive/lane11_waves01-22_original_archives.tar.xz). The bundle's SHA-256 is `0c1bf83aae510ac5456a0fb0ee1c09b277ce9e695e5ad29e6ce3d961b35e610d`; its Git blob SHA-1 `458794d1eff1a39bb5a9152368bf20f3c544bd9e` matches the independently assembled local file. Their individual SHA-256 digests and sizes are in `WAVE_ARCHIVES.sha256`.

To verify and extract locally:

```sh
python qa/lane11/verify_recovery.py qa/lane11/archive/lane11_waves01-22_original_archives.tar.xz qa/lane11/WAVE_ARCHIVES.sha256
mkdir -p /tmp/lane11-original-waves
tar -xJf qa/lane11/archive/lane11_waves01-22_original_archives.tar.xz -C /tmp/lane11-original-waves
unzip /tmp/lane11-original-waves/lane11_wave22_material_group_identity.zip -d /tmp/lane11-wave22
cd /tmp/lane11-wave22/lane11_wave22/source
python -m pytest -q test_wave22_material_group_category.py test_wave21_assignment_coverage.py test_wave20_raw_material_groups.py test_volume_material_crosswalk_wave19.py test_native_h5m_probe.py test_native_watertight_counts_wave18.py test_h5m_material_handoff.py
```

**Independently rerun on October 15, 2026:** archive integrity 22/22 ZIP SHA-256 + CRC PASS; local Wave 22 tests `61 passed in 17.55s` (Python 3.13.5). Native DAGMC/MOAB and OpenMC were not run. These are retrieved experimental artifacts, not source code integrated into Bluemira.


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
