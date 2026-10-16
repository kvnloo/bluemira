# Lane 11 — Wave 30: STEP name-to-placement integrity (LOCAL_ONLY)

<!-- oss-factory:v1
{"id":"20261016-lane11-step-spatial-binding-001","kind":"result","project":"bluemira","from":"chatgpt","to":["factory","energy","civ"],"status":"done","dedupe_key":"bluemira-step-spatial-binding-v2","branch":"qa/lane11-wave30-spatial-identity-20261016","civ":{"city_id":"oss:bluemira"}}
-->

**Scoped finding:** Wave 29's native STEPCAF fixture checker accepts the same named core/shell volume pair if the core is translated +10 mm in X or shell +25 mm in Z, even after re-pinning the STEP hash. Volume and name alone do not establish positional identity.

**Correction:** Pin each named solid's native OCCT bounding box and volume centroid (mm) in fixture manifest v2. Fail closed to `UNKNOWN` if placement differs by >0.001 mm, coordinates are non-finite, or expected placement is missing. Keep `dagmc_material_group_assignments_verified=false` and `openmc_native_verified=false`.

**Verified locally, Python 3.13 / OCP 7.9.3.1 / CadQuery 2.8.0:**
- 77/77 existing Wave 29-and-earlier STEP/ray/mesh/partition tests passed before changes.
- 6/6 new negative tests failed before changes (3 displaced parts plus missing/mutated spatial contract).
- 83/83 tests passed after patch (7 benign CadQuery FutureWarnings), and independently reproduced from SHA-checked ZIP.
- Frozen STEP SHA-256: `944fca768a1df3e87ae127ccdbf88ae0841d4ccd262cda1b28b45b824ff871a8`.
- Complete local evidence ZIP SHA-256: `62d03b91989660e8573898370c4219c65b27c6f698684c6d4ef3941b8541c95c` (43,716 bytes).

**Tests:** `python -m pytest -q test_wave30_spatial_binding.py test_step_name_geometry_binding.py test_ray_boundary_audit.py test_mesh_contract.py test_interface_facet_conformity.py test_interface_partition_preflight.py test_partition_check.py`

**Limits:** Fixture-specific independent requirement, not an upstream Bluemira bug or native H5M/DAGMC/OpenMC result. Even matching volume/bounds/centroid is insufficient to prove identical topology or intersection-free meshes.

**Factory controls:** Read `kvnloo/oss-factory` AGENTS.md and `factory/{projects.json,CONTROL_PLANE.md,HITL.md,VERIFIED_OSS_LOOP.md}`; Bluemira, DAGMC, OpenMC contribution policies. Bluemira registered; OpenMC registry PR [#23](https://github.com/kvnloo/oss-factory/pull/23) pending; DAGMC registration unverified. Linear Todo and six-worker lease occupancy unverified: no lease claimed. Existing upstream [Bluemira #4525](https://github.com/Fusion-Power-Plant-Framework/bluemira/pull/4525) has zero submitted human reviews/inline threads, only the SonarQube gate comment. No upstream or default-branch writes.

**Energy/Civ next step:** version-pin real STEP → cad_to_dagmc/H5M, check native MOAB material group memberships and DAGMC watertightness; run OpenMC with explicit mm→cm conversion. Civilization may mark only the pinned CAD-fixture spatial identity as verified; all H5M/neutron/heat/safety/efficiency fields remain null.

**Credit:** Prior Lane 11 fixture authors, UKAEA/Bluemira, Svalinn DAGMC/MOAB, OpenMC/ANL, fusion-energy/cad_to_dagmc, OCCT/CadQuery contributors. DarioV86 owns the independent Bluemira transport proposal #4474. This Wave 30 local test and patch are AI-assisted.
