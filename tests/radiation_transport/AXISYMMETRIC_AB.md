# Axisymmetric A/B benchmark for #4474

## Purpose

Use Bluemira's existing EU-DEMO single-null charged-particle regression as the
first scientific comparison for the proposed 3-D Monte Carlo transport stack.

This benchmark must distinguish **implementation parity** from **model
difference**. The existing solver and the Nichols et al. Monte Carlo model do
not use the same source model, so a single pointwise heat-flux equality test
would be misleading.

## Canonical physical fixture

Reuse without modification:

- equilibrium: `data/equilibria/EU-DEMO_EOF.json`;
- wall: `tests/radiation_transport/test_data/first_wall.json`;
- `dx_mp = 0.001 m`;
- `P_sep_particle = 100 MW`;
- near/far fractions: `0.5 / 0.5`;
- `lambda_q_near = lambda_q_far = 0.05 m` at the OMP;
- lower target split: LFS `0.75`, HFS `0.25`;
- upper target split: zero.

The legacy regression already requires its integrated wall power to reproduce
`P_sep_particle` within `2%`.

## Stage A0 — field / geometry parity

Goal: prove that the new 3-D stepping/collision stack reproduces the existing
axisymmetric open-field geometry before comparing transport models.

1. Run `ChargedParticleSolver` on the canonical fixture.
2. Extract every OMP start and first-wall endpoint from
   `flux_surfaces_ob_down` and `flux_surfaces_ob_up`.
3. Convert `Equilibrium.Bx/Bz/Bt` to Cartesian B using the axisymmetric
   cylindrical basis.
4. Revolve the same R-Z first wall with `axisymmetric_surface_hit`.
5. For each legacy start, run the seeded walker with `Dm = 0` in both parallel
   directions; keep the direction that reaches the corresponding legacy branch.
6. Compare the hit in `(R,Z)` and in wall arc length.

Acceptance:

- every selected legacy trace produces one wall hit;
- branch identity is preserved;
- endpoint error decreases when `parallel_step` is halved;
- no stochastic or power-distribution assumption participates in this stage.

Initial CI subset: first / middle / last trace from each lower branch. Expand to
all traces only after the small convergence test is stable.

## Stage A1 — legacy-weight power parity

Goal: compare deposition accounting while holding the legacy source model fixed.

Do **not** use the paper's LCFS source yet.

For each legacy OMP bin, use the same resolved flux-tube power element implied
by `_q_par` / `_analyse_SN`:

`dP ∝ 2*pi*q_parallel*(Bp/B)*R*dx_mp`.

Apply the existing `0.75 / 0.25` LFS/HFS target split. Trace those weighted
tubes through the new 3-D axisymmetric geometry with `Dm = 0`.

The legacy solver's explicit mid-plane residual is a separate bucket. Preserve
it as such rather than forcing the random walker to reproduce a workaround it
does not model.

Acceptance:

- resolved target power closes to the same weighted source power;
- LFS/HFS lower-target fractions agree with the imposed `0.75 / 0.25` split;
- legacy mid-plane residual is reported separately;
- component + residual accounting closes to `100 MW`;
- comparison is performed in common poloidal wall arc length, not array index.

## Stage A2 — new-model comparison

Only after A0/A1 pass, switch to the #4474 / Nichols model:

- initialise just inside the LCFS;
- sample poloidal angle with `1/B^2` weighting;
- random toroidal angle;
- random parallel direction;
- near and far SOL runs with their own `Dm`;
- equal macroparticle weight `PSOL/N`.

At this stage differences from `ChargedParticleSolver` are expected. Compare:

- total deposited and unresolved power;
- LFS/HFS and upper/lower region fractions;
- wall-arc deposition centroid and quantiles;
- coarse normalized wall histogram;
- sensitivity to `Dm`, seed and particle count;
- toroidal-angle invariance for the axisymmetric wall.

Do not require noisy pointwise q-perp equality.

## CI / scale policy

CI proves contracts, not production convergence:

- deterministic A0 subset;
- small fixed-seed A1 population;
- one small A2 smoke case after source sampling exists.

Large-N convergence studies and `10^6`-particle runs stay outside normal CI.

## Stop condition before production benchmark code

Do not open the A/B PR until the stacked primitives are reconciled:

1. surface hit (#4);
2. seeded walker (#5);
3. conservation oracle (#6).

The fixture/helper branch may be prepared in advance, but it must not add
another full CI queue until those three gates are understood.
