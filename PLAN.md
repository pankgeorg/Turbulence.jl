# PLAN 2 — Turbulence.jl

**Repo:** `pankgeorg/Turbulence.jl` (private).
**Depends on:** WaterLily upstream-hooks PR (PLAN 1) for effective viscosity
and scalar transport. Until that lands, develop against
`pankgeorg/WaterLily.jl#plan/upstream-hooks` directly.

## Scope

Eddy-viscosity turbulence closures for WaterLily, suitable for ship-scale
Reynolds numbers. Four models in dependency order:

1. **Smagorinsky LES** — algebraic, no extra fields. Single kernel.
2. **WALE LES** — algebraic, near-wall-correct. Single kernel.
3. **Spalart–Allmaras RANS** — one transport equation for `ν̃`.
4. **k–ω SST RANS** — two transport equations + blending function.

Each model is a function from the velocity field to a cell-centered
eddy-viscosity field `ν_t[I]` written into `flow.ν` (the upstream array-`ν`
hook). RANS models additionally transport scalar fields via the upstream
`transport!` helper.

## Non-goals

- **No Reynolds-stress models** (LRR, SSG). Marginal benefit, large code.
- **No DES / hybrid LES-RANS** for v0.1. Adds the most complexity per QoI
  improvement. Revisit once RANS + LES both work.
- **No compressibility / variable density.** Single-phase incompressible
  only; the VoF coupling lives in VoF.jl.

## API

```julia
using WaterLily, Turbulence

# Smagorinsky / WALE: algebraic, no extra state
flow_ctor = (dims, uBC; kw...) -> Turbulence.eddy_viscosity(
    Flow(dims, uBC; kw...);
    model = Smagorinsky(C_s = 0.17),
    Δ = nothing,    # filter width, defaults to local grid spacing
    wall = :sdf,    # use flow's SDF for wall damping
)

# Spalart–Allmaras: ν̃ transported as a sibling scalar field
flow_ctor = (dims, uBC; kw...) -> Turbulence.eddy_viscosity(
    Flow(dims, uBC; kw...);
    model = SpalartAllmaras(),
)

# k–ω SST: k, ω transported
flow_ctor = (dims, uBC; kw...) -> Turbulence.eddy_viscosity(
    Flow(dims, uBC; kw...);
    model = KOmegaSST(),
    k_BC = 1e-4, ω_BC = 1.0,
)

Simulation((nx, ny, nz), (U,0,0), L; ν, body, flow_ctor)
```

`Turbulence.eddy_viscosity` returns an `AbstractFlow` subtype with extra
fields (`ν_t`, plus scalars for transport models) and a `mom_step!`
override that updates `ν_t` after the corrector step.

## Algorithms (primary references)

| Model         | Reference                                                  |
|---------------|------------------------------------------------------------|
| Smagorinsky   | Smagorinsky, *Mon. Weather Rev.* 91 (1963)                 |
| WALE          | Nicoud & Ducros, *Flow Turb. Comb.* 62 (1999)              |
| SA            | Spalart & Allmaras, AIAA 92-0439 (1992)                    |
| k–ω SST       | Menter, AIAA Journal 32(8) (1994); Menter+Kuntz+Langtry 2003 |

All four are textbook (Pope, *Turbulent Flows*, 2000; Wilcox, *Turbulence
Modeling for CFD*, 2006). **Implement from the papers, not from OpenFOAM
source.**

## Validation

Three layers, fastest to slowest:

### Layer 1 — analytic / DNS data (fast, per-PR CI)

- **Decay of isotropic turbulence (Comte-Bellot–Corrsin 1971 data).** Cube,
  periodic BCs, seeded with synthetic field matching CBC spectrum. Check
  energy spectrum and decay rate at three time stations. Pass: spectral
  slope within ±10% of `k^(-5/3)` in inertial range; integral length scale
  within ±15%.
- **Decay of turbulent kinetic energy in HIT.** k(t) for k–ω SST should
  match `k₀ (1 + t/τ)^(-n)` with n ≈ 1.1.
- **Smagorinsky on Taylor–Green vortex.** At low Re the closure should
  remain inactive (`ν_t ≪ ν`). Pass: max(`ν_t/ν`) < 0.01 at Re=100.

### Layer 2 — OpenFOAM tutorial reproduction (nightly CI, Docker)

Target: `OpenFOAM/tutorials/incompressibleFluid/channel395`
(turbulent channel at Re_τ = 395, the Moser–Kim–Mansour benchmark).

Procedure:

1. Run the OpenFOAM tutorial in a Docker container (image
   `opencfd/openfoam-default:2406` or equivalent — see ShipFlow.jl
   harness). Use the `kOmegaSST` and `Smagorinsky` configs.
2. Sample mean velocity and Reynolds stresses at 16 wall-normal stations
   via `postProcess -func sample`.
3. Run the same case in WaterLily with the Turbulence.jl model and
   periodic streamwise/spanwise BCs. Use `JULIA_NUM_THREADS=auto`.
4. Compare to (a) OpenFOAM output (b) the Moser–Kim–Mansour DNS data
   (publicly hosted at https://turbulence.oden.utexas.edu/).

Pass criteria:

| Quantity                       | vs OpenFOAM   | vs DNS        |
|--------------------------------|---------------|---------------|
| u⁺(y⁺) in log layer            | ±5%           | ±10%          |
| u_rms peak location y⁺         | ±2 wall units | ±3 wall units |
| Friction Reynolds number Re_τ  | ±5%           | ±10%          |
| Wall shear stress              | ±5%           | ±10%          |

### Layer 3 — backward-facing step (release-blocking only)

Target: `OpenFOAM/tutorials/incompressibleFluid/pitzDaily` (the
Driver–Seegmiller backstep, well-instrumented).

Pass: reattachment length within ±10% of OpenFOAM result for the same
model. This catches separation/reattachment bugs that channel flow can't.

## Performance budget

Baseline: WaterLily on a 256³ Taylor–Green run, 100 steps.

| Model        | Cost vs baseline (per step) |
|--------------|-----------------------------|
| Smagorinsky  | +20%                        |
| WALE         | +25%                        |
| SA           | +50% (one extra transport)  |
| k–ω SST      | +90% (two transports + blend) |

If exceeded, investigate before merging. Don't optimize prematurely; first
build, then profile against OpenFOAM (which is the *fairer* baseline since
it does the same physics).

## Harness

- `test/runtests.jl`: Layer 1 tests only. Runs on every push, < 5 min.
- `test/openfoam/`: Layer 2 + 3. Triggered by manual workflow dispatch
  and nightly via the ShipFlow.jl harness (which holds the Docker
  orchestration — see MASTER_PLAN.md).
- Reference DNS data committed to a separate `cerulean-reference-data`
  Git LFS repo to keep this repo small. Schema documented in
  ShipFlow.jl.

## Risks & open questions

- **Wall treatment with BDIM.** Standard wall functions assume y⁺ is
  computable from a wall-normal direction. BDIM doesn't have a sharp
  wall; the SDF gives a smoothed distance. First-cut: use the SDF as y
  directly, apply damping in a band of width 5–10 cells inside the body.
  This is novel — there is no OpenFOAM equivalent to copy.
- **`ν_t` at faces vs cells.** k–ω SST's diffusion term wants
  face-interpolated `ν_t`; Smagorinsky doesn't care. Decision deferred
  to PLAN 1 (Hook 1 open questions).
- **Wall functions for k–ω SST.** Menter's automatic wall function
  switches between viscous-sublayer and log-law forms. Implement once,
  use everywhere.
- **Float32 in RANS.** `ω` ranges over many orders of magnitude. May
  force Float64 for the `ω` field while keeping velocity in Float32.

## Milestones

| # | Goal                                              | Done when                                              | Status |
|---|---------------------------------------------------|--------------------------------------------------------|--------|
| 1 | Smagorinsky working on Taylor–Green               | Layer 1 test passes                                    | ✅     |
| 2 | Smagorinsky + WALE pass channel395 vs OpenFOAM    | Layer 2 ±5% on u⁺                                      | Smag ✅ (RMS 0.028), WALE implemented + L1 tests; channel395 WALE run pending |
| 3 | SA implemented + channel395 RANS check            | Layer 2 ±5%                                            | ⛔     |
| 4 | k–ω SST implemented + channel395 + backstep       | Layer 2 + Layer 3 pass                                 | ⛔     |
| 5 | Public release (rename if needed, drop "private") | docs + version 0.1.0                                   | ⛔     |
