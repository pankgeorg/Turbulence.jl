# ----------------------------------------------------------------------------
# RANS infrastructure shared by the Spalart–Allmaras and k–ω SST closures.
#
# Two pieces the algebraic LES models did not need:
#   1. A cell-centred wall-distance field `d`, derived from the body SDF.
#   2. A positivity-preserving semi-implicit update for transported
#      scalars (ν̃ for SA; k, ω for SST), built on WaterLily's `transport!`.
# ----------------------------------------------------------------------------

using WaterLily: measure_sdf!, AbstractBody, transport!, quick

"""
    wall_distance(body, grid_size; t=0, T=Float32, mem=Array, dfloor=√eps)

Cell-centred wall-distance field `d[I]` for a RANS closure, derived from
the body signed-distance function: `d = max(sdf(body, x), dfloor)`.

In the fluid the body SDF *is* the (positive) distance to the nearest
wall, so no separate Eikonal solve is needed — this is the BDIM analogue
of a wall-distance field. Values inside the body and at the wall itself
are clamped to `dfloor` to keep `1/d` terms in the RANS sources finite.

`grid_size` is the interior size `N` (no ghost layer); the returned
array has size `N .+ 2` to match WaterLily's pressure grid.
"""
function wall_distance(body::AbstractBody, grid_size::NTuple{D};
                       t=0, T::Type=Float32, mem=Array,
                       dfloor::Real=sqrt(eps(float(real(T))))) where D
    d = zeros(T, grid_size .+ 2) |> mem
    measure_sdf!(d, body, T(t))
    fl = T(dfloor)
    WaterLily.@loop d[I] = max(d[I], fl) over I ∈ CartesianIndices(d)
    return d
end

"""
    semi_implicit_source!(φ, adv, P, Dc, dt)

Advance a transported RANS scalar one step with a positivity-preserving
semi-implicit source treatment:

```
φ⁺ = (φ + dt·(adv + P)) / (1 + dt·Dc)
```

where

- `adv[I]`  is the advection–diffusion residual `dφ/dt|transport`
  (from `WaterLily.transport!`),
- `P[I] ≥ 0` is the explicit production rate,
- `Dc[I] ≥ 0` is the *destruction coefficient* — the destruction term
  is `−Dc·φ`, treated implicitly so that `φ⁺ ≥ 0` for any `dt` whenever
  `φ ≥ 0`, `P ≥ 0`, `Dc ≥ 0` (the standard RANS positivity trick).

All arguments are cell-centred arrays of matching shape; the update is
applied over the interior (ghost cells are left to the BC pass). `P`,
`Dc`, and `adv` may alias workspace buffers — `φ` is updated in place
last so the read of `φ` happens before the write.
"""
function semi_implicit_source!(φ::AbstractArray{T}, adv, P, Dc, dt) where T
    dtT = T(dt)
    WaterLily.@loop φ[I] = (φ[I] + dtT*(adv[I] + P[I])) / (one(T) + dtT*Dc[I]) over I ∈ WaterLily.inside(φ)
    return φ
end

# ----------------------------------------------------------------------------
# Spalart–Allmaras one-equation model (Spalart & Allmaras, AIAA 92-0439, 1992).
# Implemented from the paper, not from any OpenFOAM source.
# ----------------------------------------------------------------------------

# Vorticity magnitude |ω| = √(2 Wᵢⱼ Wᵢⱼ), W = antisymmetric part of ∇u.
@inline function _vorticity_mag(::Val{D}, I, u::AbstractArray{T}) where {D,T}
    g = _grad_tensor(Val(D), I, u)
    s = zero(T)
    @inbounds for i in 1:D, j in 1:D
        w = (g[i,j] - g[j,i]) / 2
        s += w*w
    end
    return sqrt(2s)
end

# Central-difference gradient of a cell-centred scalar φ at cell I.
@inline function _grad_scalar(::Val{D}, I, φ::AbstractArray{T}) where {D,T}
    @inbounds SVector{D,T}(ntuple(j -> (φ[I+δ(j,I)] - φ[I-δ(j,I)]) / 2, D))
end

# Strain-rate magnitude S = √(2 Sᵢⱼ Sᵢⱼ), S = symmetric part of ∇u.
@inline function _strain_mag(::Val{D}, I, u::AbstractArray{T}) where {D,T}
    g = _grad_tensor(Val(D), I, u)
    s = zero(T)
    @inbounds for i in 1:D, j in 1:D
        sij = (g[i,j] + g[j,i]) / 2
        s += sij*sij
    end
    return sqrt(2s)
end

# SA viscous function fv1 = χ³/(χ³+cv1³), χ = ν̃/ν.
@inline function _sa_fv1(ν̃::T, νm, cv1³) where T
    χ³ = (max(ν̃, zero(T)) / νm)^3
    return χ³ / (χ³ + cv1³)
end

# SA per-cell source: returns (P, Dc) — explicit production (incl. the
# cb2|∇ν̃|² term) and the implicit destruction coefficient. Packaged in a
# helper so `@loop`'s symbol-grab sees a single call, not loose locals.
@inline function _sa_source(::Val{D}, I, u, ν̃arr::AbstractArray{T}, d,
                            νm, cb1, cb2, σ, κ, cv1³, cw1, cw2, cw3) where {D,T}
    ν̃I = max(ν̃arr[I], zero(T))
    χ  = ν̃I / νm
    fv1 = _sa_fv1(ν̃I, νm, cv1³)
    fv2 = one(T) - χ / (one(T) + χ*fv1)
    Ω  = _vorticity_mag(Val(D), I, u)
    dI = @inbounds d[I]
    κ²d² = κ^2 * dI^2
    S̃ = Ω + ν̃I / κ²d² * fv2
    S̃ = max(S̃, T(0.3)*Ω + eps(T))           # Spalart positivity floor
    r = min(ν̃I / (S̃ * κ²d²), T(10))
    g = r + cw2*(r^6 - r)
    fw = g * ((one(T) + cw3^6) / (g^6 + cw3^6))^(one(T)/6)
    gradν̃ = _grad_scalar(Val(D), I, ν̃arr)
    P  = cb1*S̃*ν̃I + (cb2/σ)*sum(abs2, gradν̃)
    Dc = cw1*fw*ν̃I / (dI^2)
    return P, Dc
end

"""
    SpalartAllmaras(grid_size, body; ν=1e-5, ν̃∞=3ν, T=Float32, mem=Array)

Spalart–Allmaras one-equation eddy-viscosity model. Transports the
modified viscosity `ν̃`; the eddy viscosity is `ν_t = ν̃ fv1`, and the
total effective viscosity `ν = ν_mol + ν_t` is written into the field
passed to `Flow(...; ν=model.ν)`.

`body` supplies the wall-distance field (via [`wall_distance`](@ref)).
`ν̃∞` is the freestream / initial value (3–5 ν for a fully turbulent
state; Spalart & Allmaras use ν̃∞ ≈ 3ν).

Step the model once per outer time step with [`step_sa!`](@ref) *before*
`sim_step!`, so the momentum diffusion sees the updated `ν`.

Constants are the standard set from the 1992 paper: cb1=0.1355,
cb2=0.622, σ=2/3, κ=0.41, cw2=0.3, cw3=2, cv1=7.1, with
cw1 = cb1/κ² + (1+cb2)/σ.
"""
struct SpalartAllmaras{T, A<:AbstractArray{T}}
    ν̃   :: A      # transported modified viscosity (size Ng)
    ν   :: A      # total effective viscosity ν_mol + ν_t (= flow.ν)
    d   :: A      # wall distance
    Φ   :: A      # transport workspace
    radv:: A      # advection–diffusion residual
    P   :: A      # explicit production
    Dc  :: A      # implicit destruction coefficient
    Dd  :: A      # diffusion coefficient (ν+ν̃)/σ
    ν_mol :: T
    ν̃∞  :: T
    perdir :: Tuple
    # constants
    cb1::T; cb2::T; σ::T; κ::T; cv1::T; cw1::T; cw2::T; cw3::T
end

function SpalartAllmaras(grid_size::NTuple{D}, body;
                         ν::Real=1e-5, ν̃∞::Real=3ν, perdir=(),
                         T::Type=Float32, mem=Array) where D
    Ng = grid_size .+ 2
    mk() = zeros(T, Ng) |> mem
    ν̃ = fill(T(ν̃∞), Ng) |> mem
    νf = fill(T(ν), Ng) |> mem
    d  = wall_distance(body, grid_size; T=T, mem=mem)
    κ = T(0.41); cb1 = T(0.1355); cb2 = T(0.622); σ = T(2//3)
    cw1 = cb1/κ^2 + (1+cb2)/σ
    SpalartAllmaras{T,typeof(ν̃)}(ν̃, νf, d, mk(), mk(), mk(), mk(), mk(),
        T(ν), T(ν̃∞), Tuple(perdir),
        cb1, cb2, σ, κ, T(7.1), cw1, T(0.3), T(2))
end

"""
    step_sa!(model::SpalartAllmaras, u, dt)

Advance the SA `ν̃` field one step of size `dt` under velocity `u`, then
refresh `model.ν = ν_mol + ν̃ fv1`. Call once per outer time step,
before `sim_step!`.

Implements the paper's transport equation
`Dν̃/Dt = cb1 S̃ ν̃ − cw1 fw (ν̃/d)² + (1/σ)[∂ⱼ((ν+ν̃)∂ⱼν̃) + cb2 (∂ⱼν̃)²]`
with the conservative advection+diffusion from `transport!`, the
`cb2|∇ν̃|²` term and production lumped as the explicit source, and the
destruction treated implicitly via [`semi_implicit_source!`](@ref).
"""
function step_sa!(m::SpalartAllmaras{T}, u::AbstractArray{T}, dt;
                  wallfn::Bool=false, band=(T(1), T(3))) where T
    D = ndims(u) - 1
    νm, σ, κ, cv1 = m.ν_mol, m.σ, m.κ, m.cv1
    cb1, cb2, cw1, cw2, cw3 = m.cb1, m.cb2, m.cw1, m.cw2, m.cw3
    cv1³ = cv1^3

    # Diffusion coefficient (ν+ν̃)/σ for the conservative transport.
    WaterLily.@loop m.Dd[I] = (νm + max(m.ν̃[I], zero(T))) / σ over I ∈ CartesianIndices(m.Dd)

    # Conservative advection + diffusion residual.
    WaterLily.transport!(m.radv, m.ν̃, u, m.Φ; D_diff=m.Dd, perdir=m.perdir)

    # Source terms (production + cb2-grad explicit; destruction implicit).
    # Plain interior loop: the per-cell SA closure is not yet KA-ported
    # (the momentum/Poisson @loop hot paths dominate cost; profile first).
    Dim = Val(D)
    @inbounds for I in WaterLily.inside(m.ν̃)
        P, Dc = _sa_source(Dim, I, u, m.ν̃, m.d, νm, cb1, cb2, σ, κ, cv1³, cw1, cw2, cw3)
        m.P[I] = P; m.Dc[I] = Dc
    end

    # Positivity-preserving update, then clamp + enforce BCs.
    semi_implicit_source!(m.ν̃, m.radv, m.P, m.Dc, dt)
    WaterLily.@loop m.ν̃[I] = max(m.ν̃[I], zero(T)) over I ∈ WaterLily.inside(m.ν̃)
    isempty(m.perdir) || WaterLily.perBC!(m.ν̃, m.perdir)

    # Refresh effective viscosity ν = ν_mol + ν̃ fv1.
    WaterLily.@loop m.ν[I] = νm + max(m.ν̃[I], zero(T)) * _sa_fv1(m.ν̃[I], νm, cv1³) over I ∈ CartesianIndices(m.ν)
    # Optional BDIM wall function: override ν in the wall band so the
    # diffusive flux carries the Spalding-law wall shear.
    wallfn && apply_wall_function!(m.ν, u, m.d, νm; band=band, perdir=m.perdir)
    return m.ν
end

# ----------------------------------------------------------------------------
# Menter k–ω SST (Menter 1994; Menter, Kuntz & Langtry 2003).
# Implemented from the papers, not from any OpenFOAM source.
# Two transported scalars k, ω with the F1/F2 blending and the ν_t limiter.
# ----------------------------------------------------------------------------

# F1 blending argument and value at a cell (eddy viscosity = a1 k / max(a1 ω, S F2)).
# Returns (F1, F2). d is wall distance, S strain magnitude, νm molecular ν.
@inline function _sst_blend(k::T, ω::T, d, S, νm, βstar, σω2,
                            gradk, gradω) where T
    k = max(k, zero(T)); ω = max(ω, eps(T))
    d² = d*d
    sqrtk = sqrt(k)
    CDkω = max(2*σω2/ω * sum(gradk .* gradω), T(1e-10))
    a1arg = max(sqrtk/(βstar*ω*d), 500*νm/(d²*ω))
    arg1 = min(a1arg, 4*σω2*k/(CDkω*d²))
    F1 = tanh(arg1^4)
    arg2 = max(2*sqrtk/(βstar*ω*d), 500*νm/(d²*ω))
    F2 = tanh(arg2^2)
    return F1, F2, CDkω
end

"""
    KOmegaSST(grid_size, body; ν=1e-5, k∞, ω∞, perdir=(), T=Float64, mem=Array)

Menter k–ω SST two-equation RANS model. Transports turbulent kinetic
energy `k` and specific dissipation `ω`; the eddy viscosity is
`ν_t = a1 k / max(a1 ω, S F2)` and `ν = ν_mol + ν_t` is written into the
field passed to `Flow(...; ν=model.ν)`.

`T` defaults to **Float64** because `ω` spans many orders of magnitude
near the wall (the model's stiffest field).

Step once per outer time step with [`step_sst!`](@ref) before `sim_step!`.

Constants (Menter 2003): a1=0.31, β*=0.09, κ=0.41; inner (k–ω) set
σk1=0.85, σω1=0.5, β1=0.075; outer (k–ε) set σk2=1.0, σω2=0.856,
β2=0.0828; γ = β/β* − σω κ²/√β* per set.
"""
struct KOmegaSST{T, A<:AbstractArray{T}}
    k::A; ω::A; ν::A; d::A
    Φ::A; rk::A; rω::A; Pk::A; Sωe::A; Dck::A; Dcω::A
    Ddk::A; Ddω::A; νt::A; F1::A
    ν_mol::T; k∞::T; ω∞::T; perdir::Tuple
    a1::T; βstar::T; κ::T
    σk1::T; σω1::T; β1::T; γ1::T
    σk2::T; σω2::T; β2::T; γ2::T
end

function KOmegaSST(grid_size::NTuple{D}, body;
                   ν::Real=1e-5, k∞::Real=1e-4, ω∞::Real=1.0, perdir=(),
                   T::Type=Float64, mem=Array) where D
    Ng = grid_size .+ 2
    mk() = zeros(T, Ng) |> mem
    k = fill(T(k∞), Ng) |> mem
    ω = fill(T(ω∞), Ng) |> mem
    νf = fill(T(ν), Ng) |> mem
    d  = wall_distance(body, grid_size; T=T, mem=mem)
    a1=T(0.31); βstar=T(0.09); κ=T(0.41)
    σk1=T(0.85); σω1=T(0.5); β1=T(0.075)
    σk2=T(1.0);  σω2=T(0.856); β2=T(0.0828)
    γ1 = β1/βstar - σω1*κ^2/sqrt(βstar)
    γ2 = β2/βstar - σω2*κ^2/sqrt(βstar)
    KOmegaSST{T,typeof(k)}(k, ω, νf, d,
        mk(),mk(),mk(),mk(),mk(),mk(),mk(),mk(),mk(),mk(),mk(),
        T(ν), T(k∞), T(ω∞), Tuple(perdir),
        a1, βstar, κ, σk1, σω1, β1, γ1, σk2, σω2, β2, γ2)
end

# Per-cell SST closure: returns everything the source assembly needs.
@inline function _sst_cell(::Val{D}, I, u, k_arr::AbstractArray{T}, ω_arr, d_arr,
                           νm, a1, βstar, σω2) where {D,T}
    k = max(k_arr[I], zero(T)); ω = max(ω_arr[I], eps(T))
    S = _strain_mag(Val(D), I, u)
    Ω = _vorticity_mag(Val(D), I, u)      # for the Kato–Launder production option
    gradk = _grad_scalar(Val(D), I, k_arr)
    gradω = _grad_scalar(Val(D), I, ω_arr)
    d = @inbounds d_arr[I]
    F1, F2, CDkω = _sst_blend(k, ω, d, S, νm, βstar, σω2, gradk, gradω)
    νt = a1*k / max(a1*ω, S*F2)
    νt = clamp(νt, zero(T), T(1e5)*νm)
    return (S=S, Ω=Ω, F1=F1, νt=νt, gradk=gradk, gradω=gradω)
end

"""
    step_sst!(model::KOmegaSST, u, dt; wallfn=false, band=(1,3),
              λ=quick, production=:standard)

Advance the SST `k` and `ω` fields one step of size `dt` under velocity
`u`, then refresh `model.ν = ν_mol + ν_t`. Call once per outer step,
before `sim_step!`.

- `λ` — advection limiter for the k/ω transport (`quick`, `vanLeer`, `cds`).
- `production` — turbulent-production form: `:standard` (`P_k = νt·S²`,
  baseline SST) or `:kato_launder` (`P_k = νt·S·Ω`). Kato–Launder
  suppresses spurious production where strain dominates vorticity
  (stagnation/reattachment regions); it is identical to standard in pure
  shear (S = Ω). Opt-in — not baseline SST.
- `wallfn` — apply the Spalding wall function (SA-style; SST has a native
  ω-wall treatment and normally should *not* use this).
"""
function step_sst!(m::KOmegaSST{T}, u::AbstractArray, dt;
                   wallfn::Bool=false, band=(T(1), T(3)), λ=WaterLily.quick,
                   production::Symbol=:standard) where T
    D = ndims(u) - 1
    νm = m.ν_mol; a1=m.a1; βstar=m.βstar; σω2=m.σω2
    Dim = Val(D)

    # 1. Blending, ν_t (lagged), diffusion coefficients. Store F1, νt.
    @inbounds for I in WaterLily.inside(m.k)
        c = _sst_cell(Dim, I, u, m.k, m.ω, m.d, νm, a1, βstar, σω2)
        m.F1[I] = c.F1; m.νt[I] = c.νt
        σk = c.F1*m.σk1 + (1-c.F1)*m.σk2
        σω = c.F1*m.σω1 + (1-c.F1)*m.σω2
        m.Ddk[I] = νm + σk*c.νt
        m.Ddω[I] = νm + σω*c.νt
    end
    WaterLily.perBC!(m.Ddk, m.perdir); WaterLily.perBC!(m.Ddω, m.perdir)

    # 2. Conservative advection + diffusion for each scalar (limiter λ;
    #    less-diffusive λ=vanLeer sharpens free shear layers).
    WaterLily.transport!(m.rk, m.k, u, m.Φ; D_diff=m.Ddk, perdir=m.perdir, λ=λ)
    WaterLily.transport!(m.rω, m.ω, u, m.Φ; D_diff=m.Ddω, perdir=m.perdir, λ=λ)

    # 3. Source terms (production limited; destruction implicit; ω cross-diffusion).
    @inbounds for I in WaterLily.inside(m.k)
        c = _sst_cell(Dim, I, u, m.k, m.ω, m.d, νm, a1, βstar, σω2)
        kI = max(m.k[I], zero(T)); ωI = max(m.ω[I], eps(T))
        β  = c.F1*m.β1 + (1-c.F1)*m.β2
        γ  = c.F1*m.γ1 + (1-c.F1)*m.γ2
        # Production: standard P_k = νt·S² or Kato–Launder P_k = νt·S·Ω
        # (the latter suppresses spurious production where strain ≫
        # vorticity, e.g. the reattachment stagnation region).
        Pk = production === :kato_launder ? c.νt * c.S * c.Ω : c.νt * c.S^2
        Pk = min(Pk, 10*βstar*kI*ωI)                  # production limiter
        # k: production explicit, β*·k·ω destruction implicit (Dc=β*·ω)
        m.Pk[I]  = Pk
        m.Dck[I] = βstar*ωI
        # ω: (γ/νt)Pk + cross-diffusion explicit; β·ω² implicit (Dc=β·ω)
        crossdiff = 2*(1-c.F1)*σω2/ωI * sum(c.gradk .* c.gradω)
        νt_safe = max(c.νt, eps(T))
        m.Sωe[I] = γ/νt_safe * Pk + crossdiff
        m.Dcω[I] = β*ωI
    end

    # 4. Positivity-preserving semi-implicit update.
    semi_implicit_source!(m.k, m.rk, m.Pk, m.Dck, dt)
    semi_implicit_source!(m.ω, m.rω, m.Sωe, m.Dcω, dt)

    # 5. Clamp positivity, enforce wall ω (Menter ω_wall = 60ν/(β1 d₁²)),
    #    periodic BCs.
    ωwall = 60νm / (m.β1 * T(1.0)^2)      # first off-wall cell ≈ 1 cell from wall
    @inbounds for I in WaterLily.inside(m.k)
        m.k[I] = max(m.k[I], zero(T))
        m.ω[I] = max(m.ω[I], eps(T))
        # Strong ω near the wall: where d ≤ 1.5 cells, clamp up to ω_wall.
        m.d[I] ≤ T(1.5) && (m.ω[I] = max(m.ω[I], ωwall))
    end
    isempty(m.perdir) || (WaterLily.perBC!(m.k, m.perdir); WaterLily.perBC!(m.ω, m.perdir))

    # Optional BDIM wall function: set log-layer k, ω in the wall band so
    # the recompute below and the next step's transport stay consistent.
    wallfn && wallfn_kω!(m.k, m.ω, u, m.d, νm, βstar, m.κ; band=band, perdir=m.perdir)

    # 6. Refresh effective viscosity ν = ν_mol + ν_t (recompute with new k,ω).
    @inbounds for I in WaterLily.inside(m.ν)
        c = _sst_cell(Dim, I, u, m.k, m.ω, m.d, νm, a1, βstar, σω2)
        m.ν[I] = νm + c.νt
    end
    # Authoritative momentum-side override of ν in the wall band.
    wallfn && apply_wall_function!(m.ν, u, m.d, νm; band=band, perdir=m.perdir)
    return m.ν
end

# ----------------------------------------------------------------------------
# BDIM wall function (Spalding-law eddy-viscosity override).
#
# The BDIM-smeared wall cannot carry the sublayer gradient, so a RANS
# closure under-predicts the wall shear (log-law constant B downshifts
# like roughness — see ShipFlow.jl/RESULTS-channel-sa.md). Rather than
# resolve the sublayer, we impose the correct wall shear: from the
# off-wall tangential velocity at distance d, solve Spalding's universal
# profile for u_τ, then override ν_t so (ν+ν_t)(u_t/d) = u_τ².
# ----------------------------------------------------------------------------

"""
    spalding_uτ(u_t, d, ν; κ=0.41, B=5.2, itmax=30, tol=1e-8) -> u_τ

Friction velocity from Spalding's law of the wall, given the tangential
velocity `u_t` at wall distance `d` (molecular viscosity `ν`).

Spalding (1961) is a single implicit formula spanning the viscous
sublayer, buffer, and log layers:

```
y⁺ = u⁺ + e^(−κB)[e^(κu⁺) − 1 − κu⁺ − (κu⁺)²/2 − (κu⁺)³/6]
```

With `y⁺ = d·u_τ/ν` and `u⁺ = u_t/u_τ`, substituting gives a single
equation `u⁺·y⁺(u⁺) = Re_d`, `Re_d = d·u_t/ν`, monotone in `u⁺`. Solved
by Newton with an analytic derivative and a regime-aware initial guess
(viscous `√Re_d` for small `Re_d`, log otherwise). Returns 0 for
non-positive `u_t`.
"""
@inline function spalding_uτ(u_t::T, d, νm; κ=T(0.41), B=T(5.2),
                             itmax::Int=30, tol=T(1e-8)) where T
    u_t ≤ zero(T) && return zero(T)
    Re_d = d * u_t / νm
    emκB = exp(-κ*B)
    uplus = Re_d < 100 ? sqrt(Re_d) : (log(Re_d)/κ + B)
    uplus = max(uplus, eps(T))
    for _ in 1:itmax
        ku  = κ*uplus
        ku2 = ku*ku
        Bf  = exp(ku) - 1 - ku - ku2/2 - ku2*ku/6      # Spalding bracket
        Bp  = κ*(exp(ku) - 1 - ku - ku2/2)             # d(Bf)/du⁺
        g   = uplus^2 + uplus*emκB*Bf - Re_d
        gp  = 2*uplus + emκB*(Bf + uplus*Bp)
        Δ   = g/gp
        uplus = max(uplus - Δ, eps(T))
        abs(Δ) < tol*uplus && break
    end
    return u_t / uplus
end

# Cell-centred velocity vector (face → cell average) at I.
@inline function _cell_velocity(::Val{D}, I, u::AbstractArray{T}) where {D,T}
    @inbounds SVector{D,T}(ntuple(i -> (u[I,i] + u[I+δ(i,I),i]) / 2, D))
end

# Inward wall normal n̂ = ∇d/|∇d| (d increases away from the wall).
@inline function _wall_normal(::Val{D}, I, d::AbstractArray{T}) where {D,T}
    n = _grad_scalar(Val(D), I, d)
    nn = sqrt(sum(abs2, n))
    nn < eps(T) ? n : n/nn
end

"""
    apply_wall_function!(ν, u, d, ν_mol; band=(1,3), perdir=())

Override the effective viscosity `ν` in the wall-adjacent band so the
diffusive momentum flux reproduces the Spalding-law wall shear. For each
cell with wall distance `d ∈ band`:

1. tangential velocity `u_t = |u_c − (u_c·n̂)n̂|`, `n̂ = ∇d/|∇d|`;
2. `u_τ = spalding_uτ(u_t, d, ν_mol)`;
3. set `ν_t` from `u_τ` per `mode`:
   - `:flux` (default) — `ν_t = u_τ²·d/u_t − ν_mol`, a flux match
     (`ν·u_t/d = u_τ²`). The `1/u_t` factor is self-correcting: where the
     velocity is deficient it raises `ν_t` to pump the flux back up.
     Empirically the best choice on the channel (SA log-law mean → 7.4%).
   - `:mixing` — the log-layer eddy viscosity `ν_t = κ·u_τ·d`. With the
     local (deficient) `u_t` feeding `u_τ`, it under-mixes and recovers
     little over no wall function — kept for reference.

A cosine taper ramps the override weight from 0 → 1 over the lower
`taper` fraction of the band and back to 0 over the upper fraction, so
the handoff to the model `ν_t` has no kink. The first off-wall cell
under BDIM (d ≈ 0.5) sits in the smeared region and is skipped; the
band default `(1,3)` samples the first *clean* cells. Returns `ν`.
"""
function apply_wall_function!(ν::AbstractArray{T}, u, d, ν_mol;
                              band=(T(1), T(3)), perdir=(),
                              mode::Symbol=:flux, κ::Real=T(0.41),
                              taper::Real=T(0.25)) where T
    D = ndims(u) - 1; Dim = Val(D)
    lo, hi = T(band[1]), T(band[2]); κT = T(κ); tp = T(taper)
    width = max(hi - lo, eps(T)); ramp = tp*width
    @inbounds for I in WaterLily.inside(ν)
        di = d[I]
        (lo ≤ di ≤ hi) || continue
        uc = _cell_velocity(Dim, I, u)
        n̂  = _wall_normal(Dim, I, d)
        un = sum(uc .* n̂)
        u_t = sqrt(max(sum(abs2, uc) - un^2, zero(T)))
        u_t ≤ eps(T) && continue
        uτ = spalding_uτ(u_t, di, ν_mol)
        νt_wf = mode === :flux ? max(uτ^2*di/u_t - ν_mol, zero(T)) : κT*uτ*di
        # Cosine taper weight: 0 at band edges, 1 in the core.
        w = ramp ≤ eps(T) ? one(T) :
            di < lo + ramp ? (one(T) - cos(T(π)*(di-lo)/ramp))/2 :
            di > hi - ramp ? (one(T) - cos(T(π)*(hi-di)/ramp))/2 : one(T)
        νt_model = ν[I] - ν_mol
        ν[I] = ν_mol + (one(T)-w)*νt_model + w*νt_wf
    end
    isempty(perdir) || WaterLily.perBC!(ν, perdir)
    return ν
end

"""
    wallfn_kω!(k, ω, u, d, ν_mol, βstar, κ; band=(1,3), perdir=())

k–ω wall-function companion to [`apply_wall_function!`](@ref): in the
wall band, set the log-layer equilibrium values
`k = u_τ²/√β*` and `ω = u_τ/(√β*·κ·d)` from the Spalding `u_τ`.
"""
function wallfn_kω!(k::AbstractArray{T}, ω, u, d, ν_mol, βstar, κ;
                    band=(T(1), T(3)), perdir=()) where T
    D = ndims(u) - 1; Dim = Val(D)
    lo, hi = T(band[1]), T(band[2]); sβ = sqrt(βstar)
    @inbounds for I in WaterLily.inside(k)
        di = d[I]
        (lo ≤ di ≤ hi) || continue
        uc = _cell_velocity(Dim, I, u)
        n̂  = _wall_normal(Dim, I, d)
        un = sum(uc .* n̂)
        u_t = sqrt(max(sum(abs2, uc) - un^2, zero(T)))
        u_t ≤ eps(T) && continue
        uτ = spalding_uτ(u_t, di, ν_mol)
        k[I] = uτ^2 / sβ
        ω[I] = uτ / (sβ * κ * di)
    end
    if !isempty(perdir)
        WaterLily.perBC!(k, perdir); WaterLily.perBC!(ω, perdir)
    end
    return k, ω
end
