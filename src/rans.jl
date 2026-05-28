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
function step_sa!(m::SpalartAllmaras{T}, u::AbstractArray{T}, dt) where T
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
    return m.ν
end
