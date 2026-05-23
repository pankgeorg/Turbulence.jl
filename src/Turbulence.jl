module Turbulence

using WaterLily
using WaterLily: ∂
using StaticArrays
using LinearAlgebra: tr

export Smagorinsky, WALE, update_νt!

"""
    Smagorinsky(grid_size; Cs=0.17, Δ=1, ν₀=0, T=Float32)

Classical algebraic Smagorinsky LES closure (Smagorinsky 1963):

```
ν_t = (Cₛ Δ)² · √(2 SᵢⱼSᵢⱼ)
```

with the rate-of-strain tensor `Sᵢⱼ = (∂ᵢuⱼ + ∂ⱼuᵢ)/2` evaluated at cell
centers via `WaterLily.S(I, u)`. The model carries a cell-centered total
effective-viscosity field

```
ν[I] = ν₀ + ν_t[I]
```

which is what WaterLily's `Flow` uses through PLAN 1 Hook 1
(`Flow(N, uBC; ν=model.ν)`).

# Usage
```julia
model = Smagorinsky(dims; Cs=0.17f0, ν₀=1f-5)
sim   = Simulation(dims, uBC, L; ν=model.ν, body=...)
sim_step!(sim; udf=model)   # model() refreshes ν from sim.flow.u
```

# Fields
- `Cs`  — Smagorinsky constant (0.1–0.2; default 0.17 follows Lilly)
- `Δ`   — filter width in grid units (default 1)
- `ν₀`  — molecular kinematic viscosity (Re = U L / ν₀)
- `ν`   — cell-centered total effective viscosity, shape `grid_size .+ 2`
"""
struct Smagorinsky{T, Nf<:AbstractArray{T}}
    Cs::T
    Δ::T
    ν₀::T
    ν::Nf
end

function Smagorinsky(grid_size::NTuple; Cs::Real=0.17, Δ::Real=1.0,
                     ν₀::Real=0.0, T::Type=Float32, mem=Array)
    ν = fill(T(ν₀), grid_size .+ 2) |> mem
    Smagorinsky{T, typeof(ν)}(T(Cs), T(Δ), T(ν₀), ν)
end

"""
    update_νt!(model::Smagorinsky, u)
    update_νt!(model::Smagorinsky, u, ν₀_field::AbstractArray)

Recompute the total effective viscosity in `model.ν` from velocity
field `u`. Cell-centered values only — ghost cells left at `ν₀`.

If a per-cell `ν₀_field` is supplied (e.g. the per-cell molecular
viscosity `vof.ν = μ/ρ_local` from a VoF.jl simulation), the eddy
contribution is *added* on top of that field rather than the scalar
`model.ν₀`. This is the wiring point for combined LES + VoF.
"""
function update_νt!(s::Smagorinsky, u)
    Cs²Δ² = s.Cs^2 * s.Δ^2
    @inbounds for I in WaterLily.inside(s.ν)
        Sm = WaterLily.S(I, u)
        s² = sum(abs2, Sm)
        s.ν[I] = s.ν₀ + Cs²Δ² * sqrt(2 * s²)
    end
    return s.ν
end
function update_νt!(s::Smagorinsky, u, ν₀_field::AbstractArray)
    Cs²Δ² = s.Cs^2 * s.Δ^2
    @inbounds for I in WaterLily.inside(s.ν)
        Sm = WaterLily.S(I, u)
        s² = sum(abs2, Sm)
        s.ν[I] = ν₀_field[I] + Cs²Δ² * sqrt(2 * s²)
    end
    return s.ν
end

"""
    (model::Smagorinsky)(flow, t; kwargs...)

`udf`-compatible call: refresh `model.ν` from `flow.u`. The updated
viscosity is consumed by the next `conv_diff!` call inside
`mom_step!`.
"""
(s::Smagorinsky)(flow, t; kwargs...) = (update_νt!(s, flow.u); return nothing)

# ----------------------------------------------------------------------------
# WALE (Nicoud & Ducros 1999) — wall-adapting eddy viscosity
# ----------------------------------------------------------------------------

"""
    WALE(grid_size; Cw=0.5, Δ=1, ν₀=0, T=Float32)

Wall-Adapting Local Eddy-viscosity model (Nicoud & Ducros, *Flow Turb.
& Comb.* 62: 183, 1999):

```
ν_t = (Cw Δ)² · (Sᵈᵢⱼ Sᵈᵢⱼ)^(3/2) / [(SᵢⱼSᵢⱼ)^(5/2) + (Sᵈᵢⱼ Sᵈᵢⱼ)^(5/4)]
```

where `gᵢⱼ = ∂uᵢ/∂xⱼ` is the velocity-gradient tensor, `S = (g+gᵀ)/2`
is the strain rate, and `Sᵈ = sym((g²)) - (1/3) I tr(g²)` is the
traceless symmetric part of `g²`. By construction `Sᵈ` vanishes both in
pure shear (∇u rank 2 with zero trace) and at walls, so WALE gives the
correct y³ near-wall scaling for `ν_t` without ad-hoc damping (Van
Driest-style) — the central reason for choosing it over Smagorinsky.

`Cw = 0.5` is the standard value (Nicoud & Ducros 1999); range 0.4–0.55
in literature. Δ = grid spacing (default 1 cell, matching WaterLily's
internal cell-units).

API mirrors `Smagorinsky` for interchangeability:

```julia
model = WALE(dims; Cw=0.5f0, ν₀=1f-5)
sim   = Simulation(dims, uBC, L; ν=model.ν, body=...)
sim_step!(sim; udf=model)
```
"""
struct WALE{T, Nf<:AbstractArray{T}}
    Cw::T
    Δ::T
    ν₀::T
    ν::Nf
end

function WALE(grid_size::NTuple; Cw::Real=0.5, Δ::Real=1.0,
              ν₀::Real=0.0, T::Type=Float32, mem=Array)
    ν = fill(T(ν₀), grid_size .+ 2) |> mem
    WALE{T, typeof(ν)}(T(Cw), T(Δ), T(ν₀), ν)
end

# Build the velocity-gradient tensor at cell centre, for 2D or 3D.
@inline _grad_tensor(::Val{2}, I, u) = @SMatrix [∂(i,j,I,u) for i in 1:2, j in 1:2]
@inline _grad_tensor(::Val{3}, I, u) = @SMatrix [∂(i,j,I,u) for i in 1:3, j in 1:3]

"""
    update_νt!(model::WALE, u)
    update_νt!(model::WALE, u, ν₀_field::AbstractArray)

Refresh the WALE eddy viscosity from the current velocity field `u`.
Total viscosity `ν₀ + ν_t` is written into `model.ν`. With a per-cell
`ν₀_field`, the eddy contribution is added on top of that field
(VoF + LES wiring).
"""
function update_νt!(w::WALE{T}, u::AbstractArray{Tu}) where {T, Tu}
    Cw²Δ² = w.Cw^2 * w.Δ^2
    D = ndims(u) - 1
    Dim = Val(D)
    @inbounds for I in WaterLily.inside(w.ν)
        νt = _wale_νt(Dim, I, u, Cw²Δ², D)
        w.ν[I] = w.ν₀ + T(νt)
    end
    return w.ν
end
function update_νt!(w::WALE{T}, u::AbstractArray{Tu},
                    ν₀_field::AbstractArray) where {T, Tu}
    Cw²Δ² = w.Cw^2 * w.Δ^2
    D = ndims(u) - 1
    Dim = Val(D)
    @inbounds for I in WaterLily.inside(w.ν)
        νt = _wale_νt(Dim, I, u, Cw²Δ², D)
        w.ν[I] = ν₀_field[I] + T(νt)
    end
    return w.ν
end

@inline function _wale_νt(Dim::Val{D}, I, u, Cw²Δ², ::Int) where D
    g  = _grad_tensor(Dim, I, u)
    S  = (g + g') / 2
    g² = g * g
    Sd = (g² + g²') / 2 - (tr(g²) / D) * I_identity(D, eltype(g²))
    SS  = sum(abs2, S)
    SdSd = sum(abs2, Sd)
    denom = SS^(2.5) + SdSd^(1.25)
    return denom > 0 ? Cw²Δ² * SdSd^(1.5) / denom : zero(eltype(g²))
end

# 2D/3D identity helper (avoid LinearAlgebra.I to keep things scalar
# inside @SMatrix arithmetic on the WALE hot path).
@inline I_identity(D::Int, T) = D == 2 ?
        SMatrix{2,2,T}(one(T), zero(T), zero(T), one(T)) :
        SMatrix{3,3,T}(one(T), zero(T), zero(T),
                       zero(T), one(T), zero(T),
                       zero(T), zero(T), one(T))

(w::WALE)(flow, t; kwargs...) = (update_νt!(w, flow.u); return nothing)

end # module
