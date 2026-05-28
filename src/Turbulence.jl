module Turbulence

using WaterLily
using WaterLily: ∂
using StaticArrays
using LinearAlgebra: tr

export Smagorinsky, WALE, update_νt!
export wall_distance, semi_implicit_source!
export SpalartAllmaras, step_sa!
export KOmegaSST, step_sst!

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
function update_νt!(s::Smagorinsky{T}, u) where T
    Cs²Δ² = s.Cs^2 * s.Δ^2
    WaterLily.@loop s.ν[I] = s.ν₀ +
        Cs²Δ² * sqrt(2 * sum(abs2, WaterLily.S(I, u))) over I ∈ WaterLily.inside(s.ν)
    return s.ν
end
function update_νt!(s::Smagorinsky{T}, u, ν₀_field::AbstractArray) where T
    Cs²Δ² = s.Cs^2 * s.Δ^2
    WaterLily.@loop s.ν[I] = ν₀_field[I] +
        Cs²Δ² * sqrt(2 * sum(abs2, WaterLily.S(I, u))) over I ∈ WaterLily.inside(s.ν)
    # Ghost cells of s.ν should reflect the underlying ν₀_field, not the
    # scalar s.ν₀ stored at construction. Without this the wall-bounded
    # _νf face average would mix s.ν₀ ghost with ν₀_field interior at
    # the boundary.
    _copy_ghost!(s.ν, ν₀_field)
    return s.ν
end

"""
    (model::Smagorinsky)(flow, t; ν₀_field=nothing, kwargs...)

`udf`-compatible call: refresh `model.ν` from `flow.u`. Pass
`ν₀_field` (e.g. `vof.ν`) via kwargs to use it as the per-cell
background viscosity instead of the scalar `model.ν₀`.
"""
function (s::Smagorinsky)(flow, t; ν₀_field=nothing, kwargs...)
    ν₀_field === nothing ? update_νt!(s, flow.u) :
                            update_νt!(s, flow.u, ν₀_field)
    return nothing
end

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

"""
    _grad_tensor(Val(D), I, u) -> SMatrix{D,D}

Build the velocity-gradient tensor `gᵢⱼ = ∂uᵢ/∂xⱼ` at cell `I`. Generic
in D (returns a 2×2 in 2D, 3×3 in 3D) so a 1D LES would dispatch
correctly — the error would otherwise be ugly.
"""
@inline _grad_tensor(::Val{D}, I, u) where D =
    SMatrix{D,D}(∂(i, j, I, u) for i in 1:D, j in 1:D)

"""
    update_νt!(model::WALE, u)
    update_νt!(model::WALE, u, ν₀_field::AbstractArray)

Refresh the WALE eddy viscosity from the current velocity field `u`.
Total viscosity `ν₀ + ν_t` is written into `model.ν`. With a per-cell
`ν₀_field`, the eddy contribution is added on top of that field
(VoF + LES wiring).
"""
function update_νt!(w::WALE{T}, u::AbstractArray) where T
    Cw²Δ² = w.Cw^2 * w.Δ^2
    Dim = Val(ndims(u) - 1)
    WaterLily.@loop w.ν[I] = w.ν₀ +
        T(_wale_νt(Dim, I, u, Cw²Δ²)) over I ∈ WaterLily.inside(w.ν)
    return w.ν
end
function update_νt!(w::WALE{T}, u::AbstractArray,
                    ν₀_field::AbstractArray) where T
    Cw²Δ² = w.Cw^2 * w.Δ^2
    Dim = Val(ndims(u) - 1)
    WaterLily.@loop w.ν[I] = ν₀_field[I] +
        T(_wale_νt(Dim, I, u, Cw²Δ²)) over I ∈ WaterLily.inside(w.ν)
    _copy_ghost!(w.ν, ν₀_field)   # see Smagorinsky H2 comment
    return w.ν
end

"""
    _sym(g) -> SMatrix

Symmetric part `(g + gᵀ)/2` of an `SMatrix`. Hand-written to avoid the
`g + g'` form which mixes `SMatrix` + `Adjoint{…,SMatrix}` and (on some
Julia versions) heap-promotes inside the WALE hot path.
"""
@inline _sym(g::SMatrix{D,D,T}) where {D,T} =
    SMatrix{D,D,T}((g[i,j] + g[j,i]) / 2 for i in 1:D, j in 1:D)

"""
    _wale_νt(Val(D), I, u, Cw²Δ²) -> T

Evaluate the WALE eddy-viscosity formula at cell `I`. Builds the local
velocity-gradient tensor, computes `Sᵢⱼ` and `Sᵈᵢⱼ`, and returns
`(Cw·Δ)² · (SdSd)^(3/2) / (SS^(5/2) + SdSd^(5/4))`. Returns `zero(T)`
when the denominator vanishes (uniform flow).
"""
@inline function _wale_νt(Dim::Val{D}, I, u, Cw²Δ²) where D
    g  = _grad_tensor(Dim, I, u)
    Te = eltype(g)
    S  = _sym(g)
    g² = g * g
    Sd = _sym(g²) - (tr(g²) / Te(D)) * I_identity(D, Te)
    SS   = sum(abs2, S)
    SdSd = sum(abs2, Sd)
    denom = SS^Te(2.5) + SdSd^Te(1.25)
    return denom > 0 ? Cw²Δ² * SdSd^Te(1.5) / denom : zero(Te)
end

"""
    I_identity(D, T) -> SMatrix{D,D,T}

D×D identity matrix as an `SMatrix{D,D,T}`. Local to avoid pulling in
`LinearAlgebra.I`, which can lazy-promote inside the WALE hot path.
Supports D ∈ {2, 3}.
"""
@inline I_identity(D::Int, T) = D == 2 ?
        SMatrix{2,2,T}(one(T), zero(T), zero(T), one(T)) :
        SMatrix{3,3,T}(one(T), zero(T), zero(T),
                       zero(T), one(T), zero(T),
                       zero(T), zero(T), one(T))

"""
    (model::WALE)(flow, t; ν₀_field=nothing, kwargs...)

`udf`-compatible call: refresh `model.ν` from `flow.u`. Pass
`ν₀_field` (e.g. `vof.ν`) via kwargs for the per-cell background.
"""
function (w::WALE)(flow, t; ν₀_field=nothing, kwargs...)
    ν₀_field === nothing ? update_νt!(w, flow.u) :
                            update_νt!(w, flow.u, ν₀_field)
    return nothing
end

"""
    _copy_ghost!(dst, src) -> dst

Copy only the one-cell ghost layer of `dst` from `src`. Used after the
per-cell-ν₀_field update to keep ghost values consistent with the
underlying VoF/molecular field rather than the scalar `ν₀` stored at
construction.
"""
function _copy_ghost!(dst::AbstractArray{T,D}, src::AbstractArray) where {T,D}
    sz = size(dst)
    @inbounds for I in CartesianIndices(dst)
        is_ghost = false
        for d in 1:D
            (I[d] == 1 || I[d] == sz[d]) && (is_ghost = true; break)
        end
        is_ghost && (dst[I] = src[I])
    end
    return dst
end

include("rans.jl")

end # module
