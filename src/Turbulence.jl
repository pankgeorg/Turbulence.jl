module Turbulence

using WaterLily
using StaticArrays

export Smagorinsky, update_νt!

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

Recompute the total effective viscosity in `model.ν` from velocity
field `u`. Cell-centered values only — ghost cells left at `ν₀`.
"""
function update_νt!(s::Smagorinsky, u)
    Cs²Δ² = s.Cs^2 * s.Δ^2
    @inbounds for I in WaterLily.inside(s.ν)
        Sm = WaterLily.S(I, u)
        # Frobenius norm squared: sum of squared entries.
        s² = sum(abs2, Sm)
        s.ν[I] = s.ν₀ + Cs²Δ² * sqrt(2 * s²)
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

end # module
