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
