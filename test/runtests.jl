using Test
using Turbulence
using Turbulence: update_νt!
using WaterLily
using StaticArrays

@testset "Turbulence" begin

    @testset "Smagorinsky: still water → zero ν_t" begin
        dims = (16, 16, 16)
        model = Smagorinsky(dims; Cs=0.17f0, ν₀=1f-5)
        u = zeros(Float32, (dims .+ 2)..., 3)   # quiescent
        update_νt!(model, u)
        # All interior cells should be at the molecular value.
        for I in WaterLily.inside(model.ν)
            @test model.ν[I] ≈ model.ν₀
        end
    end

    @testset "Smagorinsky: uniform shear → known ν_t" begin
        # Set up a linear shear u_x(y) = γ * y  with constant strain rate γ/2.
        # Then |S|² = 2 (S_12)² = 2*(γ/2)² = γ²/2, so √(2|S|²) = γ.
        # Therefore ν_t = (Cs Δ)² * γ.
        dims = (32, 32, 8)
        γ = 0.1f0
        u = zeros(Float32, (dims .+ 2)..., 3)
        # Apply u_x = γ * y at face locations.  u[I, 1] sits at face x = I_x - 1/2,
        # cell-center y = I_y - 1.5 → use that for shear.
        for I in CartesianIndices((1:dims[1]+2, 1:dims[2]+2, 1:dims[3]+2))
            y = I.I[2] - 1.5f0
            u[I, 1] = γ * y
        end

        Cs = 0.17f0; Δ = 1f0
        model = Smagorinsky(dims; Cs=Cs, Δ=Δ, ν₀=0f0)
        update_νt!(model, u)

        # Interior cells (away from the boundary) should have ν_t = (Cs Δ)² * γ
        νt_expected = Cs^2 * Δ^2 * γ
        # Pick cells well clear of any edge effects
        for I in CartesianIndices((6:dims[1]-4, 6:dims[2]-4, 3:dims[3]-1))
            @test isapprox(model.ν[I], νt_expected; rtol=1e-3)
        end
    end

    @testset "Smagorinsky: doubles when shear doubles" begin
        # ν_t scales linearly with |S|, so doubling γ should double ν_t.
        dims = (16, 16, 8)
        u_lo = zeros(Float32, (dims .+ 2)..., 3)
        u_hi = zeros(Float32, (dims .+ 2)..., 3)
        γ = 0.05f0
        for I in CartesianIndices((1:dims[1]+2, 1:dims[2]+2, 1:dims[3]+2))
            y = I.I[2] - 1.5f0
            u_lo[I, 1] = γ   * y
            u_hi[I, 1] = 2γ  * y
        end
        m_lo = Smagorinsky(dims; ν₀=0f0)
        m_hi = Smagorinsky(dims; ν₀=0f0)
        update_νt!(m_lo, u_lo)
        update_νt!(m_hi, u_hi)
        # Pick a deeply-interior cell.
        I = CartesianIndex(8, 8, 4)
        @test isapprox(m_hi.ν[I] / m_lo.ν[I], 2.0f0; rtol=1e-3)
    end

    @testset "Smagorinsky wiring through Flow + Hook 1" begin
        # End-to-end: build a Flow with the model's ν array, step it,
        # and verify that ν is non-trivially updated by the udf.
        dims = (32, 32, 8)
        uBC = (1f0, 0f0, 0f0)
        model = Smagorinsky(dims; Cs=0.17f0, ν₀=1f-4)
        flow = WaterLily.Flow(dims, uBC; T=Float32, ν=model.ν)

        # Inject a non-trivial velocity gradient
        for I in CartesianIndices(flow.u)
            y = I.I[2] - 1.5f0
            if I.I[end] == 1
                flow.u[I] = 0.05f0 * y
            end
        end

        ν_before = copy(model.ν)
        model(flow, 0.0)   # udf-style call
        ν_after  = model.ν

        # The model must touch the ν array.
        @test ν_after !== nothing
        @test maximum(ν_after) > maximum(ν_before)
    end

    @testset "WALE: still water → zero ν_t" begin
        dims = (16, 16, 16)
        model = WALE(dims; Cw=0.5f0, ν₀=1f-5)
        u = zeros(Float32, (dims .+ 2)..., 3)
        update_νt!(model, u)
        for I in WaterLily.inside(model.ν)
            @test model.ν[I] ≈ model.ν₀
        end
    end

    @testset "WALE: pure shear → near-zero ν_t" begin
        # WALE's defining property: in pure shear (Sᵈ = 0 by construction)
        # the eddy viscosity is exactly zero — the wall-asymptotic feature.
        # In a discrete grid we expect WALE ≪ Smagorinsky on a linear
        # shear profile.
        dims = (32, 32, 8)
        γ = 0.1f0
        u = zeros(Float32, (dims .+ 2)..., 3)
        for I in CartesianIndices((1:dims[1]+2, 1:dims[2]+2, 1:dims[3]+2))
            y = I.I[2] - 1.5f0
            u[I, 1] = γ * y
        end

        wale = WALE(dims; Cw=0.5f0, ν₀=0f0)
        smag = Smagorinsky(dims; Cs=0.17f0, ν₀=0f0)
        update_νt!(wale, u)
        update_νt!(smag, u)

        I = CartesianIndex(16, 16, 4)
        @test smag.ν[I] > 1e-5
        # WALE in exact pure shear is 0; we allow some slack for the
        # discrete stencil but expect ≪ Smagorinsky.
        @test wale.ν[I] < 0.01 * smag.ν[I]
    end

    @testset "WALE wiring through Flow + udf" begin
        dims = (32, 32, 8)
        uBC = (1f0, 0f0, 0f0)
        model = WALE(dims; Cw=0.5f0, ν₀=1f-4)
        flow = WaterLily.Flow(dims, uBC; T=Float32, ν=model.ν)
        for I in CartesianIndices(flow.u)
            y = I.I[2] - 1.5f0
            if I.I[end] == 1
                flow.u[I] = 0.05f0 * y
            end
        end
        model(flow, 0.0)
        for I in WaterLily.inside(model.ν)
            @test model.ν[I] >= model.ν₀ - 1f-7
        end
    end

end
