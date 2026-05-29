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

    @testset "Smagorinsky + per-cell ν₀ (VoF integration path)" begin
        # Background per-cell ν₀: half cells ν₀=1e-3 (water-ish),
        # half ν₀=1e-5 (air-ish).  After update_νt!(...; ν₀_field),
        # the eddy contribution should be added on top of each.
        dims = (16, 16, 8)
        γ = 0.1f0
        u = zeros(Float32, (dims .+ 2)..., 3)
        for I in CartesianIndices((1:dims[1]+2, 1:dims[2]+2, 1:dims[3]+2))
            u[I, 1] = γ * (I.I[2] - 1.5f0)
        end
        ν₀_field = fill(1f-3, dims .+ 2)
        ν₀_field[:, 1:(dims[2]÷2+1), :] .= 1f-5     # bottom half "air"
        model = Smagorinsky(dims; Cs=0.17f0, ν₀=0f0)
        update_νt!(model, u, ν₀_field)
        # Eddy contribution is the same everywhere (uniform shear).
        # So ν[I] - ν₀_field[I] should be ~ Cs²·γ for all I.
        νt_uniform = 0.17f0^2 * γ
        I_air = CartesianIndex(8, 4, 4)
        I_water = CartesianIndex(8, 14, 4)
        @test isapprox(model.ν[I_air]   - ν₀_field[I_air],   νt_uniform; rtol=1e-3)
        @test isapprox(model.ν[I_water] - ν₀_field[I_water], νt_uniform; rtol=1e-3)
        @test model.ν[I_water] > model.ν[I_air]    # water side has larger ν
    end

    @testset "WALE fires on 3D Taylor-Green-like field" begin
        # The pure-shear and rigid-rotation tests both have Sᵈ=0
        # (rank-2 trace-free), so they only exercise the denominator.
        # A 3D Taylor-Green flow has a genuine non-zero Sᵈ everywhere
        # except at symmetry planes — WALE should produce ν_t > 0 in
        # the bulk.
        dims = (16, 16, 16)
        u = zeros(Float32, (dims .+ 2)..., 3)
        kx = 2π / dims[1]; ky = 2π / dims[2]; kz = 2π / dims[3]
        for I in CartesianIndices(u)
            if I.I[end] > 3; continue; end
            x = (I.I[1] - 1.5f0)
            y = (I.I[2] - 1.5f0)
            z = (I.I[3] - 1.5f0)
            if I.I[end] == 1
                u[I] =  sin(kx*x) * cos(ky*y) * cos(kz*z)
            elseif I.I[end] == 2
                u[I] = -cos(kx*x) * sin(ky*y) * cos(kz*z)
            elseif I.I[end] == 3
                u[I] = 0f0
            end
        end
        model = WALE(dims; Cw=0.5f0, ν₀=0f0)
        update_νt!(model, u)
        # ν_t should be strictly positive somewhere in the interior.
        νt_max = 0f0
        for I in WaterLily.inside(model.ν)
            νt_max = max(νt_max, model.ν[I])
        end
        @test νt_max > 1f-4
        # Far from a symmetry plane (interior bulk) the typical ν_t
        # should be at least ~0.001.
        νt_mid = model.ν[CartesianIndex(8, 8, 8)]
        @test νt_mid > 1f-4
    end

    @testset "WALE + per-cell ν₀" begin
        dims = (16, 16, 8)
        u = zeros(Float32, (dims .+ 2)..., 3)
        for I in CartesianIndices((1:dims[1]+2, 1:dims[2]+2, 1:dims[3]+2))
            x = (I.I[1] - 1.5f0) - 8
            y = (I.I[2] - 1.5f0) - 8
            u[I, 1] = -0.05f0 * y
            u[I, 2] = +0.05f0 * x
        end
        ν₀_field = fill(0.5f0, dims .+ 2)
        model = WALE(dims; Cw=0.5f0, ν₀=0f0)
        update_νt!(model, u, ν₀_field)
        # Eddy contribution = same as without per-cell ν₀ but starting at 0.5
        model_ref = WALE(dims; Cw=0.5f0, ν₀=0f0)
        update_νt!(model_ref, u)
        for I in WaterLily.inside(model.ν)
            @test isapprox(model.ν[I] - ν₀_field[I], model_ref.ν[I];
                           atol=1f-6)
        end
    end

    # ───────────────────────── RANS infrastructure ─────────────────────────

    @testset "wall_distance from channel SDF" begin
        # Two-plate channel: sdf(x) = min(y, N_Y - y), positive inside.
        N = (8, 16, 8); N_Y = N[2]
        body = WaterLily.AutoBody((x, t) -> min(x[2], N_Y - x[2]))
        d = wall_distance(body, N; T=Float64)
        @test size(d) == N .+ 2
        # Cell-centre y for interior cell j is loc(0, I)[2] = (j - 1.5).
        # Wall distance there is min(y, N_Y - y), clamped to ≥ floor.
        for j in 2:N[2]+1
            y = (j - 1.5)
            expected = max(min(y, N_Y - y), sqrt(eps(Float64)))
            I = CartesianIndex(4, j, 4)
            @test isapprox(d[I], expected; atol=1e-6)
        end
        # Distances are strictly positive everywhere (no division blow-up).
        @test all(>(0), d)
    end

    @testset "transport! advects a blob at the flow speed" begin
        # Manufactured solution: a Gaussian scalar in a uniform periodic
        # stream. transport! gives dφ/dt; forward-Euler the field and
        # check (a) total ∑φ is conserved, (b) the centroid translates
        # at the imposed velocity.
        N = (64, 64); Ng = N .+ 2
        U = 0.5                      # uniform x-velocity
        φ = zeros(Float64, Ng); Φ = zeros(Float64, Ng)
        r = zeros(Float64, Ng)
        u = zeros(Float64, Ng..., 2); u[:, :, 1] .= U
        x0 = 24.0; σ = 5.0
        for I in CartesianIndices(φ)
            xc = I.I[1] - 1.5
            φ[I] = exp(-((xc - x0)^2) / (2σ^2))
        end
        centroid(f) = sum(I -> (I.I[1]-1.5)*f[I], CartesianIndices(f)) / sum(f)
        m0 = sum(@view φ[2:end-1, 2:end-1]); c0 = centroid(φ)
        dt = 0.2; nsteps = 40
        for _ in 1:nsteps
            WaterLily.transport!(r, φ, u, Φ; perdir=(1, 2))
            @. φ += dt * r
            WaterLily.perBC!(φ, (1, 2))
        end
        m1 = sum(@view φ[2:end-1, 2:end-1]); c1 = centroid(φ)
        # Mass conserved to round-off (periodic, conservative flux form).
        @test isapprox(m1, m0; rtol=1e-3)
        # Centroid moved by U·t (allow a little numerical diffusion slip).
        @test isapprox(c1 - c0, U * dt * nsteps; atol=0.5)
    end

    @testset "semi_implicit_source! implicit destruction stays positive" begin
        N = (8, 8); Ng = N .+ 2
        # Pure destruction, large dt·Dc: an *explicit* update
        # φ + dt(−Dc·φ) = φ(1 − dt·Dc) goes strongly negative for
        # dt·Dc > 1; the implicit form φ/(1 + dt·Dc) cannot.
        φ = fill(1.0, Ng); z = zeros(Ng); Dc = fill(10.0, Ng)
        semi_implicit_source!(φ, z, z, Dc, 1e3)
        @test all(>(0), @view φ[2:end-1, 2:end-1])
        for I in WaterLily.inside(φ)
            @test isapprox(φ[I], 1.0 / (1 + 1e3*10.0); atol=1e-12)
        end
        # With zero destruction, φ⁺ = φ + dt·(adv + P) exactly.
        φ2 = fill(2.0, Ng); Pp = fill(0.25, Ng); adv = fill(-0.1, Ng)
        semi_implicit_source!(φ2, adv, Pp, z, 4.0)
        for I in WaterLily.inside(φ2)
            @test isapprox(φ2[I], 2.0 + 4.0*(0.25 - 0.1); atol=1e-12)
        end
    end

    # ───────────────────────── Spalart–Allmaras ─────────────────────────

    @testset "SA fv1 viscous function limits" begin
        νm = 1e-3; cv1³ = 7.1^3
        # χ → 0: fv1 → 0 (laminar sublayer, ν_t suppressed)
        @test Turbulence._sa_fv1(1e-8, νm, cv1³) < 1e-6
        # χ → ∞: fv1 → 1 (ν_t → ν̃ in the fully turbulent limit)
        @test isapprox(Turbulence._sa_fv1(1e6*νm, νm, cv1³), 1.0; atol=1e-3)
        # χ = cv1: fv1 = cv1³/(cv1³+cv1³) = 1/2 exactly
        @test isapprox(Turbulence._sa_fv1(7.1*νm, νm, cv1³), 0.5; atol=1e-6)
        # negative ν̃ clamps to 0 → fv1 = 0
        @test Turbulence._sa_fv1(-1.0, νm, cv1³) == 0.0
    end

    @testset "SA: zero ν̃ gives laminar ν, stays put" begin
        N = (12, 20, 8); N_Y = N[2]
        body = WaterLily.AutoBody((x,t)->min(x[2], N_Y - x[2]))
        m = SpalartAllmaras(N, body; ν=1e-3, ν̃∞=0.0, perdir=(1,3), T=Float64)
        u = zeros(Float64, (N .+ 2)..., 3)   # quiescent
        step_sa!(m, u, 0.1)
        # ν̃ seeded at 0, no production in still fluid → ν stays molecular
        @test all(isapprox.(m.ν, 1e-3; atol=1e-9))
    end

    @testset "SA: turbulent ν̃ grows ν_t monotonically with shear" begin
        N = (12, 24, 8); N_Y = N[2]
        body = WaterLily.AutoBody((x,t)->min(x[2], N_Y - x[2]))
        function run_shear(γ)
            m = SpalartAllmaras(N, body; ν=1e-3, ν̃∞=3e-3, perdir=(1,3), T=Float64)
            u = zeros(Float64, (N .+ 2)..., 3)
            for I in CartesianIndices((1:N[1]+2,1:N[2]+2,1:N[3]+2))
                y = I.I[2] - 1.5
                u[I,1] = γ * min(y, N_Y - y)
            end
            for _ in 1:40; step_sa!(m, u, 0.05); end
            (maximum(m.ν) - 1e-3) / 1e-3      # peak ν_t/ν_mol
        end
        low  = run_shear(0.05)
        high = run_shear(0.20)
        @test high > low            # stronger shear → more eddy viscosity
        @test low ≥ 0 && isfinite(high)
    end

    # ───────────────────────────── k–ω SST ─────────────────────────────

    @testset "SST F1 blending: 1 near wall, →0 far from wall" begin
        νm = 1e-3; βstar = 0.09; σω2 = 0.856
        k = 0.1; ω = 100.0
        g0 = SVector(0.0, 0.0)                 # zero scalar gradients
        # Deep in the near-wall layer (very small d) the 500ν/(d²ω) term
        # dominates and F1, F2 → 1 (inner k–ω constants).
        F1n, F2n = Turbulence._sst_blend(k, ω, 0.05, 10.0, νm, βstar, σω2, g0, g0)
        @test F1n > 0.99
        @test F2n > 0.99
        # Far from the wall (large d): F1 → 0 (outer k–ε constants).
        F1f, _ = Turbulence._sst_blend(k, ω, 1e4, 10.0, νm, βstar, σω2, g0, g0)
        @test F1f < 0.01
        # F1 is monotone decreasing in wall distance.
        Fa, _ = Turbulence._sst_blend(k, ω, 1.0, 10.0, νm, βstar, σω2, g0, g0)
        Fb, _ = Turbulence._sst_blend(k, ω, 5.0, 10.0, νm, βstar, σω2, g0, g0)
        @test Fa ≥ Fb
    end

    @testset "SST ν_t limiter caps at a1 k /(S F2)" begin
        # In a high-strain region S F2 > a1 ω, so ν_t = a1 k/(S F2) < k/ω.
        a1 = 0.31; k = 0.5; ω = 1.0; S = 100.0; F2 = 1.0
        νt_lim = a1*k / max(a1*ω, S*F2)
        @test νt_lim ≈ a1*k/(S*F2)
        @test νt_lim < k/ω                     # limiter active
        # In low strain, the limiter is inactive → ν_t = k/ω.
        S2 = 0.0
        νt_free = a1*k / max(a1*ω, S2*F2)
        @test νt_free ≈ k/ω
    end

    @testset "SST: stable + positive, applies ω_wall" begin
        N = (12, 24); N_Y = N[2]
        body = WaterLily.AutoBody((x,t)->min(x[2], N_Y - x[2]))
        m = KOmegaSST(N, body; ν=0.1, k∞=1e-3, ω∞=1.0, perdir=(1,), T=Float64)
        u = zeros(Float64, (N .+ 2)..., 2)
        for I in CartesianIndices((1:N[1]+2, 1:N[2]+2))
            y = I.I[2] - 1.5
            u[I, 1] = 0.4 * min(y, N_Y - y)
        end
        for _ in 1:60; step_sst!(m, u, 0.02); end
        @test all(isfinite, m.k) && all(isfinite, m.ω) && all(isfinite, m.ν)
        @test all(≥(0), m.k)
        @test all(>(0), m.ω)
        @test all(m.ν .≥ 0.1 - 1e-9)           # ν ≥ ν_mol
        # Wall ω value: 60ν/(β1·d₁²) with d₁≈1 → 60·0.1/0.075 = 80.
        @test isapprox(maximum(m.ω), 60*0.1/0.075; rtol=1e-6)
    end

    @testset "SST Kato–Launder production: equals standard in pure shear" begin
        N = (16, 16); N_Y = N[2]
        body = WaterLily.AutoBody((x,t)->min(x[2], N_Y - x[2]))
        # Pure shear u_x(y)=γy: |S| = |Ω| = γ/√2·… equal magnitudes, so
        # P_k = νt·S² and νt·S·Ω coincide → the two production forms give
        # the same k,ω evolution and ν field.
        function run(prod)
            m = KOmegaSST(N, body; ν=0.05, k∞=1e-3, ω∞=1.0, perdir=(1,), T=Float64)
            u = zeros(Float64, (N .+ 2)..., 2)
            for I in CartesianIndices((1:N[1]+2, 1:N[2]+2))
                u[I,1] = 0.3 * (I.I[2] - 1.5)
            end
            for _ in 1:30; step_sst!(m, u, 0.02; production=prod); end
            copy(m.ν)
        end
        @test maximum(abs.(run(:standard) .- run(:kato_launder))) < 1e-9
        # Irrotational strain (Ω=0): Kato–Launder kills production where
        # standard would not — they must differ. u=(x, -y) ⇒ pure strain.
        N2 = (16,16); body2 = WaterLily.AutoBody((x,t)->min(x[2],16-x[2]))
        ms = KOmegaSST(N2, body2; ν=0.05, k∞=1e-3, ω∞=1.0, T=Float64)
        mk = KOmegaSST(N2, body2; ν=0.05, k∞=1e-3, ω∞=1.0, T=Float64)
        u2 = zeros(Float64, (N2 .+ 2)..., 2)
        for I in CartesianIndices((1:18,1:18))
            x = I.I[1]-1.5; y = I.I[2]-1.5
            u2[I,1] = 0.1*x; u2[I,2] = -0.1*y      # ∇·u=0, Ω=0, S≠0
        end
        step_sst!(ms, u2, 0.05; production=:standard)
        step_sst!(mk, u2, 0.05; production=:kato_launder)
        @test maximum(abs.(ms.ν .- mk.ν)) > 0      # strain-dominated ⇒ differ
    end

    # ───────────────────────── BDIM wall function ─────────────────────────

    @testset "Spalding u_τ solver round-trips the law of the wall" begin
        κ = 0.41; B = 5.2; ν = 0.01
        yp(up) = up + exp(-κ*B)*(exp(κ*up) - 1 - κ*up - (κ*up)^2/2 - (κ*up)^3/6)
        # For each u⁺ on the universal profile, fabricate (u_t, d) for a
        # known u_τ and check the solver recovers it.
        for uτ_true in (0.7, 1.3), up in (1.0, 5.0, 10.0, 16.43, 20.0)
            y = yp(up); u_t = up*uτ_true; d = y*ν/uτ_true
            uτ = Turbulence.spalding_uτ(u_t, d, ν)
            @test isapprox(uτ, uτ_true; rtol=1e-6)
        end
        # Non-positive tangential velocity → zero friction velocity.
        @test Turbulence.spalding_uτ(0.0, 1.0, ν) == 0.0
        # Viscous-sublayer limit: u_τ → √(ν u_t/d) when y⁺ ≪ 1.
        u_t = 1e-3; d = 1.0
        @test isapprox(Turbulence.spalding_uτ(u_t, d, ν), sqrt(ν*u_t/d); rtol=1e-3)
    end

end
