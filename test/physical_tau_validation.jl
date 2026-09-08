include(joinpath(@__DIR__, "..", "src", "MicrostructureAnalysis.jl"))
import .MicrostructureAnalysis as MA

using AMGX
using GLMakie
using CairoMakie
using LaTeXStrings

const Lx, Ly, H = 2.4, 8.0, 0.2
dx = [0.1, 0.01, 0.001]

function analytical_tau(α)
    return 1/cosd(α)^2
end

function tilted_channel(Nx, Ny, Δx, H, α)
    C = zeros(Int8, Nx, Ny)

    m = tand(α)
    xc = Lx / 2
    yc = Ly / 2

    @inbounds for j in 1:Ny, i in 1:Nx
        x = (i-0.5) * Δx
        y = (j-0.5) * Δx

        yline = yc + m * (x - xc)

        d = abs(y - yline) / sqrt(1 + m^2)

        if d <= H/2
            C[i,j] = 1
        end
    end

    return C
end

function main_tau_benchmark()

    AMGX.set_libAMGX_path(MA.AMGX_DLL)
    AMGX.initialize()
    try
        Nx = round(Int, Lx/dx[2])
        Ny = round(Int, Ly/dx[2])

        α = range(start=0.0, stop=70.0, length=15)

        τ_exact = zeros(Float64, length(α))
        τ_calculated_alpha = zeros(Float64, length(α))

        for i in eachindex(α)

            println("α = ", round(α[i], digits=2), "∘")

            τ_exact[i] = analytical_tau(α[i])

            C = tilted_channel(Nx, Ny, dx[2], H, α[i])

            τ_calculated_alpha[i] = MA.physical_tortuosity(C, 1; direction=1, spacings=(dx[2], dx[2]), manage_amgx=false)
        end

        α_smooth = range(0.0, 70.0, length=500)
        τ_smooth = analytical_tau.(α_smooth)

        xticks_l = ([0, 20, 40, 60], [L"0", L"20", L"40", L"60"])
        yticks_l = ([2, 4, 6, 8, 10, 12, 14],
                 [L"2", L"4", L"6", L"8", L"10", L"12", L"14"])

        fig = Figure(size=(500, 300), figure_padding=3)

        ax = Axis(fig[1, 1], xlabel=L"α\;[°]", ylabel=L"τ\;[-]", xticks=xticks_l, yticks=yticks_l,
                    xgridvisible=true, ygridvisible=true, topspinevisible=true, rightspinevisible=true,
                    xlabelsize=18, ylabelsize=18, xticklabelsize=16, yticklabelsize=16,)
        
        p_num = scatter!(ax, α, τ_calculated_alpha; marker=:circle,
                color=:white, strokecolor=:red, strokewidth=1.0)

        p_ana = lines!(ax, α_smooth, τ_smooth; linewidth=1.5, color=:black)

        axislegend(ax, [p_ana, p_num], [L"\mathrm{Analytical}\;\tau=\sec^2\alpha",
                     L"\mathrm{Numerical}"]; position=:lt, framevisible=false, labelsize=18,)
    
        xlims!(ax, 0, 70.5)
        ylims!(ax, 0, 9)
        save("test/results/tau/tau_validation.svg", fig; backend=CairoMakie)

        finally
            AMGX.finalize()
        end
end

function main_tau_grid_sensitivity()

    AMGX.set_libAMGX_path(MA.AMGX_DLL)
    AMGX.initialize()
    try
        α = 35
        τ_exact = analytical_tau(α)
        error = Float64[]
        τ_calculated_h = Float64[]

        for Δx in dx

            Nx = round(Int, Lx/Δx)
            Ny = round(Int, Ly/Δx)
            
            C = tilted_channel(Nx, Ny, Δx, H, α)
            tau = MA.physical_tortuosity(C, 1; direction=1, spacings=(Δx, Δx), manage_amgx=false)
            push!(τ_calculated_h, tau)
            push!(error, 100 * (tau - τ_exact)/τ_exact)
        end
        resolution = H./dx
        xticks_l = ([0, 20, 200], [L"0", L"20", L"200"])
        yticks_l = ([0, 20, 40], [L"0", L"20", L"40"])

        fig = Figure(size=(500, 300), figure_padding=3)

        ax = Axis(fig[1, 1], xlabel=L"H/Δx\;[-]", ylabel=L"\mathrm{Relative\ error}\;[\%]", xticks=xticks_l,
                    yticks=yticks_l, xscale=log10, xgridvisible=true, ygridvisible=true,
                    topspinevisible=true, rightspinevisible=true, xlabelsize=18, ylabelsize=18,
                    xticklabelsize=16, yticklabelsize=16,)
        
        scatterlines!(ax, resolution, error; marker=:circle, color=:black)

        hlines!(ax, [0], linestyle=:dash, color=:black)

        save("test/results/tau/tau_grid_sensitivity.svg", fig; backend=CairoMakie)
        finally
            AMGX.finalize()
        end
end

main_tau_benchmark()
main_tau_grid_sensitivity()
