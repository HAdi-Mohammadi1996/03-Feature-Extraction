
include("../src/physical_tortuosity.jl")
using Plots
using LaTeXStrings

function analytical_tau(α)
    return 1/cosd(α)^2
end

function tilted_channel(Nx, Ny, h, α)
    C = zeros(Int8, Nx, Ny)

    m = tand(α)
    yc = Ny / 2

    @inbounds for j in 1:Ny, i in 1:Nx
        yline = yc + m * (i - Nx/2)

        # perpendicular distance from point to channel centreline
        d = abs(j - yline) / sqrt(1 + m^2)

        if d <= h/2
            C[i,j] = 1
        end
    end

    return C
end

function main()

    AMGX.set_libAMGX_path(raw"C:\Users\r43341mm\AMGX\build\Release\amgxsh.dll")
    AMGX.initialize()
    try
        Nx = 240
        Ny = 800
        h = 20

        α = range(start=0.0, stop=75.0, length=15)

        τ_exact = zeros(Float64, length(α))
        τ_calculated = zeros(Float64, length(α))

        for i in eachindex(α)

            println("α = ", round(α[i], digits=2), "∘")

            τ_exact[i] = analytical_tau(α[i])

            C = tilted_channel(Nx, Ny, h, α[i])

            τ_calculated[i] = physical_tortuosity(C, 1; direction=1, spacings=(1.0, 1.0))
        end

        α_smooth = range(0.0, 75.0, length=500)
        τ_smooth = analytical_tau.(α_smooth)

        xticks_l = ([0, 20, 40, 60], [L"0", L"20", L"40", L"60"])
        yticks1 = ([2, 4, 6, 8, 10, 12, 14],
                 [L"2", L"4", L"6", L"8", L"10", L"12", L"14"])

        p = scatter(α, τ_calculated; label=L"\mathrm{Numerical}", markershape=:circle,
                    markersize=4, markercolor=:white, markerstrokecolor=:red,
                    markerstrokewidth=1.0)
        p = plot!(p, α_smooth, τ_smooth; label=L"\mathrm{Analytical}\;τ=\sec^2α",
                 linewidth=1.5, xlabel=L"α", ylabel=L"τ", xticks=xticks_l, yticks=yticks1,
                 size=(400, 400), dpi=300, framestyle=:box, grid=true, minorgrid=false,
                 legend=:topleft, linecolor=:black, tickfontsize=8, guidefontsize=10,
                 legendfontsize=8)

        display(p)

        return α, τ_exact, τ_calculated

        finally
            AMGX.finalize()
        end
end

main()

