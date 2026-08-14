
include("../src/physical_tortuosity.jl")
using Plots
include("../src/io.jl")

function analytical_tau(α)
    return 1/cosd(α)^2
end

function tilted_channel(Nx, Ny, h, α)
    C = zeros(Int8, Nx, Ny)

    m = tan(deg2rad(α))
    yc = Ny / 2

    for j in 1:Ny, i in 1:Nx
        yline = yc + m * (i - Nx/2)

        # perpendicular distance from point to channel centreline
        d = abs(j - yline) / sqrt(1 + m^2)

        if d <= h/2
            C[i,j] = 1
        end
    end

    return C
end

# C = tilted_channel(240, 800, 20, 30)
# heatmap(C')

C = load_microstructure("inputs/2.mat")


# τ_exact = analytical_tau(30)
tau_mine, _, _, _ = physical_tortuosity1(C, 1; direction=1)
tau_taufactor= physical_tortuosity(C, 1; direction=1)

# println("Analytical       = ", τ_exact)
println("mine flux       = ", tau_mine)
println("factor flux    = ", tau_taufactor)

# τ, Deff, Qin, Qout = physical_tortuosity(C, 1; direction=1)

