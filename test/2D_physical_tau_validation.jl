
include("../src/physical_tortuosity.jl")
using Plots

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

C = tilted_channel(100, 100, 10, 30)

τ, Deff, Qin, Qout = physical_tortuosity(C, 1; direction=1)

