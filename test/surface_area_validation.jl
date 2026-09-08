include(joinpath(@__DIR__, "..", "src", "MicrostructureAnalysis.jl"))
import .MicrostructureAnalysis as MA

using GLMakie
using LaTeXStrings
using CairoMakie

const Lx, Ly, Lz = 20, 20, 20
const resolution = [20, 40, 80, 160]

function make_cube(Nx, Ny, Nz; a_frac=0.5)
    C = zeros(UInt8, Nx, Ny, Nz)
    a = round(Int, a_frac*min(Nx, Ny, Nz))
    a <= min(Nx, Ny, Nz) || error("large a")
    cx = (Nx + 1) / 2
    cy = (Ny + 1) / 2
    cz = (Nz + 1) / 2

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        x = i - cx
        y = j - cy
        z = k - cz

        if abs(x) <= a/2 && abs(y) <= a/2 && abs(z) <= a/2
            C[i, j, k] = 1
        end
    end
    return C, a
end

function make_cube_rot(Nx, Ny, Nz; a_frac=0.5, θ=45)
    C = zeros(UInt8, Nx, Ny, Nz)
    a = round(Int, a_frac*min(Nx, Ny, Nz))
    a <= min(Nx, Ny, Nz) || error("large a")
    cx = (Nx + 1) / 2
    cy = (Ny + 1) / 2
    cz = (Nz + 1) / 2

    cθ = cosd(θ)
    sθ = sind(θ)

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        x = i - cx
        y = j - cy
        z = k - cz

        xr = cθ*x + sθ*y
        yr = -sθ*x + cθ*y

        if abs(xr) <= a/2 && abs(yr) <= a/2 && abs(z) <= a/2
            C[i, j, k] = 1
        end
    end
    return C, a
end

function make_sphere(Nx, Ny, Nz; r_frac=0.35)
    C = zeros(UInt8, Nx, Ny, Nz)
    r = round(Int, r_frac*min(Nx, Ny, Nz))
    r <= min(Nx, Ny, Nz)÷2 || error("large r")

    cx = (Nx + 1) / 2
    cy = (Ny + 1) / 2
    cz = (Nz + 1) / 2

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        if (i - cx)^2 + (j - cy)^2 + (k - cz)^2 <= r^2
            C[i, j, k] = 1
        end
    end
    return C, r
end

function make_multicube(Nx, Ny, Nz)
    C = zeros(UInt8, Nx, Ny, Nz)
    C[1:10, 1:10, 1:5] .= 1
    C[6:15, 6:15, 6:15] .= 2
    C[1:10, 1:10, 16:Nz] .= 1
    C[11:Nx, 6:Ny, 16:Nz] .= 3
    return C
end

const SHAPE_NAMES = [:cube, :cube_rot, :multicube, :sphere]
const PHASE_COLORS = [:gray, :blue, :green]

const GEOMETRIES = Dict(
    :cube       => () -> first(make_cube(20, 20, 20)),
    :cube_rot   => () -> first(make_cube_rot(20, 20, 20)),
    :multicube  => () -> make_multicube(20, 20, 20),
    :sphere     => () -> first(make_sphere(20, 20, 20))
)
const ANALYTICAL_REFERENCE = Dict(
    :cube       => 0.0750,
    :cube_rot   => 0.0750,
    :multicube  => 0.0500,
    :sphere     => 4*49*π/(Lx*Ly*Lz)
)

function main_benchmark()

    for i in 1:length(SHAPE_NAMES)

        shape = SHAPE_NAMES[i]
        C = GEOMETRIES[shape]()
        dx = Lx / resolution[1]
        dy = Ly / resolution[1]
        dz = Lz / resolution[1]

        ssa_calculated = MA.specific_surface_area(C, 1; spacing=(dx, dy, dz), σ=0.0)
        ssa_analytical = ANALYTICAL_REFERENCE[shape]
        error_A = 100 * abs(ssa_calculated - ssa_analytical) / ssa_analytical

        fig = Figure(size=(600, 600), figure_padding=0)
        ticks_l = ([-10, -5, 0, 5, 10], [L"-10", L"-5", L"0", L"5", L"10"])
        Label(fig[1,1], L"""
        \mathrm{ssa}_\mathrm{analytical} = %$(round(ssa_analytical, digits=3)),  
        
        \mathrm{ssa}_\mathrm{numerical} = %$(round(ssa_calculated, digits=3)),  

        ε = %$(round(error_A, digits=2))\%
        """;
        halign=:center, tellwidth=false, tellheight=false, fontsize=20)
        ax = Axis3(fig[2,1], aspect=:data, width=Relative(1.0), height=Relative(1.0), xlabel="", ylabel="", zlabel="",
                    xticks=ticks_l, yticks=ticks_l, zticks=ticks_l)

        rowsize!(fig.layout, 1, Fixed(25))
        rowgap!(fig.layout, 0)
        voxels!(ax, -10..10, -10..10, -10..10, C; gap=0.1, color=PHASE_COLORS)
        xlims!(ax, -10, 10)
        ylims!(ax, -10, 10)
        zlims!(ax, -10, 10)
        save("test/results/ssa/$(shape).png", fig; px_per_unit=3)
    end
end

function sphere_grid_resolution()
    error_A_nosmooth = Float64[]
    error_A_smooth = Float64[]
    for N in resolution
        dx = Lx / N
        dy = Ly / N
        dz = Lz / N
        C, r = make_sphere(N, N, N)
        ssa_calculated_nosmooth = MA.specific_surface_area(C, 1; spacing=(dx, dy, dz), σ=0.0)
        ssa_calculated_withsmooth = MA.specific_surface_area(C, 1; spacing=(dx, dy, dz), σ=1.0)
        ssa_analytical = 4*π*(r*dx)^2/(Lx*Ly*Lz)
        push!(error_A_nosmooth, 100 * (ssa_calculated_nosmooth - ssa_analytical) / ssa_analytical)
        push!(error_A_smooth, 100 * (ssa_calculated_withsmooth - ssa_analytical) / ssa_analytical)
    end
        fig = Figure(size=(500, 300), figure_padding=3)
        ticks_x = ([0, 50, 100, 150], [L"0", L"50", L"100", L"150"])
        ticks_y = ([-1, 0, 1, 2, 3], [L"-1", L"0", L"1", L"2", L"3"])
        ax = Axis(fig[1, 1], xlabel=L"\text{Grid Resolution}", ylabel=L"\text{Relative error }[\%]",
                    xticks=ticks_x, yticks=ticks_y,topspinevisible=true, rightspinevisible=true,
                    bottomspinevisible=true, leftspinevisible=true, xlabelsize=18, ylabelsize=18,
                    xticklabelsize=16, yticklabelsize=16,)
        hlines!(ax, [0], linestyle=:dash, color=:black)
        scatterlines!(ax, resolution, error_A_nosmooth; linewidth=2, label=L"σ=0.0", color=:black, marker=:circle)
        scatterlines!(ax, resolution, error_A_smooth; linewidth=2, label=L"σ=1.0", color=:black, marker=:rect)
        axislegend(ax; orientation=:horizontal, position=:cb, framevisible=false, labelsize=20,)
        save("test/results/ssa/sphere_grid_resolution.svg", fig; backend=CairoMakie)

end

function cube_grid_resolution()
    error_A_nosmooth = Float64[]
    error_A_smooth = Float64[]
    for N in resolution
        dx = Lx / N
        dy = Ly / N
        dz = Lz / N
        C, a = make_cube(N, N, N)
        ssa_calculated_nosmooth = MA.specific_surface_area(C, 1; spacing=(dx, dy, dz), σ=0.0)
        ssa_calculated_withsmooth = MA.specific_surface_area(C, 1; spacing=(dx, dy, dz), σ=1.0)
        ssa_analytical = 6*(a*dx)^2/(Lx*Ly*Lz)
        push!(error_A_nosmooth, 100 * (ssa_calculated_nosmooth - ssa_analytical) / ssa_analytical)
        push!(error_A_smooth, 100 * (ssa_calculated_withsmooth - ssa_analytical) / ssa_analytical)
    end
        fig = Figure(size=(500, 300), figure_padding=3)
        ticks_x = ([0, 50, 100, 150], [L"0", L"50", L"100", L"150"])
        ticks_y = ([-15, -10, -5, 0], [L"-15", L"-10", L"-5", L"0"])
        ax = Axis(fig[1, 1], xlabel=L"\text{Grid Resolution}", ylabel=L"\text{Relative error }[\%]",
                    xticks=ticks_x, yticks=ticks_y, topspinevisible=true,rightspinevisible=true,
                    bottomspinevisible=true, leftspinevisible=true, xlabelsize=18, ylabelsize=18,
                    xticklabelsize=16, yticklabelsize=16,)
        hlines!(ax, [0], linestyle=:dash, color=:black)
        scatterlines!(ax, resolution, error_A_nosmooth; linewidth=2, label=L"σ=0.0", color=:black, marker=:circle)
        scatterlines!(ax, resolution, error_A_smooth; linewidth=2, label=L"σ=1.0", color=:black, marker=:rect)
        axislegend(ax; orientation=:horizontal, position=:cb, framevisible=false, labelsize=20,)
        save("test/results/ssa/cube_grid_resolution.svg", fig; backend=CairoMakie)

end

sphere_grid_resolution()
cube_grid_resolution()
main_benchmark()
