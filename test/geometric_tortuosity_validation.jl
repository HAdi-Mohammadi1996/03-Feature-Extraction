using Printf
using Test
using Plots

include(joinpath(@__DIR__, "..", "src", "utils.jl"))
include(joinpath(@__DIR__, "..", "src", "tortuosity_geometric.jl"))

const BG = 0
const PHASE = 1
const PHASE_COLORS = Dict(BG => :white, PHASE => :steelblue)

function straight_channel_geometry(; dims=(7, 5, 5), dir=1, phase=PHASE, bg=BG)
    C = fill(bg, dims)
    mid = ntuple(i -> cld(dims[i], 2), 3)

    if dir == 1
        C[:, mid[2], mid[3]] .= phase
    elseif dir == 2
        C[mid[1], :, mid[3]] .= phase
    elseif dir == 3
        C[mid[1], mid[2], :] .= phase
    else
        error("dir must be 1, 2, or 3")
    end

    return C
end

function diagonal_touching_geometry(; n=5, phase=PHASE, bg=BG)
    C = fill(bg, n, n, 1)
    for i in 1:n
        C[i, i, 1] = phase
    end
    return C
end

function forced_detour_geometry(; phase=PHASE, bg=BG)
    C = fill(bg, 5, 5, 1)
    path = (
        (1, 1, 1),
        (2, 1, 1),
        (3, 1, 1),
        (3, 2, 1),
        (3, 3, 1),
        (4, 3, 1),
        (5, 3, 1),
    )

    for I in path
        C[I...] = phase
    end

    return C
end

function blocked_channel_geometry(; phase=PHASE, bg=BG)
    C = straight_channel_geometry(; dims=(7, 5, 5), dir=1, phase=phase, bg=bg)
    C[4, 3, 3] = bg
    return C
end

function mixed_path_geometry(; phase=PHASE, bg=BG)
    C = fill(bg, 7, 5, 1)
    C[:, 1, 1] .= phase

    detour = (
        (1, 3, 1),
        (2, 3, 1),
        (2, 4, 1),
        (3, 4, 1),
        (4, 4, 1),
        (5, 4, 1),
        (6, 4, 1),
        (6, 3, 1),
        (7, 3, 1),
    )
    for I in detour
        C[I...] = phase
    end
    return C
end

function geometry_slice(C)
    return C[:, :, cld(size(C, 3), 2)]
end

function geometry_plot(C, title)
    S = geometry_slice(C)
    return heatmap(
        1:size(S, 1),
        1:size(S, 2),
        S';
        aspect_ratio=:equal,
        color=[:white, :steelblue],
        clims=(0, 1),
        xlims=(1, size(S, 1)),
        ylims=(1, size(S, 2)),
        xlabel="x voxel",
        ylabel="y voxel",
        title=title,
        colorbar=false,
    )
end

function save_geometric_tortuosity_figures(; outdir=joinpath(@__DIR__, "figures"))
    mkpath(outdir)

    cases = (
        ("straight channel", straight_channel_geometry()),
        ("diagonal touching only", diagonal_touching_geometry()),
        ("forced detour", forced_detour_geometry()),
        ("blocked channel", blocked_channel_geometry()),
        ("mean of two paths", mixed_path_geometry()),
    )

    plots = [geometry_plot(C, title) for (title, C) in cases]
    fig = plot(plots..., layout=(2, 3), size=(1200, 750),
        plot_title="Geometric tortuosity validation geometries")

    path = joinpath(outdir, "geometric_tortuosity_geometries.png")
    savefig(fig, path)
    return path
end

function validate_geometric_tortuosity(; make_figure=true)
    println("Geometric tortuosity validation")
    println("="^80)
    println("Connectivity: 6 face-connected voxel neighbors")
    println("Method: mean inlet-to-outlet geodesic distance")
    println("-"^80)

    figure_path = make_figure ? save_geometric_tortuosity_figures() : nothing

    @testset "Geometric tortuosity" begin
        C = straight_channel_geometry()
        tau = geometric_tortuosity(C, PHASE, 1)
        @printf("%-28s tau=%10.6f expected=%10.6f\n", "straight channel", tau, 1.0)
        @test isapprox(tau, 1.0; atol=1e-12)

        C = diagonal_touching_geometry()
        tau = geometric_tortuosity(C, PHASE, 1)
        @printf("%-28s tau=%10s expected=%10s\n", "diagonal touching only", string(tau), "Inf")
        @test isinf(tau)

        C = forced_detour_geometry()
        expected = 6 / 4
        tau = geometric_tortuosity(C, PHASE, 1)
        @printf("%-28s tau=%10.6f expected=%10.6f\n", "forced detour", tau, expected)
        @test isapprox(tau, expected; atol=1e-12)

        C = mixed_path_geometry()
        expected = 7 / 6
        tau = geometric_tortuosity(C, PHASE, 1)
        @printf("%-28s tau=%10.6f expected=%10.6f\n", "mean of two paths", tau, expected)
        @test isapprox(tau, expected; atol=1e-12)

        C = blocked_channel_geometry()
        tau = geometric_tortuosity(C, PHASE, 1)
        @printf("%-28s tau=%10s expected=%10s\n", "blocked channel", string(tau), "Inf")
        @test isinf(tau)
    end

    println("="^80)
    if figure_path !== nothing
        println("Figure saved to: ", figure_path)
    end
    println("All geometric tortuosity validation checks passed.")
end

validate_geometric_tortuosity()
