using Printf
using Test
using Plots

include(joinpath(@__DIR__, "..", "src", "utils.jl"))
include(joinpath(@__DIR__, "..", "src", "percolation.jl"))
include(joinpath(@__DIR__, "..", "src", "tpb.jl"))

const BG = 0
const PHASES_TPB = (1, 2, 3)
const PHASE_COLORS = Dict(1 => :firebrick, 2 => :royalblue, 3 => :seagreen)

function no_tpb_geometry()
    return ones(Int, 3, 3, 3)
end

function single_z_edge_tpb_geometry()
    C = zeros(Int, 2, 2, 1)
    C[1, 1, 1] = 1
    C[2, 1, 1] = 2
    C[1, 2, 1] = 3
    C[2, 2, 1] = 3
    return C
end

function single_y_edge_tpb_geometry()
    C = zeros(Int, 2, 1, 2)
    C[1, 1, 1] = 1
    C[2, 1, 1] = 2
    C[1, 1, 2] = 3
    C[2, 1, 2] = 3
    return C
end

function single_x_edge_tpb_geometry()
    C = zeros(Int, 1, 2, 2)
    C[1, 1, 1] = 1
    C[1, 2, 1] = 2
    C[1, 1, 2] = 3
    C[1, 2, 2] = 3
    return C
end

function repeated_z_edge_tpb_geometry()
    C = zeros(Int, 2, 2, 3)
    for z in 1:3
        C[1, 1, z] = 1
        C[2, 1, z] = 2
        C[1, 2, z] = 3
        C[2, 2, z] = 3
    end
    return C
end

function active_x_edge_tpb_geometry()
    C = zeros(Int, 4, 2, 2)
    C[:, 1, 1] .= 1
    C[:, 2, 1] .= 2
    C[:, 1, 2] .= 3
    C[:, 2, 2] .= 3
    return C
end

function voxel_plot(C, title)
    p = nothing
    for phase in PHASES_TPB
        idx = findall(==(phase), C)
        isempty(idx) && continue

        xs = [I.I[1] for I in idx]
        ys = [I.I[2] for I in idx]
        zs = [I.I[3] for I in idx]
        kw = (
            color=PHASE_COLORS[phase],
            markershape=:square,
            markersize=7,
            markerstrokecolor=:black,
            markerstrokewidth=0.4,
            label="",
        )

        if p === nothing
            p = scatter(xs, ys, zs;
                kw...,
                title=title,
                xlim=(0.5, size(C, 1) + 0.5),
                ylim=(0.5, size(C, 2) + 0.5),
                zlim=(0.5, size(C, 3) + 0.5),
                xlabel="x",
                ylabel="y",
                zlabel="z",
                aspect_ratio=1,
                camera=(45, 30),
                legend=false,
            )
        else
            scatter!(p, xs, ys, zs; kw...)
        end
    end

    return p === nothing ? plot(title=title, legend=false) : p
end

function save_tpb_figures(; outdir=joinpath(@__DIR__, "figures"))
    mkpath(outdir)

    cases = (
        ("single z-edge TPB", single_z_edge_tpb_geometry()),
        ("single x-edge TPB", single_x_edge_tpb_geometry()),
        ("repeated z-edge TPB", repeated_z_edge_tpb_geometry()),
        ("active x-edge TPB", active_x_edge_tpb_geometry()),
    )

    plots = [voxel_plot(C, title) for (title, C) in cases]
    fig = plot(plots..., layout=(2, 2), size=(950, 800),
        plot_title="TPB validation geometries")

    path = joinpath(outdir, "tpb_geometries.png")
    savefig(fig, path)
    return path
end

function expected_tpb_density(edge_count, C; voxel_size=1.0)
    volume = length(C) * voxel_size^3
    return edge_count * voxel_size / volume
end

function validate_tpb(; make_figure=true)
    println("TPB validation")
    println("="^80)
    println("Convention: count voxel edges whose four neighboring voxels contain phases 1, 2, and 3")
    println("Density: TPB length / total volume")
    println("-"^80)

    figure_path = make_figure ? save_tpb_figures() : nothing

    @testset "TPB" begin
        C = no_tpb_geometry()
        @printf("%-30s count=%4d expected=%4d\n", "no TPB", count_tpb_edges(C), 0)
        @test count_tpb_edges(C) == 0
        @test total_tpb_density(C; voxel_size=1.0) == 0.0
        @test_throws ErrorException total_tpb_density(C; voxel_size=0.0)

        C = single_z_edge_tpb_geometry()
        expected_count = 1
        expected_density = expected_tpb_density(expected_count, C; voxel_size=1.0)
        @printf("%-30s count=%4d expected=%4d density=%8.4f\n",
            "single z-edge TPB", count_tpb_edges(C), expected_count,
            total_tpb_density(C; voxel_size=1.0))
        @test count_tpb_edges(C) == expected_count
        @test total_tpb_density(C; voxel_size=1.0) == expected_density
        @test total_tpb_density(C; voxel_size=0.5) ==
              expected_tpb_density(expected_count, C; voxel_size=0.5)

        C = single_y_edge_tpb_geometry()
        @printf("%-30s count=%4d expected=%4d\n", "single y-edge TPB", count_tpb_edges(C), 1)
        @test count_tpb_edges(C) == 1
        @test total_tpb_density(C; voxel_size=1.0) ==
              expected_tpb_density(1, C; voxel_size=1.0)

        C = single_x_edge_tpb_geometry()
        @printf("%-30s count=%4d expected=%4d\n", "single x-edge TPB", count_tpb_edges(C), 1)
        @test count_tpb_edges(C) == 1
        @test total_tpb_density(C; voxel_size=1.0) ==
              expected_tpb_density(1, C; voxel_size=1.0)

        C = repeated_z_edge_tpb_geometry()
        expected_count = 3
        @printf("%-30s count=%4d expected=%4d\n", "repeated z-edge TPB", count_tpb_edges(C), expected_count)
        @test count_tpb_edges(C) == expected_count
        @test total_tpb_density(C; voxel_size=1.0) ==
              expected_tpb_density(expected_count, C; voxel_size=1.0)

        C = single_z_edge_tpb_geometry()
        @printf("%-30s active_x=%8.4f expected=%8.4f\n",
            "inactive TPB", active_tpb_density(C, 1; voxel_size=1.0), 0.0)
        @test active_tpb_density(C, 1; voxel_size=1.0) == 0.0

        C = active_x_edge_tpb_geometry()
        expected_count = 4
        expected_density = expected_tpb_density(expected_count, C; voxel_size=1.0)
        @printf("%-30s total=%8.4f active_x=%8.4f expected=%8.4f\n",
            "active x-edge TPB",
            total_tpb_density(C; voxel_size=1.0),
            active_tpb_density(C, 1; voxel_size=1.0),
            expected_density)
        @test count_tpb_edges(C) == expected_count
        @test total_tpb_density(C; voxel_size=1.0) == expected_density
        @test active_tpb_density(C, 1; voxel_size=1.0) == expected_density
        @test active_tpb_density(C, 2; voxel_size=1.0) == 0.0
        @test_throws ErrorException active_tpb_density(C, 1; voxel_size=0.0)
    end

    println("="^80)
    if figure_path !== nothing
        println("Figure saved to: ", figure_path)
    end
    println("All TPB validation checks passed.")
end

validate_tpb()
