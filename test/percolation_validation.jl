using Printf
using Test
using Plots

include(joinpath(@__DIR__, "..", "src", "utils.jl"))
include(joinpath(@__DIR__, "..", "src", "percolation.jl"))

function straight_channel_mask(; dims=(7, 5, 5), dir=1)
    mask = falses(dims)
    mid = ntuple(i -> cld(dims[i], 2), 3)

    if dir == 1
        mask[:, mid[2], mid[3]] .= true
    elseif dir == 2
        mask[mid[1], :, mid[3]] .= true
    elseif dir == 3
        mask[mid[1], mid[2], :] .= true
    else
        error("dir must be 1, 2, or 3")
    end

    return mask
end

function channel_with_island_mask()
    mask = straight_channel_mask()
    mask[2, 5, 3] = true
    mask[3, 5, 3] = true
    return mask
end

function diagonal_touching_mask(; n=5)
    mask = falses(n, n, 1)
    for i in 1:n
        mask[i, i, 1] = true
    end
    return mask
end

function blocked_channel_mask()
    mask = straight_channel_mask()
    mask[4, 3, 3] = false
    return mask
end

function percolation_display(mask, percmask)
    shown = zeros(Int, size(mask))
    shown[mask] .= 1
    shown[percmask] .= 2
    return shown
end

function central_slice(A)
    return A[:, :, cld(size(A, 3), 2)]
end

function validation_plot(mask, dir, title)
    result = percolation_result(mask, dir)
    S = central_slice(percolation_display(mask, result.mask))

    return heatmap(
        1:size(S, 1),
        1:size(S, 2),
        S';
        aspect_ratio=:equal,
        color=[:white, :gray70, :steelblue],
        clims=(0, 2),
        xlims=(1, size(S, 1)),
        ylims=(1, size(S, 2)),
        xlabel="x voxel",
        ylabel="y voxel",
        title=title,
        colorbar=false,
    )
end

function save_percolation_figures(; outdir=joinpath(@__DIR__, "figures"))
    mkpath(outdir)

    cases = (
        ("straight channel", straight_channel_mask(), 1),
        ("channel with island", channel_with_island_mask(), 1),
        ("diagonal touching only", diagonal_touching_mask(), 1),
        ("blocked channel", blocked_channel_mask(), 1),
    )

    plots = [validation_plot(mask, dir, title) for (title, mask, dir) in cases]
    fig = plot(plots..., layout=(2, 2), size=(900, 750),
        plot_title="Percolation validation geometries")

    path = joinpath(outdir, "percolation_geometries.png")
    savefig(fig, path)
    return path
end

function validate_percolation(; make_figure=true)
    println("Percolation validation")
    println("="^80)
    println("Connectivity: 6 face-connected voxel neighbors")
    println("Fraction: percolating voxels / all phase voxels")
    println("-"^80)

    figure_path = make_figure ? save_percolation_figures() : nothing

    @testset "Percolation" begin
        mask = falses(7, 5, 5)
        result = percolation_result(mask, 1)
        @printf("%-28s fraction=%10.6f expected=%10.6f\n", "empty phase", result.fraction, 0.0)
        @test result.fraction == 0.0
        @test count(result.mask) == 0

        mask = straight_channel_mask()
        result = percolation_result(mask, 1)
        @printf("%-28s fraction=%10.6f expected=%10.6f\n", "straight channel x", result.fraction, 1.0)
        @test result.fraction == 1.0
        @test result.mask == mask

        result_y = percolation_result(mask, 2)
        @printf("%-28s fraction=%10.6f expected=%10.6f\n", "same channel y", result_y.fraction, 0.0)
        @test result_y.fraction == 0.0
        @test count(result_y.mask) == 0

        mask = channel_with_island_mask()
        result = percolation_result(mask, 1)
        expected = 7 / 9
        @printf("%-28s fraction=%10.6f expected=%10.6f\n", "channel with island", result.fraction, expected)
        @test isapprox(result.fraction, expected; atol=1e-12)
        @test count(result.mask) == 7
        @test !result.mask[2, 5, 3]
        @test !result.mask[3, 5, 3]

        mask = diagonal_touching_mask()
        result = percolation_result(mask, 1)
        @printf("%-28s fraction=%10.6f expected=%10.6f\n", "diagonal touching only", result.fraction, 0.0)
        @test result.fraction == 0.0
        @test count(result.mask) == 0

        mask = blocked_channel_mask()
        result = percolation_result(mask, 1)
        @printf("%-28s fraction=%10.6f expected=%10.6f\n", "blocked channel", result.fraction, 0.0)
        @test result.fraction == 0.0
        @test count(result.mask) == 0

        @test_throws ErrorException percolation_result(straight_channel_mask(), 4)
    end

    println("="^80)
    if figure_path !== nothing
        println("Figure saved to: ", figure_path)
    end
    println("All percolation validation checks passed.")
end

validate_percolation()
