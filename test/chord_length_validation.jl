using Printf
using Random
using Statistics
using Test
using Plots

include(joinpath(@__DIR__, "..", "src", "utils.jl"))
include(joinpath(@__DIR__, "..", "src", "chord_length.jl"))

const BG = 0
const PHASE_1 = 1
const PHASE_2 = 2

function layered_slab_geometry(; nx=30, ny=10, nz=10)
    C = fill(PHASE_1, nx, ny, nz)
    C[11:15, :, :] .= PHASE_2
    return C
end

function sphere_geometry(; n=80, radius=20, phase=PHASE_1, bg=BG)
    C = fill(bg, n, n, n)
    center = (n + 1) / 2
    r2 = radius^2

    for i in 1:n, j in 1:n, k in 1:n
        if (i - center)^2 + (j - center)^2 + (k - center)^2 <= r2
            C[i, j, k] = phase
        end
    end

    return C
end

function xray_chord_stats(C, phases; min_chord=0.0, voxel_size=1.0)
    _, ny, nz = size(C)
    sums = Dict(phase => 0.0 for phase in phases)
    counts = Dict(phase => 0 for phase in phases)

    for j in 1:ny, k in 1:nz
        p = (0.0, j - 0.5, k - 0.5)
        d = (1.0, 0.0, 0.0)
        traverse_ray_chords!(sums, counts, C, p, d, min_chord)
    end

    means = Dict(
        phase => counts[phase] == 0 ? NaN : voxel_size * sums[phase] / counts[phase]
        for phase in phases
    )

    return means, counts
end

function save_validation_figures(layered, sphere; outdir=joinpath(@__DIR__, "figures"))
    mkpath(outdir)

    layered_slice = layered[:, :, cld(size(layered, 3), 2)]
    p1 = heatmap(
        1:size(layered_slice, 1),
        1:size(layered_slice, 2),
        layered_slice';
        aspect_ratio=:equal,
        color=:viridis,
        xlims=(1, size(layered_slice, 1)),
        ylims=(1, size(layered_slice, 2)),
        xlabel="x voxel",
        ylabel="y voxel",
        title="Layered slab validation slice",
        colorbar_title="phase",
    )
    savefig(p1, joinpath(outdir, "layered_slabs.png"))

    sphere_slice = sphere[:, :, cld(size(sphere, 3), 2)]
    p2 = heatmap(
        1:size(sphere_slice, 1),
        1:size(sphere_slice, 2),
        sphere_slice';
        aspect_ratio=:equal,
        color=:viridis,
        xlims=(1, size(sphere_slice, 1)),
        ylims=(1, size(sphere_slice, 2)),
        xlabel="x voxel",
        ylabel="y voxel",
        title="Sphere validation central slice",
        colorbar_title="phase",
    )
    savefig(p2, joinpath(outdir, "sphere_slice.png"))

    return outdir
end

function validate_layered_slabs()
    C = layered_slab_geometry()

    means, counts = xray_chord_stats(C, [PHASE_1, PHASE_2]; min_chord=0.0)
    @test counts[PHASE_1] == 200
    @test counts[PHASE_2] == 100
    @test isapprox(means[PHASE_1], 12.5; atol=1e-8)
    @test isapprox(means[PHASE_2], 5.0; atol=1e-8)

    filtered_means, filtered_counts = xray_chord_stats(C, [PHASE_1, PHASE_2]; min_chord=11.0)
    @test filtered_counts[PHASE_1] == 100
    @test filtered_counts[PHASE_2] == 0
    @test isapprox(filtered_means[PHASE_1], 15.0; atol=1e-8)
    @test isnan(filtered_means[PHASE_2])

    return (
        unfiltered_means=means,
        unfiltered_counts=counts,
        filtered_means=filtered_means,
        filtered_counts=filtered_counts,
    )
end

function validate_sphere()
    radius = 20
    C = sphere_geometry(; n=80, radius=radius)
    means, counts = xray_chord_stats(C, [PHASE_1]; min_chord=0.0)

    expected = 4 * radius / 3
    @test counts[PHASE_1] > 0
    @test isapprox(means[PHASE_1], expected; rtol=0.01)

    return (
        mean=means[PHASE_1],
        expected=expected,
        relative_error=abs(means[PHASE_1] - expected) / expected,
        chord_count=counts[PHASE_1],
    )
end

function validate_chord_length(; make_figures=true)
    layered = layered_slab_geometry()
    sphere = sphere_geometry()

    figure_dir = make_figures ? save_validation_figures(layered, sphere) : nothing

    println("Chord length validation")
    println("="^80)

    slab_result = validate_layered_slabs()
    println("Layered slabs")
    @printf("  phase 1 mean, min_chord=0:  %.6f voxels, count=%d\n",
            slab_result.unfiltered_means[PHASE_1], slab_result.unfiltered_counts[PHASE_1])
    @printf("  phase 2 mean, min_chord=0:  %.6f voxels, count=%d\n",
            slab_result.unfiltered_means[PHASE_2], slab_result.unfiltered_counts[PHASE_2])
    @printf("  phase 1 mean, min_chord=11: %.6f voxels, count=%d\n",
            slab_result.filtered_means[PHASE_1], slab_result.filtered_counts[PHASE_1])
    @printf("  phase 2 count, min_chord=11: %d\n",
            slab_result.filtered_counts[PHASE_2])

    sphere_result = validate_sphere()
    println("Sphere")
    @printf("  mean chord: %.6f voxels\n", sphere_result.mean)
    @printf("  expected 4R/3: %.6f voxels\n", sphere_result.expected)
    @printf("  relative error: %.3e, count=%d\n",
            sphere_result.relative_error, sphere_result.chord_count)

    if figure_dir !== nothing
        println("Figures saved to: ", figure_dir)
    end

    println("="^80)
    println("All chord length validation checks passed.")

    return (slabs=slab_result, sphere=sphere_result, figure_dir=figure_dir)
end

results = validate_chord_length()
