using Plots
using ImageFiltering
using Statistics
using Printf
using Test

include(joinpath(@__DIR__, "geometries.jl"))
include(joinpath(@__DIR__, "..", "src", "utils.jl"))
include(joinpath(@__DIR__, "..", "src", "surface_area.jl"))

const PHASE_COLORS = Dict(1 => :gray, 2 => :blue, 3 => :green)
const SURFACE_AREA_RTOL = 0.10

function voxel_plot(C, title::String)
    p = nothing
    for phase in (1, 2, 3)
        idx = findall(==(phase), C)
        isempty(idx) && continue

        xs = [i.I[1] for i in idx]
        ys = [i.I[2] for i in idx]
        zs = [i.I[3] for i in idx]
        kw = (
            color=PHASE_COLORS[phase],
            markershape=:square,
            markersize=2,
            markerstrokecolor=:black,
            markerstrokewidth=0.2,
            label="",
        )

        if p === nothing
            p = scatter(xs, ys, zs;
                kw...,
                title=title,
                xlim=(0, NX), ylim=(0, NY), zlim=(0, NZ),
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

function save_surface_area_figure(; outdir=joinpath(@__DIR__, "figures"))
    mkpath(outdir)
    plots = [voxel_plot(GEOMETRIES[name](), string(name)) for name in SHAPE_NAMES]
    fig = plot(plots..., layout=(1, 5), size=(1600, 400),
        plot_title="Surface area test shapes")
    path = joinpath(outdir, "surface_area_shapes.png")
    savefig(fig, path)
    return path
end

function validate_surface_area(; make_figure=true)
    println("="^80)
    println("Surface Area Validation - Comparison with TauFactor Reference Values")
    println("="^80)
    println("Grid resolution: $(NX) x $(NY) x $(NZ) voxels")
    println("Method: smoothed gradient specific surface area")
    println("Smoothing parameter (sigma): 1.0")
    println("Voxel size: 1.0")
    println("Tolerance: +/- $(100 * SURFACE_AREA_RTOL)%")
    println("="^80)

    computed_areas = Dict{Symbol,Float64}()
    for name in SHAPE_NAMES
        C = GEOMETRIES[name]()
        computed_areas[name] = surface_area(C, 1; voxel_size=1.0, sigma=1.0)
    end

    figure_path = make_figure ? save_surface_area_figure() : nothing

    println()
    println("Validation Results:")
    println("="^80)
    println(rpad("Shape", 14), rpad("Computed", 14), rpad("TauFactor", 16),
        rpad("Diff (%)", 12), "Status")
    println("-"^80)

    diffs = Float64[]

    @testset "TauFactor smoothed-gradient specific surface area" begin
        for name in SHAPE_NAMES
            computed = computed_areas[name]
            reference = TAUFACTOR_REFERENCE[name]
            diff_pct = 100 * (computed - reference) / reference
            push!(diffs, abs(diff_pct))

            passed = isapprox(computed, reference; rtol=SURFACE_AREA_RTOL)
            status = passed ? "PASS" : "FAIL"

            @printf("%-14s %.5f        %.5f             %+7.2f %%   %s\n",
                String(name), computed, reference, diff_pct, status)

            @test computed ≈ reference rtol = SURFACE_AREA_RTOL
        end
    end

    println("="^80)
    println("Mean absolute error: $(round(mean(diffs), digits=2))%")
    println("Maximum absolute error: $(round(maximum(diffs), digits=2))%")
    if figure_path !== nothing
        println("Figure saved to: ", figure_path)
    end
    println("="^80)

    return computed_areas
end

results = validate_surface_area()
