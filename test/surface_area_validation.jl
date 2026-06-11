using Plots
using ImageFiltering
using Statistics
using Printf

include(joinpath(@__DIR__, "geometries.jl"))
include(joinpath(@__DIR__, "..", "src", "utils.jl"))
include(joinpath(@__DIR__, "..", "src", "surface_area.jl"))

const PHASE_COLORS = Dict(1 => :gray, 2 => :blue, 3 => :green)

function voxel_plot(C, title::String)
    p = nothing
    for phase in (1, 2, 3)
        idx = findall(==(phase), C)
        isempty(idx) && continue
        xs = [i.I[1] for i in idx]
        ys = [i.I[2] for i in idx]
        zs = [i.I[3] for i in idx]
        kw = (
            color = PHASE_COLORS[phase],
            markershape = :square,
            markersize = 2,
            markerstrokecolor = :black,
            markerstrokewidth = 0.2,
            label = "",
        )
        if p === nothing
            p = scatter(xs, ys, zs;
                kw...,
                title = title,
                xlim = (0, NX), ylim = (0, NY), zlim = (0, NZ),
                aspect_ratio = 1,
                camera = (45, 30),
                legend = false,
            )
        else
            scatter!(p, xs, ys, zs; kw...)
        end
    end
    return p === nothing ? plot(title=title, legend=false) : p
end

println("=" ^ 80)
println("Surface Area Validation - Comparison with TauFactor Reference Values")
println("=" ^ 80)
println("Grid resolution: $(NX) × $(NY) × $(NZ) voxels")
println("Smoothing parameter (sigma): 1.0")
println("Voxel size: 1.0")
println("=" ^ 80)

# Compute all surface areas
computed_areas = Dict()
println("\nComputing surface areas for all shapes...")
for name in SHAPE_NAMES
    C = GEOMETRIES[name]()
    computed_areas[name] = surface_area(C, 1; voxel_size=1.0, sigma=1.0)
end

# 3D voxel view — all shapes in one figure (like TauFactor notebook)
println("\n3D voxel visualizations:")
println("-" ^ 80)
plots = [voxel_plot(GEOMETRIES[name](), string(name)) for name in SHAPE_NAMES]
display(plot(plots..., layout=(1, 5), size=(1600, 400), plot_title="Surface area test shapes"))

# Validation table
println("\nValidation Results:")
println("=" ^ 80)
println(rpad("Shape", 14), rpad("Computed", 14), rpad("TauFactor", 16), rpad("Diff (%)", 12), "Status")
println("-" ^ 80)

let all_pass = true, diffs = Float64[]
    for name in SHAPE_NAMES
        computed = computed_areas[name]
        reference = TAUFACTOR_REFERENCE[name]
        diff_pct = 100 * (computed - reference) / reference
        push!(diffs, abs(diff_pct))
        
        # Check if within tolerance (±10%)
        status = abs(diff_pct) <= 10.0 ? "✓ PASS" : "✗ FAIL"
        if abs(diff_pct) > 10.0
            all_pass = false
        end
        @printf("%-14s %.5f        %.5f             %+7.2f %%   %s\n",
                String(name), computed, reference, diff_pct, status)
    end
    
    println("=" ^ 80)
    println("Overall Status: ", all_pass ? "✓ ALL TESTS PASSED" : "✗ SOME TESTS FAILED")
    println("Tolerance: ±10% difference from TauFactor reference")

    # Statistics
    println("\nStatistics:")
    println("  Mean absolute error: $(round(mean(diffs), digits=2))%")
    println("  Maximum absolute error: $(round(maximum(diffs), digits=2))%")
    println("=" ^ 80)
end
