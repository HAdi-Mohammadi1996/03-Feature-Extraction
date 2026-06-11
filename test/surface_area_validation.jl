using Plots
using ImageFiltering
using Statistics
using Printf

include(joinpath(@__DIR__, "geometries.jl"))
include(joinpath(@__DIR__, "..", "src", "utils.jl"))
include(joinpath(@__DIR__, "..", "src", "surface_area.jl"))

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

# Display middle slices
println("\nMiddle Slice Visualizations (z = $(NZ ÷ 2)):")
println("-" ^ 80)
mid = NZ ÷ 2
for name in SHAPE_NAMES
    C = GEOMETRIES[name]()
    p = heatmap(C[:, :, mid]', 
                title=string(name), 
                clims=(0, 3), 
                axis=false, 
                size=(300, 300),
                color=:viridis)
    display(p)
end

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
