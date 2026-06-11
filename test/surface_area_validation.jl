using Plots
using ImageFiltering

include(joinpath(@__DIR__, "geometries.jl"))
include(joinpath(@__DIR__, "..", "src", "utils.jl"))
include(joinpath(@__DIR__, "..", "src", "surface_area.jl"))

println("Surface area validation")
println("Each shape is 20x20x20 voxels. Showing middle slice, then results.\n")

mid = NZ ÷ 2
for name in SHAPE_NAMES
    C = GEOMETRIES[name]()
    p = heatmap(C[:, :, mid]', title=string(name), clims=(0, 3), axis=false)
    display(p)
end

println("\nResults (phase 1):")
println(rpad("shape", 14), rpad("computed", 12), rpad("taufactor", 12), "diff_pct")
println("-" ^ 50)

for name in SHAPE_NAMES
    C = GEOMETRIES[name]()
    computed = surface_area(C, 1; voxel_size=1.0, sigma=1.0)
    reference = TAUFACTOR_REFERENCE[name]
    diff = round(100 * (computed - reference) / reference, digits=2)
    println(rpad(string(name), 14),
        rpad(round(computed, digits=5), 12),
        rpad(round(reference, digits=5), 12),
        "$diff %")
end
