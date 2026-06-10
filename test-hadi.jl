using MAT
using Plots
using Random
using ImageFiltering
using Statistics
using SparseArrays
using LinearAlgebra

include("src/io.jl")
include("src/volume_fraction.jl")
include("src/chord_length.jl")
include("src/surface_area.jl")
include("src/percolation.jl")
include("src/tortuosity_geometric.jl")
include("src/tortuosity_physical.jl")
include("src/tpb.jl")

B = load_microstructure("test_data/2.mat", key="C")

phases = [1, 2, 3]  
dirs = [1, 2, 3]
rows = NamedTuple[]
perc_fracs = Dict{Tuple{Int,Int}, Float64}()

for phase in phases, dir in dirs
    pr = percolation_result(B .== phase, dir)
    perc_fracs[(phase, dir)] = pr.fraction
end

vf = volume_fractions(B, phases)
cld = mean_chord_lengths(B, phases; min_chord=4.0)
sf = surface_areas(B, phases)
gt = Dict(phase => mean_skip_inf([geometric_tortuosity_dijkstra(B, phase, dir; voxel_size=0.1) for dir in dirs])
    for phase in phases)
pt = Dict(phase => mean_skip_inf([physical_tortuosity(B, phase, dir; voxel_size=0.1, D0=1.0) for dir in dirs])
    for phase in phases)
tpb = total_tpb_density(B; voxel_size=0.1)
println("Loaded microstructure with size: ", size(B))
println("Volume fractions: ", vf)
println("Mean chord lengths: ", cld)
println("Surface areas: ", sf)
println("Percolation result: ", perc_fracs)
println("Geometric tortuosity: ", gt)
println("Physical tortuosity: ", pt)
println("Total TPB density: ", tpb)