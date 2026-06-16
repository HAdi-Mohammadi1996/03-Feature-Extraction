using MAT
using Random
using ImageFiltering
using Statistics
using SparseArrays
using LinearAlgebra

include(joinpath(@__DIR__, "src", "io.jl"))
include(joinpath(@__DIR__, "src", "volume_fraction.jl"))
include(joinpath(@__DIR__, "src", "chord_length.jl"))
include(joinpath(@__DIR__, "src", "surface_area.jl"))
include(joinpath(@__DIR__, "src", "percolation.jl"))
include(joinpath(@__DIR__, "src", "tortuosity_geometric.jl"))
include(joinpath(@__DIR__, "src", "tortuosity_physical_matrixfree.jl"))
include(joinpath(@__DIR__, "src", "tpb.jl"))
include(joinpath(@__DIR__, "src", "feature_extraction.jl"))

const INPUT_FILE = length(ARGS) >= 1 ? abspath(ARGS[1]) : joinpath(@__DIR__, "inputs", "2.mat")
const OUTPUT_DIR = length(ARGS) >= 2 ? abspath(ARGS[2]) : joinpath(@__DIR__, "output")
const MAT_KEY = "C"

const CONFIG = FeatureConfig(
    voxel_size=0.1,
    Nrays=5000,
    min_chord=4.0,
    phases=[1, 2, 3],
    use_all_directions=true,
    direction=1,
    surface_sigma=1.0,
    D0=1.0,
    physical_boundary_factor=2.0,
    physical_rtol=1e-8,
    physical_maxiter=10_000,
    physical_threaded=true,
    physical_thread_threshold=50_000,
    rng_seed=1,
)

isfile(INPUT_FILE) || error("Input file does not exist: $INPUT_FILE")

println("Loading: ", INPUT_FILE)
C = load_microstructure(INPUT_FILE; key=MAT_KEY)
sample = splitext(basename(INPUT_FILE))[1]

println("Microstructure size: ", size(C))
println("Detected phases: ", positive_phases(C))
println("Calculating features...")

row = extract_features(C; sample=sample, config=CONFIG)
mkpath(OUTPUT_DIR)
output_path = joinpath(OUTPUT_DIR, sample * "_features.csv")
write_features_csv(output_path, [row])

println("Finished.")
println("Results saved to: ", output_path)
