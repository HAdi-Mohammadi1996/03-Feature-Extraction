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

const INPUT_DIR = length(ARGS) >= 1 ? abspath(ARGS[1]) : joinpath(@__DIR__, "inputs")
const OUTPUT_DIR = length(ARGS) >= 2 ? abspath(ARGS[2]) : joinpath(@__DIR__, "output")
const REQUESTED_WORKERS = length(ARGS) >= 3 ? parse(Int, ARGS[3]) :
    parse(Int, get(ENV, "FEATURE_WORKERS", "1"))
const MAT_KEY = "C"

const CONFIG = FeatureConfig(
    voxel_size=0.1,
    Nrays=5000,
    min_chord=4.0,
    phases=[1, 2, 3],
    use_all_directions=false,
    direction=1,
    surface_sigma=1.0,
    D0=1.0,
    physical_boundary_factor=2.0,
    physical_rtol=1e-8,
    physical_maxiter=10_000,
    physical_threaded=REQUESTED_WORKERS == 1,
    physical_thread_threshold=50_000,
    rng_seed=1,
)

isdir(INPUT_DIR) || error("Input directory does not exist: $INPUT_DIR")

mat_files = sort(filter(
    path -> endswith(lowercase(path), ".mat"),
    readdir(INPUT_DIR; join=true),
))
isempty(mat_files) && error("No .mat files found in: $INPUT_DIR")
REQUESTED_WORKERS > 0 || error("The worker count must be positive")
REQUESTED_WORKERS <= Threads.nthreads() ||
    error("Requested $REQUESTED_WORKERS workers, but Julia has $(Threads.nthreads()) threads. " *
          "Start Julia with --threads=$REQUESTED_WORKERS.")

worker_count = min(REQUESTED_WORKERS, length(mat_files))
all_rows = Vector{NamedTuple}(undef, length(mat_files))
println("Found $(length(mat_files)) MAT file(s).")
println("Parallel sample workers: $worker_count")

function process_sample(index, input_path)
    sample = splitext(basename(input_path))[1]
    println("[$index/$(length(mat_files))] Processing: ", basename(input_path))

    C = load_microstructure(input_path; key=MAT_KEY)
    return extract_features(C; sample=sample, config=CONFIG)
end

if worker_count == 1
    for (index, input_path) in enumerate(mat_files)
        all_rows[index] = process_sample(index, input_path)
    end
else
    jobs = Channel{Int}(length(mat_files))
    for index in eachindex(mat_files)
        put!(jobs, index)
    end
    close(jobs)

    tasks = [
        Threads.@spawn begin
            for index in jobs
                all_rows[index] = process_sample(index, mat_files[index])
                GC.gc()
            end
        end
        for _ in 1:worker_count
    ]
    foreach(fetch, tasks)
end

mkpath(OUTPUT_DIR)
output_path = joinpath(OUTPUT_DIR, "all_features.csv")
write_features_csv(output_path, all_rows)

println("Finished.")
println("Combined results saved to: ", output_path)
