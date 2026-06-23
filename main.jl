# =============================================================================
# EDIT SETTINGS HERE, THEN PRESS RUN
# =============================================================================

JULIA_COMPUTE_THREADS = 48

INPUT_FILE = joinpath(@__DIR__, "inputs", "2.mat")
OUTPUT_DIR = joinpath(@__DIR__, "output")
MAT_KEY = "C"

include(joinpath(@__DIR__, "src", "runtime.jl"))

if ensure_compute_threads(@__FILE__, JULIA_COMPUTE_THREADS)
    include(joinpath(@__DIR__, "src", "load_feature_extraction.jl"))
    BLAS.set_num_threads(1)

    config = FeatureConfig(
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
        physical_max_threads=JULIA_COMPUTE_THREADS,
        recalculate_static_phase_each_timestep=true,
        static_phase=2,
        verify_static_phase=true,
        rng_seed=1,
    )

    isfile(INPUT_FILE) || error("Input file does not exist: $INPUT_FILE")

    println("Loading: ", INPUT_FILE)
    C = load_microstructure(INPUT_FILE; key=MAT_KEY)
    sample = splitext(basename(INPUT_FILE))[1]

    println("Microstructure size: ", size(C))
    println("Storage type: ", eltype(C))
    println("Detected phases: ", positive_phases(C))
    println("Julia compute threads: ", Threads.nthreads(:default))
    println("Calculating features...")

    row = extract_features(C; sample=sample, config=config)
    output_path = joinpath(OUTPUT_DIR, sample * "_features.csv")
    write_features_csv(output_path, [row])

    println("Finished.")
    println("Results saved to: ", output_path)
end
