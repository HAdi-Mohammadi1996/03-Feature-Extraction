# =============================================================================
# HYBRID CPU + GPU FEATURE EXTRACTION
# EDIT SETTINGS HERE, THEN PRESS RUN
# =============================================================================

SAMPLE_ID = 52

INPUT_DIR = joinpath("D:\\Hadi\\SharedData\\PhaseFieldResults", "$SAMPLE_ID", "mat")
OUTPUT_DIR = joinpath("D:\\Hadi\\SharedData\\PhaseFieldResults", "$SAMPLE_ID", "features")
OUTPUT_FILE = "$SAMPLE_ID.csv"
MAT_KEY = "C"

# `nothing` processes every MAT file. Use a small integer for a short test run.
MAX_FILES = nothing
PREFETCH_FILES = true

# The hybrid pipeline does not need the 48 CPU threads used by CPU physical tau.
JULIA_COMPUTE_THREADS = 16

include(joinpath(@__DIR__, "src", "runtime.jl"))

if ensure_compute_threads(@__FILE__, JULIA_COMPUTE_THREADS)
    include(joinpath(@__DIR__, "src", "load_feature_extraction.jl"))

    Base.find_package("CUDA") === nothing &&
        error("CUDA.jl is required. Install it once with: import Pkg; Pkg.add(\"CUDA\")")

    include(joinpath(@__DIR__, "src", "tortuosity_physical_gpu.jl"))
    include(joinpath(@__DIR__, "src", "feature_extraction_hybrid.jl"))
    BLAS.set_num_threads(1)

    config = FeatureConfig(
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

        # These CPU physical-tau settings are unused by this GPU entry point,
        # but remain part of the shared FeatureConfig.
        physical_threaded=false,
        physical_thread_threshold=50_000,
        physical_max_threads=1,

        # false calculates the selected phase only in the first sorted MAT file.
        recalculate_static_phase_each_timestep=false,
        static_phase=2,
        verify_static_phase=true,
        rng_seed=1,
    )

    output_path = if MAX_FILES === nothing
        joinpath(OUTPUT_DIR, OUTPUT_FILE)
    else
        stem, extension = splitext(OUTPUT_FILE)
        joinpath(OUTPUT_DIR, "$(stem)_first$(MAX_FILES)$(extension)")
    end
    println("Hybrid CPU + GPU feature extraction")
    println("Input directory: ", INPUT_DIR)
    println("Output file: ", output_path)
    println("Julia compute threads: ", Threads.nthreads(:default))
    println("Static phase: ", config.static_phase)

    run_hybrid_batch(
        INPUT_DIR;
        output_path=output_path,
        mat_key=MAT_KEY,
        config=config,
        prefetch_files=PREFETCH_FILES,
        max_files=MAX_FILES,
    )

    println()
    println("Finished.")
    println("Combined results saved to: ", output_path)
end
