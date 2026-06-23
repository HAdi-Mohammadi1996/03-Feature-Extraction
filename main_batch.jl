# =============================================================================
# EDIT SETTINGS HERE, THEN PRESS RUN
# =============================================================================

SAMPLE_ID = "69"

INPUT_DIR = raw"D:\Hadi\SharedData\PhaseFieldResults\69\mat_volume_preserving"
OUTPUT_DIR = raw"D:\Hadi\SharedData\PhaseFieldResults\69\features_new"
OUTPUT_FILE = "69_all_features.csv"
MAT_KEY = "C"

# Tuned starting point for this 96-core / 192-thread workstation.
JULIA_COMPUTE_THREADS = 48
PARALLEL_SAMPLE_WORKERS = 8
PHYSICAL_THREADS_PER_SOLVE = 6

include(joinpath(@__DIR__, "src", "runtime.jl"))

if ensure_compute_threads(@__FILE__, JULIA_COMPUTE_THREADS)
    include(joinpath(@__DIR__, "src", "load_feature_extraction.jl"))
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
        physical_threaded=true,
        physical_thread_threshold=50_000,
        physical_max_threads=PHYSICAL_THREADS_PER_SOLVE,

        # true  = recalculate every phase at every timestep (original behavior)
        # false = calculate STATIC_PHASE only for the first sorted MAT file,
        #         then reuse its phase-specific features for later timesteps
        recalculate_static_phase_each_timestep=false,
        static_phase=2,
        verify_static_phase=true,
        rng_seed=1,
    )

    output_path = joinpath(OUTPUT_DIR, OUTPUT_FILE)
    println("Input directory: ", INPUT_DIR)
    println("Output file: ", output_path)
    println("Julia compute threads: ", Threads.nthreads(:default))

    run_batch(
        INPUT_DIR;
        output_path=output_path,
        mat_key=MAT_KEY,
        config=config,
        sample_workers=PARALLEL_SAMPLE_WORKERS,
    )

    println("Finished.")
    println("Combined results saved to: ", output_path)
end
