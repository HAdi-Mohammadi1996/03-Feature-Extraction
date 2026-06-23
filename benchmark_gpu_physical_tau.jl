# =============================================================================
# GPU PHYSICAL-TORTUOSITY BENCHMARK
# Edit these settings if needed, then press Run.
# This script does not modify the production feature-extraction workflow.
# =============================================================================

SAMPLE_ID = 23
INPUT_DIR = joinpath("D:\\Hadi\\SharedData\\PhaseFieldResults", "$SAMPLE_ID", "mat")
INPUT_FILE = ""  # Empty means: use the first MAT file in sorted filename order.
MAT_KEY = "C"

PHASES_TO_BENCHMARK = [1, 3]
DIRECTION = 1
VOXEL_SIZE = 0.1
D0 = 1.0
BOUNDARY_CONDUCTANCE_FACTOR = 2.0
RTOL = 1e-8
MAXITER = 10_000

JULIA_COMPUTE_THREADS = 48
CPU_SOLVER_THREADS = 48
GPU_REPEATS = 2

RESULTS_FILE = joinpath(@__DIR__, "gpu_physical_tau_benchmark.csv")

include(joinpath(@__DIR__, "src", "runtime.jl"))

if ensure_compute_threads(@__FILE__, JULIA_COMPUTE_THREADS)
    include(joinpath(@__DIR__, "src", "load_feature_extraction.jl"))

    Base.find_package("CUDA") === nothing &&
        error("CUDA.jl is not installed. Run: import Pkg; Pkg.add(\"CUDA\")")

    include(joinpath(@__DIR__, "src", "tortuosity_physical_gpu.jl"))
    BLAS.set_num_threads(1)

    function selected_input_file()
        if !isempty(INPUT_FILE)
            return abspath(INPUT_FILE)
        end

        isdir(INPUT_DIR) || error("Input directory does not exist: $INPUT_DIR")
        files = sort(filter(
            path -> endswith(lowercase(path), ".mat"),
            readdir(INPUT_DIR; join=true),
        ))
        isempty(files) && error("No MAT files found in: $INPUT_DIR")
        return first(files)
    end

    function elapsed_seconds(f)
        start = time_ns()
        value = f()
        return value, (time_ns() - start) / 1e9
    end

    function gpu_elapsed_seconds(f)
        CUDA.synchronize()
        start = time_ns()
        value = f()
        CUDA.synchronize()
        return value, (time_ns() - start) / 1e9
    end

    function warm_up_gpu()
        C = ones(Int8, 16, 8, 8)
        percolation = percolation_result(C .== 1, 1)
        operator, rhs, initial, outlet_rows = physical_tau_system(
            percolation.mask,
            1,
            BOUNDARY_CONDUCTANCE_FACTOR,
        )
        system = GPUPhysicalTauSystem(operator, rhs, initial, outlet_rows)
        workspace = GPUPhysicalTauWorkspace(system)
        result = gpu_physical_tau!(
            workspace,
            system,
            1.0,
            size(C),
            1;
            voxel_size=VOXEL_SIZE,
            D0=D0,
            boundary_conductance_factor=BOUNDARY_CONDUCTANCE_FACTOR,
            rtol=RTOL,
            maxiter=MAXITER,
        )
        isapprox(result.tau, 1.0; atol=1e-10) ||
            error("GPU warm-up validation failed: tau = $(result.tau)")
        CUDA.synchronize()
        return nothing
    end

    function benchmark_phase(C, phase)
        println()
        println("="^80)
        println("Phase $phase")
        println("="^80)

        phase_fraction = count(==(phase), C) / length(C)
        percolation, percolation_seconds = elapsed_seconds() do
            percolation_result(C .== phase, DIRECTION)
        end
        percolation.fraction > 0 ||
            error("Phase $phase does not percolate in direction $DIRECTION")

        system_parts, setup_seconds = elapsed_seconds() do
            physical_tau_system(
                percolation.mask,
                DIRECTION,
                BOUNDARY_CONDUCTANCE_FACTOR,
            )
        end
        operator, rhs, initial, outlet_rows = system_parts
        active_voxels = length(rhs)

        cpu_result, cpu_solve_seconds = elapsed_seconds() do
            cpu_physical_tau_from_system(
                operator,
                rhs,
                initial,
                outlet_rows,
                phase_fraction,
                size(C),
                DIRECTION;
                voxel_size=VOXEL_SIZE,
                D0=D0,
                boundary_conductance_factor=BOUNDARY_CONDUCTANCE_FACTOR,
                rtol=RTOL,
                maxiter=MAXITER,
                max_threads=CPU_SOLVER_THREADS,
            )
        end

        gpu_objects, upload_seconds = gpu_elapsed_seconds() do
            system = GPUPhysicalTauSystem(operator, rhs, initial, outlet_rows)
            workspace = GPUPhysicalTauWorkspace(system)
            (system=system, workspace=workspace)
        end
        gpu_system = gpu_objects.system
        gpu_workspace = gpu_objects.workspace

        gpu_times = Float64[]
        gpu_results = NamedTuple[]
        for repeat in 1:GPU_REPEATS
            gpu_result, gpu_seconds = gpu_elapsed_seconds() do
                gpu_physical_tau!(
                    gpu_workspace,
                    gpu_system,
                    phase_fraction,
                    size(C),
                    DIRECTION;
                    voxel_size=VOXEL_SIZE,
                    D0=D0,
                    boundary_conductance_factor=BOUNDARY_CONDUCTANCE_FACTOR,
                    rtol=RTOL,
                    maxiter=MAXITER,
                )
            end
            push!(gpu_times, gpu_seconds)
            push!(gpu_results, gpu_result)
            println(
                "GPU repeat $repeat: ",
                round(gpu_seconds; digits=3),
                " s, iterations=",
                gpu_result.iterations,
            )
        end

        gpu_solve_seconds = minimum(gpu_times)
        gpu_result = gpu_results[argmin(gpu_times)]
        relative_error = abs(gpu_result.tau - cpu_result.tau) / abs(cpu_result.tau)
        gpu_end_to_end = setup_seconds + upload_seconds + gpu_solve_seconds
        cpu_end_to_end = setup_seconds + cpu_solve_seconds
        solve_speedup = cpu_solve_seconds / gpu_solve_seconds
        end_to_end_speedup = cpu_end_to_end / gpu_end_to_end

        println("Active voxels: ", active_voxels)
        println("Percolation: ", round(percolation_seconds; digits=3), " s")
        println("CPU system setup: ", round(setup_seconds; digits=3), " s")
        println("CPU solve: ", round(cpu_solve_seconds; digits=3), " s")
        println("GPU upload/allocation: ", round(upload_seconds; digits=3), " s")
        println("Best GPU solve: ", round(gpu_solve_seconds; digits=3), " s")
        println("CPU tau: ", cpu_result.tau)
        println("GPU tau: ", gpu_result.tau)
        println("Relative tau error: ", relative_error)
        println("Solve-only speedup: ", round(solve_speedup; digits=3), "x")
        println("Setup+solve speedup: ", round(end_to_end_speedup; digits=3), "x")

        return (
            sample=splitext(basename(selected_input_file()))[1],
            phase=phase,
            direction=DIRECTION,
            phase_fraction=phase_fraction,
            active_voxels=active_voxels,
            percolation_seconds=percolation_seconds,
            system_setup_seconds=setup_seconds,
            cpu_solve_seconds=cpu_solve_seconds,
            gpu_upload_seconds=upload_seconds,
            gpu_solve_seconds=gpu_solve_seconds,
            cpu_setup_plus_solve_seconds=cpu_end_to_end,
            gpu_setup_upload_plus_solve_seconds=gpu_end_to_end,
            solve_speedup=solve_speedup,
            end_to_end_speedup=end_to_end_speedup,
            cpu_tau=cpu_result.tau,
            gpu_tau=gpu_result.tau,
            relative_tau_error=relative_error,
            cpu_iterations=cpu_result.iterations,
            gpu_iterations=gpu_result.iterations,
        )
    end

    function run_benchmark()
        CUDA.functional() || error("CUDA is not functional on this machine")
        input_path = selected_input_file()
        isfile(input_path) || error("Input file does not exist: $input_path")

        println("GPU physical-tortuosity benchmark")
        println("Input: ", input_path)
        println("CPU threads: ", Threads.nthreads(:default))
        println("CPU solver threads: ", CPU_SOLVER_THREADS)
        println("GPU: ", CUDA.name(CUDA.device()))
        println("CUDA runtime: ", CUDA.runtime_version())
        println("Tolerance: ", RTOL)
        println()
        println("Loading microstructure...")

        C, load_seconds = elapsed_seconds() do
            load_microstructure(input_path; key=MAT_KEY)
        end
        println("Size: ", size(C), ", type: ", eltype(C))
        println("Load time: ", round(load_seconds; digits=3), " s")

        println("Warming GPU kernels and libraries...")
        warm_up_gpu()
        println("Warm-up complete.")

        rows = [benchmark_phase(C, phase) for phase in PHASES_TO_BENCHMARK]
        write_features_csv(RESULTS_FILE, rows)

        cpu_total = sum(row.cpu_setup_plus_solve_seconds for row in rows)
        gpu_total = sum(row.gpu_setup_upload_plus_solve_seconds for row in rows)
        combined_speedup = cpu_total / gpu_total
        maximum_error = maximum(row.relative_tau_error for row in rows)

        println()
        println("="^80)
        println("Combined phases ", join(PHASES_TO_BENCHMARK, ", "))
        println("="^80)
        println("CPU setup+solve total: ", round(cpu_total; digits=3), " s")
        println("GPU setup+upload+solve total: ", round(gpu_total; digits=3), " s")
        println("Combined physical-tau speedup: ", round(combined_speedup; digits=3), "x")
        println("Maximum relative tau error: ", maximum_error)
        println("Results saved to: ", RESULTS_FILE)

        if combined_speedup >= 1.5 && maximum_error <= 1e-7
            println("Decision: GPU integration is worthwhile.")
        elseif combined_speedup > 1.0 && maximum_error <= 1e-7
            println("Decision: GPU is faster, but the expected overall gain is modest.")
        else
            println("Decision: keep the current CPU implementation.")
        end
    end

    run_benchmark()
end
