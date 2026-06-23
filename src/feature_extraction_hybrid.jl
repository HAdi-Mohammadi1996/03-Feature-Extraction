struct HybridFeatureContext
    sample::String
    dims::NTuple{3,Int}
    calculated_phases::Vector{Int}
    masks::Dict{Int,BitArray{3}}
    vf::Dict{Int,Float64}
    percolations::Dict{Tuple{Int,Int},PercolationResult}
end

function prepare_hybrid_context(
    C::AbstractArray{<:Integer,3};
    sample="sample",
    config=FeatureConfig(),
    phase_cache::Union{Nothing,PhaseFeatureCache}=nothing,
)
    phase_cache === nothing || validate_phase_cache(C, sample, config, phase_cache)

    phases = config.phases
    dirs = directions(config)
    calculated_phases = phase_cache === nothing ?
        copy(phases) :
        [phase for phase in phases if phase != phase_cache.phase]

    masks = Dict{Int,BitArray{3}}()
    vf = Dict{Int,Float64}()
    for phase in calculated_phases
        vf[phase] = count(phase_mask!(masks, C, phase)) / length(C)
    end
    if phase_cache !== nothing
        vf[phase_cache.phase] = phase_cache.vf
    end

    percolations = Dict{Tuple{Int,Int},PercolationResult}()
    if phase_cache !== nothing
        for (dir, result) in phase_cache.percolation
            percolations[(phase_cache.phase, dir)] = result
        end
    end

    required_phases = unique(vcat(calculated_phases, [1, 2]))
    for phase in required_phases, dir in dirs
        key = (phase, dir)
        haskey(percolations, key) && continue
        percolations[key] = percolation_result(phase_mask!(masks, C, phase), dir)
    end

    return HybridFeatureContext(
        string(sample),
        size(C),
        calculated_phases,
        masks,
        vf,
        percolations,
    )
end

function compute_hybrid_cpu_features(
    C::AbstractArray{<:Integer,3},
    context::HybridFeatureContext;
    config=FeatureConfig(),
    phase_cache::Union{Nothing,PhaseFeatureCache}=nothing,
)
    calculated_phases = context.calculated_phases
    dirs = directions(config)

    chord_task = Threads.@spawn begin
        isempty(calculated_phases) ?
        Dict{Int,Float64}() :
        mean_chord_lengths(
            C,
            calculated_phases;
            Nrays=config.Nrays,
            min_chord=config.min_chord,
            voxel_size=config.voxel_size,
            rng=MersenneTwister(config.rng_seed),
        )
    end

    surface_task = Threads.@spawn begin
        isempty(calculated_phases) ?
        Dict{Int,Float64}() :
        surface_areas(
            C,
            calculated_phases;
            voxel_size=config.voxel_size,
            sigma=config.surface_sigma,
        )
    end

    geometric_task = Threads.@spawn Dict(
        phase => mean_skip_inf([
            geometric_tortuosity(
                C,
                phase,
                dir;
                voxel_size=config.voxel_size,
                percolation=context.percolations[(phase, dir)],
            )
            for dir in dirs
        ])
        for phase in calculated_phases
    )

    tpb_task = Threads.@spawn begin
        total = total_tpb_density(C; voxel_size=config.voxel_size)
        active = mean([
            active_tpb_density(
                C,
                dir;
                voxel_size=config.voxel_size,
                phase_percolation=context.percolations,
            )
            for dir in dirs
        ])
        (total=total, active=active)
    end

    chords = fetch(chord_task)
    specific_surface = fetch(surface_task)
    gt = fetch(geometric_task)
    tpb = fetch(tpb_task)
    perc = Dict(
        phase => mean([
            context.percolations[(phase, dir)].fraction
            for dir in dirs
        ])
        for phase in calculated_phases
    )

    if phase_cache !== nothing
        phase = phase_cache.phase
        chords[phase] = phase_cache.cld
        specific_surface[phase] = phase_cache.sa
        gt[phase] = phase_cache.gt
        perc[phase] = phase_cache.perc
    end

    return (
        chords=chords,
        specific_surface=specific_surface,
        gt=gt,
        perc=perc,
        tpb_total=tpb.total,
        tpb_active=tpb.active,
    )
end

function compute_hybrid_gpu_physical_tau(
    context::HybridFeatureContext;
    config=FeatureConfig(),
    phase_cache::Union{Nothing,PhaseFeatureCache}=nothing,
)
    dirs = directions(config)
    pt = Dict{Int,Float64}()
    solve_details = NamedTuple[]

    for phase in context.calculated_phases
        directional_tau = Float64[]
        for dir in dirs
            result = gpu_physical_tortuosity(
                context.percolations[(phase, dir)],
                context.vf[phase],
                context.dims,
                dir;
                voxel_size=config.voxel_size,
                D0=config.D0,
                boundary_conductance_factor=config.physical_boundary_factor,
                rtol=config.physical_rtol,
                maxiter=config.physical_maxiter,
            )
            push!(directional_tau, result.tau)
            push!(
                solve_details,
                merge(
                    (phase=phase, direction=dir),
                    result,
                ),
            )
        end
        pt[phase] = mean_skip_inf(directional_tau)
    end

    if phase_cache !== nothing
        pt[phase_cache.phase] = phase_cache.pt
    end
    return (pt=pt, solve_details=solve_details)
end

function warm_up_hybrid_gpu(config::FeatureConfig)
    C = ones(Int8, 16, 8, 8)
    percolation = percolation_result(C .== 1, 1)
    result = gpu_physical_tortuosity(
        percolation,
        1.0,
        size(C),
        1;
        voxel_size=config.voxel_size,
        D0=config.D0,
        boundary_conductance_factor=config.physical_boundary_factor,
        rtol=config.physical_rtol,
        maxiter=config.physical_maxiter,
    )
    isapprox(result.tau, 1.0; atol=1e-10) ||
        error("GPU warm-up validation failed: tau = $(result.tau)")
    return nothing
end

function extract_features_hybrid(
    C::AbstractArray{<:Integer,3};
    sample="sample",
    config=FeatureConfig(),
    phase_cache::Union{Nothing,PhaseFeatureCache}=nothing,
)
    start = time_ns()
    context = prepare_hybrid_context(
        C;
        sample=sample,
        config=config,
        phase_cache=phase_cache,
    )
    preparation_seconds = (time_ns() - start) / 1e9

    cpu_task = Threads.@spawn begin
        cpu_start = time_ns()
        features = compute_hybrid_cpu_features(
            C,
            context;
            config=config,
            phase_cache=phase_cache,
        )
        (features=features, seconds=(time_ns() - cpu_start) / 1e9)
    end

    gpu_start = time_ns()
    gpu_features = compute_hybrid_gpu_physical_tau(
        context;
        config=config,
        phase_cache=phase_cache,
    )
    gpu_seconds = (time_ns() - gpu_start) / 1e9
    cpu_result = fetch(cpu_task)

    row = assemble_feature_row(
        sample,
        config.phases;
        vf=context.vf,
        chords=cpu_result.features.chords,
        specific_surface=cpu_result.features.specific_surface,
        gt=cpu_result.features.gt,
        pt=gpu_features.pt,
        perc=cpu_result.features.perc,
        tpb_total=cpu_result.features.tpb_total,
        tpb_active=cpu_result.features.tpb_active,
    )

    updated_cache = phase_cache
    if phase_cache === nothing && !config.recalculate_static_phase_each_timestep
        updated_cache = make_phase_feature_cache(
            C,
            sample,
            config,
            context.masks,
            context.percolations;
            vf=context.vf,
            chords=cpu_result.features.chords,
            specific_surface=cpu_result.features.specific_surface,
            gt=cpu_result.features.gt,
            pt=gpu_features.pt,
            perc=cpu_result.features.perc,
        )
    end

    return (
        row=row,
        phase_cache=updated_cache,
        timing=(
            preparation_seconds=preparation_seconds,
            cpu_features_seconds=cpu_result.seconds,
            gpu_physical_seconds=gpu_seconds,
            total_seconds=(time_ns() - start) / 1e9,
        ),
        gpu_solve_details=gpu_features.solve_details,
    )
end

function hybrid_mat_files(input_dir::AbstractString, max_files)
    isdir(input_dir) || error("Input directory does not exist: $input_dir")
    files = sort(filter(
        path -> endswith(lowercase(path), ".mat"),
        readdir(input_dir; join=true),
    ))
    isempty(files) && error("No .mat files found in: $input_dir")
    if max_files === nothing
        return files
    end
    max_files > 0 || error("max_files must be positive or nothing")
    return files[1:min(max_files, length(files))]
end

function run_hybrid_batch(
    input_dir::AbstractString;
    output_path::AbstractString,
    mat_key="C",
    config=FeatureConfig(),
    prefetch_files=true,
    max_files=nothing,
)
    CUDA.functional() || error("CUDA is not functional on this machine")
    files = hybrid_mat_files(input_dir, max_files)
    rows = Vector{NamedTuple}(undef, length(files))
    phase_cache = nothing

    println("Found $(length(files)) MAT file(s).")
    println("GPU: ", CUDA.name(CUDA.device()))
    println("Warming GPU kernels...")
    warm_up_hybrid_gpu(config)
    println("GPU warm-up complete.")
    batch_start = time_ns()

    function process_loaded(index, path, C, load_seconds)
        sample = splitext(basename(path))[1]
        println()
        println("[$index/$(length(files))] Processing: ", basename(path))
        result = extract_features_hybrid(
            C;
            sample=sample,
            config=config,
            phase_cache=phase_cache,
        )
        rows[index] = result.row

        timing = result.timing
        println(
            "  load=", round(load_seconds; digits=2), " s",
            ", prepare=", round(timing.preparation_seconds; digits=2), " s",
            ", CPU features=", round(timing.cpu_features_seconds; digits=2), " s",
            ", GPU physical=", round(timing.gpu_physical_seconds; digits=2), " s",
            ", overlapped total=", round(timing.total_seconds; digits=2), " s",
        )
        for detail in result.gpu_solve_details
            println(
                "    phase $(detail.phase), dir $(detail.direction): ",
                round(detail.gpu_solve_seconds; digits=2),
                " s GPU solve, tau=",
                detail.tau,
            )
        end
        elapsed = (time_ns() - batch_start) / 1e9
        average = elapsed / index
        remaining = average * (length(files) - index)
        println(
            "  elapsed=", round(elapsed / 60; digits=1), " min",
            ", estimated remaining=", round(remaining / 60; digits=1), " min",
        )
        return result.phase_cache
    end

    if prefetch_files && length(files) > 1
        loaded = Channel{Any}(1)
        loader = Threads.@spawn begin
            try
                for (index, path) in enumerate(files)
                    start = time_ns()
                    C = load_microstructure(path; key=mat_key)
                    put!(
                        loaded,
                        (
                            index=index,
                            path=path,
                            C=C,
                            load_seconds=(time_ns() - start) / 1e9,
                            error=nothing,
                        ),
                    )
                end
            catch error
                put!(loaded, (error=(error, catch_backtrace()),))
            finally
                close(loaded)
            end
        end

        for item in loaded
            if item.error !== nothing
                error, backtrace = item.error
                Base.showerror(stderr, error, backtrace)
                println(stderr)
                throw(error)
            end
            phase_cache = process_loaded(
                item.index,
                item.path,
                item.C,
                item.load_seconds,
            )
        end
        fetch(loader)
    else
        for (index, path) in enumerate(files)
            start = time_ns()
            C = load_microstructure(path; key=mat_key)
            load_seconds = (time_ns() - start) / 1e9
            phase_cache = process_loaded(index, path, C, load_seconds)
        end
    end

    write_features_csv(output_path, rows)
    return output_path
end
