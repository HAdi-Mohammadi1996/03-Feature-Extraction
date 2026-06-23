struct FeatureConfig
    voxel_size::Float64
    Nrays::Int
    min_chord::Float64
    phases::Vector{Int}
    use_all_directions::Bool
    direction::Int
    surface_sigma::Float64
    D0::Float64
    physical_boundary_factor::Float64
    physical_rtol::Float64
    physical_maxiter::Int
    physical_threaded::Bool
    physical_thread_threshold::Int
    physical_max_threads::Int
    recalculate_static_phase_each_timestep::Bool
    static_phase::Int
    verify_static_phase::Bool
    rng_seed::Int
end

function FeatureConfig(;
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
    physical_max_threads=0,
    recalculate_static_phase_each_timestep=true,
    static_phase=2,
    verify_static_phase=true,
    rng_seed=1,
)
    voxel_size > 0 || error("voxel_size must be positive")
    Nrays > 0 || error("Nrays must be positive")
    min_chord >= 0 || error("min_chord must be non-negative")
    isempty(phases) && error("phases must not be empty")
    all(>(0), phases) || error("phases must contain positive integer labels")
    allunique(phases) || error("phases must not contain duplicates")
    direction in (1, 2, 3) || error("direction must be 1, 2, or 3")
    physical_boundary_factor > 0 || error("physical_boundary_factor must be positive")
    physical_rtol > 0 || error("physical_rtol must be positive")
    physical_maxiter > 0 || error("physical_maxiter must be positive")
    physical_thread_threshold > 0 || error("physical_thread_threshold must be positive")
    physical_max_threads >= 0 || error("physical_max_threads must be non-negative")

    sorted_phases = sort!(Int.(collect(phases)))
    recalculate_static_phase_each_timestep || static_phase in sorted_phases ||
        error("static_phase must be included in phases")

    return FeatureConfig(
        float(voxel_size),
        Int(Nrays),
        float(min_chord),
        sorted_phases,
        Bool(use_all_directions),
        Int(direction),
        float(surface_sigma),
        float(D0),
        float(physical_boundary_factor),
        float(physical_rtol),
        Int(physical_maxiter),
        Bool(physical_threaded),
        Int(physical_thread_threshold),
        Int(physical_max_threads),
        Bool(recalculate_static_phase_each_timestep),
        Int(static_phase),
        Bool(verify_static_phase),
        Int(rng_seed),
    )
end

struct PhaseFeatureCache
    phase::Int
    reference_sample::String
    reference_mask::BitArray{3}
    vf::Float64
    cld::Float64
    sa::Float64
    gt::Float64
    pt::Float64
    perc::Float64
    percolation::Dict{Int,PercolationResult}
end

directions(config::FeatureConfig) =
    config.use_all_directions ? (1, 2, 3) : (config.direction,)

function positive_phases(C::AbstractArray{<:Integer,3})
    return [phase for phase in sort!(collect(unique(vec(C)))) if phase > 0]
end

function phase_mask!(masks::Dict{Int,BitArray{3}}, C, phase::Int)
    return get!(masks, phase) do
        BitArray(C .== phase)
    end
end

function phase_cache_matches(C, cache::PhaseFeatureCache)
    size(C) == size(cache.reference_mask) || return false
    @inbounds for index in eachindex(C, cache.reference_mask)
        ((C[index] == cache.phase) == cache.reference_mask[index]) || return false
    end
    return true
end

function validate_phase_cache(C, sample, config, cache)
    cache.phase == config.static_phase ||
        error("The phase cache is for phase $(cache.phase), not phase $(config.static_phase)")
    if config.verify_static_phase && !phase_cache_matches(C, cache)
        error(
            "Static phase $(cache.phase) changed in sample '$sample' relative to " *
            "reference sample '$(cache.reference_sample)'. Set " *
            "recalculate_static_phase_each_timestep=true to calculate every timestep.",
        )
    end
end

function assemble_feature_row(
    sample,
    phases;
    vf,
    chords,
    specific_surface,
    gt,
    pt,
    perc,
    tpb_total,
    tpb_active,
)
    names = Symbol[:sample]
    values = Any[string(sample)]
    feature_groups = (
        ("vf", vf),
        ("cld", chords),
        ("sa", specific_surface),
        ("gt", gt),
        ("pt", pt),
        ("perc", perc),
    )
    for (prefix, feature) in feature_groups, phase in phases
        push!(names, Symbol(prefix, phase))
        push!(values, feature[phase])
    end
    append!(names, (:tpb, :atpb))
    append!(values, (tpb_total, tpb_active))
    return NamedTuple{Tuple(names)}(Tuple(values))
end

function make_phase_feature_cache(
    C,
    sample,
    config,
    masks,
    percolations;
    vf,
    chords,
    specific_surface,
    gt,
    pt,
    perc,
)
    phase = config.static_phase
    phase_percolation = Dict(
        dir => percolations[(phase, dir)]
        for dir in directions(config)
    )
    return PhaseFeatureCache(
        phase,
        string(sample),
        copy(phase_mask!(masks, C, phase)),
        vf[phase],
        chords[phase],
        specific_surface[phase],
        gt[phase],
        pt[phase],
        perc[phase],
        phase_percolation,
    )
end

function _extract_features(
    C::AbstractArray{<:Integer,3};
    sample="sample",
    config=FeatureConfig(),
    phase_cache::Union{Nothing,PhaseFeatureCache}=nothing,
    physical_max_threads=config.physical_max_threads,
)
    phase_cache === nothing || validate_phase_cache(C, sample, config, phase_cache)
    physical_max_threads >= 0 || error("physical_max_threads must be non-negative")

    phases = config.phases
    dirs = directions(config)
    calculated_phases = phase_cache === nothing ?
        phases :
        [phase for phase in phases if phase != phase_cache.phase]

    masks = Dict{Int,BitArray{3}}()
    vf = Dict{Int,Float64}()
    chords = Dict{Int,Float64}()
    specific_surface = Dict{Int,Float64}()
    gt = Dict{Int,Float64}()
    pt = Dict{Int,Float64}()
    perc = Dict{Int,Float64}()

    for phase in calculated_phases
        vf[phase] = count(phase_mask!(masks, C, phase)) / length(C)
    end

    if !isempty(calculated_phases)
        merge!(
            chords,
            mean_chord_lengths(
                C,
                calculated_phases;
                Nrays=config.Nrays,
                min_chord=config.min_chord,
                voxel_size=config.voxel_size,
                rng=MersenneTwister(config.rng_seed),
            ),
        )
        merge!(
            specific_surface,
            surface_areas(
                C,
                calculated_phases;
                voxel_size=config.voxel_size,
                sigma=config.surface_sigma,
            ),
        )
    end

    percolations = Dict{Tuple{Int,Int},PercolationResult}()
    if phase_cache !== nothing
        for (dir, result) in phase_cache.percolation
            percolations[(phase_cache.phase, dir)] = result
        end
    end

    required_percolation_phases = unique(vcat(calculated_phases, [1, 2]))
    for phase in required_percolation_phases, dir in dirs
        key = (phase, dir)
        haskey(percolations, key) && continue
        percolations[key] = percolation_result(phase_mask!(masks, C, phase), dir)
    end

    for phase in calculated_phases
        gt[phase] = mean_skip_inf([
            geometric_tortuosity(
                C,
                phase,
                dir;
                voxel_size=config.voxel_size,
                percolation=percolations[(phase, dir)],
            )
            for dir in dirs
        ])
        pt[phase] = mean_skip_inf([
            physical_tortuosity_matrixfree(
                C,
                phase,
                dir;
                voxel_size=config.voxel_size,
                D0=config.D0,
                boundary_conductance_factor=config.physical_boundary_factor,
                rtol=config.physical_rtol,
                maxiter=config.physical_maxiter,
                threaded=config.physical_threaded,
                thread_threshold=config.physical_thread_threshold,
                max_threads=physical_max_threads,
                percolation=percolations[(phase, dir)],
                phase_fraction=vf[phase],
            )
            for dir in dirs
        ])
        perc[phase] = mean([percolations[(phase, dir)].fraction for dir in dirs])
    end

    if phase_cache !== nothing
        phase = phase_cache.phase
        vf[phase] = phase_cache.vf
        chords[phase] = phase_cache.cld
        specific_surface[phase] = phase_cache.sa
        gt[phase] = phase_cache.gt
        pt[phase] = phase_cache.pt
        perc[phase] = phase_cache.perc
    end

    tpb_total = total_tpb_density(C; voxel_size=config.voxel_size)
    tpb_active = mean([
        active_tpb_density(
            C,
            dir;
            voxel_size=config.voxel_size,
            phase_percolation=percolations,
        )
        for dir in dirs
    ])
    row = assemble_feature_row(
        sample,
        phases;
        vf=vf,
        chords=chords,
        specific_surface=specific_surface,
        gt=gt,
        pt=pt,
        perc=perc,
        tpb_total=tpb_total,
        tpb_active=tpb_active,
    )
    context = (
        masks=masks,
        vf=vf,
        chords=chords,
        specific_surface=specific_surface,
        gt=gt,
        pt=pt,
        perc=perc,
        percolations=percolations,
    )
    return row, context
end

function extract_features(
    C::AbstractArray{<:Integer,3};
    sample="sample",
    config=FeatureConfig(),
    phase_cache::Union{Nothing,PhaseFeatureCache}=nothing,
)
    row, _ = _extract_features(
        C;
        sample=sample,
        config=config,
        phase_cache=phase_cache,
    )
    return row
end

function extract_features_with_phase_cache(
    C::AbstractArray{<:Integer,3};
    sample="sample",
    config=FeatureConfig(),
    physical_max_threads=config.physical_max_threads,
)
    row, context = _extract_features(
        C;
        sample=sample,
        config=config,
        physical_max_threads=physical_max_threads,
    )
    cache = make_phase_feature_cache(
        C,
        sample,
        config,
        context.masks,
        context.percolations;
        vf=context.vf,
        chords=context.chords,
        specific_surface=context.specific_surface,
        gt=context.gt,
        pt=context.pt,
        perc=context.perc,
    )
    return row, cache
end

function run_sample(
    input_path::AbstractString;
    output_dir="output",
    mat_key="C",
    config=FeatureConfig(),
)
    C = load_microstructure(input_path; key=mat_key)
    sample = splitext(basename(input_path))[1]
    row = extract_features(C; sample=sample, config=config)
    output_path = joinpath(output_dir, sample * "_features.csv")
    write_features_csv(output_path, [row])
    return output_path
end

function run_batch(
    input_dir::AbstractString;
    output_path::AbstractString,
    mat_key="C",
    config=FeatureConfig(),
    sample_workers=1,
)
    isdir(input_dir) || error("Input directory does not exist: $input_dir")
    sample_workers > 0 || error("sample_workers must be positive")
    sample_workers <= Threads.nthreads(:default) ||
        error(
            "Requested $sample_workers sample workers, but Julia has only " *
            "$(Threads.nthreads(:default)) compute threads.",
        )

    mat_files = sort(filter(
        path -> endswith(lowercase(path), ".mat"),
        readdir(input_dir; join=true),
    ))
    isempty(mat_files) && error("No .mat files found in: $input_dir")

    all_rows = Vector{NamedTuple}(undef, length(mat_files))
    first_parallel_index = 1
    phase_cache = nothing

    println("Found $(length(mat_files)) MAT file(s).")
    if !config.recalculate_static_phase_each_timestep
        reference_path = first(mat_files)
        reference_sample = splitext(basename(reference_path))[1]
        println(
            "[1/$(length(mat_files))] Processing static-phase reference: ",
            basename(reference_path),
        )
        C = load_microstructure(reference_path; key=mat_key)
        reference_solver_threads = config.physical_threaded ?
            Threads.nthreads(:default) :
            1
        println("Reference physical-solver tasks per solve: ", reference_solver_threads)
        all_rows[1], phase_cache = extract_features_with_phase_cache(
            C;
            sample=reference_sample,
            config=config,
            physical_max_threads=reference_solver_threads,
        )
        first_parallel_index = 2
        println(
            "Reusing phase $(config.static_phase) features from '$reference_sample' ",
            "for later timesteps.",
        )
    end

    remaining_indices = first_parallel_index:length(mat_files)
    worker_count = min(sample_workers, length(remaining_indices))
    println("Parallel sample workers: $worker_count")
    println(
        "Physical-solver tasks per solve: ",
        config.physical_threaded ?
        (config.physical_max_threads == 0 ? Threads.nthreads(:default) :
         min(config.physical_max_threads, Threads.nthreads(:default))) :
        1,
    )

    function process_sample(index)
        input_path = mat_files[index]
        sample = splitext(basename(input_path))[1]
        println("[$index/$(length(mat_files))] Processing: ", basename(input_path))
        C = load_microstructure(input_path; key=mat_key)
        return extract_features(
            C;
            sample=sample,
            config=config,
            phase_cache=phase_cache,
        )
    end

    if worker_count == 1
        for index in remaining_indices
            all_rows[index] = process_sample(index)
        end
    elseif worker_count > 1
        jobs = Channel{Int}(length(remaining_indices))
        for index in remaining_indices
            put!(jobs, index)
        end
        close(jobs)

        tasks = [
            Threads.@spawn begin
                for index in jobs
                    all_rows[index] = process_sample(index)
                end
            end
            for _ in 1:worker_count
        ]
        foreach(fetch, tasks)
    end

    write_features_csv(output_path, all_rows)
    return output_path
end
