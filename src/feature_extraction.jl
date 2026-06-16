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
    return FeatureConfig(
        float(voxel_size),
        Int(Nrays),
        float(min_chord),
        sort!(Int.(collect(phases))),
        Bool(use_all_directions),
        Int(direction),
        float(surface_sigma),
        float(D0),
        float(physical_boundary_factor),
        float(physical_rtol),
        Int(physical_maxiter),
        Bool(physical_threaded),
        Int(physical_thread_threshold),
        Int(rng_seed),
    )
end

directions(config::FeatureConfig) = config.use_all_directions ? (1, 2, 3) : (config.direction,)

function positive_phases(C::AbstractArray{<:Integer,3})
    return [phase for phase in sort!(collect(unique(vec(C)))) if phase > 0]
end

function extract_features(C::AbstractArray{<:Integer,3}; sample="sample", config=FeatureConfig())
    phases = config.phases
    dirs = directions(config)

    vf = volume_fractions(C, phases)
    chords = mean_chord_lengths(
        C,
        phases;
        Nrays=config.Nrays,
        min_chord=config.min_chord,
        voxel_size=config.voxel_size,
        rng=MersenneTwister(config.rng_seed),
    )
    specific_surface = surface_areas(
        C,
        phases;
        voxel_size=config.voxel_size,
        sigma=config.surface_sigma,
    )

    perc_fracs = Dict{Tuple{Int,Int},Float64}()
    for phase in phases, dir in dirs
        perc_fracs[(phase, dir)] = percolation_result(C .== phase, dir).fraction
    end

    gt = Dict(
        phase => mean_skip_inf([
            geometric_tortuosity(C, phase, dir; voxel_size=config.voxel_size)
            for dir in dirs
        ])
        for phase in phases
    )
    pt = Dict(
        phase => mean_skip_inf([
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
            )
            for dir in dirs
        ])
        for phase in phases
    )
    perc = Dict(
        phase => mean([perc_fracs[(phase, dir)] for dir in dirs])
        for phase in phases
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

    tpb_total = total_tpb_density(C; voxel_size=config.voxel_size)
    tpb_active = mean([
        active_tpb_density(C, dir; voxel_size=config.voxel_size)
        for dir in dirs
    ])
    append!(names, (:tpb, :atpb))
    append!(values, (tpb_total, tpb_active))

    return NamedTuple{Tuple(names)}(Tuple(values))
end

function run_sample(input_path::AbstractString; output_dir="output", mat_key="C", config=FeatureConfig())
    C = load_microstructure(input_path; key=mat_key)
    sample = splitext(basename(input_path))[1]
    row = extract_features(C; sample=sample, config=config)
    mkpath(output_dir)
    output_path = joinpath(output_dir, sample * "_features.csv")
    write_features_csv(output_path, [row])
    return output_path
end
