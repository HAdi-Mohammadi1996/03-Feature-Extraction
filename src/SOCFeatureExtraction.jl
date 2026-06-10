module SOCFeatureExtraction

using ImageFiltering
using LinearAlgebra
using MAT
using Random
using SparseArrays
using Statistics

include("utils.jl")
include("io.jl")
include("volume_fraction.jl")
include("percolation.jl")
include("chord_length.jl")
include("surface_area.jl")
include("tortuosity_geometric.jl")
include("tortuosity_physical.jl")
include("tpb.jl")

export Config,
    load_microstructure,
    extract_features,
    write_features_csv,
    run_sample,
    volume_fractions,
    mean_chord_lengths,
    surface_areas,
    percolation_result,
    geometric_tortuosity,
    physical_tortuosity,
    total_tpb_density,
    active_tpb_density

struct Config
    voxel_size::Float64
    Nrays::Int
    min_chord::Int
    isotropic::Bool
    direction::Int
    surface_sigma::Float64
    D0::Float64
    rng_seed::Int
end

function Config(; voxel_size=0.1, Nrays=5000, min_chord=4, isotropic=true,
    direction=1, surface_sigma=1.0, D0=1.0, rng_seed=1)
    @assert voxel_size > 0
    @assert Nrays > 0
    @assert min_chord >= 0
    @assert direction in (1, 2, 3)
    return Config(float(voxel_size), Nrays, min_chord, isotropic, direction,
        float(surface_sigma), float(D0), rng_seed)
end

directions(cfg::Config) = cfg.isotropic ? (cfg.direction,) : (1, 2, 3)

function extract_features(C::AbstractArray{<:Integer,3}; sample="sample", config=Config())
    phases = sort!(collect(unique(vec(C))))
    phases = [p for p in phases if p > 0]
    dirs = directions(config)
    vf = volume_fractions(C, phases)
    chords = mean_chord_lengths(C, phases; Nrays=config.Nrays,
        min_chord=config.min_chord, voxel_size=config.voxel_size,
        rng=MersenneTwister(config.rng_seed))
    sa = surface_areas(C, phases; voxel_size=config.voxel_size,
        sigma=config.surface_sigma)

    rows = NamedTuple[]
    perc_fracs = Dict{Tuple{Int,Int}, Float64}()

    for phase in phases, dir in dirs
        pr = percolation_result(C .== phase, dir)
        perc_fracs[(phase, dir)] = pr.fraction
    end

    for phase in phases
        gt = mean_skip_inf([geometric_tortuosity(C, phase, dir;
            voxel_size=config.voxel_size) for dir in dirs])
        pt = mean_skip_inf([physical_tortuosity(C, phase, dir;
            voxel_size=config.voxel_size, D0=config.D0) for dir in dirs])
        pf = mean([perc_fracs[(phase, dir)] for dir in dirs])
        push!(rows, feature_row(sample, string(phase), vf[phase], chords[phase],
            sa[phase], gt, pt, pf, missing, missing))
    end

    tpb_total = total_tpb_density(C; voxel_size=config.voxel_size)
    tpb_active = mean([active_tpb_density(C, dir; voxel_size=config.voxel_size)
        for dir in dirs])
    push!(rows, feature_row(sample, "global", missing, missing, missing,
        missing, missing, missing, tpb_total, tpb_active))
    return rows
end

function run_sample(input_path::AbstractString; output_dir="output", config=Config())
    C = load_microstructure(input_path)
    sample = splitext(basename(input_path))[1]
    rows = extract_features(C; sample=sample, config=config)
    mkpath(output_dir)
    output_path = joinpath(output_dir, sample * "_features.csv")
    write_features_csv(output_path, rows)
    return output_path
end

end
