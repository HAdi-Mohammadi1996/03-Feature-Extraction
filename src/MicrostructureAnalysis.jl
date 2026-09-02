module MicrostructureAnalysis
using TOML
using LinearAlgebra
using SparseArrays
using CUDA
using AMGX
using ImageFiltering
using ImageMorphology
using MAT

const CONFIG = TOML.parsefile(normpath(joinpath(@__DIR__, "..", "config.toml")))
const AMGX_DLL = CONFIG["AMGX_DLL"]

include("io.jl")
include("volume_fraction.jl")
include("surface_area.jl")
include("tpb.jl")
include("connectivity.jl")
include("physical_tortuosity.jl")


export  load_microstructure,
        write_features_csv,
        volume_fraction,
        specific_surface_area,
        total_tpb_density,
        physical_tortuosity,
        is_percolated,
        removing_isolated_particles
end