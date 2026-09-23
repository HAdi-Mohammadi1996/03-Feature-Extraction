using TOML
using AMGX

config = TOML.parsefile(joinpath(@__DIR__, "config.toml"))

const SAMPLE_ID = config["SAMPLE_ID"]
const MAT_KEY = config["MAT_KEY"]
const STATIC_PHASE = config["STATIC_PHASE"]
static_properties = nothing
static_mask = nothing

const INPUT_FILE = joinpath(config["DATA_DIR"], SAMPLE_ID, "mat")
const OUTPUT_DIR = joinpath(config["DATA_DIR"], SAMPLE_ID, "features", "$SAMPLE_ID.csv")
const AMGX_DLL = config["AMGX_DLL"]

const DIRECTION = config["DIRECTION"]
const SPACINGS = (config["SPACING_X"], config["SPACING_Y"], config["SPACING_Z"])
const SSA_SIGMA = config["SSA_SIGMA"]

include(joinpath(@__DIR__, "src", "MicrostructureAnalysis.jl"))
import .MicrostructureAnalysis as MA

function main()

    files = sort(filter(f -> endswith(lowercase(f), ".mat"), readdir(INPUT_FILE; join=true)))
    isempty(files) && error("No .mat files found in $INPUT_FILE")

    rows = []

    AMGX.set_libAMGX_path(AMGX_DLL)
    AMGX.initialize()

    try
        for file in files 
            println("\n processing: ", basename(file))
            C = MA.load_microstructure(file; key=MAT_KEY)

            props = Vector{Any}(undef, 3)

            for p in 1:3
                if p == STATIC_PHASE && !isnothing(static_properties)
                    @assert (C .== p) == static_mask

                    props[p] = static_properties
                    continue
                end

                props[p] = (
                    vf = MA.volume_fraction(C, p),
                    ssa = MA.specific_surface_area(C, p; spacing=SPACINGS, σ=SSA_SIGMA),
                    tau = MA.physical_tortuosity(C, p; direction=DIRECTION, spacings=SPACINGS, manage_amgx=false)
                )

                if p == STATIC_PHASE
                    static_properties = props[p]
                    static_mask = C .== p
                end
            end

            tpb = MA.total_tpb_density(C; spacing=SPACINGS)

            push!(rows, (file=basename(file), vf1=props[1].vf, vf2=props[2].vf, vf3=props[3].vf,
                         ssa1=props[1].ssa, ssa2=props[2].ssa, ssa3=props[3].ssa,
                         tau1=props[1].tau, tau2=props[2].tau, tau3=props[3].tau, tpb=tpb))
        end
        
        mkpath(dirname(OUTPUT_DIR))
        MA.write_features_csv(OUTPUT_DIR, rows)

        println("\nSaved: ", OUTPUT_DIR)

    finally
        AMGX.finalize_plugins()
        AMGX.finalize()
    end
    
end

main()
