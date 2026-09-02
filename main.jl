using TOML
using AMGX

config = TOML.parsefile(joinpath(@__DIR__, "config.toml"))

const SAMPLE_ID = config["SAMPLE_ID"]
const MAT_KEY = config["MAT_KEY"]

const INPUT_FILE = joinpath(@__DIR__, SAMPLE_ID)
const OUTPUT_DIR = joinpath(@__DIR__, "output", "$SAMPLE_ID.csv")
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
            vf1 = MA.volume_fraction(C, 1)
            vf2 = MA.volume_fraction(C, 2)
            vf3 = MA.volume_fraction(C, 3)
        
            ssa1 = MA.specific_surface_area(C, 1; spacing=SPACINGS, σ=SSA_SIGMA)
            ssa2 = MA.specific_surface_area(C, 2; spacing=SPACINGS, σ=SSA_SIGMA)
            ssa3 = MA.specific_surface_area(C, 3; spacing=SPACINGS, σ=SSA_SIGMA)

            tpb = MA.total_tpb_density(C; spacing=SPACINGS)

            # Assuming isotropic samples, therefore, only one direction is used for tau
            tau1 = MA.physical_tortuosity(C, 1; direction=DIRECTION, spacings=SPACINGS, manage_amgx=false)
            tau2 = MA.physical_tortuosity(C, 2; direction=DIRECTION, spacings=SPACINGS, manage_amgx=false)
            tau3 = MA.physical_tortuosity(C, 3; direction=DIRECTION, spacings=SPACINGS, manage_amgx=false)

            push!(rows, (file=basename(file), vf1=vf1, vf2=vf2, vf3=vf3,
                         ssa1=ssa1, ssa2=ssa2, ssa3=ssa3, tau1=tau1, tau2=tau2, tau3=tau3, tpb=tpb))
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
