SAMPLE_ID = "inputs"
INPUT_FILE = joinpath(@__DIR__, SAMPLE_ID)
OUTPUT_DIR = joinpath(@__DIR__, "output", "$SAMPLE_ID.csv")
MAT_KEY = "C"

include(joinpath(@__DIR__, "src", "MicrostructureAnalysis.jl"))
import .MicrostructureAnalysis as MA
using AMGX

const AMGX_DLL = raw"C:\Users\r43341mm\AMGX\build\Release\amgxsh.dll"

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
        
            ssa1 = MA.specific_surface_area(C, 1; spacing=(1.0, 1.0, 1.0), σ=0.0)
            ssa2 = MA.specific_surface_area(C, 2; spacing=(1.0, 1.0, 1.0), σ=0.0)
            ssa3 = MA.specific_surface_area(C, 3; spacing=(1.0, 1.0, 1.0), σ=0.0)

            tpb = MA.total_tpb_density(C; spacing=(1.0, 1.0, 1.0))

            # Assuming isotropic samples, therefore, only one direction is used for tau
            tau1 = MA.physical_tortuosity(C, 1; direction=1, spacings=(1.0, 1.0, 1.0), manage_amgx=false)
            tau2 = MA.physical_tortuosity(C, 2; direction=1, spacings=(1.0, 1.0, 1.0), manage_amgx=false)
            tau3 = MA.physical_tortuosity(C, 3; direction=1, spacings=(1.0, 1.0, 1.0), manage_amgx=false)

            push!(rows, (file=basename(file), vf1=vf1, vf2=vf2, vf3=vf3,
                         ssa1=ssa1, ssa2=ssa2, ssa3=ssa3, tau1=tau1, tau2=tau2, tau3=tau3, tpb=tpb))
        end
        
        mkpath(dirname(OUTPUT_DIR))
        MA.write_features_csv(OUTPUT_DIR, rows)

        println("\nSaved: ", OUTPUT_DIR)

    finally
        AMGX.finalize()
    end
    
end

main()
