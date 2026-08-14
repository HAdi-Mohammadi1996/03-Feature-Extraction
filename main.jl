
INPUT_FILE = joinpath(@__DIR__, "inputs", "3.mat")
# OUTPUT_DIR = joinpath(@__DIR__, "output")
MAT_KEY = "C"

include(joinpath(@__DIR__, "src", "MicrostructureAnalysis.jl"))
using .MicrostructureAnalysis
using AMGX

const AMGX_DLL = raw"C:\Users\r43341mm\AMGX\build\Release\amgxsh.dll"

function main()

    AMGX.set_libAMGX_path(AMGX_DLL)
    AMGX.initialize()

    try
        isfile(INPUT_FILE) || error("Input file does not exist: $INPUT_FILE")
        println("Loading: ", INPUT_FILE)
        C = load_microstructure(INPUT_FILE; key=MAT_KEY)
        phases = unique(C)
        println("Microstructure size: ", size(C))
        println("Calculating features...")
        vf  = volume_fraction(C, 1)
        ssa = specific_surface_area(C, 1; spacing=(1.0, 1.0, 1.0))
        tpb = total_tpb_density(C; spacing=(1.0, 1.0, 1.0))
        tau = physical_tortuosity(C, 1; direction=1, spacings=(1.0, 1.0, 1.0))

        println("vf1 = $vf")
        println("ssa_1 = $ssa")
        println("TPB = $tpb")
        println("tau phase 1 in direction 1 = $tau")

    finally
        AMGX.finalize()
    end
    
end

main()
