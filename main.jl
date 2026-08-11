
INPUT_FILE = joinpath(@__DIR__, "inputs", "2.mat")
# OUTPUT_DIR = joinpath(@__DIR__, "output")
MAT_KEY = "C"

include(joinpath(@__DIR__, "src", "MicrostructureAnalysis.jl"))
using .MicrostructureAnalysis
# using CUDA
# CUDA.functional(true) || error("CUDA is not functional.")

isfile(INPUT_FILE) || error("Input file does not exist: $INPUT_FILE")

println("Loading: ", INPUT_FILE)
C = load_microstructure(INPUT_FILE; key=MAT_KEY)
phases = unique(C)
# sample = splitext(basename(INPUT_FILE))[1]

println("Microstructure size: ", size(C))
# C = CuArray(C)
println("Calculating features...")
vf  = volume_fraction(C, 1)
ssa = specific_surface_area(C, 1; spacing=(1.0, 1.0, 1.0))
tpb = total_tpb_density(C; spacing=(1.0, 1.0, 1.0))
println("vf1 = $vf")
println("ssa_1 = $ssa")
println("TPB = $tpb")

# row = extract_features(C; sample=sample, config=config)
# output_path = joinpath(OUTPUT_DIR, sample * "_features.csv")
# write_features_csv(output_path, [row])

# println("Finished.")
# println("Results saved to: ", output_path)
