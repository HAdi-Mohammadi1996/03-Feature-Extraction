using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using SOCFeatureExtraction

function parse_bool(s)
    lowercase(string(s)) in ("true", "1", "yes", "y")
end

input_dir = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "..")
output_dir = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "..", "output")
voxel_size = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.1
Nrays = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 5000
isotropic = length(ARGS) >= 5 ? parse_bool(ARGS[5]) : true
direction = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 1

cfg = Config(voxel_size=voxel_size, Nrays=Nrays, isotropic=isotropic,
    direction=direction)

files = filter(path -> lowercase(splitext(path)[2]) == ".mat", readdir(input_dir; join=true))
isempty(files) && error("No .mat files found in $input_dir")

mkpath(output_dir)
for file in files
    println("Processing ", file)
    out = run_sample(file; output_dir=output_dir, config=cfg)
    println("Wrote ", out)
end
