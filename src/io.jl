function load_microstructure(path::AbstractString; key="C")
    data = matread(path)
    haskey(data, key) || error("MAT file does not contain key '$key'")
    C = data[key]
    ndims(C) == 3 || error("Expected '$key' to be a 3D array")
    return Array{Int,3}(round.(Int, C))
end

function write_features_csv(path::AbstractString, rows)
    headers = (:sample, :phase, :volume_fraction, :mean_chord_length_um,
        :surface_area_um2, :geometric_tortuosity, :physical_tortuosity,
        :percolation_fraction, :total_tpb_density_um_inv2,
        :active_tpb_density_um_inv2)
    open(path, "w") do io
        println(io, join(string.(headers), ","))
        for row in rows
            println(io, join((csv_value(getproperty(row, h)) for h in headers), ","))
        end
    end
    return path
end
