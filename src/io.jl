function load_microstructure(path::AbstractString; key="C")
    data = matread(path)
    haskey(data, key) || error("MAT file does not contain key '$key'")
    C = data[key]
    ndims(C) == 3 || error("Expected '$key' to be a 3D array")

    if eltype(C) <: Integer
        return Array(C)
    end

    all(isinteger, C) || error("Expected '$key' to contain integer phase labels")
    return round.(Int, C)
end

function write_features_csv(path::AbstractString, rows)
    isempty(rows) && error("Cannot write an empty feature table")
    headers = propertynames(first(rows))
    all(propertynames(row) == headers for row in rows) ||
        error("All feature rows must have the same columns")

    output_dir = dirname(abspath(path))
    mkpath(output_dir)
    temporary_path = tempname(output_dir)

    open(temporary_path, "w") do io
        println(io, join(string.(headers), ","))
        for row in rows
            println(io, join((csv_value(getproperty(row, h)) for h in headers), ","))
        end
    end
    mv(temporary_path, path; force=true)
    return path
end
