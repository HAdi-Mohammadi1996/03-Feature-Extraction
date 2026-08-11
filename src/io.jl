using MAT
function load_microstructure(path; key="C")
    C = matread(path)[key]
    ndims(C) ∈ (2, 3) || error("'$key' must be a 2D or 3D array")
    all(isinteger, C) || error("'$key' must contain integer phase labels")
    return Int8.(C)
end

function write_features_csv(path, rows)
    headers = propertynames(first(rows))
    open(path, "w") do f
        println(f, join(headers, ","))
        for row in rows
            println(f, join(row, ","))
        end
    end
end
