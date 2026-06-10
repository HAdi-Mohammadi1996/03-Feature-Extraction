function physical_tortuosity(C::AbstractArray{<:Integer,3}, phase::Integer, dir::Int;
    voxel_size=0.1, D0=1.0)
    dims = size(C)
    eps_phase = count(==(phase), C) / length(C)
    eps_phase == 0 && return Inf

    percmask = percolation_result(C .== phase, dir).mask
    count(percmask) == 0 && return Inf

    ids = zeros(Int, dims)
    coords = CartesianIndex{3}[]
    for I in CartesianIndices(percmask)
        if percmask[I]
            push!(coords, I)
            ids[I] = length(coords)
        end
    end

    n = length(coords)
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    rhs = zeros(Float64, n)
    gface = D0 * voxel_size

    for (row, CI) in enumerate(coords)
        I = Tuple(CI)
        diag = 0.0
        for (dx, dy, dz) in FACE_DIRS
            J = (I[1] + dx, I[2] + dy, I[3] + dz)
            if inbounds3(J, dims)
                col = ids[J...]
                if col > 0
                    diag += gface
                    push!(rows, row); push!(cols, col); push!(vals, -gface)
                end
            elseif dir == 1 && dx != 0 || dir == 2 && dy != 0 || dir == 3 && dz != 0
                inlet = (dir == 1 && dx < 0) || (dir == 2 && dy < 0) || (dir == 3 && dz < 0)
                diag += gface
                rhs[row] += inlet ? gface : 0.0
            end
        end
        push!(rows, row); push!(cols, row); push!(vals, diag)
    end

    A = sparse(rows, cols, vals, n, n)
    c = A \ rhs
    outlet_flux = 0.0
    for (row, CI) in enumerate(coords)
        I = Tuple(CI)
        if axis_value(I, dir) == dims[dir]
            outlet_flux += gface * c[row]
        end
    end

    other = [dims[i] for i in 1:3 if i != dir]
    area = prod(other) * voxel_size^2
    length_sample = dims[dir] * voxel_size
    Deff = outlet_flux * length_sample / area
    return Deff <= 0 ? Inf : eps_phase * D0 / Deff
end
