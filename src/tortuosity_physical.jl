function transport_percolation_mask(mask::AbstractArray{Bool,3}, dir::Int)
    dims = size(mask)
    count(mask) == 0 && return falses(dims)

    from_inlet = boundary_reachable_mask(mask, dir, 1)
    from_outlet = boundary_reachable_mask(mask, dir, dims[dir])
    return from_inlet .& from_outlet
end

function physical_tortuosity(C::AbstractArray{<:Integer,3}, phase::Integer, dir::Int;
    voxel_size=0.1, D0=1.0, boundary_conductance_factor::Float64=2.0)
    dir in (1, 2, 3) || error("dir must be 1, 2, or 3")
    voxel_size > 0 || error("voxel_size must be positive")
    D0 > 0 || error("D0 must be positive")
    boundary_conductance_factor > 0 ||
        error("boundary_conductance_factor must be positive")

    dims = size(C)
    eps_phase = count(==(phase), C) / length(C)
    eps_phase == 0 && return Inf

    percmask = transport_percolation_mask(C .== phase, dir)
    count(percmask) == 0 && return Inf

    active_indices = findall(vec(percmask))
    n = length(active_indices)
    n <= typemax(Int32) || error("Too many active voxels for Int32 indexing")

    ids = zeros(Int32, dims)
    for (row, idx) in enumerate(active_indices)
        ids[idx] = row
    end

    rows = Int[]
    cols = Int[]
    vals = Float64[]
    sizehint!(rows, 7n)
    sizehint!(cols, 7n)
    sizehint!(vals, 7n)
    rhs = zeros(Float64, n)
    gface = D0 * voxel_size
    gboundary = boundary_conductance_factor * gface

    for (row, idx) in enumerate(active_indices)
        I = coord(idx, dims)
        diag = 0.0
        for (dx, dy, dz) in FACE_DIRS
            J = (I[1] + dx, I[2] + dy, I[3] + dz)
            step_dir = dx != 0 ? 1 : dy != 0 ? 2 : 3
            if inbounds3(J, dims)
                col = Int(ids[J...])
                if col > 0
                    diag += gface
                    push!(rows, row); push!(cols, col); push!(vals, -gface)
                end
            elseif step_dir == dir
                inlet = (dir == 1 && dx < 0) || (dir == 2 && dy < 0) || (dir == 3 && dz < 0)
                diag += gboundary
                rhs[row] += inlet ? gboundary : 0.0
            end
        end
        push!(rows, row); push!(cols, row); push!(vals, diag)
    end

    A = sparse(rows, cols, vals, n, n)
    c = cholesky(Symmetric(A)) \ rhs
    outlet_flux = 0.0
    for (row, idx) in enumerate(active_indices)
        I = coord(idx, dims)
        if axis_value(I, dir) == dims[dir]
            outlet_flux += gboundary * c[row]
        end
    end

    other = [dims[i] for i in 1:3 if i != dir]
    area = prod(other) * voxel_size^2
    length_sample = dims[dir] * voxel_size
    Deff = outlet_flux * length_sample / area
    return Deff <= 0 ? Inf : eps_phase * D0 / Deff
end
