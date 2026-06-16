function boundary_reachable_mask(
    mask::AbstractArray{Bool,3},
    dir::Int,
    boundary::Int,
)
    dims = size(mask)
    reached = falses(dims)
    queue = Int[]

    for I in CartesianIndices(mask)
        if mask[I] && axis_value(Tuple(I), dir) == boundary
            reached[I] = true
            push!(queue, LinearIndices(dims)[I])
        end
    end

    head = 1
    while head <= length(queue)
        idx = queue[head]
        head += 1
        I = coord(idx, dims)

        for (dx, dy, dz) in FACE_DIRS
            J = (I[1] + dx, I[2] + dy, I[3] + dz)
            inbounds3(J, dims) || continue
            mask[J...] || continue
            reached[J...] && continue
            reached[J...] = true
            push!(queue, lin(J, dims))
        end
    end

    return reached
end

function percolation_result(mask::AbstractArray{Bool,3}, dir::Int)
    dir in (1, 2, 3) || error("dir must be 1, 2, or 3")

    dims = size(mask)
    total = count(mask)
    percmask = falses(dims)
    total == 0 && return PercolationResult(0.0, percmask)

    from_inlet = boundary_reachable_mask(mask, dir, 1)
    from_outlet = boundary_reachable_mask(mask, dir, dims[dir])
    percmask .= from_inlet .& from_outlet

    return PercolationResult(count(percmask) / total, percmask)
end
