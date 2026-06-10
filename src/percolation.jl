function connected_components_mask(mask::AbstractArray{Bool,3})
    dims = size(mask)
    labels = zeros(Int, dims)
    clusters = Vector{Vector{Int}}()
    current = 0
    queue = Int[]

    for I in CartesianIndices(mask)
        mask[I] || continue
        labels[I] == 0 || continue
        current += 1
        component = Int[]
        empty!(queue)
        labels[I] = current
        push!(queue, LinearIndices(dims)[I])
        head = 1
        while head <= length(queue)
            idx = queue[head]
            head += 1
            push!(component, idx)
            x, y, z = Tuple(CartesianIndices(dims)[idx])
            for (dx, dy, dz) in FACE_DIRS
                J = (x + dx, y + dy, z + dz)
                inbounds3(J, dims) || continue
                mask[J...] || continue
                labels[J...] == 0 || continue
                labels[J...] = current
                push!(queue, lin(J, dims))
            end
        end
        push!(clusters, component)
    end
    return labels, clusters
end

function percolation_result(mask::AbstractArray{Bool,3}, dir::Int)
    dims = size(mask)
    total = count(mask)
    percmask = falses(dims)
    total == 0 && return PercolationResult(0.0, percmask)

    _, clusters = connected_components_mask(mask)
    for component in clusters
        touches_inlet = false
        touches_outlet = false
        for idx in component
            I = coord(idx, dims)
            a = axis_value(I, dir)
            touches_inlet |= a == 1
            touches_outlet |= a == dims[dir]
            touches_inlet && touches_outlet && break
        end
        if touches_inlet && touches_outlet
            for idx in component
                percmask[coord(idx, dims)...] = true
            end
        end
    end
    return PercolationResult(count(percmask) / total, percmask)
end
