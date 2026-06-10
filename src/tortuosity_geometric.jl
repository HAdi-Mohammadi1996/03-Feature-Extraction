function heuristic_to_outlet(I, dims, dir)
    return max(0.0, dims[dir] - axis_value(I, dir))
end

function geometric_tortuosity(C::AbstractArray{<:Integer,3}, phase::Integer, dir::Int;
    voxel_size=0.1, method=:astar)
    method == :astar || return geometric_tortuosity_dijkstra(C, phase, dir;
        voxel_size=voxel_size)

    dims = size(C)
    mask = C .== phase
    any(mask) || return Inf
    start_voxels = [LinearIndices(dims)[I] for I in CartesianIndices(mask)
        if mask[I] && axis_value(Tuple(I), dir) == 1]
    isempty(start_voxels) && return Inf

    dist = fill(Inf, dims)
    heap = Tuple{Float64,Int}[]
    for idx in start_voxels
        I = coord(idx, dims)
        dist[I...] = 0.0
        minheap_push!(heap, (heuristic_to_outlet(I, dims, dir), idx))
    end

    while !isempty(heap)
        _, idx = minheap_pop!(heap)
        I = coord(idx, dims)
        g = dist[I...]
        axis_value(I, dir) == dims[dir] && return g / max(dims[dir] - 1, 1)
        for (dx, dy, dz) in NEIGHBOR26
            J = (I[1] + dx, I[2] + dy, I[3] + dz)
            inbounds3(J, dims) || continue
            mask[J...] || continue
            step = sqrt(dx^2 + dy^2 + dz^2)
            alt = g + step
            if alt < dist[J...]
                dist[J...] = alt
                f = alt + heuristic_to_outlet(J, dims, dir)
                minheap_push!(heap, (f, lin(J, dims)))
            end
        end
    end
    return Inf
end

function geometric_tortuosity_dijkstra(C::AbstractArray{<:Integer,3}, phase::Integer,
    dir::Int; voxel_size=0.1)
    dims = size(C)
    mask = C .== phase
    any(mask) || return Inf
    start_voxels = [LinearIndices(dims)[I] for I in CartesianIndices(mask)
        if mask[I] && axis_value(Tuple(I), dir) == 1]
    isempty(start_voxels) && return Inf

    dist = fill(Inf, dims)
    heap = Tuple{Float64,Int}[]
    for idx in start_voxels
        I = coord(idx, dims)
        dist[I...] = 0.0
        minheap_push!(heap, (0.0, idx))
    end

    while !isempty(heap)
        g, idx = minheap_pop!(heap)
        I = coord(idx, dims)
        g > dist[I...] && continue
        axis_value(I, dir) == dims[dir] && return g / max(dims[dir] - 1, 1)
        for (dx, dy, dz) in NEIGHBOR26
            J = (I[1] + dx, I[2] + dy, I[3] + dz)
            inbounds3(J, dims) || continue
            mask[J...] || continue
            step = sqrt(dx^2 + dy^2 + dz^2)
            alt = g + step
            if alt < dist[J...]
                dist[J...] = alt
                minheap_push!(heap, (alt, lin(J, dims)))
            end
        end
    end
    return Inf
end
