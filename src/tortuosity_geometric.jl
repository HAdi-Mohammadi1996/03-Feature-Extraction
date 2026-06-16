function geometric_tortuosity(C::AbstractArray{<:Integer,3}, phase::Integer, dir::Int;
    voxel_size=0.1)
    dir in (1, 2, 3) || error("dir must be 1, 2, or 3")
    voxel_size > 0 || error("voxel_size must be positive")

    dims = size(C)
    mask = C .== phase
    any(mask) || return Inf

    unreachable = typemax(Int32)
    distance = fill(unreachable, dims)
    queue = Int[]

    for I in CartesianIndices(mask)
        if mask[I] && axis_value(Tuple(I), dir) == dims[dir]
            distance[I] = 0
            push!(queue, LinearIndices(dims)[I])
        end
    end
    isempty(queue) && return Inf

    head = 1
    while head <= length(queue)
        idx = queue[head]
        head += 1
        I = coord(idx, dims)
        next_distance = distance[I...] + 1

        for (dx, dy, dz) in FACE_DIRS
            J = (I[1] + dx, I[2] + dy, I[3] + dz)
            inbounds3(J, dims) || continue
            mask[J...] || continue
            distance[J...] == unreachable || continue

            distance[J...] = next_distance
            push!(queue, lin(J, dims))
        end
    end

    total_distance = 0.0
    inlet_count = 0
    for I in CartesianIndices(mask)
        if mask[I] && axis_value(Tuple(I), dir) == 1 && distance[I] != unreachable
            total_distance += distance[I]
            inlet_count += 1
        end
    end
    inlet_count == 0 && return Inf

    straight_distance = max(dims[dir] - 1, 1)
    return (total_distance / inlet_count) / straight_distance
end
