function geometric_tortuosity_mask(mask::AbstractArray{Bool,3}, dir::Int)
    dir in (1, 2, 3) || error("dir must be 1, 2, or 3")

    any(mask) || return Inf

    dims = size(mask)
    nx, ny, nz = dims
    nxy = nx * ny
    mask_values = vec(mask)
    unreachable = typemax(Int32)
    distance_values = fill(unreachable, length(mask))
    queue = Int[]

    outlet = dims[dir]
    if dir == 1
        for z in 1:nz, y in 1:ny
            index = outlet + (y - 1) * nx + (z - 1) * nxy
            if mask_values[index]
                distance_values[index] = 0
                push!(queue, index)
            end
        end
    elseif dir == 2
        for z in 1:nz, x in 1:nx
            index = x + (outlet - 1) * nx + (z - 1) * nxy
            if mask_values[index]
                distance_values[index] = 0
                push!(queue, index)
            end
        end
    else
        for y in 1:ny, x in 1:nx
            index = x + (y - 1) * nx + (outlet - 1) * nxy
            if mask_values[index]
                distance_values[index] = 0
                push!(queue, index)
            end
        end
    end
    isempty(queue) && return Inf

    head = 1
    while head <= length(queue)
        index = queue[head]
        head += 1
        next_distance = distance_values[index] + 1

        shifted = index - 1
        x = shifted % nx + 1
        y = (shifted ÷ nx) % ny + 1
        z = shifted ÷ nxy + 1

        if x < nx
            visit = index + 1
            if mask_values[visit] && distance_values[visit] == unreachable
                distance_values[visit] = next_distance
                push!(queue, visit)
            end
        end
        if x > 1
            visit = index - 1
            if mask_values[visit] && distance_values[visit] == unreachable
                distance_values[visit] = next_distance
                push!(queue, visit)
            end
        end
        if y < ny
            visit = index + nx
            if mask_values[visit] && distance_values[visit] == unreachable
                distance_values[visit] = next_distance
                push!(queue, visit)
            end
        end
        if y > 1
            visit = index - nx
            if mask_values[visit] && distance_values[visit] == unreachable
                distance_values[visit] = next_distance
                push!(queue, visit)
            end
        end
        if z < nz
            visit = index + nxy
            if mask_values[visit] && distance_values[visit] == unreachable
                distance_values[visit] = next_distance
                push!(queue, visit)
            end
        end
        if z > 1
            visit = index - nxy
            if mask_values[visit] && distance_values[visit] == unreachable
                distance_values[visit] = next_distance
                push!(queue, visit)
            end
        end
    end

    total_distance = 0.0
    inlet_count = 0
    if dir == 1
        for z in 1:nz, y in 1:ny
            index = 1 + (y - 1) * nx + (z - 1) * nxy
            if mask_values[index] && distance_values[index] != unreachable
                total_distance += distance_values[index]
                inlet_count += 1
            end
        end
    elseif dir == 2
        for z in 1:nz, x in 1:nx
            index = x + (z - 1) * nxy
            if mask_values[index] && distance_values[index] != unreachable
                total_distance += distance_values[index]
                inlet_count += 1
            end
        end
    else
        for y in 1:ny, x in 1:nx
            index = x + (y - 1) * nx
            if mask_values[index] && distance_values[index] != unreachable
                total_distance += distance_values[index]
                inlet_count += 1
            end
        end
    end
    inlet_count == 0 && return Inf

    straight_distance = max(dims[dir] - 1, 1)
    return (total_distance / inlet_count) / straight_distance
end

function geometric_tortuosity(
    C::AbstractArray{<:Integer,3},
    phase::Integer,
    dir::Int;
    voxel_size=0.1,
    percolation::Union{Nothing,PercolationResult}=nothing,
)
    voxel_size > 0 || error("voxel_size must be positive")
    mask = percolation === nothing ? C .== phase : percolation.mask
    return geometric_tortuosity_mask(mask, dir)
end
