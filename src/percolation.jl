function boundary_reachable_mask(
    mask::AbstractArray{Bool,3},
    dir::Int,
    boundary::Int,
)
    dims = size(mask)
    boundary in (1, dims[dir]) || error("boundary must be 1 or size(mask, dir)")

    nx, ny, nz = dims
    nxy = nx * ny
    mask_values = vec(mask)
    reached = falses(dims)
    reached_values = vec(reached)
    queue = Int[]

    if dir == 1
        for z in 1:nz, y in 1:ny
            index = boundary + (y - 1) * nx + (z - 1) * nxy
            if mask_values[index]
                reached_values[index] = true
                push!(queue, index)
            end
        end
    elseif dir == 2
        for z in 1:nz, x in 1:nx
            index = x + (boundary - 1) * nx + (z - 1) * nxy
            if mask_values[index]
                reached_values[index] = true
                push!(queue, index)
            end
        end
    else
        for y in 1:ny, x in 1:nx
            index = x + (y - 1) * nx + (boundary - 1) * nxy
            if mask_values[index]
                reached_values[index] = true
                push!(queue, index)
            end
        end
    end

    head = 1
    while head <= length(queue)
        index = queue[head]
        head += 1

        shifted = index - 1
        x = shifted % nx + 1
        y = (shifted ÷ nx) % ny + 1
        z = shifted ÷ nxy + 1

        if x < nx
            visit = index + 1
            if mask_values[visit] && !reached_values[visit]
                reached_values[visit] = true
                push!(queue, visit)
            end
        end
        if x > 1
            visit = index - 1
            if mask_values[visit] && !reached_values[visit]
                reached_values[visit] = true
                push!(queue, visit)
            end
        end
        if y < ny
            visit = index + nx
            if mask_values[visit] && !reached_values[visit]
                reached_values[visit] = true
                push!(queue, visit)
            end
        end
        if y > 1
            visit = index - nx
            if mask_values[visit] && !reached_values[visit]
                reached_values[visit] = true
                push!(queue, visit)
            end
        end
        if z < nz
            visit = index + nxy
            if mask_values[visit] && !reached_values[visit]
                reached_values[visit] = true
                push!(queue, visit)
            end
        end
        if z > 1
            visit = index - nxy
            if mask_values[visit] && !reached_values[visit]
                reached_values[visit] = true
                push!(queue, visit)
            end
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
