function block_phase(C, I, active12)
    v = C[I...]
    if active12 !== nothing && (v == 1 || v == 2) && !active12[v][I...]
        return 0
    end
    return v
end

is_tpb_edge(vals) = all(phase -> phase in vals, (1, 2, 3))

function count_tpb_edges(C::AbstractArray{<:Integer,3}; active12=nothing)
    nx, ny, nz = size(C)
    count_edges = 0

    for x in 1:nx, y in 1:ny-1, z in 1:nz-1
        vals = (block_phase(C, (x, y, z), active12),
            block_phase(C, (x, y + 1, z), active12),
            block_phase(C, (x, y, z + 1), active12),
            block_phase(C, (x, y + 1, z + 1), active12))
        count_edges += is_tpb_edge(vals) ? 1 : 0
    end
    for y in 1:ny, x in 1:nx-1, z in 1:nz-1
        vals = (block_phase(C, (x, y, z), active12),
            block_phase(C, (x + 1, y, z), active12),
            block_phase(C, (x, y, z + 1), active12),
            block_phase(C, (x + 1, y, z + 1), active12))
        count_edges += is_tpb_edge(vals) ? 1 : 0
    end
    for z in 1:nz, x in 1:nx-1, y in 1:ny-1
        vals = (block_phase(C, (x, y, z), active12),
            block_phase(C, (x + 1, y, z), active12),
            block_phase(C, (x, y + 1, z), active12),
            block_phase(C, (x + 1, y + 1, z), active12))
        count_edges += is_tpb_edge(vals) ? 1 : 0
    end
    return count_edges
end

function total_tpb_density(C::AbstractArray{<:Integer,3}; voxel_size=0.1)
    voxel_size > 0 || error("voxel_size must be positive")
    volume = length(C) * voxel_size^3
    return count_tpb_edges(C) * voxel_size / volume
end

function active_tpb_density(C::AbstractArray{<:Integer,3}, dir::Int; voxel_size=0.1)
    voxel_size > 0 || error("voxel_size must be positive")
    active12 = Dict(1 => percolation_result(C .== 1, dir).mask,
        2 => percolation_result(C .== 2, dir).mask)
    volume = length(C) * voxel_size^3
    return count_tpb_edges(C; active12=active12) * voxel_size / volume
end
