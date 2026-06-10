function derivative_at(A, I, dir, dx)
    dims = size(A)
    i, j, k = Tuple(I)
    if dir == 1
        i == 1 && return (A[i + 1, j, k] - A[i, j, k]) / dx
        i == dims[1] && return (A[i, j, k] - A[i - 1, j, k]) / dx
        return (A[i + 1, j, k] - A[i - 1, j, k]) / (2dx)
    elseif dir == 2
        j == 1 && return (A[i, j + 1, k] - A[i, j, k]) / dx
        j == dims[2] && return (A[i, j, k] - A[i, j - 1, k]) / dx
        return (A[i, j + 1, k] - A[i, j - 1, k]) / (2dx)
    else
        k == 1 && return (A[i, j, k + 1] - A[i, j, k]) / dx
        k == dims[3] && return (A[i, j, k] - A[i, j, k - 1]) / dx
        return (A[i, j, k + 1] - A[i, j, k - 1]) / (2dx)
    end
end

function surface_area(C::AbstractArray{<:Integer,3}, phase::Integer;
    voxel_size=0.1, sigma=1.0)
    indicator = Float64.(C .== phase)
    smooth = imfilter(indicator, KernelFactors.gaussian((sigma, sigma, sigma)), Pad(:replicate))
    total = 0.0
    for I in CartesianIndices(smooth)
        gx = derivative_at(smooth, I, 1, voxel_size)
        gy = derivative_at(smooth, I, 2, voxel_size)
        gz = derivative_at(smooth, I, 3, voxel_size)
        total += sqrt(gx^2 + gy^2 + gz^2)
    end
    return total * voxel_size^3 / (prod(size(C)) * voxel_size^3)
end

function surface_areas(C::AbstractArray{<:Integer,3}, phases=PHASES;
    voxel_size=0.1, sigma=1.0)
    return Dict(phase => surface_area(C, phase; voxel_size=voxel_size, sigma=sigma)
        for phase in phases)
end
