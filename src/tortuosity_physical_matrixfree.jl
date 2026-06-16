struct PhysicalTauOperator
    neighbors::Matrix{Int32}
    diagonal::Vector{Float64}
    inv_diagonal::Vector{Float64}
end

struct PhysicalTauThreadPlan
    ntasks::Int
    chunk::Int
    sums1::Vector{Float64}
    sums2::Vector{Float64}
end

function PhysicalTauThreadPlan(n::Int, threaded::Bool, threshold::Int)
    ntasks = threaded && n >= threshold ? min(Threads.nthreads(), n) : 1
    return PhysicalTauThreadPlan(ntasks, cld(n, ntasks), zeros(ntasks), zeros(ntasks))
end

@inline function task_range(plan::PhysicalTauThreadPlan, task::Int, n::Int)
    first = (task - 1) * plan.chunk + 1
    last = min(task * plan.chunk, n)
    return first:last
end

@inline function finish_sum(sums::Vector{Float64}, ntasks::Int)
    total = 0.0
    @inbounds for task in 1:ntasks
        total += sums[task]
    end
    return total
end

@inline function operator_row_value(
    neighbors::Matrix{Int32},
    diagonal::Vector{Float64},
    x::AbstractVector{Float64},
    row::Int,
)
    value = diagonal[row] * x[row]
    col = neighbors[1, row]; col == 0 || (value -= x[col])
    col = neighbors[2, row]; col == 0 || (value -= x[col])
    col = neighbors[3, row]; col == 0 || (value -= x[col])
    col = neighbors[4, row]; col == 0 || (value -= x[col])
    col = neighbors[5, row]; col == 0 || (value -= x[col])
    col = neighbors[6, row]; col == 0 || (value -= x[col])
    return value
end

function LinearAlgebra.mul!(
    y::AbstractVector{Float64},
    operator::PhysicalTauOperator,
    x::AbstractVector{Float64},
)
    neighbors = operator.neighbors
    diagonal = operator.diagonal

    @inbounds for row in eachindex(diagonal)
        y[row] = operator_row_value(neighbors, diagonal, x, row)
    end
    return y
end

function operator_and_dot!(
    y::Vector{Float64},
    operator::PhysicalTauOperator,
    x::Vector{Float64},
    plan::PhysicalTauThreadPlan,
)
    neighbors = operator.neighbors
    diagonal = operator.diagonal
    n = length(x)

    if plan.ntasks == 1
        total = 0.0
        @inbounds for row in 1:n
            value = operator_row_value(neighbors, diagonal, x, row)
            y[row] = value
            total += x[row] * value
        end
        return total
    end

    Threads.@threads :static for task in 1:plan.ntasks
        subtotal = 0.0
        @inbounds for row in task_range(plan, task, n)
            value = operator_row_value(neighbors, diagonal, x, row)
            y[row] = value
            subtotal += x[row] * value
        end
        plan.sums1[task] = subtotal
    end
    return finish_sum(plan.sums1, plan.ntasks)
end

function initialize_residual!(
    residual::Vector{Float64},
    operator::PhysicalTauOperator,
    x::Vector{Float64},
    rhs::Vector{Float64},
    plan::PhysicalTauThreadPlan,
)
    neighbors = operator.neighbors
    diagonal = operator.diagonal
    n = length(rhs)

    if plan.ntasks == 1
        residual_norm2 = 0.0
        rhs_norm2 = 0.0
        @inbounds for row in 1:n
            r = rhs[row] - operator_row_value(neighbors, diagonal, x, row)
            residual[row] = r
            residual_norm2 += r * r
            rhs_norm2 += rhs[row] * rhs[row]
        end
        return residual_norm2, rhs_norm2
    end

    Threads.@threads :static for task in 1:plan.ntasks
        residual_sum = 0.0
        rhs_sum = 0.0
        @inbounds for row in task_range(plan, task, n)
            r = rhs[row] - operator_row_value(neighbors, diagonal, x, row)
            residual[row] = r
            residual_sum += r * r
            rhs_sum += rhs[row] * rhs[row]
        end
        plan.sums1[task] = residual_sum
        plan.sums2[task] = rhs_sum
    end
    return finish_sum(plan.sums1, plan.ntasks), finish_sum(plan.sums2, plan.ntasks)
end

function initialize_direction!(
    z::Vector{Float64},
    direction::Vector{Float64},
    residual::Vector{Float64},
    inv_diagonal::Vector{Float64},
    plan::PhysicalTauThreadPlan,
)
    n = length(residual)

    if plan.ntasks == 1
        rz = 0.0
        @inbounds @simd for i in 1:n
            zi = residual[i] * inv_diagonal[i]
            z[i] = zi
            direction[i] = zi
            rz += residual[i] * zi
        end
        return rz
    end

    Threads.@threads :static for task in 1:plan.ntasks
        subtotal = 0.0
        @inbounds for i in task_range(plan, task, n)
            zi = residual[i] * inv_diagonal[i]
            z[i] = zi
            direction[i] = zi
            subtotal += residual[i] * zi
        end
        plan.sums1[task] = subtotal
    end
    return finish_sum(plan.sums1, plan.ntasks)
end

function update_solution_residual!(
    x::Vector{Float64},
    residual::Vector{Float64},
    direction::Vector{Float64},
    product::Vector{Float64},
    alpha::Float64,
    plan::PhysicalTauThreadPlan,
)
    n = length(x)

    if plan.ntasks == 1
        residual_norm2 = 0.0
        @inbounds @simd for i in 1:n
            x[i] += alpha * direction[i]
            r = residual[i] - alpha * product[i]
            residual[i] = r
            residual_norm2 += r * r
        end
        return residual_norm2
    end

    Threads.@threads :static for task in 1:plan.ntasks
        subtotal = 0.0
        @inbounds for i in task_range(plan, task, n)
            x[i] += alpha * direction[i]
            r = residual[i] - alpha * product[i]
            residual[i] = r
            subtotal += r * r
        end
        plan.sums1[task] = subtotal
    end
    return finish_sum(plan.sums1, plan.ntasks)
end

function precondition_and_dot!(
    z::Vector{Float64},
    residual::Vector{Float64},
    inv_diagonal::Vector{Float64},
    plan::PhysicalTauThreadPlan,
)
    n = length(residual)

    if plan.ntasks == 1
        rz = 0.0
        @inbounds @simd for i in 1:n
            zi = residual[i] * inv_diagonal[i]
            z[i] = zi
            rz += residual[i] * zi
        end
        return rz
    end

    Threads.@threads :static for task in 1:plan.ntasks
        subtotal = 0.0
        @inbounds for i in task_range(plan, task, n)
            zi = residual[i] * inv_diagonal[i]
            z[i] = zi
            subtotal += residual[i] * zi
        end
        plan.sums1[task] = subtotal
    end
    return finish_sum(plan.sums1, plan.ntasks)
end

function update_direction!(
    direction::Vector{Float64},
    z::Vector{Float64},
    beta::Float64,
    plan::PhysicalTauThreadPlan,
)
    n = length(direction)

    if plan.ntasks == 1
        @inbounds @simd for i in 1:n
            direction[i] = z[i] + beta * direction[i]
        end
        return direction
    end

    Threads.@threads :static for task in 1:plan.ntasks
        @inbounds for i in task_range(plan, task, n)
            direction[i] = z[i] + beta * direction[i]
        end
    end
    return direction
end

function physical_tau_system(
    percmask::AbstractArray{Bool,3},
    dir::Int,
    boundary_conductance_factor::Float64,
)
    dims = size(percmask)
    nx, ny, nz = dims
    nxy = nx * ny
    active_indices = findall(vec(percmask))
    n = length(active_indices)
    n <= typemax(Int32) || error("Too many active voxels for Int32 indexing")

    ids = zeros(Int32, length(percmask))
    for (row, idx) in enumerate(active_indices)
        ids[idx] = row
    end

    neighbors = zeros(Int32, 6, n)
    diagonal = zeros(Float64, n)
    inv_diagonal = zeros(Float64, n)
    rhs = zeros(Float64, n)
    initial = zeros(Float64, n)
    outlet_rows = Int32[]

    for (row, idx) in enumerate(active_indices)
        shifted = idx - 1
        x = shifted % nx + 1
        y = (shifted ÷ nx) % ny + 1
        z = shifted ÷ nxy + 1
        axis_position = dir == 1 ? x : dir == 2 ? y : z
        initial[row] = 1.0 - (axis_position - 0.5) / dims[dir]
        axis_position == dims[dir] && push!(outlet_rows, row)

        diag = 0.0
        rhs_row = 0.0

        if x < nx
            col = ids[idx + 1]
            if col > 0
                neighbors[1, row] = col
                diag += 1.0
            end
        elseif dir == 1
            diag += boundary_conductance_factor
        end
        if x > 1
            col = ids[idx - 1]
            if col > 0
                neighbors[2, row] = col
                diag += 1.0
            end
        elseif dir == 1
            diag += boundary_conductance_factor
            rhs_row += boundary_conductance_factor
        end

        if y < ny
            col = ids[idx + nx]
            if col > 0
                neighbors[3, row] = col
                diag += 1.0
            end
        elseif dir == 2
            diag += boundary_conductance_factor
        end
        if y > 1
            col = ids[idx - nx]
            if col > 0
                neighbors[4, row] = col
                diag += 1.0
            end
        elseif dir == 2
            diag += boundary_conductance_factor
            rhs_row += boundary_conductance_factor
        end

        if z < nz
            col = ids[idx + nxy]
            if col > 0
                neighbors[5, row] = col
                diag += 1.0
            end
        elseif dir == 3
            diag += boundary_conductance_factor
        end
        if z > 1
            col = ids[idx - nxy]
            if col > 0
                neighbors[6, row] = col
                diag += 1.0
            end
        elseif dir == 3
            diag += boundary_conductance_factor
            rhs_row += boundary_conductance_factor
        end

        diagonal[row] = diag
        inv_diagonal[row] = 1.0 / diag
        rhs[row] = rhs_row
    end

    return PhysicalTauOperator(neighbors, diagonal, inv_diagonal), rhs, initial, outlet_rows
end

function jacobi_pcg!(
    x::Vector{Float64},
    operator::PhysicalTauOperator,
    rhs::Vector{Float64};
    rtol::Float64,
    atol::Float64,
    maxiter::Int,
    threaded::Bool,
    thread_threshold::Int,
)
    n = length(rhs)
    residual = similar(rhs)
    z = similar(rhs)
    direction = similar(rhs)
    product = similar(rhs)
    plan = PhysicalTauThreadPlan(n, threaded, thread_threshold)

    residual_norm2, rhs_norm2 = initialize_residual!(residual, operator, x, rhs, plan)
    rhs_norm = sqrt(rhs_norm2)
    tolerance = max(atol, rtol * rhs_norm)
    residual_norm = sqrt(residual_norm2)
    if residual_norm <= tolerance
        relative_residual = rhs_norm == 0 ? residual_norm : residual_norm / rhs_norm
        return true, 0, relative_residual
    end

    inv_diagonal = operator.inv_diagonal
    rz = initialize_direction!(z, direction, residual, inv_diagonal, plan)

    for iteration in 1:maxiter
        denominator = operator_and_dot!(product, operator, direction, plan)
        denominator > 0 && isfinite(denominator) ||
            error("PCG failed because the diffusion operator is not positive definite")

        alpha = rz / denominator

        residual_norm = sqrt(update_solution_residual!(
            x,
            residual,
            direction,
            product,
            alpha,
            plan,
        ))
        if residual_norm <= tolerance
            relative_residual = rhs_norm == 0 ? residual_norm : residual_norm / rhs_norm
            return true, iteration, relative_residual
        end

        rz_new = precondition_and_dot!(z, residual, inv_diagonal, plan)
        beta = rz_new / rz
        update_direction!(direction, z, beta, plan)
        rz = rz_new
    end

    relative_residual = rhs_norm == 0 ? residual_norm : residual_norm / rhs_norm
    return false, maxiter, relative_residual
end

"""
    physical_tortuosity_matrixfree(C, phase, dir; kwargs...)

Calculate physical tortuosity without assembling a sparse matrix. The diffusion
equations are solved with Jacobi-preconditioned conjugate gradient on a compact
six-neighbor stencil.

Set `threaded=true` and start Julia with `--threads=N` to parallelize the PCG
stencil and vector-update loops. Small systems below `thread_threshold` run
serially to avoid thread overhead.

Set `return_info=true` to return the tortuosity, convergence flag, iteration
count, and final relative residual.
"""
function physical_tortuosity_matrixfree(
    C::AbstractArray{<:Integer,3},
    phase::Integer,
    dir::Int;
    voxel_size=0.1,
    D0=1.0,
    boundary_conductance_factor::Float64=2.0,
    rtol::Float64=1e-8,
    atol::Float64=0.0,
    maxiter::Int=10_000,
    threaded::Bool=true,
    thread_threshold::Int=50_000,
    return_info::Bool=false,
)
    dir in (1, 2, 3) || error("dir must be 1, 2, or 3")
    voxel_size > 0 || error("voxel_size must be positive")
    D0 > 0 || error("D0 must be positive")
    boundary_conductance_factor > 0 ||
        error("boundary_conductance_factor must be positive")
    rtol > 0 || error("rtol must be positive")
    atol >= 0 || error("atol must be non-negative")
    maxiter > 0 || error("maxiter must be positive")
    thread_threshold > 0 || error("thread_threshold must be positive")

    dims = size(C)
    eps_phase = count(==(phase), C) / length(C)
    if eps_phase == 0
        info = (tau=Inf, converged=true, iterations=0, relative_residual=0.0)
        return return_info ? info : info.tau
    end

    percolation = percolation_result(C .== phase, dir)
    if percolation.fraction == 0.0
        info = (tau=Inf, converged=true, iterations=0, relative_residual=0.0)
        return return_info ? info : info.tau
    end

    operator, rhs, concentration, outlet_rows = physical_tau_system(
        percolation.mask,
        dir,
        boundary_conductance_factor,
    )
    converged, iterations, relative_residual = jacobi_pcg!(
        concentration,
        operator,
        rhs;
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
        threaded=threaded,
        thread_threshold=thread_threshold,
    )
    converged || error(
        "Matrix-free physical tortuosity did not converge in $maxiter iterations " *
        "(relative residual = $relative_residual)",
    )

    gboundary = boundary_conductance_factor * D0 * voxel_size
    outlet_flux = 0.0
    @inbounds for row in outlet_rows
        outlet_flux += concentration[row]
    end
    outlet_flux *= gboundary

    area_voxels = dir == 1 ? dims[2] * dims[3] :
        dir == 2 ? dims[1] * dims[3] :
        dims[1] * dims[2]
    area = area_voxels * voxel_size^2
    length_sample = dims[dir] * voxel_size
    Deff = outlet_flux * length_sample / area
    tau = Deff <= 0 ? Inf : eps_phase * D0 / Deff

    info = (
        tau=tau,
        converged=converged,
        iterations=iterations,
        relative_residual=relative_residual,
    )
    return return_info ? info : info.tau
end
