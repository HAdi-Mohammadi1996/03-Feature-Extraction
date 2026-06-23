using CUDA

CUDA.allowscalar(false)

struct GPUPhysicalTauSystem{T}
    neighbors::CuMatrix{Int32}
    diagonal::CuVector{T}
    inv_diagonal::CuVector{T}
    rhs::CuVector{T}
    initial::CuVector{T}
    outlet_rows::CuVector{Int32}
end

mutable struct GPUPhysicalTauWorkspace{T}
    concentration::CuVector{T}
    residual::CuVector{T}
    preconditioned::CuVector{T}
    direction::CuVector{T}
    product::CuVector{T}
end

function GPUPhysicalTauSystem(
    operator::PhysicalTauOperator,
    rhs::Vector{Float64},
    initial::Vector{Float64},
    outlet_rows::Vector{Int32},
)
    return GPUPhysicalTauSystem(
        CuArray(operator.neighbors),
        CuArray(operator.diagonal),
        CuArray(operator.inv_diagonal),
        CuArray(rhs),
        CuArray(initial),
        CuArray(outlet_rows),
    )
end

function GPUPhysicalTauWorkspace(system::GPUPhysicalTauSystem{T}) where T
    n = length(system.rhs)
    return GPUPhysicalTauWorkspace(
        similar(system.initial),
        CUDA.zeros(T, n),
        CUDA.zeros(T, n),
        CUDA.zeros(T, n),
        CUDA.zeros(T, n),
    )
end

function free_gpu_physical_tau!(
    workspace::GPUPhysicalTauWorkspace,
    system::GPUPhysicalTauSystem,
)
    for array in (
        workspace.concentration,
        workspace.residual,
        workspace.preconditioned,
        workspace.direction,
        workspace.product,
        system.neighbors,
        system.diagonal,
        system.inv_diagonal,
        system.rhs,
        system.initial,
        system.outlet_rows,
    )
        CUDA.unsafe_free!(array)
    end
    return nothing
end

function gpu_operator_kernel!(y, neighbors, diagonal, x, n)
    row = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if row <= n
        @inbounds begin
            value = diagonal[row] * x[row]
            col = neighbors[1, row]; col == 0 || (value -= x[Int(col)])
            col = neighbors[2, row]; col == 0 || (value -= x[Int(col)])
            col = neighbors[3, row]; col == 0 || (value -= x[Int(col)])
            col = neighbors[4, row]; col == 0 || (value -= x[Int(col)])
            col = neighbors[5, row]; col == 0 || (value -= x[Int(col)])
            col = neighbors[6, row]; col == 0 || (value -= x[Int(col)])
            y[row] = value
        end
    end
    return
end

function gpu_operator!(
    y::CuVector{T},
    system::GPUPhysicalTauSystem{T},
    x::CuVector{T};
    threads=256,
) where T
    n = length(x)
    blocks = cld(n, threads)
    @cuda threads=threads blocks=blocks gpu_operator_kernel!(
        y,
        system.neighbors,
        system.diagonal,
        x,
        n,
    )
    return y
end

function gpu_jacobi_pcg!(
    workspace::GPUPhysicalTauWorkspace{T},
    system::GPUPhysicalTauSystem{T};
    rtol::Float64,
    atol::Float64,
    maxiter::Int,
) where T
    concentration = workspace.concentration
    residual = workspace.residual
    preconditioned = workspace.preconditioned
    direction = workspace.direction
    product = workspace.product

    copyto!(concentration, system.initial)
    gpu_operator!(product, system, concentration)
    @. residual = system.rhs - product

    rhs_norm = norm(system.rhs)
    tolerance = max(atol, rtol * rhs_norm)
    residual_norm = norm(residual)
    if residual_norm <= tolerance
        relative_residual = rhs_norm == 0 ? residual_norm : residual_norm / rhs_norm
        return (
            converged=true,
            iterations=0,
            relative_residual=Float64(relative_residual),
        )
    end

    @. preconditioned = residual * system.inv_diagonal
    copyto!(direction, preconditioned)
    residual_preconditioned = dot(residual, preconditioned)

    for iteration in 1:maxiter
        gpu_operator!(product, system, direction)
        denominator = dot(direction, product)
        denominator > 0 && isfinite(denominator) ||
            error("GPU PCG failed because the diffusion operator is not positive definite")

        alpha = residual_preconditioned / denominator
        @. concentration = concentration + alpha * direction
        @. residual = residual - alpha * product

        residual_norm = norm(residual)
        if residual_norm <= tolerance
            relative_residual = rhs_norm == 0 ? residual_norm : residual_norm / rhs_norm
            return (
                converged=true,
                iterations=iteration,
                relative_residual=Float64(relative_residual),
            )
        end

        @. preconditioned = residual * system.inv_diagonal
        new_residual_preconditioned = dot(residual, preconditioned)
        beta = new_residual_preconditioned / residual_preconditioned
        @. direction = preconditioned + beta * direction
        residual_preconditioned = new_residual_preconditioned
    end

    relative_residual = rhs_norm == 0 ? residual_norm : residual_norm / rhs_norm
    return (
        converged=false,
        iterations=maxiter,
        relative_residual=Float64(relative_residual),
    )
end

function physical_tau_from_concentration(
    concentration,
    outlet_rows,
    phase_fraction::Real,
    dims,
    dir::Int;
    voxel_size::Real,
    D0::Real,
    boundary_conductance_factor::Real,
)
    outlet_sum = sum(concentration[outlet_rows])
    gboundary = boundary_conductance_factor * D0 * voxel_size
    outlet_flux = Float64(outlet_sum) * gboundary

    area_voxels = dir == 1 ? dims[2] * dims[3] :
        dir == 2 ? dims[1] * dims[3] :
        dims[1] * dims[2]
    area = area_voxels * voxel_size^2
    length_sample = dims[dir] * voxel_size
    effective_diffusivity = outlet_flux * length_sample / area
    return effective_diffusivity <= 0 ?
        Inf :
        Float64(phase_fraction) * D0 / effective_diffusivity
end

function gpu_physical_tau!(
    workspace::GPUPhysicalTauWorkspace,
    system::GPUPhysicalTauSystem,
    phase_fraction::Real,
    dims,
    dir::Int;
    voxel_size=0.1,
    D0=1.0,
    boundary_conductance_factor=2.0,
    rtol=1e-8,
    atol=0.0,
    maxiter=10_000,
)
    convergence = gpu_jacobi_pcg!(
        workspace,
        system;
        rtol=Float64(rtol),
        atol=Float64(atol),
        maxiter=Int(maxiter),
    )
    convergence.converged || error(
        "GPU physical tortuosity did not converge in $maxiter iterations " *
        "(relative residual = $(convergence.relative_residual))",
    )

    tau = physical_tau_from_concentration(
        workspace.concentration,
        system.outlet_rows,
        phase_fraction,
        dims,
        dir;
        voxel_size=voxel_size,
        D0=D0,
        boundary_conductance_factor=boundary_conductance_factor,
    )
    return merge((tau=tau,), convergence)
end

function gpu_physical_tortuosity(
    percolation::PercolationResult,
    phase_fraction::Real,
    dims,
    dir::Int;
    voxel_size=0.1,
    D0=1.0,
    boundary_conductance_factor=2.0,
    rtol=1e-8,
    atol=0.0,
    maxiter=10_000,
)
    if phase_fraction == 0 || percolation.fraction == 0
        return (
            tau=Inf,
            converged=true,
            iterations=0,
            relative_residual=0.0,
            system_setup_seconds=0.0,
            gpu_upload_seconds=0.0,
            gpu_solve_seconds=0.0,
        )
    end

    setup_start = time_ns()
    operator, rhs, initial, outlet_rows = physical_tau_system(
        percolation.mask,
        dir,
        Float64(boundary_conductance_factor),
    )
    system_setup_seconds = (time_ns() - setup_start) / 1e9

    CUDA.synchronize()
    upload_start = time_ns()
    system = GPUPhysicalTauSystem(operator, rhs, initial, outlet_rows)
    workspace = GPUPhysicalTauWorkspace(system)
    CUDA.synchronize()
    gpu_upload_seconds = (time_ns() - upload_start) / 1e9

    try
        CUDA.synchronize()
        solve_start = time_ns()
        result = gpu_physical_tau!(
            workspace,
            system,
            phase_fraction,
            dims,
            dir;
            voxel_size=voxel_size,
            D0=D0,
            boundary_conductance_factor=boundary_conductance_factor,
            rtol=rtol,
            atol=atol,
            maxiter=maxiter,
        )
        CUDA.synchronize()
        gpu_solve_seconds = (time_ns() - solve_start) / 1e9
        return merge(
            result,
            (
                system_setup_seconds=system_setup_seconds,
                gpu_upload_seconds=gpu_upload_seconds,
                gpu_solve_seconds=gpu_solve_seconds,
            ),
        )
    finally
        free_gpu_physical_tau!(workspace, system)
    end
end

function cpu_physical_tau_from_system(
    operator::PhysicalTauOperator,
    rhs::Vector{Float64},
    initial::Vector{Float64},
    outlet_rows::Vector{Int32},
    phase_fraction::Real,
    dims,
    dir::Int;
    voxel_size=0.1,
    D0=1.0,
    boundary_conductance_factor=2.0,
    rtol=1e-8,
    atol=0.0,
    maxiter=10_000,
    max_threads=0,
)
    concentration = copy(initial)
    converged, iterations, relative_residual = jacobi_pcg!(
        concentration,
        operator,
        rhs;
        rtol=Float64(rtol),
        atol=Float64(atol),
        maxiter=Int(maxiter),
        threaded=true,
        thread_threshold=1,
        max_threads=Int(max_threads),
    )
    converged || error(
        "CPU physical tortuosity did not converge in $maxiter iterations " *
        "(relative residual = $relative_residual)",
    )

    tau = physical_tau_from_concentration(
        concentration,
        outlet_rows,
        phase_fraction,
        dims,
        dir;
        voxel_size=voxel_size,
        D0=D0,
        boundary_conductance_factor=boundary_conductance_factor,
    )
    return (
        tau=tau,
        converged=converged,
        iterations=iterations,
        relative_residual=relative_residual,
    )
end
