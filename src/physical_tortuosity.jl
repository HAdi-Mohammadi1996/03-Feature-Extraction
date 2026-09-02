const AMGX_CONFIG_PATH = normpath(joinpath(@__DIR__, "..", "amgx.json"))
const config_string = read(AMGX_CONFIG_PATH, String)

function createD(connected_mask)
    return padarray(connected_mask, Fill(false, ntuple(_ -> 1, ndims(connected_mask)))) 
end

@inline function face_D(D, I)
    @inbounds begin
        if ndims(D) == 3
            i, j, k = I.I
            De = D[i, j, k] & D[i+1, j, k]
            Dw = D[i, j, k] & D[i-1, j, k]
            Dn = D[i, j, k] & D[i, j+1, k]
            Ds = D[i, j, k] & D[i, j-1, k]
            Dt = D[i, j, k] & D[i, j, k+1]
            Db = D[i, j, k] & D[i, j, k-1]

            return De, Dw, Dn, Ds, Dt, Db
        else
            i, j = I.I
            De = D[i, j] & D[i+1, j]
            Dw = D[i, j] & D[i-1, j]
            Dn = D[i, j] & D[i, j+1]
            Ds = D[i, j] & D[i, j-1]

            return De, Dw, Dn, Ds
        end
    end
end

function phase_ids(connected_mask)
    ids = zeros(Int32, size(connected_mask))
    n = 0

    @inbounds for i in eachindex(connected_mask)
        if connected_mask[i]
            n += 1
            ids[i] = n
        end
    end
    return ids, n
end

function matrix_assembly(C, phase, direction, spacings)

    direction ∈ 1:ndims(C) || error("Invalid trasport direction")

    C_connected = removing_isolated_particles(C, phase, direction)

    ids, N = phase_ids(C_connected)
    D = createD(C_connected)

    Nx = size(C, 1)
    Ny = size(C, 2)
    Nz = ndims(C) == 3 ? size(C, 3) : 1

    dx = spacings[1]
    dy = spacings[2]
    dz = ndims(C) == 3 ? spacings[3] : 1.0

    I = Int[]
    J = Int[]
    V = Float64[]

    @inbounds for index in CartesianIndices(C_connected)

        p = ids[index]
        p == 0 && continue
        if ndims(C) == 3
            i, j, k = index.I
            De, Dw, Dn, Ds, Dt, Db = face_D(D, index)
            diag = De/dx^2 + Dw/dx^2 + Dn/dy^2 + Ds/dy^2 + Dt/dz^2 + Db/dz^2
            push!(I, p); push!(J, p); push!(V, diag)

            if De
                pe = ids[i+1, j, k]
                push!(I, p); push!(J, pe); push!(V, -1.0/dx^2)
            end

            if Dw
                pw = ids[i-1, j, k]
                push!(I, p); push!(J, pw); push!(V, -1.0/dx^2)
            end

            if Dn
                pn = ids[i, j+1, k]
                push!(I, p); push!(J, pn); push!(V, -1.0/dy^2)
            end

            if Ds
                ps = ids[i, j-1, k]
                push!(I, p); push!(J, ps); push!(V, -1.0/dy^2)
            end

            if Dt
                pt = ids[i, j, k+1]
                push!(I, p); push!(J, pt); push!(V, -1.0/dz^2)
            end

            if Db
                pb = ids[i, j, k-1]
                push!(I, p); push!(J, pb); push!(V, -1.0/dz^2)
            end
        else
            i, j = index.I
            De, Dw, Dn, Ds = face_D(D, index)
            diag = De/dx^2 + Dw/dx^2 + Dn/dy^2 + Ds/dy^2
            push!(I, p); push!(J, p); push!(V, diag)

            if De
                pe = ids[i+1, j]
                push!(I, p); push!(J, pe); push!(V, -1.0/dx^2)
            end

            if Dw
                pw = ids[i-1, j]
                push!(I, p); push!(J, pw); push!(V, -1.0/dx^2)
            end

            if Dn
                pn = ids[i, j+1]
                push!(I, p); push!(J, pn); push!(V, -1.0/dy^2)
            end

            if Ds
                ps = ids[i, j-1]
                push!(I, p); push!(J, ps); push!(V, -1.0/dy^2)
            end
        end
    end

    # Now adding the Drichlet BC effect
    b = zeros(Float64, N)

    if ndims(C) == 3
        if direction == 1
            @inbounds for k in 1:Nz, j in 1:Ny
                p = ids[1, j, k]
                if p != 0
                    push!(I, p); push!(J, p); push!(V, 2.0/dx^2)
                    b[p] = 2.0/dx^2
                end
            end
           @inbounds for k in 1:Nz, j in 1:Ny
                p = ids[Nx, j, k]
                if p != 0
                    push!(I, p); push!(J, p); push!(V, 2.0/dx^2)
                end
            end
        elseif direction == 2
            @inbounds for k in 1:Nz, i in 1:Nx
                p = ids[i, 1, k]
                if p != 0
                    push!(I, p); push!(J, p); push!(V, 2.0/dy^2)
                    b[p] = 2.0/dy^2
                end
            end
           @inbounds for k in 1:Nz, i in 1:Nx
                p = ids[i, Ny, k]
                if p != 0
                    push!(I, p); push!(J, p); push!(V, 2.0/dy^2)
                end
            end
        else
            @inbounds for j in 1:Ny, i in 1:Nx
                p = ids[i, j, 1]
                if p != 0
                    push!(I, p); push!(J, p); push!(V, 2.0/dz^2)
                    b[p] = 2.0/dz^2
                end
            end
           @inbounds for j in 1:Ny, i in 1:Nx
                p = ids[i, j, Nz]
                if p != 0
                    push!(I, p); push!(J, p); push!(V, 2.0/dz^2)
                end
            end
        end
    else
        if direction == 1
            @inbounds for j in 1:Ny
                p = ids[1, j]
                if p != 0
                    push!(I, p); push!(J, p); push!(V, 2.0/dx^2)
                    b[p] = 2.0/dx^2
                end
            end
           @inbounds for j in 1:Ny
                p = ids[Nx, j]
                if p != 0
                    push!(I, p); push!(J, p); push!(V, 2.0/dx^2)
                end
            end
        elseif direction == 2
            @inbounds for i in 1:Nx
                p = ids[i, 1]
                if p != 0
                    push!(I, p); push!(J, p); push!(V, 2.0/dy^2)
                    b[p] = 2.0/dy^2
                end
            end
           @inbounds for i in 1:Nx
                p = ids[i, Ny]
                if p != 0
                    push!(I, p); push!(J, p); push!(V, 2.0/dy^2)
                end
            end
        end
    end

    A = sparse(I, J, V, N, N)
    return A, b, ids, C_connected
end

function calculate_tortuosity(C, C_connected, ϕ, phase, spacings, direction)

    # τ = ε * D0 / D_eff, where D0 = 1.0
    # Here, along direction="i" : D_eff = -<j_i> / (ΔC/L)
    # <j_i> = is the mean of slice by slice flux calculation along direction i
    # L = length from inlet to outlet
    # ΔC = difference in inlet and outlet concerntration which is 1.0 here

    Ndir = size(C, direction)
    h = spacings[direction]
    L = Ndir * h

    Area = length(C) / Ndir

    flux = zeros(Float64, Ndir-1)

    @inbounds for n in 1:(Ndir-1)

        C1 = selectdim(C_connected, direction, n)
        C2 = selectdim(C_connected, direction, n+1)

        ϕ1 = selectdim(ϕ, direction, n)
        ϕ2 = selectdim(ϕ, direction, n+1)

        Q = 0.0

        @inbounds for i in eachindex(C1)
            if C1[i] && C2[i]
                Q += (ϕ1[i] - ϕ2[i])/h
            end
        end
        flux[n] = Q / Area
    end

    mean_flux = sum(flux) / (Ndir-1)

    Deff = L * mean_flux
    ε = volume_fraction(C, phase)
    τ = ε / Deff
    return τ
end

function solve_amgx(A, b, manage_amgx)

    if manage_amgx
        AMGX.set_libAMGX_path(MA.AMGX_DLL)
        AMGX.initialize()
    end
    
    config    = AMGX.Config(config_string)
    resources = AMGX.Resources(config)
    matrix = AMGX.AMGXMatrix(resources, AMGX.dDDI)
    rhs    = AMGX.AMGXVector(resources, AMGX.dDDI)
    x_amgx = AMGX.AMGXVector(resources, AMGX.dDDI)
    solver = AMGX.Solver(resources, AMGX.dDDI, config)

    try
        @assert issymmetric(A)
        # A is symmetric, so its CSC storage can be used as CSR
        # after converting Julia's 1-based indices to AMGX's 0-based indices.
        # Convert Julia CSC -> GPU CSR
        @assert size(A, 1) <= typemax(Cint)
        @assert nnz(A) <= typemax(Cint)

        row_ptr = Cint.(A.colptr .- 1)
        col_idx = Cint.(A.rowval .- 1)
        values  = A.nzval

        @assert row_ptr[1] == 0
        @assert row_ptr[end] == nnz(A)  
        
        # Upload matrix and RHS to AMGX
        println("Uploading to AMGX...")
        AMGX.upload!(matrix, row_ptr, col_idx, values)
        AMGX.upload!(rhs, b)
        # Initial guess x = 0
        AMGX.set_zero!(x_amgx, length(b))
    
        # Solver setup
        println("AMGX setup...")
        AMGX.setup!(solver, matrix)

        # Solve Ax = b
        println("AMGX solve...")
        AMGX.solve!(x_amgx, solver, rhs)

        status = AMGX.get_status(solver)
        niter  = AMGX.get_iterations_number(solver)

        println("AMGX status     = ", status)
        println("AMGX iterations = ", niter)

        status == AMGX.SUCCESS ||
            @warn "AMGX did not converge successfully" status

        # GPU -> CPU
        x = Vector(x_amgx)

        return x

    finally
        # IMPORTANT: close in dependency order
        close(solver)
        close(x_amgx)
        close(rhs)
        close(matrix)
        close(resources)
        close(config)
        if manage_amgx
            AMGX.finalize_plugins()
            AMGX.finalize()
        end
    end
end

function physical_tortuosity(C, phase; direction = 1, spacings=ntuple(_ -> 1.0, ndims(C)), manage_amgx=true)

    if !is_percolated(C, phase, direction)
        println("Phase $phase does not percolate in direction $direction.")
        println("τ = Inf")
        return Inf
    end

    A, b, ids, C_connected = matrix_assembly(C, phase, direction, spacings)

    println("\n--- AMGX solve ---")
    x = solve_amgx(A, b, manage_amgx)
    ϕ = zeros(Float64, size(C))

    @inbounds for i in eachindex(ids)
        p = ids[i]
        if p != 0
            ϕ[i] = x[p]
        end
    end

    τ = calculate_tortuosity(C, C_connected, ϕ, phase, spacings, direction)

    if !isfinite(τ) || τ < 1.0
        @warn "Invalid tortuosity obtained" phase direction τ
        return Inf
    end

    return τ
end
