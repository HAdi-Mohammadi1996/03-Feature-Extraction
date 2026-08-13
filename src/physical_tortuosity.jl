using ImageFiltering
using SparseArrays
using Krylov
using LinearAlgebra
using CUDA
using CUDA.cuSPARSE

include("volume_fraction.jl")

function createD(C, phase)
    D = C .== phase
    return padarray(D, Fill(false, ntuple(_ -> 1, ndims(C)))) 
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

function matrix_assembely(C, phase, direction)

    direction ∈ 1:ndims(C) || error("Invalid trasport direction")

    N = length(C)
    Nx = size(C, 1)
    Ny = size(C, 2)
    Nz = ndims(C) == 3 ? size(C, 3) : 1

    I = Int[]
    J = Int[]
    V = Float64[]

    D = createD(C, phase)

    @inbounds for index in CartesianIndices(C)
        if ndims(C) == 3
            if C[index] != phase
                p = LinearIndices(C)[index]
                push!(I, p); push!(J, p); push!(V, 1.0)
                continue
            end

            if C[index] == phase
                # i, j, k = index.I
                # p =  i + (j-1)*Nx + (k-1)*Nx*Ny
                p = LinearIndices(C)[index]
                De, Dw, Dn, Ds, Dt, Db = face_D(D, index)
                diag = De + Dw + Dn + Ds + Dt + Db
                push!(I, p); push!(J, p); push!(V, diag)

                if De
                    pe = p + 1
                    push!(I, p); push!(J, pe); push!(V, -1.0)
                end

                if Dw
                    pw = p -1 
                    push!(I, p); push!(J, pw); push!(V, -1.0)
                end

                if Dn
                    pn = p + Nx
                    push!(I, p); push!(J, pn); push!(V, -1.0)
                end

                if Ds
                    ps = p - Nx
                    push!(I, p); push!(J, ps); push!(V, -1.0)
                end

                if Dt
                    pt = p + Nx*Ny
                    push!(I, p); push!(J, pt); push!(V, -1.0)
                end

                if Db
                    pb = p - Nx*Ny
                    push!(I, p); push!(J, pb); push!(V, -1.0)
                end
            end
    
        else
            if C[index] != phase
                p = LinearIndices(C)[index]
                push!(I, p); push!(J, p); push!(V, 1.0)
                continue
            end
            if C[index] == phase
                # i, j = index.I
                # p =  i + (j-1)*Nx
                p = LinearIndices(C)[index]
                De, Dw, Dn, Ds = face_D(D, index)
                diag = De + Dw + Dn + Ds
                push!(I, p); push!(J, p); push!(V, diag)

                if De
                    pe = p + 1
                    push!(I, p); push!(J, pe); push!(V, -1.0)
                end

                if Dw
                    pw = p -1 
                    push!(I, p); push!(J, pw); push!(V, -1.0)
                end

                if Dn
                    pn = p + Nx
                    push!(I, p); push!(J, pn); push!(V, -1.0)
                end

                if Ds
                    ps = p - Nx
                    push!(I, p); push!(J, ps); push!(V, -1.0)
                end
            end
        end
    end

    # Now adding the Drichlet BC effect
    b = zeros(Float64, N)

    if ndims(C) == 3
        if direction == 1
            @inbounds for k in 1:Nz, j in 1:Ny
                if C[CartesianIndex(1, j, k)] == phase
                    p = 1 + (j-1)*Nx + (k-1)*Nx*Ny
                    push!(I, p); push!(J, p); push!(V, 2.0)
                    b[p] = 2.0
                end
                if C[CartesianIndex(Nx, j, k)] == phase
                    p = Nx + (j-1)*Nx + (k-1)*Nx*Ny
                    push!(I, p); push!(J, p); push!(V, 2.0)
                end
            end
        elseif direction == 2
            @inbounds for k in 1:Nz, i in 1:Nx
                if C[CartesianIndex(i, 1, k)] == phase
                    p = i + (k-1)*Nx*Ny
                    push!(I, p); push!(J, p); push!(V, 2.0)
                    b[p] = 2.0
                end
                if C[CartesianIndex(i, Ny, k)] == phase
                    p = i + (Ny-1)*Nx + (k-1)*Nx*Ny
                    push!(I, p); push!(J, p); push!(V, 2.0)
                end
            end
        else
            @inbounds for j in 1:Ny, i in 1:Nx
                if C[CartesianIndex(i, j, 1)] == phase
                    p = i + (j-1)*Nx
                    push!(I, p); push!(J, p); push!(V, 2.0)
                    b[p] = 2.0
                end
                if C[CartesianIndex(i, j, Nz)] == phase
                    p = i + (j-1)*Nx + (Nz-1)*Nx*Ny
                    push!(I, p); push!(J, p); push!(V, 2.0)
                end
            end
        end
    else
        if direction == 1
            @inbounds for j in 1:Ny
                if C[CartesianIndex(1, j)] == phase
                    p = 1 + (j-1)*Nx
                    push!(I, p); push!(J, p); push!(V, 2.0)
                    b[p] = 2.0
                end
                if C[CartesianIndex(Nx, j)] == phase
                    p = Nx + (j-1)*Nx
                    push!(I, p); push!(J, p); push!(V, 2.0)
                end
            end
        elseif direction == 2
            @inbounds for i in 1:Nx
                if C[CartesianIndex(i, 1)] == phase
                    p = i
                    push!(I, p); push!(J, p); push!(V, 2.0)
                    b[p] = 2.0
                end
                if C[CartesianIndex(i, Ny)] == phase
                    p = i + (Ny-1)*Nx
                    push!(I, p); push!(J, p); push!(V, 2.0)
                end
            end
        end
    end

    A = sparse(I, J, V, N, N)
    return A, b
end

function calculate_tortuosity(C, ϕ, phase; direction=1,
                               spacing=ntuple(_ -> 1.0, ndims(C)))

    Ndir = size(C, direction)
    h = spacing[direction]

    # Physical transport length
    L = Ndir * h

    # Total cross-sectional measure
    transverse = filter(!=(direction), 1:ndims(C))
    Across = prod(size(C,d) * spacing[d] for d in transverse)

    # Area/length of one voxel face normal to transport direction
    Aface = prod(spacing[d] for d in transverse)

    Qin  = 0.0
    Qout = 0.0

    @inbounds for I in CartesianIndices(C)
        C[I] == phase || continue

        if I[direction] == 1
            Qin += 2.0 * (1.0 - ϕ[I]) / h * Aface
        end

        if I[direction] == Ndir
            Qout += 2.0 * ϕ[I] / h * Aface
        end
    end

    # Average inlet/outlet flux magnitude
    Q = (Qin + Qout) / 2

    Deff = Q * L / Across       # Δϕ = 1
    ε = volume_fraction(C, phase)

    τ = ε / Deff                # D0 = 1

    return τ, Deff, Qin, Qout
end

function physical_tortuosity(C, phase; direction = 1)

    A, b = matrix_assembely(C, phase, direction)

    # A_gpu = CuSparseMatrixCSR(A)
    # b_gpu = CuVector(b)

    # x_gpu, stats_gpu = minres_qlp(A_gpu, b_gpu, atol=1e-12, rtol=1e-12, itmax=1000)
    x, stats = minres_qlp(A, b, atol=1e-12, rtol=1e-12, itmax=1000)
    # x = Array(x_gpu)
    ϕ = reshape(x, size(C))
    τ, Deff, Qin, Qout = calculate_tortuosity(C, ϕ, phase; direction=1)

    return τ, Deff, Qin, Qout
end
