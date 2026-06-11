using SparseArrays
using LinearAlgebra
import LinearSolve as LS

function tau_physical_hadi(C, phase; dir=1, voxel_size=0.1, D0=1.0)
    dims = size(C)
    ϵ_phase = count(==(phase), C) / length(C)
    ϵ_phase == 0 && return Inf
    
    D = C .== phase
    De = zeros(Float64, dims)
    Dw = zeros(Float64, dims)
    Dn = zeros(Float64, dims)
    Ds = zeros(Float64, dims)
    Dt = zeros(Float64, dims)
    Db = zeros(Float64, dims)
    Ce = zeros(Float64, dims)
    Cw = zeros(Float64, dims)
    Cn = zeros(Float64, dims)
    Cs = zeros(Float64, dims)
    Ct = zeros(Float64, dims)
    Cb = zeros(Float64, dims)
    f = zeros(Float64, dims)

    for i in 1:dims[1]
        for j in 1:dims[2]
            for k in 1:dims[3]
                i == dims[1] ? De[i, j, k] = 0.0 : De[i, j, k] = min(D[i, j, k], D[i+1, j, k])
                i == 1       ? Dw[i, j, k] = 0.0 : Dw[i, j, k] = min(D[i, j, k], D[i-1, j, k])
                j == dims[2] ? Dn[i, j, k] = 0.0 : Dn[i, j, k] = min(D[i, j, k], D[i, j+1, k])
                j == 1       ? Ds[i, j, k] = 0.0 : Ds[i, j, k] = min(D[i, j, k], D[i, j-1, k])
                k == dims[3] ? Dt[i, j, k] = 0.0 : Dt[i, j, k] = min(D[i, j, k], D[i, j, k+1])
                k == 1       ? Db[i, j, k] = 0.0 : Db[i, j, k] = min(D[i, j, k], D[i, j, k-1])
            end
        end
    end

    # effect of Drichlet boundary conditions
    if dir == 1
        Ce[end, :, :] .= 2 * D[end, :, :]
        Cw[1, :, :] .= 2 * D[1, :, :]
        f[1, :, :] .= 2.0 * D[1, :, :] * 1.0
        f[end, :, :] .= 2.0 * D[end, :, :] * 0.0
    elseif dir == 2
        Cn[:, end, :] .= 2 * D[:, end, :]
        Cs[:, 1, :] .= 2 * D[:, 1, :]
        f[:, 1, :] .= 2.0 * D[:, 1, :] * 1.0
        f[:, end, :] .= 2.0 * D[:, end, :] * 0.0
    else
        Ct[:, :, end] .= 2 * D[:, :, end]
        Cb[:, :, 1] .= 2 * D[:, :, 1]
        f[:, :, 1] .= 2.0 * D[:, :, 1] * 1.0
        f[:, :, end] .= 2.0 * D[:, :, end] * 0.0
    end

    A = spdiagm(0 => vec(-(De + Dw + Dn + Ds + Dt + Db + Ce + Cw + Cn + Cs + Ct + Cb)), 
                1 => vec(De[1:end-1, :, :]), -1 => vec(Dw[2:end, :, :]), 
                dims[1] => vec(Dn[:, 1:end-1, :]), -dims[1] => vec(Ds[:, 2:end, :]), 
                dims[1]*dims[2] => vec(Dt[:, :, 1:end-1]), -dims[1]*dims[2] => vec(Db[:, :, 2:end]))
    dropzeros(A)
    b = vec(f)
    prob = LS.LinearProblem(A, b)
    sol = LS.solve(prob, LS.UMFPACKFactorization())
    ϕ = sol.u
    return ϕ

end