@inline function is_tpb(a, b, c, d, phases)
    return all(p -> a == p || b == p || c == p || d == p, phases)
end

function total_tpb_density(C; spacing=(1.0, 1.0, 1.0))
    ndims(C) == 3  || error("The volume must be 3D")
    phases = unique(C)
    length(phases) == 3 || error("The volume must have excatly 3 phases")

    Nx, Ny, Nz = size(C)
    tpb_x = tpb_y = tpb_z = 0

    for k in 1:(Nz-1), j in 1:(Ny-1)
        @inbounds @simd for i in 1:Nx
            a = C[i, j, k]
            b = C[i, j+1, k]
            c = C[i, j, k+1]
            d = C[i, j+1, k+1]

            tpb_x += is_tpb(a, b, c, d, phases)
        end
    end

    for k in 1:(Nz-1), j in 1:Ny
        @inbounds @simd for i in 1:(Nx-1)
            a = C[i, j, k]
            b = C[i+1, j, k]
            c = C[i, j, k+1]
            d = C[i+1, j, k+1]

            tpb_y += is_tpb(a, b, c, d, phases)
        end
    end

    for k in 1:Nz, j in 1:(Ny-1)
        @inbounds @simd for i in 1:(Nx-1)
            a = C[i, j, k]
            b = C[i+1, j, k]
            c = C[i, j+1, k]
            d = C[i+1, j+1, k]

            tpb_z += is_tpb(a, b, c, d, phases)
        end
    end

    return (tpb_x * spacing[1] + tpb_y * spacing[2] + tpb_z * spacing[3]) / (length(C) * prod(spacing))

end

