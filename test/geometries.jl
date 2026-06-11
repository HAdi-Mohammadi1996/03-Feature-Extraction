# Shapes from TauFactor surface-area notebook (20 x 20 x 20 grid)
# https://taufactor.readthedocs.io/en/latest/notebooks/02-surface-areas.html

const NX, NY, NZ = 20, 20, 20

function make_cube()
    C = zeros(Int, NX, NY, NZ)
    C[6:15, 6:15, 6:15] .= 1
    return C
end

function rotate_z_nn(A::AbstractArray{T,3}, angle_deg::Real) where T
    nx, ny, nz = size(A)
    out = zeros(T, nx, ny, nz)
    θ = deg2rad(angle_deg)
    c, s = cos(θ), sin(θ)
    cx, cy = nx / 2 + 0.5, ny / 2 + 0.5
    for k in 1:nz, j in 1:ny, i in 1:nx
        x, y = i - cx, j - cy
        is = round(Int, c * x + s * y + cx)
        js = round(Int, -s * x + c * y + cy)
        if 1 <= is <= nx && 1 <= js <= ny
            out[i, j, k] = A[is, js, k]
        end
    end
    return out
end

function make_cube_rot()
    return rotate_z_nn(make_cube(), 45.0)
end

function make_multicube()
    C = zeros(Int, NX, NY, NZ)
    C[1:10, 1:10, 1:5] .= 1
    C[6:15, 6:15, 6:15] .= 2
    C[1:10, 1:10, 16:NZ] .= 1
    C[11:NX, 6:NY, 16:NZ] .= 3
    return C
end

function make_diagonal()
    C = zeros(Int, NX, NY, NZ)
    for i in 1:NX, j in 1:NY, k in 1:NZ
        plane = (i - 0.5) / NX + (j - 0.5) / NY + (k - 0.5) / NZ
        if plane <= 1.05 || plane > 2.0
            C[i, j, k] = 1
        end
    end
    return C
end

function make_sphere()
    C = zeros(Int, NX, NY, NZ)
    r = min(NX, NY, NZ) * 0.5 - 3
    r² = r^2
    for i in 1:NX, j in 1:NY, k in 1:NZ
        d² = (i - NX / 2 + 0.5)^2 + (j - NY / 2 + 0.5)^2 + (k - NZ / 2 + 0.5)^2
        C[i, j, k] = d² <= r² ? 1 : 0
    end
    return C
end

const SHAPE_NAMES = [:cube, :cube_rot, :multicube, :diagonal, :sphere]

const GEOMETRIES = Dict(
    :cube => make_cube,
    :cube_rot => make_cube_rot,
    :multicube => make_multicube,
    :diagonal => make_diagonal,
    :sphere => make_sphere,
)

# TauFactor gradient method with smoothing (reference values)
const TAUFACTOR_REFERENCE = Dict(
    :cube => 0.06547,
    :cube_rot => 0.07063,
    :multicube => 0.04587,
    :diagonal => 0.08502,
    :sphere => 0.07770,
)
