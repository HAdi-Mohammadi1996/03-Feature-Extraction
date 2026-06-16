using Printf
using Test
using SparseArrays
using LinearAlgebra

include(joinpath(@__DIR__, "..", "src", "utils.jl"))
include(joinpath(@__DIR__, "..", "src", "percolation.jl"))
include(joinpath(@__DIR__, "..", "src", "tortuosity_physical.jl"))

const BG = 0
const PHASE = 1

function straight_channel_geometry(dims, dir; phase=PHASE, bg=BG)
    C = fill(bg, dims)
    transverse = [axis for axis in 1:3 if axis != dir]
    ranges = Dict(
        transverse[1] => 2:min(4, dims[transverse[1]]),
        transverse[2] => 2:min(5, dims[transverse[2]]),
    )

    if dir == 1
        C[:, ranges[2], ranges[3]] .= phase
    elseif dir == 2
        C[ranges[1], :, ranges[3]] .= phase
    else
        C[ranges[1], ranges[2], :] .= phase
    end
    return C
end

function blocked_channel_geometry(dims, dir; phase=PHASE, bg=BG)
    C = straight_channel_geometry(dims, dir; phase=phase, bg=bg)
    middle = cld(dims[dir], 2)
    if dir == 1
        C[middle, :, :] .= bg
    elseif dir == 2
        C[:, middle, :] .= bg
    else
        C[:, :, middle] .= bg
    end
    return C
end

function validate_physical_tortuosity()
    println("Physical tortuosity validation")
    println("="^80)
    println("Transverse boundaries: non-periodic, zero flux")
    println("Transport boundaries: Dirichlet with half-cell conductance")
    println("Linear solver: sparse Cholesky")
    println("-"^80)

    @testset "Physical tortuosity" begin
        for dir in 1:3
            dims = (8, 9, 10)

            C = ones(Int, dims)
            tau = physical_tortuosity(C, PHASE, dir; voxel_size=1.0, D0=1.0)
            @printf("full phase dir=%d             tau=%10.6f expected=%10.6f\n",
                dir, tau, 1.0)
            @test isapprox(tau, 1.0; atol=1e-10)

            C = straight_channel_geometry(dims, dir)
            tau = physical_tortuosity(C, PHASE, dir; voxel_size=1.0, D0=1.0)
            @printf("straight channel dir=%d       tau=%10.6f expected=%10.6f\n",
                dir, tau, 1.0)
            @test isapprox(tau, 1.0; atol=1e-10)

            C = blocked_channel_geometry(dims, dir)
            tau = physical_tortuosity(C, PHASE, dir; voxel_size=1.0, D0=1.0)
            @printf("blocked channel dir=%d        tau=%10s expected=%10s\n",
                dir, string(tau), "Inf")
            @test isinf(tau)
        end

        C = straight_channel_geometry((8, 9, 10), 1)
        reference = physical_tortuosity(C, PHASE, 1; voxel_size=1.0, D0=1.0)
        scaled = physical_tortuosity(C, PHASE, 1; voxel_size=0.25, D0=3.5)
        @printf("parameter scaling            tau=%10.6f reference=%10.6f\n",
            scaled, reference)
        @test isapprox(scaled, reference; atol=1e-10)

        @test_throws ErrorException physical_tortuosity(C, PHASE, 4)
        @test_throws ErrorException physical_tortuosity(C, PHASE, 1; voxel_size=0.0)
        @test_throws ErrorException physical_tortuosity(C, PHASE, 1; D0=0.0)
    end

    println("="^80)
    println("All physical tortuosity validation checks passed.")
end

validate_physical_tortuosity()
