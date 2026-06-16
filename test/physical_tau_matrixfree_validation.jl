using Printf
using Test
using SparseArrays
using LinearAlgebra

include(joinpath(@__DIR__, "..", "src", "utils.jl"))
include(joinpath(@__DIR__, "..", "src", "percolation.jl"))
include(joinpath(@__DIR__, "..", "src", "tortuosity_physical.jl"))
include(joinpath(@__DIR__, "..", "src", "tortuosity_physical_matrixfree.jl"))

const BG = 0
const PHASE = 1

function straight_channel_geometry(dims, dir)
    C = fill(BG, dims)
    if dir == 1
        C[:, 2:4, 2:5] .= PHASE
    elseif dir == 2
        C[2:4, :, 2:5] .= PHASE
    else
        C[2:4, 2:5, :] .= PHASE
    end
    return C
end

function blocked_channel_geometry(dims, dir)
    C = straight_channel_geometry(dims, dir)
    middle = cld(dims[dir], 2)
    if dir == 1
        C[middle, :, :] .= BG
    elseif dir == 2
        C[:, middle, :] .= BG
    else
        C[:, :, middle] .= BG
    end
    return C
end

function staggered_barrier_geometry(n)
    C = ones(Int8, n, n, n)
    for (barrier, x) in enumerate((div(n, 4) + 1, div(n, 2) + 1, div(3n, 4) + 1))
        C[x, :, :] .= BG
        if isodd(barrier)
            C[x, 1:div(n, 2), :] .= PHASE
        else
            C[x, (div(n, 2) + 1):n, :] .= PHASE
        end
    end
    return C
end

function validate_matrixfree_physical_tortuosity()
    println("Matrix-free physical tortuosity validation")
    println("="^80)

    @testset "Matrix-free physical tortuosity" begin
        dims = (8, 9, 10)
        for dir in 1:3
            full = ones(Int8, dims)
            full_info = physical_tortuosity_matrixfree(
                full,
                PHASE,
                dir;
                voxel_size=1.0,
                return_info=true,
            )
            @printf(
                "full phase dir=%d        tau=%10.7f iterations=%4d residual=%8.2e\n",
                dir,
                full_info.tau,
                full_info.iterations,
                full_info.relative_residual,
            )
            @test full_info.converged
            @test isapprox(full_info.tau, 1.0; atol=1e-10)

            channel = straight_channel_geometry(dims, dir)
            channel_tau = physical_tortuosity_matrixfree(
                channel,
                PHASE,
                dir;
                voxel_size=1.0,
            )
            @test isapprox(channel_tau, 1.0; atol=1e-10)

            blocked = blocked_channel_geometry(dims, dir)
            @test isinf(physical_tortuosity_matrixfree(blocked, PHASE, dir))
        end

        for n in (20, 30, 40)
            C = staggered_barrier_geometry(n)
            direct = physical_tortuosity(C, PHASE, 1; voxel_size=1.0)
            matrixfree = physical_tortuosity_matrixfree(
                C,
                PHASE,
                1;
                voxel_size=1.0,
                rtol=1e-10,
                return_info=true,
            )
            relative_error = abs(matrixfree.tau - direct) / direct
            @printf(
                "staggered %2d^3           tau=%10.7f iterations=%4d error=%8.2e\n",
                n,
                matrixfree.tau,
                matrixfree.iterations,
                relative_error,
            )
            @test matrixfree.converged
            @test matrixfree.relative_residual <= 1e-10
            @test relative_error <= 1e-8
        end

        C = staggered_barrier_geometry(20)
        reference = physical_tortuosity_matrixfree(C, PHASE, 1; voxel_size=1.0)
        scaled = physical_tortuosity_matrixfree(
            C,
            PHASE,
            1;
            voxel_size=0.25,
            D0=3.5,
        )
        @test isapprox(scaled, reference; rtol=1e-10)

        serial = physical_tortuosity_matrixfree(
            C,
            PHASE,
            1;
            rtol=1e-10,
            threaded=false,
        )
        threaded = physical_tortuosity_matrixfree(
            C,
            PHASE,
            1;
            rtol=1e-10,
            threaded=true,
            thread_threshold=1,
        )
        @test isapprox(threaded, serial; rtol=1e-12)

        @test_throws ErrorException physical_tortuosity_matrixfree(C, PHASE, 4)
        @test_throws ErrorException physical_tortuosity_matrixfree(C, PHASE, 1; rtol=0.0)
        @test_throws ErrorException physical_tortuosity_matrixfree(C, PHASE, 1; maxiter=0)
        @test_throws ErrorException physical_tortuosity_matrixfree(C, PHASE, 1; thread_threshold=0)
    end

    println("="^80)
    println("All matrix-free physical tortuosity checks passed.")
end

validate_matrixfree_physical_tortuosity()
