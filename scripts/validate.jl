using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using SOCFeatureExtraction
using Random
using Test

function straight_channel(n=12)
    C = fill(2, n, n, n)
    C[:, 5:8, 5:8] .= 1
    C[:, 1:2, :] .= 3
    return C
end

function single_block(n=10)
    C = fill(1, n, n, n)
    C[:, 1:div(n, 2), :] .= 2
    C[:, :, 1:2] .= 3
    return C
end

function layered(n=9)
    C = fill(1, n, n, n)
    C[div(n, 3)+1:2 * div(n, 3), :, :] .= 2
    C[2 * div(n, 3)+1:end, :, :] .= 3
    return C
end

@testset "SOCFeatureExtraction validation" begin
    C = straight_channel()
    vf = volume_fractions(C, (1, 2, 3))
    @test vf[1] > 0
    @test isapprox(sum(values(vf)), 1.0; atol=1e-12)
    @test percolation_result(C .== 1, 1).fraction == 1.0
    @test isapprox(geometric_tortuosity(C, 1, 1), 1.0; atol=1e-12)
    @test physical_tortuosity(C, 1, 1) >= 1.0

    B = single_block()
    @test volume_fractions(B, (1, 2, 3))[2] > 0
    @test percolation_result(B .== 2, 1).fraction == 1.0
    @test total_tpb_density(B) >= 0

    L = layered()
    @test percolation_result(L .== 1, 1).fraction == 0.0
    @test isinf(geometric_tortuosity(L, 1, 1))

    cfg = Config(Nrays=200, rng_seed=4)
    rows = extract_features(C; sample="straight_channel", config=cfg)
    @test length(rows) == 4
    chords = mean_chord_lengths(fill(1, 8, 8, 8), (1,);
        Nrays=20, min_chord=1, rng=Random.MersenneTwister(2))
    @test !isnan(chords[1])
end

println("Validation checks passed.")
