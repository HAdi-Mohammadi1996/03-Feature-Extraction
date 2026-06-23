using Test

include(joinpath(@__DIR__, "..", "src", "load_feature_extraction.jl"))
include(joinpath(@__DIR__, "..", "src", "tortuosity_physical_gpu.jl"))
include(joinpath(@__DIR__, "..", "src", "feature_extraction_hybrid.jl"))

function hybrid_validation_geometry()
    n = 24
    C = fill(Int8(1), n, n, n)
    C[9:12, :, :] .= 2
    C[17:24, :, :] .= 3

    for x in (6, 15)
        C[x, 1:10, :] .= 3
        C[x, 11:24, :] .= 1
    end
    return C
end

function validate_hybrid_features()
    CUDA.functional() || error("CUDA is not functional")
    config = FeatureConfig(
        voxel_size=1.0,
        Nrays=300,
        min_chord=0.0,
        phases=[1, 2, 3],
        use_all_directions=false,
        direction=1,
        physical_rtol=1e-10,
        physical_threaded=true,
        physical_thread_threshold=1,
        physical_max_threads=4,
        recalculate_static_phase_each_timestep=false,
        static_phase=2,
        verify_static_phase=true,
        rng_seed=1,
    )

    warm_up_hybrid_gpu(config)
    reference = hybrid_validation_geometry()
    changed = copy(reference)
    changed[1:2, 1:8, :] .= 3
    changed[23:24, 1:8, :] .= 1

    hybrid_reference = extract_features_hybrid(
        reference;
        sample="reference",
        config=config,
    )
    cpu_reference = extract_features(
        reference;
        sample="reference",
        config=config,
    )

    hybrid_changed = extract_features_hybrid(
        changed;
        sample="changed",
        config=config,
        phase_cache=hybrid_reference.phase_cache,
    )
    cpu_changed = extract_features(
        changed;
        sample="changed",
        config=config,
        phase_cache=hybrid_reference.phase_cache,
    )

    @testset "Hybrid CPU/GPU feature extraction" begin
        for (hybrid, cpu) in (
            (hybrid_reference.row, cpu_reference),
            (hybrid_changed.row, cpu_changed),
        )
            @test propertynames(hybrid) == propertynames(cpu)
            for name in propertynames(cpu)
                name == :sample && continue
                @test isapprox(
                    getproperty(hybrid, name),
                    getproperty(cpu, name);
                    rtol=1e-10,
                    atol=1e-12,
                    nans=true,
                )
            end
        end
    end
end

validate_hybrid_features()
